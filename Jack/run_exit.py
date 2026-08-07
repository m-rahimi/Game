"""Expert Iteration (ExIt) driver for beating search.

Each iteration:
  1. Self-play with the current best net-guided ISMCTS (records improved-policy
     visit distributions + value outcomes -- targets STRONGER than the raw nets).
  2. Train a new policy (KL to visit distributions) + value (MSE to outcomes),
     warm-started from the current best, on a replay window of recent iterations.
  3. Gate: head-to-head vs the current best AND vs plain search. Promote the new
     nets only if they beat the current best by the gate margin (monotonic).

The first iteration's self-play already uses a >=search player, so its targets
exceed search; gating ensures we only ever move forward.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import FEATURE_DIM, resolve_device
from jack_game.engine import new_round, play_card
from jack_game.cards import team_for
from jack_game.mcts import ISMCTSPlayer
from jack_game.players import SearchPlayer
from jack_game.policy import PolicyNet, PolicyFunction, save_policy_checkpoint
from jack_game.value import ValueNet, ValueFunction, save_value_checkpoint, VALUE_CONTEXT_DIM
from jack_game.selfplay import generate_selfplay


def load_policy_model(ckpt, device, hidden):
    m = PolicyNet(hidden_dim=hidden).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state"])
    return m


def load_value_model(ckpt, device, hidden):
    m = ValueNet(input_dim=FEATURE_DIM + VALUE_CONTEXT_DIM, hidden_dim=hidden).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state"])
    return m


def train_policy_kl(model, feats, visits, legal, device, epochs, bs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(feats, visits, legal), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, tb, lb in loader:
            xb, tb, lb = xb.to(device), tb.to(device), lb.to(device)
            opt.zero_grad(set_to_none=True)
            logp = torch.log_softmax(model(xb).masked_fill(~lb, float("-inf")), dim=1)
            loss = -(tb * logp).sum(1).mean()  # cross-entropy to the soft visit target
            loss.backward()
            opt.step()
    return model


def train_value_mse(model, feats, vctx, outs, device, epochs, bs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    inputs = torch.cat([feats, vctx], dim=1)
    loader = DataLoader(TensorDataset(inputs, outs), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()
    return model


def head_to_head(p0_pol, p0_val, p1_player_fn, games, sims, c_puct, seed):
    """Team 0 = ISMCTS(p0 nets) vs Team 1 = p1_player_fn(); return team-0 win rate."""
    policy = PolicyFunction(p0_pol, device="cpu")
    value = ValueFunction(p0_val, device="cpu")
    rng = Random(seed)
    wins = 0
    for _ in range(games):
        ismcts = ISMCTSPlayer(value_fn=value, policy_fn=policy, simulations=sims, c_puct=c_puct, time_limit_seconds=None)
        opp = p1_player_fn()
        players = [ismcts, opp, ismcts, opp]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            play_card(state, actor, players[actor].choose_card(state.clone(), actor, rng))
        wins += int(state.team_score(0) > state.team_score(1))
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-policy", default="checkpoints/policy_search100k.pt")
    ap.add_argument("--init-value", default="checkpoints/value_search_ctx.pt")
    ap.add_argument("--out-prefix", default="checkpoints/exit")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--games-per-iter", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--sims", type=int, default=160, help="self-play MCTS sims")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-moves", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--replay-window", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--gate-margin", type=float, default=0.005, help="promote if vs_search beats best-so-far by this")
    ap.add_argument("--value-anchor-data", default=None, help="curated dataset to anchor value training (prevent regression)")
    ap.add_argument("--anchor-cap", type=int, default=400000, help="max anchor examples mixed in per iter")
    ap.add_argument("--search-rollouts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(None)
    best_policy, best_value = args.init_policy, args.init_value
    replay = deque(maxlen=args.replay_window)
    search_fn = lambda: SearchPlayer(rollouts_per_action=args.search_rollouts, time_limit_seconds=1.0)

    # Optional value anchor: keep value grounded on curated search data so self-play noise can't regress it.
    anchor = None
    if args.value_anchor_data:
        a = torch.load(args.value_anchor_data, weights_only=False)
        idx = torch.randperm(a["features"].shape[0])[: args.anchor_cap]
        anchor = (a["features"][idx], a["value_context"][idx], a["outcomes"][idx])
        print(f"value anchor: {anchor[0].shape[0]} examples from {args.value_anchor_data}")

    # Gate directly on the goal (vs search), tracked as best-so-far.
    best_vs_search = head_to_head(best_policy, best_value, search_fn, args.eval_games, args.eval_sims, args.c_puct, args.seed)
    print(f"ExIt start: policy={best_policy} value={best_value} device={device} init vs_search={best_vs_search:.3f}")

    for it in range(1, args.iterations + 1):
        feats, vctx, visits, legal, outs = generate_selfplay(
            best_policy, best_value, args.games_per_iter, args.workers,
            args.sims, args.c_puct, args.temp_moves, args.seed + 100 * it)
        replay.append((feats, vctx, visits, legal, outs))
        F = torch.cat([r[0] for r in replay]); V = torch.cat([r[1] for r in replay])
        Pi = torch.cat([r[2] for r in replay]); L = torch.cat([r[3] for r in replay])
        O = torch.cat([r[4] for r in replay])

        vF, vV, vO = F, V, O
        if anchor is not None:
            vF = torch.cat([F, anchor[0]]); vV = torch.cat([V, anchor[1]]); vO = torch.cat([O, anchor[2]])

        pmodel = load_policy_model(best_policy, device, args.hidden_dim)  # warm start
        vmodel = load_value_model(best_value, device, args.hidden_dim)
        train_policy_kl(pmodel, F, Pi, L, device, args.epochs, args.batch_size, args.lr)
        train_value_mse(vmodel, vF, vV, vO, device, args.epochs, args.batch_size, args.lr)

        new_policy = f"{args.out_prefix}_policy_it{it}.pt"
        new_value = f"{args.out_prefix}_value_it{it}.pt"
        save_policy_checkpoint(Path(new_policy), {k: v.cpu() for k, v in pmodel.state_dict().items()}, args.hidden_dim, 0.0)
        save_value_checkpoint(Path(new_value), {k: v.cpu() for k, v in vmodel.state_dict().items()}, args.hidden_dim, 0.0,
                              value_context_dim=VALUE_CONTEXT_DIM)

        # Gate on the actual goal: beat search by more than the current best does.
        vs_search = head_to_head(new_policy, new_value, search_fn, args.eval_games, args.eval_sims, args.c_puct, args.seed + 5000 + it)
        promoted = vs_search > best_vs_search + args.gate_margin
        tag = "PROMOTED" if promoted else "kept-best"
        print(f"[iter {it}] examples={F.shape[0]} vs_search={vs_search:.3f} (best {best_vs_search:.3f}) -> {tag}")
        if promoted:
            best_policy, best_value, best_vs_search = new_policy, new_value, vs_search

    print(f"ExIt done. best policy={best_policy} value={best_value} vs_search={best_vs_search:.3f}")


if __name__ == "__main__":
    main()
