"""Data-scaling curve: how much search data does distillation need?

Loads a search self-play dataset, trains a policy net and a value net on
increasing subsets, and reports for each size:
  - policy top-1 imitation accuracy (held-out)   "does it pick search's move?"
  - value MSE (held-out)
  - greedy-policy win rate vs heuristic and vs search

Pure distillation can at best *tie* search (it imitates it); the curve shows how
fast it saturates, i.e. the data budget where more search games stop helping.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import resolve_device
from jack_game.engine import new_round, play_round
from jack_game.players import HeuristicPlayer, SearchPlayer
from jack_game.policy import PolicyNet, save_policy_checkpoint, PolicyFunction
from jack_game.value import ValueNet, save_value_checkpoint
from jack_game.cards import team_for


class GreedyPolicyPlayer:
    def __init__(self, checkpoint_path: str):
        self.policy = PolicyFunction(checkpoint_path)

    def choose_card(self, state, player, rng):
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return legal[0]
        priors = self.policy(state, player)
        best_idx = max(priors, key=priors.get)
        return next(c for c in legal if c.index == best_idx)


def train_value(features, outcomes, device, hidden, epochs, bs, lr):
    model = ValueNet(hidden_dim=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(features, outcomes), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def train_policy(features, actions, legal, device, hidden, epochs, bs, lr):
    model = PolicyNet(hidden_dim=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(features, actions, legal), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, ab, lb in loader:
            xb, ab, lb = xb.to(device), ab.to(device), lb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).masked_fill(~lb, float("-inf"))
            loss = nn.functional.cross_entropy(logits, ab)
            loss.backward()
            opt.step()
    return model


def win_rate(team0_player, team1_player, games, seed):
    rng = Random(seed)
    wins = 0
    for _ in range(games):
        players = [team0_player, team1_player, team0_player, team1_player]
        state = play_round(players, seed=rng.randrange(10**9))
        wins += int(state.team_score(0) > state.team_score(1))
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_24k.pt")
    ap.add_argument("--sizes", default="1000,3000,10000,24000", help="games per subset")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--search-rollouts", type=int, default=24)
    ap.add_argument("--search-time-limit", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(None)
    torch.manual_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    feats, acts, legal, outs = data["features"], data["actions"], data["legal"], data["outcomes"]

    # Shuffle once, carve a held-out tail for metrics.
    perm = torch.randperm(feats.shape[0])
    feats, acts, legal, outs = feats[perm], acts[perm], legal[perm], outs[perm]
    n_hold = max(5000, feats.shape[0] // 10)
    hf, ha, hl, ho = (t[:n_hold].to(device) for t in (feats, acts, legal, outs))
    tf, ta, tl, to = feats[n_hold:], acts[n_hold:], legal[n_hold:], outs[n_hold:]

    heuristic = HeuristicPlayer()
    search = SearchPlayer(rollouts_per_action=args.search_rollouts, time_limit_seconds=args.search_time_limit)

    print(f"data={args.data} total_examples={feats.shape[0]} holdout={n_hold} device={device}")
    print(f"{'games':>7} {'pol_acc':>8} {'val_mse':>8} {'vs_heur':>8} {'vs_search':>10}")
    for size in [int(s) for s in args.sizes.split(",")]:
        n = min(size * 52, tf.shape[0])
        pmodel = train_policy(tf[:n], ta[:n], tl[:n], device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
        vmodel = train_value(tf[:n], to[:n], device, args.hidden_dim, args.epochs, args.batch_size, args.lr)

        pmodel.eval(); vmodel.eval()
        with torch.no_grad():
            logits = pmodel(hf).masked_fill(~hl, float("-inf"))
            pol_acc = float((logits.argmax(1) == ha).float().mean())
            val_mse = float(nn.functional.mse_loss(vmodel(hf), ho))

        save_policy_checkpoint(Path(f"checkpoints/scale_policy_{size}.pt"),
                               {k: v.cpu() for k, v in pmodel.state_dict().items()}, args.hidden_dim, 0.0)
        greedy = GreedyPolicyPlayer(f"checkpoints/scale_policy_{size}.pt")
        wr_heur = win_rate(greedy, heuristic, args.eval_games, args.seed + 1)
        wr_search = win_rate(greedy, search, args.eval_games, args.seed + 2)
        print(f"{size:>7} {pol_acc:>8.3f} {val_mse:>8.3f} {wr_heur:>8.3f} {wr_search:>10.3f}")


if __name__ == "__main__":
    main()
