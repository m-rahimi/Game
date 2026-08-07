"""Self-play blunder evaluation for belief-trust configurations.

Plays N logged self-play games (same seeded deals for every config -> paired
comparisons), then runs the true-deal/uniform-belief oracle analysis and prints
the blunder bucket table. Lets us tune calibration/λ without human games.

Workers run nets on CPU (CUDA cannot be used in forked workers; single-row MLP
inference is fast on CPU anyway).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from random import Random
from statistics import mean

import torch

from analyze_blunders import analyze_decision, verdict
from jack_game.engine import new_round, play_card
from jack_game.gamelog import GameLog, replay_decisions

CFG: dict = {}


def build_player():
    torch.set_num_threads(1)
    from jack_game.mcts import ISMCTSPlayer
    from jack_game.policy import PolicyFunction
    from jack_game.value import ValueFunction

    belief_fn = None
    if CFG["belief_checkpoint"]:
        from jack_game.belief import BeliefFunction

        belief_fn = BeliefFunction(CFG["belief_checkpoint"], device="cpu")
    return ISMCTSPlayer(
        value_fn=ValueFunction(CFG["value_checkpoint"], device="cpu"),
        policy_fn=PolicyFunction(CFG["policy_checkpoint"], device="cpu"),
        simulations=CFG["simulations"],
        belief_fn=belief_fn,
        belief_lambda=CFG["belief_lambda"],
        belief_lambda_by_trick=CFG["belief_lambda_by_trick"],
    )


_PLAYER = None


def play_one_game(game_seed: int) -> dict:
    global _PLAYER
    if _PLAYER is None:
        _PLAYER = build_player()
    rng = Random(game_seed)
    state = new_round(seed=game_seed, leader=0)
    record = {"meta": {"label": CFG["label"]}, "hands": [[str(c) for c in h] for h in state.hands],
              "leader": 0, "moves": [], "flags": []}
    while not state.round_over:
        actor = state.current_player
        card, visits, q = _PLAYER.choose_with_stats(state.clone(), actor, rng)
        move = {"seat": actor, "card": str(card), "human": False}
        if visits:
            move["visits"] = {str(i): n for i, n in visits.items()}
        record["moves"].append(move)
        play_card(state, actor, card)
    record["scores"] = [state.team_score(0), state.team_score(1)]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--simulations", type=int, default=1000)
    ap.add_argument("--policy-checkpoint", default="checkpoints/policy_search100k.pt")
    ap.add_argument("--value-checkpoint", default="checkpoints/value_search_ctx.pt")
    ap.add_argument("--belief-checkpoint", default=None)
    ap.add_argument("--belief-lambda", type=float, default=0.5)
    ap.add_argument("--belief-lambda-schedule", default=None,
                    help="'acc' = per-trick top1 from the checkpoint, or 13 comma floats")
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--deals", type=int, default=48)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--log-out", default=None)
    args = ap.parse_args()

    schedule = None
    if args.belief_lambda_schedule == "acc":
        ck = torch.load(args.belief_checkpoint, map_location="cpu", weights_only=False)
        schedule = [float(x) for x in ck["trick_top1"]]
    elif args.belief_lambda_schedule:
        schedule = [float(x) for x in args.belief_lambda_schedule.split(",")]
        assert len(schedule) == 13

    CFG.update(label=args.label, simulations=args.simulations,
               policy_checkpoint=args.policy_checkpoint, value_checkpoint=args.value_checkpoint,
               belief_checkpoint=args.belief_checkpoint, belief_lambda=args.belief_lambda,
               belief_lambda_by_trick=schedule)

    seeds = [args.seed_base + i for i in range(args.games)]
    with Pool(processes=min(args.workers, args.games)) as pool:
        games = pool.map(play_one_game, seeds)
    if args.log_out:
        GameLog(args.log_out).games = games  # not appended; write fresh
        json.dump({"games": games}, open(args.log_out, "w"), indent=1)

    jobs = []
    for gi, game in enumerate(games):
        for idx, state, move in replay_decisions(game):
            if len(state.legal_actions(int(move["seat"]))) < 2:
                continue
            jobs.append((gi, idx, game, args.deals, 1 + 97 * len(jobs)))
    with Pool(processes=48) as pool:
        results = pool.map(analyze_decision, jobs)

    n_games = len(games)
    blunders = [r for r in results if verdict(r, 0.5) != "ok"]
    hr = [r for r in blunders if verdict(r, 0.5) == "HAND-READING"]
    fixable = [r for r in hr if r["trick"] >= 2]
    plain = [r for r in blunders if verdict(r, 0.5) == "PLAIN"]
    total_regret = sum(r["regret_true"] for r in results)
    print(f"\n=== {args.label} ({n_games} games, sims={args.simulations}) ===")
    print(f"decisions={len(results)}  blunders={len(blunders)} ({len(blunders)/len(results):.0%})")
    print(f"fixable HR (trick3+): {len(fixable)} blunders, "
          f"{sum(r['regret_true'] for r in fixable)/n_games:.2f} regret pts/game")
    print(f"plain:                {len(plain)} blunders, "
          f"{sum(r['regret_true'] for r in plain)/n_games:.2f} regret pts/game")
    print(f"TOTAL regret/game: {total_regret/n_games:.2f}   mean/decision: {total_regret/len(results):+.3f}")


if __name__ == "__main__":
    main()
