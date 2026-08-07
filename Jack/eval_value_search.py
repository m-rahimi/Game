"""Head-to-head: value-guided search (Team 0) vs plain heuristic-rollout search (Team 1).

Both use the same rollout budget so the comparison is apples-to-apples. Reports
win rate, score diff, Jack capture, and average decision time for each side.
"""
from __future__ import annotations

import argparse
from random import Random
from time import monotonic

from jack_game.cards import team_for
from jack_game.engine import new_round, play_card
from jack_game.players import SearchPlayer
from jack_game.value import ValueFunction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-checkpoint", default="checkpoints/value_heuristic.pt")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=0.5)
    ap.add_argument("--rollout-depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    value_fn = ValueFunction(args.value_checkpoint)
    rng = Random(args.seed)

    wins = diff = 0
    jacks0 = jacks1 = 0
    move_time = {0: 0.0, 1: 0.0}
    move_count = {0: 0, 1: 0}

    for _ in range(args.games):
        value_search = SearchPlayer(
            rollouts_per_action=args.rollouts, time_limit_seconds=args.time_limit,
            value_fn=value_fn, rollout_depth=args.rollout_depth,
        )
        plain_search = SearchPlayer(
            rollouts_per_action=args.rollouts, time_limit_seconds=args.time_limit,
        )
        players = [value_search, plain_search, value_search, plain_search]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            t0 = monotonic()
            card = players[actor].choose_card(state.clone(), actor, rng)
            move_time[team_for(actor)] += monotonic() - t0
            move_count[team_for(actor)] += 1
            play_card(state, actor, card)
        wins += int(state.team_score(0) > state.team_score(1))
        diff += state.team_score(0) - state.team_score(1)
        for r in state.completed_tricks:
            for c in r.penalties:
                if team_for(r.winner) == 0:
                    jacks0 += 1
                else:
                    jacks1 += 1

    n = args.games
    print(f"value-checkpoint: {args.value_checkpoint}")
    print(f"settings: {args.games} games, rollouts={args.rollouts}, time_limit={args.time_limit}s, depth={args.rollout_depth}")
    print(f"value-search (team0) win rate vs plain search: {wins/n:.3f}")
    print(f"value-search score diff vs plain search:       {diff/n:+.3f}")
    print(f"penalty Jacks captured/game: value-search {jacks0/n:.2f}  plain {jacks1/n:.2f}")
    print(f"avg decision time: value-search {1000*move_time[0]/max(1,move_count[0]):.1f}ms  "
          f"plain {1000*move_time[1]/max(1,move_count[1]):.1f}ms")


if __name__ == "__main__":
    main()
