"""Evaluate the greedy search-distilled policy net (instant, no search) vs an opponent."""
from __future__ import annotations

import argparse
from random import Random
from time import monotonic

from jack_game.cards import team_for
from jack_game.engine import new_round, play_card
from jack_game.players import HeuristicPlayer, SearchPlayer
from scaling_curve import GreedyPolicyPlayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-checkpoint", default="checkpoints/policy_search.pt")
    ap.add_argument("--opponent", choices=["search", "heuristic"], default="search")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--search-rollouts", type=int, default=24)
    ap.add_argument("--search-time-limit", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    greedy = GreedyPolicyPlayer(args.policy_checkpoint)
    opp = (HeuristicPlayer() if args.opponent == "heuristic"
           else SearchPlayer(rollouts_per_action=args.search_rollouts, time_limit_seconds=args.search_time_limit))
    rng = Random(args.seed)

    wins = diff = jacks0 = jacks1 = 0
    for _ in range(args.games):
        players = [greedy, opp, greedy, opp]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            play_card(state, actor, players[actor].choose_card(state.clone(), actor, rng))
        wins += int(state.team_score(0) > state.team_score(1))
        diff += state.team_score(0) - state.team_score(1)
        for r in state.completed_tricks:
            for c in r.penalties:
                if team_for(r.winner) == 0:
                    jacks0 += 1
                else:
                    jacks1 += 1
    n = args.games
    print(f"greedy policy ({args.policy_checkpoint}) vs {args.opponent}, {n} games")
    print(f"  win rate:   {wins/n:.3f}")
    print(f"  score diff: {diff/n:+.3f}")
    print(f"  penalty Jacks/game: policy {jacks0/n:.2f}  {args.opponent} {jacks1/n:.2f}")


if __name__ == "__main__":
    main()
