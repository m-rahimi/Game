"""Greedy policy vs greedy policy head-to-head — the 0-simulation test.

Both sides play the argmax of their policy net over legal moves. No value net,
no search, no determinization — a pure comparison of the two policies as
standalone decision-makers. Seats are swapped every other game; same seeded
deals for both seatings via one RNG stream.

Motivation (2026-07-31): if the 572-dim nets are genuinely better decision-
makers, the advantage must show at ZERO sims, where nothing can wash it out.

Run:
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY eval_greedy_h2h.py \
     --policy-a checkpoints/policy_attr572_g196000.pt \
     --policy-b checkpoints/policy_search100k.pt --games 2000
"""
from __future__ import annotations

import argparse
from math import sqrt
from random import Random

from jack_game.cards import team_for
from jack_game.engine import new_round, play_card
from jack_game.policy import GreedyPolicyPlayer, PolicyFunction  # canonical player lives in the package


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", default="checkpoints/policy_attr572_g196000.pt",
                    help="the tracked side (win rate reported for A)")
    ap.add_argument("--policy-b", default="checkpoints/policy_search100k.pt")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pa = GreedyPolicyPlayer(PolicyFunction(args.policy_a))
    pb = GreedyPolicyPlayer(PolicyFunction(args.policy_b))
    print(f"A: {args.policy_a} (feature_dim={pa.policy_fn.feature_dim})")
    print(f"B: {args.policy_b} (feature_dim={pb.policy_fn.feature_dim})")

    rng = Random(args.seed)
    a_wins = ties = 0
    a_diff = 0
    jacks_a = jacks_b = 0
    for g in range(args.games):
        a_is_team0 = (g % 2 == 0)
        players = [pa, pb, pa, pb] if a_is_team0 else [pb, pa, pb, pa]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            card = players[actor].choose_card(state, actor, rng)
            play_card(state, actor, card)
        a_team = 0 if a_is_team0 else 1
        sa, sb = state.team_score(a_team), state.team_score(1 - a_team)
        a_wins += int(sa > sb)
        ties += int(sa == sb)
        a_diff += sa - sb
        for r in state.completed_tricks:
            for c in r.penalties:
                if (team_for(r.winner) == 0) == a_is_team0:
                    jacks_a += 1
                else:
                    jacks_b += 1
        if (g + 1) % 500 == 0:
            print(f"  {g+1}/{args.games}: A win {a_wins/(g+1):.3f}", flush=True)

    n = args.games
    win = a_wins / n
    se = sqrt(win * (1 - win) / n)
    print(f"\ngames={n} (seats swapped every other game)")
    print(f"A win rate : {win:.3f}  (SE {se:.3f}, z vs 0.5 = {(win-0.5)/se:+.2f}, ties {ties/n:.3f})")
    print(f"A score diff: {a_diff/n:+.3f} / game")
    print(f"penalty Jacks/game: A {jacks_a/n:.2f}  B {jacks_b/n:.2f}")


if __name__ == "__main__":
    main()
