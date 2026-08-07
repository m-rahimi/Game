"""Asymmetric head-to-head: greedy policy (0 sims) vs ISMCTS (N sims).

Measures how much play-time search is actually worth over the best standalone
policy: side A plays pure argmax of its policy net; side B runs full net-guided
ISMCTS. Seats swapped every other game. Win rate reported for A (greedy).

Run:
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY eval_greedy_vs_ismcts.py \
     --greedy-policy checkpoints/policy_attr572_g196000.pt \
     --ismcts-policy checkpoints/policy_search100k.pt \
     --ismcts-value  checkpoints/value_search_ctx.pt \
     --simulations 200 --games 200
"""
from __future__ import annotations

import argparse
from math import sqrt
from random import Random

from jack_game.cards import team_for
from jack_game.engine import new_round, play_card
from jack_game.mcts import ISMCTSPlayer
from jack_game.policy import PolicyFunction
from jack_game.value import ValueFunction
from eval_greedy_h2h import GreedyPolicyPlayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy-policy", default="checkpoints/policy_attr572_g196000.pt")
    ap.add_argument("--ismcts-policy", default="checkpoints/policy_search100k.pt")
    ap.add_argument("--ismcts-value", default="checkpoints/value_search_ctx.pt")
    ap.add_argument("--simulations", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    greedy = GreedyPolicyPlayer(PolicyFunction(args.greedy_policy))
    pol = PolicyFunction(args.ismcts_policy)
    val = ValueFunction(args.ismcts_value)
    print(f"A (greedy, 0 sims): {args.greedy_policy} (dim={greedy.policy_fn.feature_dim})")
    print(f"B (ISMCTS, {args.simulations} sims): {args.ismcts_policy} + {args.ismcts_value} "
          f"(dim={pol.feature_dim})")

    rng = Random(args.seed)
    a_wins = ties = 0
    a_diff = 0
    for g in range(args.games):
        ismcts = ISMCTSPlayer(value_fn=val, policy_fn=pol,
                              simulations=args.simulations, c_puct=args.c_puct)
        a_is_team0 = (g % 2 == 0)
        players = [greedy, ismcts, greedy, ismcts] if a_is_team0 else [ismcts, greedy, ismcts, greedy]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            card = players[actor].choose_card(state.clone(), actor, rng)
            play_card(state, actor, card)
        a_team = 0 if a_is_team0 else 1
        sa, sb = state.team_score(a_team), state.team_score(1 - a_team)
        a_wins += int(sa > sb)
        ties += int(sa == sb)
        a_diff += sa - sb
        if (g + 1) % 50 == 0:
            print(f"  {g+1}/{args.games}: greedy win {a_wins/(g+1):.3f}", flush=True)

    n = args.games
    win = a_wins / n
    se = sqrt(max(win * (1 - win), 1e-9) / n)
    print(f"\ngames={n} (seats swapped)")
    print(f"greedy (A) win rate : {win:.3f}  (SE {se:.3f}, ties {ties/n:.3f})")
    print(f"greedy (A) score diff: {a_diff/n:+.3f} / game")


if __name__ == "__main__":
    main()
