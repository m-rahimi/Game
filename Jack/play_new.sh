#!/bin/bash
# Play against the NEW distilled model — zero simulations, instant moves.
# The AI is the greedy policy net distilled from ISMCTS@800 visit distributions
# (checkpoints/policy_distill800_l3.pt): ~1 ms/move at the strength of the
# net-guided search at ~600 sims (beats plain search 0.533 / +0.36 per game).
#
# Usage: ./play_new.sh          (you are player 0, logs to data/my_games_distill.json)
#        ./play_new.sh 2        (sit at seat 2 instead)
# In-game: type a card like JC / 10H / AS, 'hand' to reprint, 'flag' to mark
# the last AI move as a suspected blunder (review with ./review.sh or
# analyze_blunders.py --log data/my_games_distill.json).
cd "$(dirname "$0")"
SEAT="${1:-0}"
python -m jack_game play --ai policy \
  --policy-checkpoint checkpoints/policy_distill800_l3.pt \
  --player "$SEAT" --log data/my_games_distill.json
