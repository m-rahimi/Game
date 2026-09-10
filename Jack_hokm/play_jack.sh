#!/bin/bash
# Play Jack4 — asks which variant you want, then loads the matching model.
#
#   classic (no trump) -> checkpoints/student_ens_l3.pt      (572-dim, current best)
#   Hokm    (trump)    -> checkpoints/student_hokm_l3.pt     (580-dim, trump-aware)
#
# The two are DIFFERENT GAMES with different feature widths, so a classic net cannot
# play Hokm at all — hence two checkpoints and the prompt.
#
# Usage: ./play_jack.sh        (you are player 0)
#        ./play_jack.sh 2      (sit at seat 2)
# In-game: type a card like JC / 10H / AS, 'hand' to reprint your hand,
# 'flag' to mark the last AI move as a suspected blunder.
# Review afterwards: python analyze_blunders.py --log data/my_games_jack.json
#
# Note: one run is a single 13-trick round, not a match to 21.
cd "$(dirname "$0")"
SEAT="${1:-0}"
CLASSIC=checkpoints/student_ens_l3.pt
HOKM=checkpoints/student_hokm_l3.pt

if [ ! -f "$HOKM" ]; then
  echo "note: $HOKM does not exist yet (Hokm model still training)."
  echo "      Choosing Hokm will fail until it does; classic play is unaffected."
  echo
fi

python -m jack_game play --ai policy \
  --ask-hokm \
  --policy-checkpoint "$CLASSIC" \
  --hokm-policy-checkpoint "$HOKM" \
  --player "$SEAT" --log data/my_games_jack.json
