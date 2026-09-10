#!/usr/bin/env bash
# Launch Jack with Kivy graphics — asks Hokm yes/no at startup, then plays to 21.
#
# Usage: ./play_gui.sh          (you are player 0)
#        ./play_gui.sh 2        (sit at seat 2 instead)
#
# Skip the dialog with extra args after --:
#   ./play_gui.sh -- --hokm
#   ./play_gui.sh 2 -- --no-hokm

set -euo pipefail
cd "$(dirname "$0")"

AI="${AI:-policy}"
CLASSIC_CHECKPOINT="${CLASSIC_CHECKPOINT:-checkpoints/student_ens_l3.pt}"
HOKM_CHECKPOINT="${HOKM_CHECKPOINT:-checkpoints/student_hokm_iimc_sT035.pt}"
SIMULATIONS="${SIMULATIONS:-400}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-checkpoints/value_search_ctx.pt}"
BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-checkpoints/belief_search40k_cal.pt}"
ROLLOUTS="${ROLLOUTS:-24}"
ENV_NAME="${ENV_NAME:-game}"
SEAT="${1:-0}"

if [[ "$SEAT" =~ ^[0-3]$ ]]; then
  shift || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Error: conda env '$ENV_NAME' not found."
  exit 1
fi

conda activate "$ENV_NAME"

if ! python -c "import kivy" >/dev/null 2>&1; then
  echo "Error: Kivy is not installed in the '$ENV_NAME' conda env."
  exit 1
fi

TORCH_AIS="policy ismcts bc dmc dmc_guarded search_value"
if [[ " $TORCH_AIS " == *" $AI "* ]] && ! python -c "import torch" >/dev/null 2>&1; then
  echo "Error: AI=$AI requires PyTorch."
  exit 1
fi

export KIVY_NO_ARGS=1

echo "Engine: $AI (player seat $SEAT)"
python -m jack_gui \
  --ai "$AI" \
  --player "$SEAT" \
  --policy-checkpoint "$CLASSIC_CHECKPOINT" \
  --hokm-policy-checkpoint "$HOKM_CHECKPOINT" \
  --simulations "$SIMULATIONS" \
  --rollouts "$ROLLOUTS" \
  --log data/my_games_jack_gui.json \
  "$@"
