#!/usr/bin/env bash
# Launch Jack5 with Kivy graphics — same distilled policy as play_new.sh.
#
# Usage: ./play_gui.sh          (you are player 0)
#        ./play_gui.sh 2        (sit at seat 2 instead)
#
# Override engine via environment:
#   AI=heuristic ./play_gui.sh
#   AI=ismcts SIMULATIONS=400 ./play_gui.sh
#
# Torch required for: policy, ismcts, bc, dmc, dmc_guarded, search_value

set -euo pipefail
cd "$(dirname "$0")"

AI="${AI:-policy}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-checkpoints/policy_distill800_l3.pt}"
SIMULATIONS="${SIMULATIONS:-400}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-checkpoints/value_search_ctx.pt}"
BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-checkpoints/belief_search40k_cal.pt}"
ROLLOUTS="${ROLLOUTS:-24}"
ENV_NAME="${ENV_NAME:-game}"
SEAT="${1:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Error: conda env '$ENV_NAME' not found."
  echo "Create it with: cd ../11game && ./setup_conda_env.sh"
  exit 1
fi

conda activate "$ENV_NAME"

if ! python -c "import kivy" >/dev/null 2>&1; then
  echo "Error: Kivy is not installed in the '$ENV_NAME' conda env."
  echo "Install with: cd ../11game && ./setup_conda_env.sh"
  exit 1
fi

TORCH_AIS="policy ismcts bc dmc dmc_guarded search_value"
if [[ " $TORCH_AIS " == *" $AI "* ]] && ! python -c "import torch" >/dev/null 2>&1; then
  echo "Error: AI=$AI requires PyTorch, which is not installed."
  echo "Install with: python -m pip install torch"
  echo "Or use: AI=heuristic ./play_gui.sh"
  exit 1
fi

export KIVY_NO_ARGS=1

echo "Engine: $AI (player seat $SEAT)"
ARGS=(
  --ai "$AI"
  --player "$SEAT"
  --policy-checkpoint "$POLICY_CHECKPOINT"
  --simulations "$SIMULATIONS"
  --rollouts "$ROLLOUTS"
)
if [[ "$AI" == "ismcts" ]]; then
  ARGS+=(
    --value-checkpoint "$VALUE_CHECKPOINT"
    --belief-checkpoint "$BELIEF_CHECKPOINT"
  )
fi
python -m jack_gui "${ARGS[@]}"
