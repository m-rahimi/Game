#!/usr/bin/env bash
# Build Jack as an Android APK using Buildozer.
#
# First-time setup:
#   brew install --cask temurin@17
#   brew install autoconf automake libtool pkgconf cmake
#   python3 -m pip install --user buildozer cython
#
# Export ONNX models (once, or after retraining):
#   python export_policy_onnx.py --checkpoint checkpoints/student_ens_l3.pt --out checkpoints/student_ens_l3.onnx
#   python export_policy_onnx.py --checkpoint checkpoints/student_hokm_iimc_sT035.pt --out checkpoints/student_hokm_iimc_sT035.onnx
#
# Build debug APK:
#   ./build_android.sh
#
# Output: bin/jack-0.1-arm64-v8a_armeabi-v7a-debug.apk

set -euo pipefail
cd "$(dirname "$0")"

# p4a runs `python -m venv venv` without --clear; a half-upgraded pip in an
# existing venv breaks the next build (ImportError: open_rich_spinner).
deactivate 2>/dev/null || true
unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
VENV_DIR=".buildozer/android/platform/build-arm64-v8a_armeabi-v7a/build/venv"
if [[ -d "$VENV_DIR" ]]; then
  echo "Removing stale p4a venv ($VENV_DIR)..."
  rm -rf "$VENV_DIR"
fi

P4A_BUILD=".buildozer/android/platform/python-for-android/pythonforandroid/build.py"
if [[ -f "$P4A_BUILD" ]] && ! grep -q "'venv', '--clear'" "$P4A_BUILD"; then
  echo "Patching p4a to recreate venv with --clear (fixes pip upgrade on 2nd arch)..."
  sed -i '' "s/shprint(host_python, '-m', 'venv', 'venv')/shprint(host_python, '-m', 'venv', 'venv', '--clear')/" "$P4A_BUILD"
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/libtool/libexec/gnubin:$HOME/Library/Python/3.9/bin:$HOME/.local/bin:$PATH"

if ! /usr/libexec/java_home -V 2>/dev/null | grep -q .; then
  echo "Error: Java JDK not found. Install: brew install --cask temurin@17"
  exit 1
fi

export JAVA_HOME="$(/usr/libexec/java_home 2>/dev/null || true)"
unset PIP_CACHE_DIR

missing_tools=()
for tool in autoconf automake libtool pkg-config cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing_tools+=("$tool")
  fi
done
if ((${#missing_tools[@]} > 0)); then
  echo "Error: missing native build tools: ${missing_tools[*]}"
  echo "  brew install autoconf automake libtool pkgconf cmake"
  exit 1
fi

if ! command -v buildozer >/dev/null 2>&1; then
  echo "Installing buildozer..."
  python3 -m pip install --user buildozer cython
fi

CLASSIC_ONNX="checkpoints/student_ens_l3.onnx"
HOKM_ONNX="checkpoints/student_hokm_iimc_sT035.onnx"

if [[ ! -f "$CLASSIC_ONNX" ]] || [[ ! -f "$HOKM_ONNX" ]]; then
  echo "Exporting ONNX models..."
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME:-game}" 2>/dev/null || true
  fi
  if ! python -c "import torch" >/dev/null 2>&1; then
    echo "Error: PyTorch required to export ONNX. Run in conda env 'game' with torch installed."
    exit 1
  fi
  [[ -f "$CLASSIC_ONNX" ]] || python export_policy_onnx.py --checkpoint checkpoints/student_ens_l3.pt --out "$CLASSIC_ONNX"
  [[ -f "$HOKM_ONNX" ]] || python export_policy_onnx.py --checkpoint checkpoints/student_hokm_iimc_sT035.pt --out "$HOKM_ONNX"
fi

MODE="${1:-debug}"

if pgrep -f "buildozer.*android" >/dev/null 2>&1; then
  echo "Error: another buildozer android build is already running."
  exit 1
fi

if [[ "$MODE" == "release" ]]; then
  echo "Building release AAB..."
  buildozer -v android release
else
  echo "Building debug APK..."
  buildozer -v android debug
fi

echo ""
echo "Done. APK/AAB files are in: $(pwd)/bin/"
ls -la bin/ 2>/dev/null || true
