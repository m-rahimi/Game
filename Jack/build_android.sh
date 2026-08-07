#!/usr/bin/env bash
# Build Jack5 as an Android APK using Buildozer.
#
# First-time setup:
#   1. Install Java JDK 17+ (required for Android SDK tools)
#      macOS: brew install --cask temurin@17
#   2. Install native build tools (python-for-android compiles Python/Kivy)
#      macOS: brew install autoconf automake libtool pkgconf cmake
#   3. pip install buildozer cython
#      python3 -m pip install --user buildozer cython
#   4. export PATH="$HOME/Library/Python/3.9/bin:$HOME/.local/bin:$PATH"
#
# Build debug APK (install on connected device):
#   ./build_android.sh
#
# Long builds: run in tmux/screen so closing the terminal doesn't kill it.
#   tmux new -s jack5 './build_android.sh 2>&1 | tee /tmp/jack5-buildozer.log'
#
# Build release AAB (Google Play):
#   ./build_android.sh release
#
# Output: bin/jack5-0.1-arm64-v8a_armeabi-v7a-debug.apk  (debug)
# Notes:
#   - Run only ONE build at a time (parallel builds corrupt .buildozer state).
#   - First build downloads SDK/NDK and can take 30–60+ minutes.
#   - Android runs the PyTorch policy via ONNX (export with export_policy_onnx.py).
#   - Desktop ./play_gui.sh still uses the .pt checkpoint directly with PyTorch.
#   - Re-export after retraining: python export_policy_onnx.py

set -euo pipefail
cd "$(dirname "$0")"

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/libtool/libexec/gnubin:$HOME/Library/Python/3.9/bin:$HOME/.local/bin:$PATH"

if ! /usr/libexec/java_home -V 2>/dev/null | grep -q .; then
  echo "Error: Java JDK not found. Install one first, e.g.:"
  echo "  brew install --cask temurin@17"
  exit 1
fi

export JAVA_HOME="$(/usr/libexec/java_home 2>/dev/null || true)"

# Avoid broken pip cache / partial upgrades during p4a venv bootstrap.
unset PIP_CACHE_DIR

missing_tools=()
for tool in autoconf automake libtool pkg-config cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing_tools+=("$tool")
  fi
done
if ((${#missing_tools[@]} > 0)); then
  echo "Error: missing native build tools: ${missing_tools[*]}"
  echo "Install them first, e.g.:"
  echo "  brew install autoconf automake libtool pkgconf cmake"
  exit 1
fi

if ! command -v buildozer >/dev/null 2>&1; then
  echo "Installing buildozer..."
  python3 -m pip install --user buildozer cython
fi

MODE="${1:-debug}"

if pgrep -f "buildozer.*android" >/dev/null 2>&1; then
  echo "Error: another buildozer android build is already running."
  echo "Wait for it to finish, or stop it first with: pkill -f 'buildozer.*android'"
  echo "Monitor progress: tail -f /tmp/jack5-buildozer.log"
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
