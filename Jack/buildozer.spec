[app]

title = Jack5
package.name = jack5
package.domain = com.mohammadrahimi

source.dir = .
source.main = main.py

source.include_exts = py,png,jpg,jpeg,kv,atlas,mp3,json,onnx,data

source.include_dirs = png,jack_game,jack_gui,checkpoints

source.exclude_dirs = data,.buildozer,bin,.git

source.exclude_patterns = analyze_*,eval_*,train_*,gen_*,*_experiment.py,scaling_curve.py,play_vs_search.py,run_exit.py,calibrate_belief.py,scale_attr_curve.py,g11/*,*.sh,data/*,.DS_Store,checkpoints/*.pt

version = 0.1

# numpy for ONNX input arrays; PyTorch model runs via ONNX Runtime on Android.
# Pin Python 3.11 — p4a default (3.14) breaks host venv pip upgrade on macOS.
# Both hostpython3 and python3 must match the same version.
requirements = hostpython3==3.11.9,python3==3.11.9,kivy,pillow,numpy

orientation = landscape

fullscreen = 0

icon.filename = %(source.dir)s/png/icon.png
icon.adaptive_foreground.filename = %(source.dir)s/png/icon.png
icon.adaptive_background.filename = %(source.dir)s/png/icon.png

android.presplash_color = #197C54

android.minapi = 24
android.api = 33
android.archs = arm64-v8a, armeabi-v7a
android.accept_sdk_license = True
android.allow_backup = True

android.gradle_dependencies = com.microsoft.onnxruntime:onnxruntime-android:1.22.0

[buildozer]

log_level = 2
warn_on_root = 1
