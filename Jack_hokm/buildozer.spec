[app]

title = Jack
package.name = jack
package.domain = com.mohammadrahimi

source.dir = .
source.main = main.py

source.include_exts = py,png,jpg,jpeg,kv,atlas,mp3,json,onnx,data

source.include_dirs = png,jack_game,jack_gui,checkpoints

source.exclude_dirs = data,.buildozer,bin,.git

source.exclude_patterns = analyze_*,eval_*,train_*,gen_*,*_experiment.py,*.sh,data/*,.DS_Store,checkpoints/*.pt

version = 0.1

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
