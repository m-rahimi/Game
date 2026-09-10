import os
import sys

os.environ.setdefault("KIVY_NO_ARGS", "1")

from jack_gui.config import parse_args

if __name__ == "__main__":
    config = parse_args(sys.argv[1:])
    from jack_gui.main import run_app

    run_app(config)
