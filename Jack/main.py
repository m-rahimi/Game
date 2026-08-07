"""Android / desktop entry point for the Jack5 Kivy app."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KIVY_NO_ARGS", "1")


def _default_config():
    from pathlib import Path

    from kivy.utils import platform as kivy_platform

    from jack_gui.config import GuiConfig

    if kivy_platform == "android":
        root = Path(__file__).resolve().parent
        onnx_path = root / "checkpoints/policy_distill800_l3.onnx"
        return GuiConfig(
            ai="policy",
            human_seat=0,
            policy_checkpoint=str(onnx_path),
            debug_ai=True,
        )
    return None


if __name__ == "__main__":
    config = _default_config()
    if config is None:
        from jack_gui.config import parse_args

        config = parse_args(sys.argv[1:])
    from jack_gui.main import run_app

    run_app(config)
