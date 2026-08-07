from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Jack5 project root (parent of jack_gui/)."""
    return Path(__file__).resolve().parent.parent


def card_png_folder() -> Path:
    return project_root() / "png"


def card_back_png() -> str:
    return str(card_png_folder() / "blue.png")


def icon_png() -> str:
    return str(card_png_folder() / "icon.png")


def card_background_png() -> str:
    return str(project_root() / "card_background.png")


def card_sound_mp3() -> str:
    return str(project_root() / "sounds" / "card.mp3")
