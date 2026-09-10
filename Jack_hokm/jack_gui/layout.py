from __future__ import annotations

from kivy.config import Config
from kivy.core.window import Window

DESIGN_WIDTH = int(Config.get("graphics", "width"))
DESIGN_HEIGHT = int(Config.get("graphics", "height"))
CARD_RATIO = 323 / 222
# Tighter packing than 11game default (uses 5 slots); applies to bottom/top hands.
HAND_SPAN_SLOTS = 3.2
HAND_GAP = 0.006
SIDE_STEP_RATIO = 0.14
# Card sizing at the design (original) window; height/window-height ratio stays fixed on resize.
CARD_WIDTH_RATIO = 0.115
CARD_MAX_HEIGHT_RATIO = 0.275

# ---------------------------------------------------------------------------
# Animation & pacing (seconds) — tune these to find your preferred feel.
# Set CARD_ANIM_DURATION to 0 for instant card placement (no fly-in).
# ---------------------------------------------------------------------------
CARD_ANIM_DURATION = 0.25       # 0.25 hand / opponent -> trick (table)
HAND_RELAYOUT_DURATION = 0.2    # 0.2 remaining hand cards slide together
TRICK_COLLECT_DURATION = 0.35  # 0.35 trick -> won-trick pile
AI_MOVE_DELAY = 0.35            # pause before an AI seat plays
TRICK_FINISH_DELAY = 0.7        # 0.7 cards sit on table before collection
ROUND_END_DELAY = 0.6            # 0.6 pause after last trick before round overlay
ROUND_BANNER_DURATION = 1.2       # "Next round..." / tie-break banner
TRUMP_REVEAL_HOLD = 1.0           # center trump image before fly to score area
TRUMP_MOVE_DURATION = 0.45        # animate trump to badge above match icon


def _design_card_height() -> float:
    """Card height at the configured design window (before resize)."""
    card_w = DESIGN_WIDTH * CARD_WIDTH_RATIO
    card_h = card_w * CARD_RATIO
    max_h = DESIGN_HEIGHT * CARD_MAX_HEIGHT_RATIO
    return min(card_h, max_h)


# Keep card length (height) / window height constant at every viewport size.
CARD_HEIGHT_WINDOW_RATIO = _design_card_height() / DESIGN_HEIGHT if DESIGN_HEIGHT else CARD_MAX_HEIGHT_RATIO
# Side hands: edge margin (window-width fraction) + optional nudge toward table center.
SIDE_LEFT_EDGE_X_RATIO = 0.015
SIDE_LEFT_CENTER_INSET_RATIO = 0.06
SIDE_RIGHT_EDGE_X_RATIO = 0.10
SIDE_RIGHT_INSET_RATIO = 0.55
# How far trick cards sit from table center (fraction of card width/height).
TRICK_SLOT_INSET = 0.45

# Layout design toggle: "default" (see layout_saved.py) or "trial" (hands left, piles right).
LAYOUT_DESIGN = "trial"

# --- default (saved in layout_saved.py) ---
_HAND_SHIFT_X_RATIO_DEFAULT = 0.04
# --- trial: shift all hands left ---
_HAND_SHIFT_X_RATIO_TRIAL = -0.08

COLLECT_STACK_X_STEP = 8
COLLECT_STACK_Y_STEP = 6


def border_pad(card_width: int) -> float:
    """One fourth of card width clearance from the window border."""
    return card_width / 4


def display_seat(seat: int, human_seat: int) -> int:
    return (seat - human_seat) % 4


def configure_window() -> None:
    Window.clearcolor = (25 / 255, 124 / 255, 84 / 255, 1)


def viewport_size() -> tuple[float, float]:
    """Current drawable size; prefer Window metrics (tracks resize/rotation)."""
    return float(Window.width), float(Window.height)


def layout_scale() -> float:
    """Uniform scale factor relative to the default Kivy design size."""
    width, height = viewport_size()
    if DESIGN_WIDTH <= 0 or DESIGN_HEIGHT <= 0:
        return 1.0
    return min(width / DESIGN_WIDTH, height / DESIGN_HEIGHT)


def card_size() -> tuple[int, int]:
    _, win_h = viewport_size()
    card_h = win_h * CARD_HEIGHT_WINDOW_RATIO
    card_w = card_h / CARD_RATIO
    return int(card_w), int(card_h)


def seat_zone(seat: int, human_seat: int) -> str:
    zones = ["bottom", "left", "top", "right"]
    return zones[(seat - human_seat) % 4]


FULL_HAND_COUNT = 13


def hand_card_spacing(width: int) -> float:
    """Fixed edge-to-edge gap between hand cards (same spacing at every count)."""
    win_w, _ = viewport_size()
    card_x = win_w * 0.05
    gap = win_w * HAND_GAP
    card_x_max = card_x + (width + gap) * HAND_SPAN_SLOTS
    return (card_x_max - card_x) / (FULL_HAND_COUNT - 1) - width


def _hand_margin(count: int, width: int) -> float:
    """Legacy helper for top hand; player hand uses hand_card_spacing instead."""
    return hand_card_spacing(width)


def _horizontal_row(
    count: int,
    card_w: int,
    card_h: int,
    start_x: float,
    y: float,
) -> list[tuple[float, float]]:
    margin = _hand_margin(count, card_w)
    return [(start_x + (card_w + margin) * idx, y) for idx in range(count)]


def hand_shift_x() -> float:
    ratio = _HAND_SHIFT_X_RATIO_TRIAL if LAYOUT_DESIGN == "trial" else _HAND_SHIFT_X_RATIO_DEFAULT
    return viewport_size()[0] * ratio


def player_hand_layout(count: int) -> tuple[tuple[int, int], list[tuple[float, float]]]:
    win_w, win_h = viewport_size()
    width, height = card_size()
    spacing = hand_card_spacing(width)
    step = width + spacing
    total_width = width + max(count - 1, 0) * step
    start_x = (win_w - total_width) / 2 + hand_shift_x()
    y = win_h * 0.05
    positions = [(start_x + step * idx, y) for idx in range(count)]
    return (width, height), positions


def top_hand_layout(count: int) -> tuple[tuple[int, int], list[tuple[float, float]]]:
    win_w, win_h = viewport_size()
    width, height = card_size()
    total_width = width + max(count - 1, 0) * (width + _hand_margin(count, width))
    start_x = (win_w - total_width) / 2 + hand_shift_x()
    y = win_h * 0.95 - height
    return (width, height), _horizontal_row(count, width, height, start_x, y)


def side_hand_layout(count: int, side: str) -> tuple[tuple[int, int], list[tuple[float, float]]]:
    """P1 / P3: vertical stack on left/right; edge margin scales with window width."""
    win_w, win_h = viewport_size()
    width, height = card_size()
    step = height * SIDE_STEP_RATIO
    total_height = height + max(count - 1, 0) * step
    start_y = (win_h - total_height) / 2

    if side == "left":
        x = win_w * SIDE_LEFT_EDGE_X_RATIO + width * SIDE_LEFT_CENTER_INSET_RATIO
    else:
        x = win_w * (1 - SIDE_RIGHT_EDGE_X_RATIO) - width * (1 + SIDE_RIGHT_INSET_RATIO)

    positions = [(x, start_y + step * idx) for idx in range(count)]
    return (width, height), positions


def trick_card_position(display: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """Center trick slot for compass display index 0=bottom, 1=left, 2=top, 3=right."""
    win_w, win_h = viewport_size()
    width, height = card_size()
    cx = win_w * 0.5 - width / 2 + hand_shift_x()
    cy = win_h * 0.5 - height / 2
    inset = TRICK_SLOT_INSET
    offsets = {
        0: (0, -height * inset),
        1: (-width * inset, 0),
        2: (0, height * inset),
        3: (width * inset, 0),
    }
    dx, dy = offsets.get(display, (0, 0))
    return (width, height), (cx + dx, cy + dy)


def trick_animation_start(display: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """Start position for card fly-in animation (from the player's side)."""
    (width, height), (end_x, end_y) = trick_card_position(display)
    pull = {
        0: (0, -height * 1.6),
        1: (-width * 1.6, 0),
        2: (0, height * 1.6),
        3: (width * 1.6, 0),
    }
    dx, dy = pull.get(display, (0, 0))
    return (width, height), (end_x + dx, end_y + dy)


def trick_collect_layout(
    team: int,
    _trick_index: int = 0,
) -> tuple[tuple[int, int], tuple[float, float], float]:
    """Collected trick pile. default: team0 bottom-left, team1 top-left; trial: right-bottom / right-top."""
    win_w, win_h = viewport_size()
    width, height = card_size()
    pad = border_pad(width)

    if LAYOUT_DESIGN == "trial":
        x = win_w - width - pad
        if team == 0:
            y = pad
        else:
            y = win_h - height - pad
    else:
        x = pad
        if team == 0:
            y = pad
        else:
            y = win_h - height - pad

    return (width, height), (x, y), 0.0


def round_score_layout(team: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """Live round score beside the collected-trick pile for this team."""
    card_w, card_h = card_size()
    (_, _), (pile_x, pile_y), _ = trick_collect_layout(team)
    box_w = int(viewport_size()[0] * 0.08 * 0.5)
    box_h = int(box_w * 1.1)
    gap = card_w * 0.12

    if LAYOUT_DESIGN == "trial":
        x = pile_x - box_w - gap
    else:
        x = pile_x + card_w + gap
    y = pile_y + (card_h - box_h) / 2
    return (box_w, box_h), (x, y)


PLAYED_JACK_SIZE_SCALE = 0.62
PLAYED_JACK_OVERLAP_RATIO = 0.38


def played_jack_stack_layout(index: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """Top-left stack: first jack leftmost, each next overlaps the previous."""
    card_w, card_h = card_size()
    mini_w = int(card_w * PLAYED_JACK_SIZE_SCALE)
    mini_h = int(card_h * PLAYED_JACK_SIZE_SCALE)
    pad = border_pad(card_w)
    step = mini_w * (1 - PLAYED_JACK_OVERLAP_RATIO)
    x = pad + index * step
    _, win_h = viewport_size()
    y = win_h - mini_h - pad
    return (mini_w, mini_h), (x, y)
