"""Saved layout constants — restore these in layout.py to revert the trial design."""

# Hand positioning
HAND_SHIFT_X_RATIO = 0.04
HAND_SPAN_SLOTS = 3.2
HAND_GAP = 0.006
SIDE_STEP_RATIO = 0.14
SIDE_CENTER_INSET_RATIO = 0.55

# Card sizing
CARD_WIDTH_RATIO = 0.11
CARD_MAX_HEIGHT_RATIO = 0.275

# Trick / collection
TRICK_SLOT_INSET = 0.45
COLLECT_TEAM0 = "bottom-left"  # x = pad + stack, y = pad + stack
COLLECT_TEAM1 = "top-left"  # x = pad + stack, y = height - pad - stack
COLLECT_STACK_X_STEP = 8
COLLECT_STACK_Y_STEP = 6

# Hand suit order (hand_widget.py)
HAND_SUIT_ORDER = ("C", "H", "S", "D")

# Collection uses blue.png card back (card_widget.set_card_back -> card_back_png)
