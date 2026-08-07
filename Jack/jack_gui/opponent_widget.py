from __future__ import annotations

from kivy.uix.floatlayout import FloatLayout

from jack_game.cards import Card

from .card_widget import CardWidget
from .layout import seat_zone, side_hand_layout, top_hand_layout, viewport_size


class OpponentWidget(FloatLayout):
    """P1 left, P2 top, P3 right — card backs only."""

    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, 1), **kwargs)
        self.human_seat = 0
        self.card_widgets: dict[int, list[CardWidget]] = {}

    def relayout(self, *_args):
        self._relayout_all()

    def refresh(self, hands: list[list[Card]], human_seat: int):
        self.human_seat = human_seat
        for widgets in self.card_widgets.values():
            for widget in widgets:
                self.remove_widget(widget)
        self.card_widgets = {}

        for seat, hand in enumerate(hands):
            if seat == human_seat:
                continue
            zone = seat_zone(seat, human_seat)
            if zone == "bottom":
                continue

            widgets: list[CardWidget] = []
            for idx in range(len(hand)):
                widget = CardWidget(card=None, name="Opponent", group=idx, rotation=0)
                widget.size_hint = (None, None)
                widgets.append(widget)
                self.add_widget(widget)
            self.card_widgets[seat] = widgets

        self._relayout_all()

    def _relayout_all(self):
        if viewport_size()[0] <= 0 or viewport_size()[1] <= 0:
            return
        for seat, widgets in self.card_widgets.items():
            zone = seat_zone(seat, self.human_seat)
            count = len(widgets)
            if zone == "top":
                size, positions = top_hand_layout(count)
            elif zone == "left":
                size, positions = side_hand_layout(count, "left")
            elif zone == "right":
                size, positions = side_hand_layout(count, "right")
            else:
                continue
            for idx, widget in enumerate(widgets):
                widget.snap_layout(positions[idx], size=size, rotation=0)
