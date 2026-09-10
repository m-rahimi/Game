from __future__ import annotations

from kivy.uix.floatlayout import FloatLayout

from jack_game.cards import Card

from .card_widget import CardWidget
from .layout import display_seat, trick_animation_start, trick_card_position, viewport_size, CARD_ANIM_DURATION


class TrickWidget(FloatLayout):
    def __init__(self, human_seat: int = 0, **kwargs):
        super().__init__(size_hint=(1, 1), **kwargs)
        self.human_seat = human_seat
        self.card_widgets: list[CardWidget] = []

    def relayout(self, *_args):
        if viewport_size()[0] <= 0 or viewport_size()[1] <= 0:
            return
        for widget in self.card_widgets:
            size, pos = trick_card_position(widget.group)
            widget.snap_layout(pos, size=size, rotation=0)

    def clear(self):
        for widget in self.card_widgets:
            self.remove_widget(widget)
        self.card_widgets = []

    def refresh(self, current_trick: list[tuple[int, Card]], human_seat: int):
        self.human_seat = human_seat
        self.clear()
        for player, card in current_trick:
            display = display_seat(player, human_seat)
            self._place_card(card, display)

    def animate_from_hand(self, widget: CardWidget, display: int, callback):
        size, end_pos = trick_card_position(display)
        widget.name = "Trick"
        widget.group = display
        widget.playable = False
        widget.rotation = 0
        widget.size_hint = (None, None)
        widget.size = size
        if widget.parent:
            widget.parent.remove_widget(widget)
        self.add_widget(widget)
        self.card_widgets.append(widget)
        widget.animate_to(end_pos, duration=CARD_ANIM_DURATION, callback=callback)

    def animate_from_opponent(self, card: Card, display: int, callback):
        size, start_pos = trick_animation_start(display)
        _, end_pos = trick_card_position(display)
        widget = CardWidget(card=card, name="Opponent", group=display, rotation=0)
        widget.size_hint = (None, None)
        widget.size = size
        widget.pos = start_pos
        self.add_widget(widget)
        self.card_widgets.append(widget)
        widget.animate_to(end_pos, duration=CARD_ANIM_DURATION, callback=callback)

    def collect_trick(self, winning_team: int, collection, callback):
        widgets = list(self.card_widgets)
        self.card_widgets = []
        collection.absorb_trick(widgets, winning_team, callback)

    def _place_card(self, card: Card, display: int):
        size, pos = trick_card_position(display)
        widget = CardWidget(card=card, name="Trick", group=display, rotation=0)
        widget.size_hint = (None, None)
        widget.size = size
        widget.pos = pos
        self.add_widget(widget)
        self.card_widgets.append(widget)
