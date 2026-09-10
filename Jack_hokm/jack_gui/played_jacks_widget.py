from __future__ import annotations

from kivy.uix.floatlayout import FloatLayout

from jack_game.cards import Card
from jack_game.engine import GameState

from .card_widget import CardWidget
from .layout import played_jack_stack_layout, viewport_size


class PlayedJacksWidget(FloatLayout):
    """Shows face-up penalty jacks in play order (top-left, overlapping stack)."""

    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, 1), **kwargs)
        self._widgets: dict[Card, CardWidget] = {}
        self._order: list[Card] = []

    def relayout(self, *_args):
        self._relayout()

    def clear(self):
        for widget in self._widgets.values():
            self.remove_widget(widget)
        self._widgets = {}
        self._order = []

    @staticmethod
    def _ordered_jacks(state: GameState) -> list[Card]:
        jacks: list[Card] = []
        for trick in state.completed_tricks:
            for _, card in trick.plays:
                if card.is_penalty_jack:
                    jacks.append(card)
        for _, card in state.current_trick:
            if card.is_penalty_jack:
                jacks.append(card)
        return jacks

    def refresh(self, state: GameState):
        self._order = self._ordered_jacks(state)
        current = set(self._order)

        for card in list(self._widgets):
            if card not in current:
                self.remove_widget(self._widgets.pop(card))

        for card in self._order:
            if card not in self._widgets:
                widget = CardWidget(
                    card=card,
                    name="PlayedJack",
                    playable=False,
                )
                widget.size_hint = (None, None)
                self._widgets[card] = widget
                self.add_widget(widget)

        self._relayout()

    def _relayout(self, *_args):
        if viewport_size()[0] <= 0 or viewport_size()[1] <= 0:
            return
        for index, card in enumerate(self._order):
            widget = self._widgets.get(card)
            if widget is None:
                continue
            size, pos = played_jack_stack_layout(index)
            widget.snap_layout(pos, size=size, rotation=0)
            self.remove_widget(widget)
            self.add_widget(widget)
