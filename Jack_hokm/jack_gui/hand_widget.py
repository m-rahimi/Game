from __future__ import annotations

from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout

from jack_game.cards import Card

from .card_widget import CardWidget
from .layout import player_hand_layout, viewport_size, HAND_RELAYOUT_DURATION

# Left-to-right suit order in the player's hand.
HAND_SUIT_ORDER = ("C", "H", "S", "D")


def hand_sort_key(card: Card) -> tuple[int, int]:
    return (HAND_SUIT_ORDER.index(card.suit), card.rank)


class PlayerHandWidget(FloatLayout):
    """Shows all 13 human cards at the bottom."""

    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, 1), **kwargs)
        self.card_widgets: list[CardWidget] = []

    def relayout(self, *_args):
        self.relayout_player_hand()

    def refresh(self, hand: list[Card], legal_cards: set[Card] | None = None):
        legal_cards = legal_cards or set()
        for widget in self.card_widgets:
            self.remove_widget(widget)
        self.card_widgets = []

        sorted_hand = sorted(hand, key=hand_sort_key)
        for idx, card in enumerate(sorted_hand):
            widget = CardWidget(
                card=card,
                name="Player1",
                group=idx,
                playable=card in legal_cards,
            )
            widget.size_hint = (None, None)
            self.card_widgets.append(widget)
            self.add_widget(widget)

        self.relayout_player_hand()

    def detach_widget(self, widget: CardWidget):
        if widget in self.card_widgets:
            self.card_widgets.remove(widget)
        self.remove_widget(widget)
        self.relayout_player_hand(animate=True)

    def relayout_player_hand(self, animate: bool = False):
        if not self.card_widgets:
            return
        if viewport_size()[0] <= 0 or viewport_size()[1] <= 0:
            return
        size, positions = player_hand_layout(len(self.card_widgets))
        width, height = size
        for idx, widget in enumerate(self.card_widgets):
            new_pos = positions[idx]
            if animate:
                widget.cancel_animations()
                widget.size = (width, height)
                Animation(pos=new_pos, duration=HAND_RELAYOUT_DURATION).start(widget)
            else:
                widget.snap_layout(new_pos, size=(width, height), rotation=0)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        for child in self.children:
            if child.dispatch("on_touch_down", touch):
                return True
        return False
