from __future__ import annotations

from jack_game.cards import Card, SUIT_NAMES

RANK_NAMES = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "jack",
    12: "queen",
    13: "king",
    14: "ace",
}


def card_png_path(card: Card | None) -> str:
    from .assets import card_back_png, card_png_folder

    if card is None:
        return card_back_png()
    rank = RANK_NAMES[card.rank]
    suit = SUIT_NAMES[card.suit]
    return str(card_png_folder() / f"{rank}_of_{suit}.png")
