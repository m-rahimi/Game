from __future__ import annotations

from dataclasses import dataclass


SUITS = ("C", "D", "H", "S")
RANKS = tuple(range(2, 15))
RANK_NAMES = {
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
SUIT_NAMES = {
    "C": "clubs",
    "D": "diamonds",
    "H": "hearts",
    "S": "spades",
}
PENALTY_BY_SUIT = {
    "C": -4,
    "H": -3,
    "S": -2,
    "D": -1,
}


@dataclass(frozen=True, order=True)
class Card:
    suit: str
    rank: int

    @property
    def index(self) -> int:
        return SUITS.index(self.suit) * len(RANKS) + RANKS.index(self.rank)

    @property
    def is_penalty_jack(self) -> bool:
        return self.rank == 11

    @property
    def penalty(self) -> int:
        if not self.is_penalty_jack:
            return 0
        return PENALTY_BY_SUIT[self.suit]

    @property
    def penalty_severity(self) -> int:
        return -self.penalty

    def short(self) -> str:
        return f"{RANK_NAMES.get(self.rank, str(self.rank))}{self.suit}"

    def __str__(self) -> str:
        return self.short()


DECK = tuple(Card(suit, rank) for suit in SUITS for rank in RANKS)
JACKS = tuple(Card(suit, 11) for suit in SUITS)


def card_from_text(text: str) -> Card:
    text = text.strip().upper()
    if len(text) < 2:
        raise ValueError(f"Invalid card text: {text!r}")
    suit = text[-1]
    rank_text = text[:-1]
    ranks = {"J": 11, "Q": 12, "K": 13, "A": 14}
    rank = ranks.get(rank_text, int(rank_text) if rank_text.isdigit() else -1)
    card = Card(suit, rank)
    if card not in DECK:
        raise ValueError(f"Invalid card text: {text!r}")
    return card


def team_for(player: int) -> int:
    return player % 2


def partner_of(player: int) -> int:
    return (player + 2) % 4


def card_sort_key(card: Card) -> tuple[int, int]:
    return (SUITS.index(card.suit), card.rank)
