from __future__ import annotations

from .cards import DECK, JACKS, SUITS, Card
from .engine import GameState


def infer_void_suits(state: GameState) -> list[set[str]]:
    voids = [set() for _ in range(4)]
    tricks = [result.plays for result in state.completed_tricks]
    if state.current_trick:
        tricks.append(tuple(state.current_trick))
    for plays in tricks:
        if not plays:
            continue
        led_suit = plays[0][1].suit
        for player, card in plays[1:]:
            if card.suit != led_suit:
                voids[player].add(led_suit)
    return voids


def unseen_cards(state: GameState, observer: int | None = None) -> set[Card]:
    seen = set(state.played_cards)
    if observer is not None:
        seen.update(state.hands[observer])
    return set(DECK) - seen


def unseen_penalty_jacks(state: GameState) -> set[Card]:
    return set(JACKS) - state.played_cards


def cards_remaining_by_suit(state: GameState) -> dict[str, int]:
    remaining = set(DECK) - state.played_cards
    return {suit: sum(1 for card in remaining if card.suit == suit) for suit in SUITS}


def trick_penalty_cards(state: GameState) -> list[Card]:
    return [card for _, card in state.current_trick if card.is_penalty_jack]
