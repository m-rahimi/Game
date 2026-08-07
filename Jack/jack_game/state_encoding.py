"""Torch-free state feature encoding for mobile ONNX inference."""

from __future__ import annotations

from .cards import Card, partner_of, team_for
from .engine import GameState
from .history import infer_void_suits

LEGACY_FEATURE_DIM = 179
FEATURE_DIM = 260
NUM_CARDS = 52
NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_TRICKS = 13
SUIT_ORDER = ["C", "D", "H", "S"]
HISTORY_DIMS_PER_CARD = 7
FEATURE_DIM_ATTR = FEATURE_DIM + NUM_CARDS * (HISTORY_DIMS_PER_CARD - 1)


def unseen_penalty_jacks_from_played(played_cards: set[Card]) -> list[Card]:
    jacks = [Card("C", 11), Card("D", 11), Card("H", 11), Card("S", 11)]
    return [card for card in jacks if card not in played_cards]


def build_modern_features(
    player: int,
    trick: int,
    team: int,
    hand: list[Card],
    played_cards: set[Card] | list[Card],
    current_trick: list[Card],
    legal: list[Card],
    void_suits: list[set[str]] | None = None,
    current_winner: int | None = None,
    current_player: int | None = None,
) -> tuple[list[float], list[bool]]:
    vector = [0.0] * FEATURE_DIM
    played_set = set(played_cards)
    legal_indexes = {card.index for card in legal}
    trick_cards = list(current_trick)
    current_penalties = [card for card in trick_cards if card.is_penalty_jack]
    if void_suits is None:
        void_suits = [set() for _ in range(NUM_PLAYERS)]

    offset = 0
    for card in hand:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for card in played_set:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for card in trick_cards:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for index in legal_indexes:
        vector[offset + index] = 1.0
    offset += NUM_CARDS
    for card in unseen_penalty_jacks_from_played(played_set):
        vector[offset + card.index] = 1.0
    offset += NUM_SUITS
    for seat in range(NUM_PLAYERS):
        for suit_index, suit in enumerate(SUIT_ORDER):
            vector[offset + seat * NUM_SUITS + suit_index] = 1.0 if suit in void_suits[seat] else 0.0
    offset += NUM_PLAYERS * NUM_SUITS
    relation = 0
    if current_winner is not None:
        if current_winner == player:
            relation = 1
        elif current_winner == partner_of(player):
            relation = 2
        else:
            relation = 3
    vector[offset + relation] = 1.0
    offset += 4
    if current_player is None:
        current_player = player
    vector[offset + current_player] = 1.0
    offset += NUM_PLAYERS
    vector[offset + team] = 1.0
    offset += 2
    vector[offset + min(NUM_TRICKS - 1, trick)] = 1.0
    offset += NUM_TRICKS
    vector[offset + min(NUM_PLAYERS - 1, len(trick_cards))] = 1.0
    offset += NUM_PLAYERS
    led_suit = trick_cards[0].suit if trick_cards else None
    if led_suit is not None:
        vector[offset + SUIT_ORDER.index(led_suit)] = 1.0
    offset += NUM_SUITS
    vector[offset] = sum(card.penalty_severity for card in current_penalties) / 10.0

    illegal = [index not in legal_indexes for index in range(NUM_CARDS)]
    return vector, illegal


def build_history_block(state: GameState, player: int) -> list[float]:
    block = [0.0] * (NUM_CARDS * HISTORY_DIMS_PER_CARD)
    tricks: list[tuple[int, tuple[tuple[int, Card], ...]]] = [
        (index, trick.plays) for index, trick in enumerate(state.completed_tricks)
    ]
    if state.current_trick:
        tricks.append((state.trick_index, tuple(state.current_trick)))
    for trick_index, plays in tricks:
        led_suit = plays[0][1].suit
        for position, (seat, card) in enumerate(plays):
            base = card.index * HISTORY_DIMS_PER_CARD
            block[base + (seat - player) % NUM_PLAYERS] = 1.0
            block[base + 4] = 1.0 if position == 0 else 0.0
            block[base + 5] = 1.0 if card.suit == led_suit else 0.0
            block[base + 6] = trick_index / (NUM_TRICKS - 1)
    return block


def build_attributed_features(state: GameState, player: int) -> tuple[list[float], list[bool]]:
    modern, illegal = build_modern_features(
        player=player,
        trick=state.trick_index,
        team=team_for(player),
        hand=state.hands[player],
        played_cards=state.played_cards,
        current_trick=[card for _, card in state.current_trick],
        legal=state.legal_actions(player),
        void_suits=infer_void_suits(state),
        current_winner=state.current_winner(),
        current_player=state.current_player,
    )
    history = build_history_block(state, player)
    vector = modern[:NUM_CARDS] + history + modern[2 * NUM_CARDS :]
    return vector, illegal


def encode_state_from_game_state(
    state: GameState,
    player: int,
    feature_dim: int = FEATURE_DIM,
) -> tuple[list[float], list[bool]]:
    if feature_dim == FEATURE_DIM_ATTR:
        return build_attributed_features(state, player)
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"mobile encoding supports feature_dim={FEATURE_DIM} or {FEATURE_DIM_ATTR}, got {feature_dim}")
    return build_modern_features(
        player=player,
        trick=state.trick_index,
        team=team_for(player),
        hand=state.hands[player],
        played_cards=state.played_cards,
        current_trick=[card for _, card in state.current_trick],
        legal=state.legal_actions(player),
        void_suits=infer_void_suits(state),
        current_winner=state.current_winner(),
        current_player=state.current_player,
    )
