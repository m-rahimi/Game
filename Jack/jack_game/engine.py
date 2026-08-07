from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from .cards import DECK, Card, card_sort_key, team_for


@dataclass(frozen=True)
class TrickResult:
    leader: int
    plays: tuple[tuple[int, Card], ...]
    winner: int
    points: int
    penalties: tuple[Card, ...]


@dataclass
class GameState:
    hands: list[list[Card]]
    leader: int = 0
    current_player: int = 0
    current_trick: list[tuple[int, Card]] = field(default_factory=list)
    scores: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    played_cards: set[Card] = field(default_factory=set)
    completed_tricks: list[TrickResult] = field(default_factory=list)

    def clone(self) -> "GameState":
        return GameState(
            hands=[hand.copy() for hand in self.hands],
            leader=self.leader,
            current_player=self.current_player,
            current_trick=self.current_trick.copy(),
            scores=self.scores.copy(),
            played_cards=set(self.played_cards),
            completed_tricks=self.completed_tricks.copy(),
        )

    @property
    def trick_index(self) -> int:
        return len(self.completed_tricks)

    @property
    def round_over(self) -> bool:
        return len(self.completed_tricks) == 13

    def legal_actions(self, player: int | None = None) -> list[Card]:
        actor = self.current_player if player is None else player
        hand = self.hands[actor]
        if not self.current_trick:
            return sorted(hand, key=card_sort_key)
        led_suit = self.current_trick[0][1].suit
        suited = [card for card in hand if card.suit == led_suit]
        return sorted(suited or hand, key=card_sort_key)

    def current_winner(self) -> int | None:
        if not self.current_trick:
            return None
        return trick_winner(self.current_trick)

    def current_winning_card(self) -> Card | None:
        if not self.current_trick:
            return None
        led_suit = self.current_trick[0][1].suit
        return max((card for _, card in self.current_trick if card.suit == led_suit), key=lambda c: c.rank)

    def team_score(self, team: int) -> int:
        return self.scores[team] + self.scores[team + 2]

    def team_diff_for(self, player: int) -> int:
        own = team_for(player)
        return self.team_score(own) - self.team_score(1 - own)


class PlayerPolicy(Protocol):
    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        ...


def deal(seed: int | None = None) -> list[list[Card]]:
    rng = Random(seed)
    deck = list(DECK)
    rng.shuffle(deck)
    return [sorted(deck[i * 13 : (i + 1) * 13], key=card_sort_key) for i in range(4)]


def new_round(seed: int | None = None, leader: int = 0) -> GameState:
    return GameState(hands=deal(seed), leader=leader, current_player=leader)


def trick_winner(plays: list[tuple[int, Card]] | tuple[tuple[int, Card], ...]) -> int:
    led_suit = plays[0][1].suit
    return max(((player, card) for player, card in plays if card.suit == led_suit), key=lambda pc: pc[1].rank)[0]


def play_card(state: GameState, player: int, card: Card) -> TrickResult | None:
    if player != state.current_player:
        raise ValueError(f"Player {player} cannot act; current player is {state.current_player}")
    if card not in state.legal_actions(player):
        raise ValueError(f"Illegal card {card} for player {player}")

    state.hands[player].remove(card)
    state.current_trick.append((player, card))
    state.played_cards.add(card)

    if len(state.current_trick) < 4:
        state.current_player = (player + 1) % 4
        return None

    winner = trick_winner(state.current_trick)
    penalties = tuple(card for _, card in state.current_trick if card.is_penalty_jack)
    points = 1 + sum(card.penalty for card in penalties)
    state.scores[winner] += points
    result = TrickResult(
        leader=state.leader,
        plays=tuple(state.current_trick),
        winner=winner,
        points=points,
        penalties=penalties,
    )
    state.completed_tricks.append(result)
    state.current_trick.clear()
    state.leader = winner
    state.current_player = winner
    return result


def play_round(players: list[PlayerPolicy], seed: int | None = None, leader: int = 0) -> GameState:
    rng = Random(seed)
    state = new_round(seed, leader)
    while not state.round_over:
        player = state.current_player
        card = players[player].choose_card(state.clone(), player, rng)
        play_card(state, player, card)
    return state
