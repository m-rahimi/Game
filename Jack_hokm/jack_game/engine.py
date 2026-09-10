from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from .cards import DECK, SUITS, Card, card_sort_key, team_for


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
    # Hokm (trump) suit for this round, or None for the classic no-trump game.
    # Public information: every seat knows it from the first lead, so it is a plain
    # feature rather than something the belief model has to infer. ``None`` keeps every
    # pre-Hokm code path, checkpoint and recorded result exactly as it was.
    trump: str | None = None

    def clone(self) -> "GameState":
        return GameState(
            hands=[hand.copy() for hand in self.hands],
            leader=self.leader,
            current_player=self.current_player,
            current_trick=self.current_trick.copy(),
            scores=self.scores.copy(),
            played_cards=set(self.played_cards),
            completed_tricks=self.completed_tricks.copy(),
            trump=self.trump,
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
        return trick_winner(self.current_trick, self.trump)

    def current_winning_card(self) -> Card | None:
        if not self.current_trick:
            return None
        return max(_contenders(self.current_trick, self.trump), key=lambda pc: pc[1].rank)[1]

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


HOKM_SEED_SALT = 0x484F4B4D  # "HOKM"


def trump_for_seed(seed: int | None) -> str:
    """The round's Hokm suit, derived from the deal seed.

    Deriving it from the seed rather than drawing it inside :func:`deal` has two
    consequences we want. The deal itself stays byte-identical to the classic game for
    the same seed, so a Hokm round and a no-trump round can be compared on literally the
    same 52 cards. And duplicate-deal (paired) scoring keeps working: both playings of a
    pair share a seed, so they share a trump — without that the two halves would not be
    the same deal and the variance cancellation that every gate in this project relies
    on would silently break.
    """
    return Random(HOKM_SEED_SALT if seed is None else seed ^ HOKM_SEED_SALT).choice(SUITS)


def new_round(seed: int | None = None, leader: int = 0, hokm: bool = False,
              trump: str | None = None) -> GameState:
    """``hokm=True`` picks the trump suit from ``seed``; ``trump`` forces a specific one."""
    if trump is None and hokm:
        trump = trump_for_seed(seed)
    return GameState(hands=deal(seed), leader=leader, current_player=leader, trump=trump)


def _contenders(plays, trump: str | None) -> list[tuple[int, Card]]:
    """The cards still in the running for the trick.

    Trump beats every non-trump, so if anyone has played the trump suit only trumps
    contend; otherwise the led suit does. When the trump suit *is* the led suit both
    branches select the same cards, which is why this needs no special case.
    """
    if trump is not None:
        trumps = [(player, card) for player, card in plays if card.suit == trump]
        if trumps:
            return trumps
    led_suit = plays[0][1].suit
    return [(player, card) for player, card in plays if card.suit == led_suit]


def trick_winner(plays: list[tuple[int, Card]] | tuple[tuple[int, Card], ...],
                 trump: str | None = None) -> int:
    return max(_contenders(plays, trump), key=lambda pc: pc[1].rank)[0]


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

    winner = trick_winner(state.current_trick, state.trump)
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


def play_round(players: list[PlayerPolicy], seed: int | None = None, leader: int = 0,
               hokm: bool = False, trump: str | None = None) -> GameState:
    rng = Random(seed)
    state = new_round(seed, leader, hokm=hokm, trump=trump)
    while not state.round_over:
        player = state.current_player
        card = players[player].choose_card(state.clone(), player, rng)
        play_card(state, player, card)
    return state
