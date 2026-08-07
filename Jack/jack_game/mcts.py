"""Information-Set MCTS guided by a policy prior and a value net (ExIt Phase 2).

Single-observer determinized ISMCTS: one shared tree keyed by public action
history; each simulation samples a fresh determinization of the hidden hands,
descends via PUCT (restricted to the moves legal in that determinization),
expands a leaf, and scores it with the value net. The move played is the most
visited root action.

Values are stored from Team 0's perspective and normalised by VALUE_SCALE so the
exploitation term sits on the same scale as the PUCT exploration term; the player
to move flips the sign when selecting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import Random
from time import monotonic
from typing import Callable, Optional

from .cards import RANKS, SUITS, Card, team_for
from .engine import GameState, play_card
from .players import HeuristicPlayer, sample_information_set, sample_information_set_weighted

ValueFn = Callable[[GameState, int], float]
PolicyFn = Callable[[GameState, int], dict]
BeliefFn = Callable[[GameState, int], object]  # -> (52, 3) probs over hidden seats

VALUE_SCALE = 8.0  # ~ std of final team-score diff; maps value into ~[-1, 1]


def index_to_card(index: int) -> Card:
    return Card(SUITS[index // len(RANKS)], RANKS[index % len(RANKS)])


class Node:
    __slots__ = ("expanded", "player", "P", "N", "W", "N_total", "children")

    def __init__(self) -> None:
        self.expanded = False
        self.player: int | None = None
        self.P: dict[int, float] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.N_total = 0
        self.children: dict[int, "Node"] = {}

    def expand(self, player: int, legal_idx: list[int], priors: dict[int, float]) -> None:
        self.expanded = True
        self.player = player
        for a in legal_idx:
            self.P[a] = priors.get(a, 1.0 / len(legal_idx))

    def select(self, legal_idx: list[int], c_puct: float) -> int:
        # PUCT, evaluated from the perspective of the player to move at this node.
        sign = 1.0 if team_for(self.player) == 0 else -1.0
        sqrt_total = sqrt(self.N_total + 1)
        best_score = -1e18
        best_a = legal_idx[0]
        for a in legal_idx:
            n = self.N.get(a, 0)
            q = (self.W.get(a, 0.0) / n) if n > 0 else 0.0
            prior = self.P.get(a, 1.0 / len(legal_idx))
            score = sign * q + c_puct * prior * sqrt_total / (1 + n)
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


@dataclass
class ISMCTSPlayer:
    value_fn: ValueFn
    policy_fn: PolicyFn
    simulations: int = 200
    c_puct: float = 1.5
    time_limit_seconds: float | None = None
    belief_fn: BeliefFn | None = None  # optional: belief-weighted determinization
    belief_lambda: float = 0.5  # belief vs neutral-dealer mix (1.0 = pure belief)
    # Optional per-trick λ override (13 values): trust reads more as the net's
    # placement accuracy rises over the game (~0.37 early -> ~0.63 late).
    belief_lambda_by_trick: list[float] | None = None
    fallback: HeuristicPlayer = field(default_factory=HeuristicPlayer)

    def effective_lambda(self, trick_index: int) -> float:
        if self.belief_lambda_by_trick:
            return self.belief_lambda_by_trick[max(0, min(trick_index, 12))]
        return self.belief_lambda

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        return self.choose_with_visits(state, player, rng)[0]

    def choose_with_visits(self, state: GameState, player: int, rng: Random) -> tuple[Card, dict[int, int]]:
        """Return (chosen most-visited card, root visit counts per action index).

        The visit counts are the improved-policy target used by ExIt self-play.
        """
        card, visits, _ = self.choose_with_stats(state, player, rng)
        return card, visits

    def choose_with_stats(self, state: GameState, player: int, rng: Random) -> tuple[Card, dict[int, int], dict[int, float]]:
        """Return (chosen card, root visit counts, root mean-Q per action).

        Q is from the acting player's team perspective, rescaled back to
        team-score-diff units (used by the blunder-review game log).
        """
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return legal[0], {legal[0].index: 1}, {}
        try:
            root = Node()
            # One belief-net call per decision; every simulation reuses the probs.
            card_probs = self.belief_fn(state, player).tolist() if self.belief_fn else None
            lam = self.effective_lambda(state.trick_index)
            deadline = monotonic() + self.time_limit_seconds if self.time_limit_seconds else None
            for _ in range(self.simulations):
                if deadline and monotonic() > deadline:
                    break
                self._simulate(root, state, player, rng, card_probs, lam)
            if not root.N:
                fallback = self.fallback.choose_card(state, player, rng)
                return fallback, {fallback.index: 1}, {}
            # Most-visited root action (ties broken toward lower penalty risk via index).
            best = max(root.N.items(), key=lambda kv: (kv[1], -index_to_card(kv[0]).rank))[0]
            sign = 1.0 if team_for(player) == 0 else -1.0
            q = {a: sign * VALUE_SCALE * root.W[a] / root.N[a] for a in root.N if root.N[a] > 0}
            return index_to_card(best), dict(root.N), q
        except Exception:
            fallback = self.fallback.choose_card(state, player, rng)
            return fallback, {fallback.index: 1}, {}

    def _simulate(self, root: Node, root_state: GameState, observer: int, rng: Random,
                  card_probs: list[list[float]] | None = None, belief_weight: float = 0.5) -> None:
        if card_probs is not None:
            state = sample_information_set_weighted(root_state, observer, rng, card_probs,
                                                    belief_weight=belief_weight)
        else:
            state = sample_information_set(root_state, observer, rng)
        node = root
        path: list[tuple[Node, int]] = []
        while True:
            if state.round_over:
                leaf_value = state.team_diff_for(0) / VALUE_SCALE
                break
            player = state.current_player
            legal_idx = [card.index for card in state.legal_actions(player)]
            if not node.expanded:
                node.expand(player, legal_idx, self.policy_fn(state, player))
                leaf_value = self.value_fn(state, 0) / VALUE_SCALE
                break
            action = node.select(legal_idx, self.c_puct)
            path.append((node, action))
            play_card(state, player, index_to_card(action))
            child = node.children.get(action)
            if child is None:
                child = Node()
                node.children[action] = child
            node = child
        for parent, action in path:
            parent.N_total += 1
            parent.N[action] = parent.N.get(action, 0) + 1
            parent.W[action] = parent.W.get(action, 0.0) + leaf_value
