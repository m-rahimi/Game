from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from statistics import mean
from time import monotonic
from typing import Callable, Optional

from .cards import DECK, Card, card_sort_key, partner_of, team_for
from .engine import GameState, play_card
from .history import infer_void_suits, trick_penalty_cards, unseen_penalty_jacks

# A value function maps (state, player) to the predicted final team-score diff
# from that player's perspective. Used to truncate rollouts in SearchPlayer.
ValueFn = Callable[[GameState, int], float]


class RandomPlayer:
    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        legal = state.legal_actions(player)
        return rng.choice(legal)


@dataclass
class HeuristicPlayer:
    """Team-aware rule player focused on avoiding bad Jack captures."""

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        legal = state.legal_actions(player)
        scored = [(self.card_score(state, player, card), card) for card in legal]
        return min(scored, key=lambda item: (item[0], item[1].rank, item[1].index))[1]

    def card_score(self, state: GameState, player: int, card: Card) -> float:
        if not state.current_trick:
            return self._lead_score(state, card)
        led_suit = state.current_trick[0][1].suit
        if card.suit == led_suit:
            return self._follow_score(state, player, card)
        return self._discard_score(state, player, card)

    def _lead_score(self, state: GameState, card: Card) -> float:
        unseen_jacks = unseen_penalty_jacks(state)
        jack = Card(card.suit, 11)
        danger = jack.penalty_severity if jack in unseen_jacks else 0
        high_card_risk = max(0, card.rank - 10) * 0.6
        low_control_bonus = card.rank * 0.03
        return danger * 10 + high_card_risk + low_control_bonus + card.index * 0.001

    def _follow_score(self, state: GameState, player: int, card: Card) -> float:
        winner = state.current_winner()
        partner_winning = winner == partner_of(player)
        opponent_winning = winner is not None and team_for(winner) != team_for(player)
        penalty_load = sum(c.penalty_severity for c in trick_penalty_cards(state))
        would_win = would_win_current_trick(state, card)
        winning_rank_cost = card.rank * 0.1
        losing_rank_cost = card.rank * 0.02

        if penalty_load:
            if would_win:
                if partner_winning:
                    return 80 + penalty_load * 8 + winning_rank_cost
                return 120 + penalty_load * 12 + winning_rank_cost
            if opponent_winning:
                return -40 - penalty_load * 6 + losing_rank_cost
            return -10 - penalty_load + losing_rank_cost

        if would_win:
            return 4 + winning_rank_cost
        jack = Card(card.suit, 11)
        dangerous_low_card = jack in unseen_penalty_jacks(state) and card.rank < 11
        preserve_low_penalty = 2 if dangerous_low_card else 0
        return losing_rank_cost + preserve_low_penalty

    def _discard_score(self, state: GameState, player: int, card: Card) -> float:
        winner = state.current_winner()
        partner_winning = winner == partner_of(player)
        opponent_winning = winner is not None and team_for(winner) != team_for(player)

        if card.is_penalty_jack:
            if opponent_winning:
                return -100 - card.penalty_severity * 10
            if partner_winning:
                return 100 + card.penalty_severity * 10
            return 30 + card.penalty_severity * 4

        jack = Card(card.suit, 11)
        dangerous_suit = jack in unseen_penalty_jacks(state)
        rank_risk = card.rank * (0.8 if dangerous_suit else 0.35)
        singleton_bonus = -3 if sum(1 for c in state.hands[player] if c.suit == card.suit) == 1 else 0
        return -rank_risk + singleton_bonus + card.index * 0.001


@dataclass
class SearchPlayer:
    rollouts_per_action: int = 24
    lambda_clubs: float = 2.5
    lambda_hearts: float = 0.25
    time_limit_seconds: float | None = 2.0
    fallback: HeuristicPlayer = field(default_factory=HeuristicPlayer)
    # When set, rollouts are truncated after `rollout_depth` further completed
    # tricks and the leaf is scored by `value_fn` instead of playing to the end.
    # The learned value already accounts for penalty Jacks, so the lambda risk
    # terms are dropped on this path.
    value_fn: Optional[ValueFn] = None
    rollout_depth: int = 2

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return legal[0]
        deadline = monotonic() + self.time_limit_seconds if self.time_limit_seconds else None
        try:
            action_scores = []
            deals = [sample_information_set(state, player, rng) for _ in range(self.rollouts_per_action)]
            for card in legal:
                if deadline and monotonic() > deadline:
                    break
                score = self._score_action(state, player, card, deals, rng)
                if score is not None:
                    action_scores.append((score, card))
            if not action_scores:
                return self.fallback.choose_card(state, player, rng)
            return max(action_scores, key=lambda item: (item[0], -item[1].rank, -item[1].index))[1]
        except Exception:
            return self.fallback.choose_card(state, player, rng)

    def _score_action(self, state, player, card, deals, rng) -> float | None:
        if self.value_fn is not None:
            values = []
            for sampled in deals:
                rollout_state = sampled.clone()
                play_card(rollout_state, player, card)
                rollout_round(rollout_state, self.fallback, rng, max_tricks=self.rollout_depth)
                if rollout_state.round_over:
                    values.append(rollout_state.team_diff_for(player))
                else:
                    values.append(self.value_fn(rollout_state, player))
            return mean(values) if values else None
        values = []
        club_risks = []
        heart_risks = []
        for sampled in deals:
            rollout_state = sampled.clone()
            play_card(rollout_state, player, card)
            rollout_round(rollout_state, self.fallback, rng)
            values.append(rollout_state.team_diff_for(player))
            club_risks.append(team_captured_penalty(rollout_state, player, Card("C", 11)))
            heart_risks.append(team_captured_penalty(rollout_state, player, Card("H", 11)))
        if not values:
            return None
        return mean(values) - self.lambda_clubs * mean(club_risks) - self.lambda_hearts * mean(heart_risks)


def would_win_current_trick(state: GameState, card: Card) -> bool:
    if not state.current_trick:
        return True
    led_suit = state.current_trick[0][1].suit
    if card.suit != led_suit:
        return False
    winning_card = state.current_winning_card()
    return winning_card is None or card.rank > winning_card.rank


def rollout_round(
    state: GameState,
    player_policy: HeuristicPlayer,
    rng: Random,
    max_tricks: int | None = None,
) -> None:
    """Play out the round with ``player_policy``.

    If ``max_tricks`` is given, stop after that many *further* tricks complete
    (so a value function can score the truncated leaf) instead of playing to the
    end. ``max_tricks`` counts tricks completed from the current point forward.
    """
    start = len(state.completed_tricks)
    while not state.round_over:
        if max_tricks is not None and len(state.completed_tricks) - start >= max_tricks:
            return
        actor = state.current_player
        card = player_policy.choose_card(state.clone(), actor, rng)
        play_card(state, actor, card)


def sample_information_set(state: GameState, observer: int, rng: Random) -> GameState:
    sampled = state.clone()
    known = set(state.played_cards) | set(state.hands[observer])
    unknown = [card for card in DECK if card not in known]
    rng.shuffle(unknown)
    voids = infer_void_suits(state)

    sampled.hands = [[] for _ in range(4)]
    sampled.hands[observer] = state.hands[observer].copy()
    remaining_counts = {p: len(state.hands[p]) for p in range(4) if p != observer}

    for player in sorted(remaining_counts, key=lambda p: len(voids[p]), reverse=True):
        allowed = [card for card in unknown if card.suit not in voids[player]]
        if len(allowed) < remaining_counts[player]:
            allowed = unknown
        chosen = allowed[: remaining_counts[player]]
        sampled.hands[player] = sorted(chosen, key=card_sort_key)
        for card in chosen:
            unknown.remove(card)

    for player in range(4):
        sampled.hands[player].sort(key=card_sort_key)
    return sampled


def sample_information_set_weighted(
    state: GameState,
    observer: int,
    rng: Random,
    card_probs: list[list[float]],
    belief_weight: float = 0.5,
    max_attempts: int = 20,
) -> GameState:
    """Belief-weighted determinization: like ``sample_information_set`` but each
    unseen card is dealt to a hidden seat by a mix of the belief and the neutral
    capacity-proportional dealer:

        weight = belief_weight * belief + (1 - belief_weight) * capacity

    ``card_probs``: row = card index, columns = relative seats [left opponent,
    partner, right opponent] from the observer (``BeliefFunction`` output).
    Tempering (belief_weight < 1) keeps world diversity — pure belief sampling
    concentrates simulations on modal deals, which measurably degrades deep
    searches. Hand sizes and inferred voids are respected exactly; rare
    dead-ends retry and then fall back to the uniform sampler."""
    known = set(state.played_cards) | set(state.hands[observer])
    unknown = [card for card in DECK if card not in known]
    voids = infer_void_suits(state)
    seats = [(observer + rel) % 4 for rel in (1, 2, 3)]
    capacities = [len(state.hands[seat]) for seat in seats]

    for _ in range(max_attempts):
        rng.shuffle(unknown)
        caps = list(capacities)
        hands: list[list[Card]] = [[], [], []]
        dead_end = False
        for card in unknown:
            allowed = [caps[i] > 0 and card.suit not in voids[seats[i]] for i in range(3)]
            belief = [card_probs[card.index][i] if allowed[i] else 0.0 for i in range(3)]
            base = [float(caps[i]) if allowed[i] else 0.0 for i in range(3)]
            belief_total, base_total = sum(belief), sum(base)
            if belief_total > 0.0 and base_total > 0.0:
                weights = [
                    belief_weight * belief[i] / belief_total + (1.0 - belief_weight) * base[i] / base_total
                    for i in range(3)
                ]
            else:  # belief mass all blocked: fall back to capacity weights
                weights = base
            total = sum(weights)
            if total <= 0.0:
                dead_end = True
                break
            pick = rng.random() * total
            for i in range(3):
                pick -= weights[i]
                if pick <= 0.0 or i == 2:
                    if weights[i] == 0.0:  # numerical edge: take first allowed
                        i = next(j for j in range(3) if weights[j] > 0.0)
                    hands[i].append(card)
                    caps[i] -= 1
                    break
        if not dead_end:
            sampled = state.clone()
            sampled.hands = [[] for _ in range(4)]
            sampled.hands[observer] = sorted(state.hands[observer], key=card_sort_key)
            for i, seat in enumerate(seats):
                sampled.hands[seat] = sorted(hands[i], key=card_sort_key)
            return sampled
    return sample_information_set(state, observer, rng)


def team_captured_penalty(state: GameState, player: int, penalty_card: Card) -> int:
    own_team = team_for(player)
    for result in state.completed_tricks:
        if penalty_card in result.penalties:
            return int(team_for(result.winner) == own_team)
    return 0
