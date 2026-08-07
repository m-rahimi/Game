from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .cards import Card, SUIT_NAMES, team_for
from .engine import GameState, play_card, play_round
from .players import HeuristicPlayer, RandomPlayer, would_win_current_trick


@dataclass(frozen=True)
class SimulationMetrics:
    games: int
    team_win_rate: float
    team_score_diff: float
    jack_capture_rate: dict[str, float]


def evaluate_team_vs_random(policy, games: int = 1000, seed: int = 1) -> SimulationMetrics:
    rng = Random(seed)
    wins = 0
    diff_total = 0
    captures = {suit: 0 for suit in SUIT_NAMES}
    players = [policy, RandomPlayer(), policy, RandomPlayer()]
    for _ in range(games):
        state = play_round(players, seed=rng.randrange(10**9))
        team0 = state.team_score(0)
        team1 = state.team_score(1)
        wins += int(team0 > team1)
        diff_total += team0 - team1
        for result in state.completed_tricks:
            for card in result.penalties:
                if team_for(result.winner) == 0:
                    captures[card.suit] += 1
    return SimulationMetrics(
        games=games,
        team_win_rate=wins / games,
        team_score_diff=diff_total / games,
        jack_capture_rate={SUIT_NAMES[suit]: captures[suit] / games for suit in captures},
    )


def jack_danger_probes(policy=None, seed: int = 1) -> dict[str, bool | str]:
    policy = policy or HeuristicPlayer()
    rng = Random(seed)
    return {
        "fourth_to_act_dodges_jc": _probe_fourth_to_act_dodge(policy, rng),
        "opponent_winning_dump_jc": _probe_dump_on_opponent(policy, rng),
        "partner_winning_keeps_jc": _probe_avoid_partner_dump(policy, rng),
        "leading_avoids_live_jc_suit": _probe_avoid_dangerous_club_lead(policy, rng),
    }


def _probe_fourth_to_act_dodge(policy, rng: Random) -> bool:
    state = GameState(
        hands=[
            [],
            [],
            [],
            [Card("C", 10), Card("C", 14), Card("D", 2)],
        ],
        leader=0,
        current_player=3,
        current_trick=[(0, Card("C", 12)), (1, Card("C", 11)), (2, Card("C", 13))],
        played_cards={Card("C", 12), Card("C", 11), Card("C", 13)},
    )
    card = policy.choose_card(state, 3, rng)
    return card == Card("C", 10) and not would_win_current_trick(state, card)


def _probe_dump_on_opponent(policy, rng: Random) -> bool:
    state = GameState(
        hands=[[], [], [Card("C", 11), Card("D", 2)], []],
        leader=3,
        current_player=2,
        current_trick=[(3, Card("H", 14)), (0, Card("H", 2)), (1, Card("H", 3))],
        played_cards={Card("H", 14), Card("H", 2), Card("H", 3)},
    )
    return policy.choose_card(state, 2, rng) == Card("C", 11)


def _probe_avoid_partner_dump(policy, rng: Random) -> bool:
    state = GameState(
        hands=[[], [], [Card("C", 11), Card("D", 2)], []],
        leader=0,
        current_player=2,
        current_trick=[(0, Card("H", 14)), (1, Card("H", 2))],
        played_cards={Card("H", 14), Card("H", 2)},
    )
    return policy.choose_card(state, 2, rng) == Card("D", 2)


def _probe_avoid_dangerous_club_lead(policy, rng: Random) -> bool:
    state = GameState(
        hands=[[Card("C", 2), Card("H", 2), Card("D", 9)], [], [], []],
        leader=0,
        current_player=0,
    )
    return policy.choose_card(state, 0, rng).suit != "C"


def print_report(games: int = 1000, seed: int = 1) -> None:
    metrics = evaluate_team_vs_random(HeuristicPlayer(), games=games, seed=seed)
    probes = jack_danger_probes(seed=seed)
    print(f"games: {metrics.games}")
    print(f"heuristic_team_win_rate_vs_random: {metrics.team_win_rate:.3f}")
    print(f"heuristic_team_score_diff_vs_random: {metrics.team_score_diff:.3f}")
    for suit, rate in metrics.jack_capture_rate.items():
        print(f"heuristic_{suit}_jack_capture_rate: {rate:.3f}")
    for name, passed in probes.items():
        print(f"{name}: {passed}")
