"""Match session: multiple rounds until a team reaches 21."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .engine import GameState

MATCH_TARGET = 21


@dataclass
class MatchState:
    team_totals: list[int]
    next_leader: int
    round_number: int = 0
    tiebreak: bool = False

    @classmethod
    def new(cls, rng: Random) -> MatchState:
        return cls(team_totals=[0, 0], next_leader=rng.randrange(4))

    def rotate_leader(self) -> None:
        self.next_leader = (self.next_leader + 1) % 4

    def apply_round(self, state: GameState) -> int | None | str:
        """Apply round scores. Returns winning team (0/1), 'tiebreak', or None to continue."""
        round_t0 = state.team_score(0)
        round_t1 = state.team_score(1)

        if self.tiebreak:
            if round_t0 > round_t1:
                return 0
            if round_t1 > round_t0:
                return 1
            return None

        self.team_totals[0] += round_t0
        self.team_totals[1] += round_t1
        t0, t1 = self.team_totals

        if t0 < MATCH_TARGET and t1 < MATCH_TARGET:
            return None
        if t0 >= MATCH_TARGET and t1 >= MATCH_TARGET:
            if t0 == MATCH_TARGET and t1 == MATCH_TARGET:
                return "tiebreak"
            return 0 if t0 > t1 else 1
        return 0 if t0 >= MATCH_TARGET else 1

    def begin_round(self) -> None:
        self.round_number += 1
