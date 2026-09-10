"""Game logging and replay for human-vs-model blunder review.

``GameLog`` records complete games played through the CLI: the true deal, every
move (with the ISMCTS root visit counts and Q values for the model's decisions),
human flags ("that AI move looked wrong"), and final scores. The log is enough
to reconstruct the exact pre-move state of every decision (``replay_decisions``),
which lets an offline oracle re-score the model's choices against the true hands.

JSON on disk; multiple sessions append to the same file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .cards import Card, card_from_text
from .engine import GameState, play_card


class GameLog:
    def __init__(self, path: str, meta: dict | None = None) -> None:
        self.path = Path(path)
        self.meta = meta or {}
        self.games: list[dict] = []
        if self.path.exists():
            self.games = json.loads(self.path.read_text()).get("games", [])
        self._current: dict | None = None
        self._last_ai_move: int | None = None

    def start_game(self, state: GameState) -> None:
        self._current = {
            "meta": dict(self.meta),
            "hands": [[str(card) for card in hand] for hand in state.hands],
            "leader": state.leader,
            "moves": [],
            "flags": [],
        }
        self._last_ai_move = None

    def record_move(self, seat: int, card: Card, human: bool,
                    visits: dict[int, int] | None = None,
                    q: dict[int, float] | None = None) -> None:
        assert self._current is not None, "start_game first"
        move: dict = {"seat": seat, "card": str(card), "human": human}
        if visits:
            move["visits"] = {str(index): count for index, count in visits.items()}
        if q:
            move["q"] = {str(index): round(value, 3) for index, value in q.items()}
        self._current["moves"].append(move)
        if not human:
            self._last_ai_move = len(self._current["moves"]) - 1

    def flag_last_ai_move(self) -> str | None:
        """Mark the most recent AI move as suspicious. Returns its description."""
        if self._current is None or self._last_ai_move is None:
            return None
        if self._last_ai_move not in self._current["flags"]:
            self._current["flags"].append(self._last_ai_move)
        move = self._current["moves"][self._last_ai_move]
        return f"P{move['seat']} {move['card']}"

    def end_game(self, state: GameState) -> None:
        assert self._current is not None, "start_game first"
        self._current["scores"] = [state.team_score(0), state.team_score(1)]
        self.games.append(self._current)
        self._current = None
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"games": self.games}, indent=1))


def initial_state(game: dict) -> GameState:
    hands = [[card_from_text(text) for text in hand] for hand in game["hands"]]
    leader = int(game.get("leader", 0))
    return GameState(hands=hands, leader=leader, current_player=leader)


def replay_decisions(game: dict) -> Iterator[tuple[int, GameState, dict]]:
    """Yield (move_index, pre-move state clone, move record) for every move."""
    state = initial_state(game)
    for index, move in enumerate(game["moves"]):
        yield index, state.clone(), move
        play_card(state, int(move["seat"]), card_from_text(move["card"]))
