"""Ensemble-as-teacher: a cheap stand-in for the IIMC operator (item 9, 2026-08-31).

Why this is the teacher to use
------------------------------
`ORACLE_IIMC_PIPELINE.md` §2.10 found that an ensemble of *architecturally diverse*
members plays at **0.589 per round** (edge +0.958) — the closest anything in the project
has come to the 0.610 target, and above every single net. It costs one forward pass per
member, ~2 ms, against the IIMC operator's ~2.3 s per decision. That is a **~1000x
cheaper teacher for a comparable level** (IIMC: 0.606-0.618).

The IIMC route's bottleneck is signal scarcity: 60,000 rows, ~19% of them carrying any
deviation at all, at ~114 min per 20,000 games. At 2 ms a label, millions of rows are
affordable in one afternoon — and visit-distribution distillation has already shown on
this project that a *soft* target at that scale breaks the argmax bottleneck (the
distilled greedy net reached ISMCTS@600 strength at ~1 ms/move).

What the members must be
------------------------
§2.10's decisive finding: **diversity, not member quality, is what pays.** `a+b`
(+0.948) beat `a+c` (+0.886) even though `ens_b` alone is far weaker than `ens_c` (0.530
vs 0.574), because a and c share a lineage and a shape (512x3) and therefore err
together. Pick members that differ in architecture and init, not the N strongest
checkpoints — and never equal-weight a much weaker member in (2D+2A went *backwards*,
+0.761 vs +0.843).

Relationship to `eval_ensemble.py`: that script has its own `EnsemblePlayer` and
produced the §2.10 numbers. It is left untouched so those measurements stay reproducible;
this module is the labelling-side implementation, which additionally exposes the averaged
distribution (the training target) and shares the state encoding across members.
"""
from __future__ import annotations

from random import Random

import torch

from .bc import NUM_CARDS, encode_state_from_game_state
from .cards import Card
from .engine import GameState
from .exploit import ActorSpec, PolicyActor, build_net
from .mcts import index_to_card


class EnsembleTeacher:
    """Averages the members' temperature-1.0 move probabilities.

    Averaging PROBABILITIES (not logits) matches the §2.10 measurement, whose default
    mode is ``prob``. The distinction matters: the geometric mean (logit averaging) is
    dominated by whichever member is most confident that a move is bad, while the
    arithmetic mean lets one member's support keep a move alive. The ensemble's edge
    comes from members disagreeing, so the forgiving average is the right one.
    """

    def __init__(self, specs: list[ActorSpec]) -> None:
        if not specs:
            raise ValueError("an ensemble needs at least one member")
        self.actors = [PolicyActor(spec) for spec in specs]
        self.models = [actor.model for actor in self.actors]
        # Members that share an observable width (and none reading the oracle block)
        # can share one encoding pass instead of repeating it per member.
        dims = {actor.observable_dim for actor in self.actors}
        self.shared_dim = dims.pop() if len(dims) == 1 else None
        if any(actor.oracle for actor in self.actors):
            self.shared_dim = None

    @torch.no_grad()
    def distribution(self, state: GameState, player: int) -> tuple[list[float], list[bool]]:
        """-> (averaged probability over the 52 cards, legal mask).

        Zero on illegal moves, sums to 1 over legal ones — the soft distillation target.
        """
        if self.shared_dim is not None:
            features, illegal = encode_state_from_game_state(
                state, player, feature_dim=self.shared_dim)
            legal_mask = [not flag for flag in illegal]
            legal_indexes = [i for i, ok in enumerate(legal_mask) if ok]
            x = torch.tensor([features], dtype=torch.float32)
            total = torch.zeros(len(legal_indexes))
            for model in self.models:
                logits = model(x).squeeze(0)
                total += torch.softmax(logits[legal_indexes], dim=0)
        else:
            rng = Random(0)  # unused at temperature 0, but act_full wants one
            legal_mask = None
            total = None
            legal_indexes = []
            for actor in self.actors:
                _, _, _, mask, _, distribution = actor.act_full(state, player, rng)
                if legal_mask is None:
                    legal_mask = mask
                    legal_indexes = [i for i, ok in enumerate(mask) if ok]
                    total = torch.zeros(len(legal_indexes))
                total += torch.tensor([distribution[i] for i in legal_indexes])

        total /= float(len(self.models))
        probabilities = [0.0] * NUM_CARDS
        for position, index in enumerate(legal_indexes):
            probabilities[index] = float(total[position])
        return probabilities, legal_mask

    def choose(self, state: GameState, player: int, rng: Random,
               temperature: float = 0.0) -> Card:
        """Argmax by default; ``temperature > 0`` samples, for state diversity."""
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return legal[0]
        probabilities, _ = self.distribution(state, player)
        indexes = [card.index for card in legal]
        if temperature <= 0.0:
            return index_to_card(max(indexes, key=lambda i: probabilities[i]))
        weights = [probabilities[i] ** (1.0 / temperature) for i in indexes]
        total = sum(weights)
        if total <= 0.0:
            return index_to_card(indexes[rng.randrange(len(indexes))])
        threshold = rng.random() * total
        accumulated = 0.0
        for index, weight in zip(indexes, weights):
            accumulated += weight
            if threshold <= accumulated:
                return index_to_card(index)
        return index_to_card(indexes[-1])


def teacher_from_paths(paths: list[str]) -> EnsembleTeacher:
    from .exploit import spec_from_checkpoint
    return EnsembleTeacher([spec_from_checkpoint(p, temperature=0.0) for p in paths])


__all__ = ["EnsembleTeacher", "teacher_from_paths", "build_net"]
