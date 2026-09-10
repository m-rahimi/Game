"""Oracle -> student distillation (Step 2A of the search-free-RL roadmap, 2026-08-28).

The problem
-----------
`checkpoints/exploiter_oracle.pt` plays at +1.34 pts/round (61.4% per round) against
the frozen `policy_distill800_l3` — the user's target level. But it CHEATS: 156 of
its 728 inputs are the ground-truth locations of every hidden card. It cannot ship.

The student
-----------
One net, 572 observable inputs, that must reach the same level by *inferring* what
the teacher is told. Trained on two targets, both free in self-play because the
generator deals the cards:

  policy head (52)  <- KL to the teacher's move distribution
  belief head (52x3) <- cross-entropy to the TRUE hidden-card locations

Why the belief head is load-bearing: without it the student is asked to predict moves
that depend on information it does not have, and the gradient merely pushes it toward
the teacher's *average* move over all hands consistent with its observation — a
blurrier policy, not inference. The belief head supplies 52 dense labels per decision
instead of 1, on exactly the missing variable, forcing the trunk to encode where the
cards are. That representation is what the policy head needs in order to imitate a
teacher whose choices depend on the hands.

At inference only the policy head runs. :func:`export_policy_state` drops the belief
head and emits a plain ``PolicyNet`` checkpoint, so every existing tool
(``PolicyFunction``, ``GreedyPolicyPlayer``, ``PolicyActor``, ``train_exploiter.py
--target``, the ``policy`` CLI player) loads it unchanged: one net, one forward pass,
~1 ms/move. The belief head is training scaffolding, not a second model.

The irreducible gap: where a situation is genuinely ambiguous the teacher KNOWS and
the student can only guess, so the student cannot fully match +1.34 by imitation
alone. How much it recovers is the measurement; RL (Step 2C) closes the remainder.
"""
from __future__ import annotations

from multiprocessing import Pool
from random import Random

import torch
from torch import nn

from .bc import FEATURE_DIM_ATTR, NUM_CARDS
from .belief import BELIEF_CLASSES, belief_loss, baseline_log_probs
from .engine import new_round, play_card
from .exploit import (
    ActorSpec,
    PolicyActor,
    build_oracle_card_loc,
    exploiter_seat_set,
)
from .policy import PolicyNet


# --------------------------------------------------------------------------
# the student network
# --------------------------------------------------------------------------

class StudentNet(nn.Module):
    """Shared trunk -> policy head (52) + belief head (52 x 3).

    The trunk plus the policy head are laid out to be *exactly* a ``PolicyNet``, so
    :func:`export_policy_state` is a pure key rename with no numerical change.
    """

    def __init__(self, input_dim: int = FEATURE_DIM_ATTR, hidden_dim: int = 512,
                 num_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, NUM_CARDS)
        self.belief_head = nn.Linear(hidden_dim, NUM_CARDS * BELIEF_CLASSES)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.trunk(x)
        belief = self.belief_head(latent).view(-1, NUM_CARDS, BELIEF_CLASSES)
        return self.policy_head(latent), belief


def export_policy_state(student: StudentNet) -> dict:
    """Student trunk + policy head -> a plain ``PolicyNet`` state dict.

    ``PolicyNet`` is ``[Linear, ReLU] * num_layers + [Linear]``, so its linear layers
    sit at ``net.0, net.2, ..., net.2*num_layers``. The student's trunk uses the same
    even indices, and its policy head becomes the final one.
    """
    state = {}
    for key, value in student.trunk.state_dict().items():
        state[f"net.{key}"] = value.detach().cpu().clone()
    final = 2 * student.num_layers
    state[f"net.{final}.weight"] = student.policy_head.weight.detach().cpu().clone()
    state[f"net.{final}.bias"] = student.policy_head.bias.detach().cpu().clone()
    return state


def export_policy_net(student: StudentNet) -> PolicyNet:
    """The deployable net: same policy logits as the student, no belief head."""
    model = PolicyNet(input_dim=student.input_dim, hidden_dim=student.hidden_dim,
                      num_layers=student.num_layers)
    model.load_state_dict(export_policy_state(student))
    model.eval()
    return model


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)


def distill_policy_loss(logits: torch.Tensor, teacher: torch.Tensor,
                        legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (cross-entropy to minimise, true KL for reporting).

    ``teacher`` is a probability vector over the 52 cards, zero on illegal actions.
    The gradient comes from the cross-entropy term; the teacher's own entropy is a
    constant, added back only so the reported number is a real KL (>= 0, 0 iff equal).
    """
    student_logp = masked_log_softmax(logits, legal)
    safe_logp = student_logp.masked_fill(~legal, 0.0)  # 0 * -inf would be nan
    cross_entropy = -(teacher * safe_logp).sum(dim=1).mean()
    with torch.no_grad():
        teacher_logp = torch.log(teacher.clamp(min=1e-12))
        teacher_entropy = -(teacher * teacher_logp.masked_fill(~legal, 0.0)).sum(dim=1).mean()
    return cross_entropy, cross_entropy - teacher_entropy


def student_loss(student: StudentNet, features: torch.Tensor, teacher: torch.Tensor,
                 legal: torch.Tensor, card_loc: torch.Tensor, belief_weight: float):
    policy_logits, belief_logits = student(features)
    cross_entropy, kl = distill_policy_loss(policy_logits, teacher, legal)
    belief = belief_loss(belief_logits, card_loc)
    return cross_entropy + belief_weight * belief, kl.detach(), belief.detach()


@torch.no_grad()
def teacher_agreement(logits: torch.Tensor, teacher: torch.Tensor,
                      legal: torch.Tensor) -> float:
    """Fraction of decisions where the student's argmax matches the teacher's."""
    student_choice = logits.masked_fill(~legal, float("-inf")).argmax(dim=1)
    return float((student_choice == teacher.argmax(dim=1)).float().mean())


@torch.no_grad()
def belief_logloss(belief_logits: torch.Tensor, card_loc: torch.Tensor) -> float:
    return float(belief_loss(belief_logits, card_loc))


@torch.no_grad()
def baseline_belief_logloss(features: torch.Tensor, card_loc: torch.Tensor) -> float:
    """The no-inference reference: the implicit belief of uniform determinization
    (capacity-proportional, void-masked). The belief head must beat this to have
    learned any hand reading at all. Comparable to the standalone belief net's
    0.9992 vs baseline 1.0518 (see jack-hand-reading)."""
    return float(belief_loss(baseline_log_probs(features[:, :FEATURE_DIM_ATTR]), card_loc))


# --------------------------------------------------------------------------
# data generation: oracle teacher vs frozen target
# --------------------------------------------------------------------------

def _oracle_chunk(arg):
    (games, seed, game_base, teacher_spec, target_spec, seats) = arg
    torch.set_num_threads(1)  # avoid thread oversubscription across many workers
    rng = Random(seed)
    teacher = PolicyActor(teacher_spec)
    target = PolicyActor(target_spec)

    feats: list[list[float]] = []
    dists: list[list[float]] = []
    locs: list[list[int]] = []
    legals: list[list[bool]] = []
    outcomes: list[float] = []
    game_ids: list[int] = []

    for index in range(games):
        team = index % 2  # alternate which partnership the teacher holds
        teacher_seats = exploiter_seat_set(team, seats)
        state = new_round(seed=rng.randrange(10 ** 9), leader=index % 4)
        pending: list[tuple[int, int]] = []  # (row, acting seat)
        while not state.round_over:
            seat = state.current_player
            legal = state.legal_actions(seat)
            if len(legal) == 1:
                play_card(state, seat, legal[0])  # forced: nothing to imitate
                continue
            if seat not in teacher_seats:
                card, *_ = target.act(state, seat, rng)
                play_card(state, seat, card)
                continue
            card, _, features, legal_mask, _, distribution = teacher.act_full(state, seat, rng)
            # The teacher's inputs are [572 observable | 156 oracle]; the student
            # only ever gets the observable prefix.
            feats.append(features[:FEATURE_DIM_ATTR])
            dists.append(distribution)
            locs.append(build_oracle_card_loc(state, seat))
            legals.append(legal_mask)
            pending.append((len(outcomes), seat))
            outcomes.append(0.0)
            game_ids.append(game_base + index)
            play_card(state, seat, card)
        for row, seat in pending:
            outcomes[row] = float(state.team_diff_for(seat))

    return (
        torch.tensor(feats, dtype=torch.float16),
        torch.tensor(dists, dtype=torch.float16),
        torch.tensor(locs, dtype=torch.uint8),
        torch.tensor(legals, dtype=torch.bool),
        torch.tensor(outcomes, dtype=torch.float16),
        torch.tensor(game_ids, dtype=torch.int32),
    )


def generate_oracle_data(teacher_spec: ActorSpec, target_spec: ActorSpec, games: int,
                         workers: int, seed: int, seats: int = 2) -> dict:
    """Play ``games`` rounds of oracle-teacher vs frozen-target and record, for every
    non-forced teacher decision: observable features (572), the teacher's move
    distribution (52), the true hidden-card labels (52), the legal mask, the round
    outcome (kept for a value head in Step 2C) and a game id for game-level splits.
    """
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    tasks = []
    base = 0
    for count in per:
        if count > 0:
            tasks.append((count, seed + 7919 * len(tasks), base, teacher_spec,
                          target_spec, seats))
            base += count

    parts: list[tuple] = []
    if len(tasks) == 1:
        parts.append(_oracle_chunk(tasks[0]))
    else:
        with Pool(processes=len(tasks)) as pool:
            for result in pool.imap_unordered(_oracle_chunk, tasks):
                parts.append(result)

    keys = ("features", "teacher", "card_loc", "legal", "outcome", "game_id")
    data: dict[str, torch.Tensor] = {}
    for position, key in enumerate(keys):
        pieces = [part[position] for part in parts if part[position].numel() > 0]
        data[key] = torch.cat(pieces) if pieces else torch.empty(0)
    return data


# --------------------------------------------------------------------------
# game-level split (never row-level: rows in a game share the outcome)
# --------------------------------------------------------------------------

def game_level_split(game_id: torch.Tensor, holdout_fraction: float, seed: int):
    """-> (train_index, holdout_index), split on whole games.

    Row-level splits leak in this project: rows within a game are correlated and
    share the outcome, which rewards memorisation. Established methodology rule.
    """
    unique = torch.unique(game_id)
    generator = torch.Generator().manual_seed(seed)
    order = unique[torch.randperm(len(unique), generator=generator)]
    cut = max(1, int(len(order) * holdout_fraction))
    holdout_games = set(order[:cut].tolist())
    mask = torch.tensor([int(g) in holdout_games for g in game_id.tolist()])
    return (~mask).nonzero(as_tuple=True)[0], mask.nonzero(as_tuple=True)[0]
