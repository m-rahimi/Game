"""Belief network: predict where the hidden cards sit (Step 2 of hand-reading).

``BeliefNet`` maps the 572-dim attributed features to a per-card distribution
over the three hidden hands (relative seats: 1=left opponent, 2=partner,
3=right opponent). Targets come from search self-play where the true deal is
known (``gen_search_data.py --record-hands``).

The bar it must clear (Gate A) is ``baseline_log_probs``: the capacity-
proportional dealer with void constraints — the belief implicitly used by
``sample_information_set`` today. Beating it = genuine hand-reading beyond
counting and voids.

Evaluation MUST use game-level splits (52 correlated rows per game; row splits
leak — see BEATING_SEARCH_PROGRESS.md).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .bc import (
    FEATURE_DIM_ATTR,
    HISTORY_DIMS_PER_CARD,
    NUM_CARDS,
    NUM_PLAYERS,
    NUM_SUITS,
    encode_state_from_game_state,
    resolve_device,
)

BELIEF_CLASSES = 3  # hidden hands: left opponent, partner, right opponent

# Offsets into the 572-dim attributed feature vector.
_HIST_START = NUM_CARDS
_HIST_END = NUM_CARDS + NUM_CARDS * HISTORY_DIMS_PER_CARD          # 416
_VOIDS_START = _HIST_END + NUM_CARDS + NUM_CARDS + NUM_SUITS       # 524 (skip trick/legal/jacks)
_ACTOR_START = _VOIDS_START + NUM_PLAYERS * NUM_SUITS + 4          # 544 (skip winner relation)


class BeliefNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM_ATTR, hidden_dim: int = 512, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, NUM_CARDS * BELIEF_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, NUM_CARDS, BELIEF_CLASSES)


def belief_loss(logits: torch.Tensor, card_loc: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over HIDDEN cards only (card_loc 1..3 -> class 0..2)."""
    hidden = card_loc > 0
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, BELIEF_CLASSES),
        (card_loc.long().clamp(min=1) - 1).reshape(-1),
        reduction="none",
    ).view_as(card_loc)
    return (ce * hidden).sum() / hidden.sum().clamp(min=1)


def hidden_counts_and_voids(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive, from attributed features, each hidden seat's remaining hand size
    (N, 3) and its void flags (N, 3, 4 suits), both in RELATIVE seat order.

    Assumes the features were built for the ACTING player (observer == the
    current-player block), which is true for every encoding site in the
    pipeline — datasets, PolicyFunction/ValueFunction/BeliefFunction all encode
    for the player to move."""
    n = features.shape[0]
    hist = features[:, _HIST_START:_HIST_END].view(n, NUM_CARDS, HISTORY_DIMS_PER_CARD)
    played_by = hist[:, :, :NUM_PLAYERS].sum(dim=1)                  # (N, 4 relative seats)
    counts = (13.0 - played_by[:, 1:])                               # (N, 3)

    voids_abs = features[:, _VOIDS_START:_VOIDS_START + NUM_PLAYERS * NUM_SUITS].view(n, NUM_PLAYERS, NUM_SUITS)
    actor = features[:, _ACTOR_START:_ACTOR_START + NUM_PLAYERS].argmax(dim=1)  # absolute actor seat
    rel_voids = torch.stack(
        [voids_abs[torch.arange(n), (actor + r) % NUM_PLAYERS] for r in (1, 2, 3)], dim=1
    )                                                                # (N, 3, 4)
    return counts, rel_voids


def baseline_log_probs(features: torch.Tensor) -> torch.Tensor:
    """(N, 52, 3) log-probs of the no-inference dealer: P(card -> seat) proportional
    to that seat's remaining hand size, zeroed where the seat is void in the
    card's suit. This is the implicit belief of uniform determinization."""
    counts, rel_voids = hidden_counts_and_voids(features)
    # card index // 13 = suit index (C,D,H,S ordering of Card.index)
    suit_of_card = torch.arange(NUM_CARDS) // 13                     # (52,)
    allowed = 1.0 - rel_voids[:, :, suit_of_card].transpose(1, 2)    # (N, 52, 3)
    weights = counts[:, None, :] * allowed                           # (N, 52, 3)
    weights = weights.clamp(min=1e-9)
    return (weights / weights.sum(dim=2, keepdim=True)).log()


class BeliefFunction:
    """Callable: ``belief(state, player) -> (52, 3) probs`` over hidden hands
    (relative seats: left opponent, partner, right opponent)."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM_ATTR))
        # Calibration temperature (fit on the game-split holdout; 1.0 = raw).
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.model = BeliefNet(
            input_dim=self.feature_dim,
            hidden_dim=int(checkpoint.get("hidden_dim", 512)),
            num_layers=int(checkpoint.get("num_layers", 2)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def __call__(self, state, player: int) -> torch.Tensor:
        features, _ = encode_state_from_game_state(state, player, feature_dim=self.feature_dim)
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            return torch.softmax(self.model(x)[0] / self.temperature, dim=1).cpu()


def save_belief_checkpoint(path: Path, model_state, hidden_dim: int, num_layers: int,
                           best_loss: float, extra: dict | None = None) -> None:
    payload = {
        "model_state": model_state,
        "feature_dim": FEATURE_DIM_ATTR,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "best_loss": best_loss,
        "model_kind": "belief",
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
