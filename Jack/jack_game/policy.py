"""Policy network for net-guided search (ExIt Phase 2).

``PolicyNet`` maps a state to logits over the 52 cards. ``PolicyFunction`` loads a
checkpoint and returns a prior distribution over the *legal* actions, used as the
PUCT prior in ISMCTS. Bootstrapped by behaviour-cloning a teacher; later refined
on MCTS visit distributions in the self-play loop (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bc import FEATURE_DIM, NUM_CARDS, encode_state_from_game_state, resolve_device
from .cards import RANKS, SUITS, Card
from .engine import new_round, play_card
from .value import build_teacher  # heuristic / search / random self-play teachers


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        # num_layers = hidden layers; default (2) reproduces the original layer
        # indices (net.0/net.2/net.4) so existing checkpoints load unchanged.
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, NUM_CARDS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainPolicyResult:
    train_examples: int
    best_loss: float
    output_path: str


class PolicyFunction:
    """Callable prior: ``policy(state, player) -> {action_index: prob}`` over legal moves."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        num_layers = int(checkpoint.get("num_layers", 2))
        self.model = PolicyNet(input_dim=self.feature_dim, hidden_dim=hidden_dim,
                               num_layers=num_layers).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def __call__(self, state, player: int) -> dict[int, float]:
        legal = state.legal_actions(player)
        legal_indexes = [card.index for card in legal]
        features, _ = encode_state_from_game_state(state, player, feature_dim=self.feature_dim)
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            logits = self.model(x).squeeze(0)
        legal_logits = torch.tensor([float(logits[i]) for i in legal_indexes])
        probs = torch.softmax(legal_logits, dim=0)
        return {index: float(p) for index, p in zip(legal_indexes, probs)}


class GreedyPolicyPlayer:
    """Plays the argmax of the policy net over legal moves — zero simulations.

    With the distilled policy (`policy_distill800_*.pt`, trained on ISMCTS@800
    visit distributions) this is the fast production-strength player:
    ~1 ms/move at the strength of the net-guided search at ~600 sims.
    Accepts a checkpoint path or an already-loaded ``PolicyFunction``.
    """

    def __init__(self, policy_fn: "PolicyFunction | str") -> None:
        self.policy_fn = PolicyFunction(policy_fn) if isinstance(policy_fn, str) else policy_fn

    def choose_card(self, state, player: int, rng: Random) -> Card:
        probs = self.policy_fn(state, player)
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        return Card(SUITS[best // len(RANKS)], RANKS[best % len(RANKS)])


def generate_policy_examples(games: int, seed: int, teacher: str, rollouts: int, teacher_checkpoint: str | None = None):
    """Self-play with ``teacher``; record (features, chosen action, legal mask)."""
    rng = Random(seed)
    teacher_player = build_teacher(teacher, rollouts, teacher_checkpoint)
    feature_rows: list[list[float]] = []
    actions: list[int] = []
    legal_masks: list[list[bool]] = []
    for _ in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            features, illegal = encode_state_from_game_state(state, actor)
            card = teacher_player.choose_card(state.clone(), actor, rng)
            feature_rows.append(features)
            actions.append(card.index)
            legal_masks.append([not flag for flag in illegal])  # True where legal
            play_card(state, actor, card)
    return feature_rows, actions, legal_masks


def train_policy(
    output_path: str,
    games: int = 2000,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    teacher: str = "heuristic",
    rollouts: int = 24,
    teacher_checkpoint: str | None = None,
    seed: int = 1,
    device: str | None = None,
) -> TrainPolicyResult:
    device_obj = resolve_device(device)
    torch.manual_seed(seed)
    feature_rows, action_list, legal_masks = generate_policy_examples(
        games=games, seed=seed, teacher=teacher, rollouts=rollouts, teacher_checkpoint=teacher_checkpoint
    )

    features = torch.tensor(feature_rows, dtype=torch.float32)
    actions = torch.tensor(action_list, dtype=torch.long)
    legal = torch.tensor(legal_masks, dtype=torch.bool)
    loader = DataLoader(TensorDataset(features, actions, legal), batch_size=batch_size, shuffle=True)

    model = PolicyNet(hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for batch_features, batch_actions, batch_legal in loader:
            batch_features = batch_features.to(device_obj)
            batch_actions = batch_actions.to(device_obj)
            batch_legal = batch_legal.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            logits = logits.masked_fill(~batch_legal, float("-inf"))  # never predict illegal moves
            loss = nn.functional.cross_entropy(logits, batch_actions)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * batch_features.shape[0]
            seen += batch_features.shape[0]
        epoch_loss = loss_total / max(1, seen)
        print(f"epoch={epoch} examples={seen} ce={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    save_policy_checkpoint(
        Path(output_path), best_state, hidden_dim, best_loss,
        extra={"teacher": teacher, "games": games, "epochs": epochs},
    )
    return TrainPolicyResult(train_examples=len(action_list), best_loss=best_loss, output_path=output_path)


def save_policy_checkpoint(path: Path, model_state, hidden_dim: int, best_loss: float, extra=None,
                           feature_dim: int = FEATURE_DIM, num_layers: int = 2) -> None:
    payload = {
        "model_state": model_state,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "best_loss": best_loss,
        "model_kind": "policy",
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
