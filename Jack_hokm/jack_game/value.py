"""State-value network for net-guided search (ExIt Phase 1).

``ValueNet`` predicts the final team-score differential from the acting player's
perspective. ``ValueFunction`` loads a checkpoint and is callable as
``value(state, player) -> float``; it is plugged into ``SearchPlayer`` to replace
full heuristic rollouts with a truncated rollout + learned leaf value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bc import FEATURE_DIM, encode_state_from_game_state, resolve_device
from .cards import SUITS, team_for
from .engine import GameState, new_round, play_card
from .players import HeuristicPlayer, RandomPlayer, SearchPlayer

# Extra inputs for the VALUE net only (the policy doesn't need them: scoring is
# additive, so optimal future play is independent of sunk score). The value
# target is sunk_diff + future_diff, so without the sunk part the net must guess
# it from played cards (impossible — it can't see who won past tricks). We feed
# the sunk part directly: running team-score diff + per-suit Jack-capture status.
VALUE_CONTEXT_DIM = 1 + 2 * len(SUITS)  # score diff + (mine, opp) per suit = 9


def value_context_features(state: GameState, player: int) -> list[float]:
    my_team = team_for(player)
    feats = [state.team_diff_for(player) / 13.0]
    captured: dict[str, int] = {}
    for result in state.completed_tricks:
        for card in result.penalties:
            captured[card.suit] = team_for(result.winner)
    for suit in SUITS:
        team = captured.get(suit)
        feats.append(1.0 if team == my_team else 0.0)                       # my team took this Jack
        feats.append(1.0 if (team is not None and team != my_team) else 0.0)  # opponents took it
    return feats


class ValueNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        # num_layers = hidden layers; default (2) reproduces the original layer
        # indices (net.0/net.2/net.4) so existing checkpoints load unchanged.
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainValueResult:
    train_examples: int
    best_loss: float
    output_path: str


class ValueFunction:
    """Callable state-value: ``value(state, player) -> predicted final team diff``."""

    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        # Checkpoints predating the value-context fix lack this key -> 0 (legacy).
        self.value_context_dim = int(checkpoint.get("value_context_dim", 0))
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        num_layers = int(checkpoint.get("num_layers", 2))
        self.model = ValueNet(input_dim=self.feature_dim + self.value_context_dim, hidden_dim=hidden_dim,
                              num_layers=num_layers).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def __call__(self, state, player: int) -> float:
        features, _ = encode_state_from_game_state(state, player, feature_dim=self.feature_dim)
        if self.value_context_dim:
            features = features + value_context_features(state, player)
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            return float(self.model(x).item())


def build_teacher(name: str, rollouts: int = 24, checkpoint: str | None = None):
    if name == "random":
        return RandomPlayer()
    if name == "heuristic":
        return HeuristicPlayer()
    if name == "search":
        return SearchPlayer(rollouts_per_action=rollouts, time_limit_seconds=2.0)
    if name in ("dmc", "dmc_guarded"):
        if checkpoint is None:
            raise ValueError(f"{name} teacher requires --teacher-checkpoint")
        from .dmc import DMCPlayer, GuardedDMCPlayer

        return GuardedDMCPlayer(checkpoint) if name == "dmc_guarded" else DMCPlayer(checkpoint)
    raise ValueError(f"Unknown value teacher: {name}")


def generate_value_examples(
    games: int,
    seed: int,
    teacher: str,
    rollouts: int,
    teacher_checkpoint: str | None = None,
) -> list[tuple[list[float], float]]:
    """Self-play with ``teacher`` in all seats; label each decision state with the
    realized final team-score diff from that actor's perspective (Monte-Carlo value)."""
    rng = Random(seed)
    # One stateless teacher instance reused for every seat and game (avoids reloading
    # a neural checkpoint thousands of times).
    teacher_player = build_teacher(teacher, rollouts, teacher_checkpoint)
    examples: list[tuple[list[float], float]] = []
    for _ in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        pending: list[tuple[list[float], int]] = []
        while not state.round_over:
            actor = state.current_player
            features, _ = encode_state_from_game_state(state, actor)
            pending.append((features, actor))
            card = teacher_player.choose_card(state.clone(), actor, rng)
            play_card(state, actor, card)
        for features, actor in pending:
            examples.append((features, float(state.team_diff_for(actor))))
    return examples


def train_value(
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
) -> TrainValueResult:
    device_obj = resolve_device(device)
    torch.manual_seed(seed)
    examples = generate_value_examples(
        games=games, seed=seed, teacher=teacher, rollouts=rollouts, teacher_checkpoint=teacher_checkpoint
    )

    features = torch.tensor([row for row, _ in examples], dtype=torch.float32)
    targets = torch.tensor([value for _, value in examples], dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=True)

    model = ValueNet(hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device_obj)
            batch_targets = batch_targets.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_features)
            loss = loss_fn(preds, batch_targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * batch_features.shape[0]
            seen += batch_features.shape[0]
        epoch_loss = loss_total / max(1, seen)
        print(f"epoch={epoch} examples={seen} mse={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    save_value_checkpoint(
        Path(output_path),
        best_state,
        hidden_dim,
        best_loss,
        extra={"teacher": teacher, "games": games, "epochs": epochs},
    )
    return TrainValueResult(train_examples=len(examples), best_loss=best_loss, output_path=output_path)


def save_value_checkpoint(
    path: Path,
    model_state: dict[str, torch.Tensor] | None,
    hidden_dim: int,
    best_loss: float,
    extra: dict[str, object] | None = None,
    value_context_dim: int = 0,
    feature_dim: int = FEATURE_DIM,
    num_layers: int = 2,
) -> None:
    payload = {
        "model_state": model_state,
        "feature_dim": feature_dim,
        "value_context_dim": value_context_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "best_loss": best_loss,
        "model_kind": "value",
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
