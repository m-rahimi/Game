from __future__ import annotations

import csv
import warnings

warnings.filterwarnings("ignore", message="CUDA initialization:")
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cards import Card, card_from_text, partner_of, team_for
from .engine import GameState, new_round, play_card
from .history import infer_void_suits, trick_penalty_cards, unseen_penalty_jacks
from .players import HeuristicPlayer, RandomPlayer, SearchPlayer

LEGACY_FEATURE_DIM = 179
FEATURE_DIM = 260
NUM_CARDS = 52
NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_TRICKS = 13
CARDS_PER_SUIT = 13
SUIT_ORDER = ["C", "D", "H", "S"]

# Attributed-history encoding: the modern 260 vector with its unattributed
# "played cards" 52-multi-hot replaced by a 7-dim record per card, capturing
# WHO played it and in what context — the belief-relevant signal the 260 lacks
# (it only kept voids). Per card: [relative seat one-hot (me/left/partner/right),
# was the lead of its trick, followed the led suit (0 = free discard, the most
# informative play type), trick index when played / 12]. All zeros if unseen.
HISTORY_DIMS_PER_CARD = 7
FEATURE_DIM_ATTR = FEATURE_DIM + NUM_CARDS * (HISTORY_DIMS_PER_CARD - 1)  # 572

# Hokm (trump) encoding: the 572 attributed vector plus an 8-dim trump block.
# A separate width, not an extension of 572, so every existing checkpoint keeps
# loading and every published no-trump result stays reproducible.
#
# Four of the eight are the trump suit one-hot; the other four are derived quantities a
# feedforward net cannot cheaply recover from the one-hot alone, and which decide most
# trump decisions:
#   [4] led suit IS trump (following suit and ruffing are the same act this trick)
#   [5] the trick has already been ruffed (a trump is currently winning off-suit)
#   [6] trumps in my hand / 13
#   [7] trumps still unseen by me / 13   (how much ruffing power is still out there)
# The project's precedent for explicit derived inputs is `value_context_features`,
# where 9 hand-computed dims cut held-out MSE almost in half.
TRUMP_BLOCK_DIM = 8
FEATURE_DIM_HOKM = FEATURE_DIM_ATTR + TRUMP_BLOCK_DIM  # 580


class BCPolicyNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_CARDS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainBCResult:
    train_examples: int
    val_examples: int
    best_val_accuracy: float
    output_path: str



@dataclass
class TeacherBatch:
    features: torch.Tensor
    illegal_masks: torch.Tensor
    actions: torch.Tensor
    danger_mask: torch.Tensor


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")


def build_teacher(
    name: str,
    rollouts: int = 24,
    lambda_clubs: float = 2.5,
    lambda_hearts: float = 0.25,
):
    if name == "random":
        return RandomPlayer()
    if name == "heuristic":
        return HeuristicPlayer()
    if name == "search":
        return SearchPlayer(
            rollouts_per_action=rollouts,
            lambda_clubs=lambda_clubs,
            lambda_hearts=lambda_hearts,
            time_limit_seconds=2.0,
        )
    raise ValueError(f"Unknown teacher policy: {name}")


class BCPlayer:
    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        self.model = BCPolicyNet(input_dim=self.feature_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        legal = state.legal_actions(player)
        features, mask = encode_state_from_game_state(state, player, feature_dim=self.feature_dim)
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(x).squeeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
            masked = logits.masked_fill(mask_t, -1e9)
            best_index = int(masked.argmax().item())
        for card in legal:
            if card.index == best_index:
                return card
        return legal[0]


def train_bc(
    data_path: str,
    output_path: str,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    val_fraction: float = 0.1,
    seed: int = 1,
    device: str | None = None,
) -> TrainBCResult:
    device_obj = resolve_device(device)
    torch.manual_seed(seed)

    features, illegal_masks, actions = load_bc_csv(data_path, feature_dim=FEATURE_DIM)
    total = features.shape[0]
    order = torch.randperm(total)
    val_count = max(1, int(total * val_fraction))
    val_idx = order[:val_count]
    train_idx = order[val_count:]

    train_ds = TensorDataset(features[train_idx], illegal_masks[train_idx], actions[train_idx])
    val_ds = TensorDataset(features[val_idx], illegal_masks[val_idx], actions[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = BCPolicyNet(hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device_obj)
        val_loss, val_acc = evaluate_bc(model, val_loader, loss_fn, device_obj)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "feature_dim": FEATURE_DIM,
            "hidden_dim": hidden_dim,
            "best_val_accuracy": best_acc,
            "data_path": data_path,
            "training_mode": "csv",
        },
        output_path,
    )
    return TrainBCResult(
        train_examples=len(train_ds),
        val_examples=len(val_ds),
        best_val_accuracy=best_acc,
        output_path=output_path,
    )




def train_bc_online(
    output_path: str,
    iterations: int = 5,
    games_per_iter: int = 100,
    epochs_per_iter: int = 1,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    val_fraction: float = 0.1,
    seed: int = 1,
    device: str | None = None,
    teacher_team0: str = "search",
    teacher_team1: str = "heuristic",
    focus_team: int | None = 0,
    rollouts: int = 24,
    lambda_clubs: float = 2.5,
    lambda_hearts: float = 0.25,
    replay_size: int = 50000,
    danger_multiplier: int = 4,
    save_every: int = 1,
) -> TrainBCResult:
    device_obj = resolve_device(device)
    torch.manual_seed(seed)
    rng = Random(seed)
    model = BCPolicyNet(hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    total_train_examples = 0
    total_val_examples = 0
    replay: list[TeacherBatch] = []

    for iteration in range(1, iterations + 1):
        fresh = generate_teacher_dataset(
            games=games_per_iter,
            seed=rng.randrange(10**9),
            teacher_team0=teacher_team0,
            teacher_team1=teacher_team1,
            focus_team=focus_team,
            rollouts=rollouts,
            lambda_clubs=lambda_clubs,
            lambda_hearts=lambda_hearts,
        )
        replay.append(fresh)
        replay = trim_teacher_replay(replay, replay_size)
        combined = concat_teacher_batches(replay)
        train_batch, val_batch = split_teacher_batch(combined, val_fraction)
        train_batch = oversample_danger_examples(train_batch, danger_multiplier)

        total_train_examples += int(train_batch.features.shape[0])
        total_val_examples += int(val_batch.features.shape[0])

        train_loader = DataLoader(
            TensorDataset(train_batch.features, train_batch.illegal_masks, train_batch.actions),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_batch.features, val_batch.illegal_masks, val_batch.actions),
            batch_size=batch_size,
        )

        for epoch in range(1, epochs_per_iter + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device_obj)
            val_loss, val_acc = evaluate_bc(model, val_loader, loss_fn, device_obj)
            print(
                f"iter={iteration} epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        if save_every > 0 and iteration % save_every == 0:
            snapshot_path = Path(output_path).with_suffix(f".iter{iteration:03d}.pt")
            save_teacher_checkpoint(
                snapshot_path,
                best_state,
                hidden_dim=hidden_dim,
                best_val_accuracy=best_acc,
                extra={
                    "training_mode": "online",
                    "teacher_team0": teacher_team0,
                    "teacher_team1": teacher_team1,
                    "focus_team": focus_team,
                    "games_per_iter": games_per_iter,
                    "iterations": iterations,
                    "replay_size": replay_size,
                    "danger_multiplier": danger_multiplier,
                    "iteration": iteration,
                },
            )

    save_teacher_checkpoint(
        Path(output_path),
        best_state,
        hidden_dim=hidden_dim,
        best_val_accuracy=best_acc,
        extra={
            "training_mode": "online",
            "teacher_team0": teacher_team0,
            "teacher_team1": teacher_team1,
            "focus_team": focus_team,
            "games_per_iter": games_per_iter,
            "iterations": iterations,
            "replay_size": replay_size,
            "danger_multiplier": danger_multiplier,
        },
    )
    return TrainBCResult(
        train_examples=total_train_examples,
        val_examples=total_val_examples,
        best_val_accuracy=best_acc,
        output_path=output_path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_total = 0.0
    correct = 0
    seen = 0
    for x, illegal, y in loader:
        x = x.to(device)
        illegal = illegal.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).masked_fill(illegal, -1e9)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item()) * x.shape[0]
        correct += int((logits.argmax(dim=1) == y).sum().item())
        seen += x.shape[0]
    return loss_total / seen, correct / seen


def evaluate_bc(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, illegal, y in loader:
            x = x.to(device)
            illegal = illegal.to(device)
            y = y.to(device)
            logits = model(x).masked_fill(illegal, -1e9)
            loss = loss_fn(logits, y)
            loss_total += float(loss.item()) * x.shape[0]
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += x.shape[0]
    return loss_total / seen, correct / seen




def generate_teacher_dataset(
    games: int,
    seed: int,
    teacher_team0: str,
    teacher_team1: str,
    focus_team: int | None,
    rollouts: int,
    lambda_clubs: float,
    lambda_hearts: float,
) -> TeacherBatch:
    rng = Random(seed)
    examples: list[tuple[list[float], list[bool], int, bool]] = []
    players = [
        build_teacher(teacher_team0, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts),
        build_teacher(teacher_team1, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts),
        build_teacher(teacher_team0, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts),
        build_teacher(teacher_team1, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts),
    ]
    for _ in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            action = players[actor].choose_card(state.clone(), actor, rng)
            if focus_team is None or team_for(actor) == focus_team:
                features, illegal = encode_state_from_game_state(state, actor, feature_dim=FEATURE_DIM)
                examples.append((features, illegal, action.index, is_danger_state(state, actor)))
            play_card(state, actor, action)
    return examples_to_tensors(examples)


def examples_to_tensors(
    examples: list[tuple[list[float], list[bool], int, bool]]
) -> TeacherBatch:
    features = torch.zeros((len(examples), FEATURE_DIM), dtype=torch.float32)
    illegal_masks = torch.ones((len(examples), NUM_CARDS), dtype=torch.bool)
    actions = torch.empty((len(examples),), dtype=torch.long)
    danger_mask = torch.zeros((len(examples),), dtype=torch.bool)
    for i, (vector, illegal, action_index, danger) in enumerate(examples):
        features[i] = torch.tensor(vector, dtype=torch.float32)
        illegal_masks[i] = torch.tensor(illegal, dtype=torch.bool)
        actions[i] = action_index
        danger_mask[i] = danger
    return TeacherBatch(features=features, illegal_masks=illegal_masks, actions=actions, danger_mask=danger_mask)


def concat_teacher_batches(batches: list[TeacherBatch]) -> TeacherBatch:
    if not batches:
        raise ValueError("replay buffer is empty")
    return TeacherBatch(
        features=torch.cat([batch.features for batch in batches], dim=0),
        illegal_masks=torch.cat([batch.illegal_masks for batch in batches], dim=0),
        actions=torch.cat([batch.actions for batch in batches], dim=0),
        danger_mask=torch.cat([batch.danger_mask for batch in batches], dim=0),
    )


def split_teacher_batch(batch: TeacherBatch, val_fraction: float) -> tuple[TeacherBatch, TeacherBatch]:
    total = int(batch.features.shape[0])
    if total <= 1:
        empty = TeacherBatch(
            features=batch.features[:0],
            illegal_masks=batch.illegal_masks[:0],
            actions=batch.actions[:0],
            danger_mask=batch.danger_mask[:0],
        )
        return batch, empty
    order = torch.randperm(total)
    val_count = max(1, int(total * val_fraction))
    if val_count >= total:
        val_count = total - 1
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    return take_teacher_batch(batch, train_idx), take_teacher_batch(batch, val_idx)


def take_teacher_batch(batch: TeacherBatch, indices: torch.Tensor) -> TeacherBatch:
    return TeacherBatch(
        features=batch.features[indices],
        illegal_masks=batch.illegal_masks[indices],
        actions=batch.actions[indices],
        danger_mask=batch.danger_mask[indices],
    )


def oversample_danger_examples(batch: TeacherBatch, multiplier: int) -> TeacherBatch:
    if multiplier <= 1 or batch.features.shape[0] == 0:
        return batch
    danger_idx = torch.nonzero(batch.danger_mask, as_tuple=False).squeeze(-1)
    if danger_idx.numel() == 0:
        return batch
    extra_idx = danger_idx.repeat_interleave(multiplier - 1)
    return TeacherBatch(
        features=torch.cat([batch.features, batch.features[extra_idx]], dim=0),
        illegal_masks=torch.cat([batch.illegal_masks, batch.illegal_masks[extra_idx]], dim=0),
        actions=torch.cat([batch.actions, batch.actions[extra_idx]], dim=0),
        danger_mask=torch.cat([batch.danger_mask, batch.danger_mask[extra_idx]], dim=0),
    )


def trim_teacher_replay(replay: list[TeacherBatch], max_examples: int) -> list[TeacherBatch]:
    if max_examples <= 0:
        return replay
    trimmed = list(replay)
    total = sum(int(batch.features.shape[0]) for batch in trimmed)
    while trimmed and total > max_examples:
        total -= int(trimmed[0].features.shape[0])
        trimmed.pop(0)
    return trimmed


def save_teacher_checkpoint(
    path: str | Path,
    model_state: dict[str, torch.Tensor] | None,
    hidden_dim: int,
    best_val_accuracy: float,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "model_state": model_state,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": hidden_dim,
        "best_val_accuracy": best_val_accuracy,
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def is_danger_state(state: GameState, player: int) -> bool:
    legal = state.legal_actions(player)
    if trick_penalty_cards(state):
        return True
    if not state.current_trick and bool(unseen_penalty_jacks(state)):
        return True
    if any(card.is_penalty_jack for card in legal):
        return True
    if state.current_trick and len(legal) > 1:
        return True
    return False


def load_bc_csv(path: str, feature_dim: int = FEATURE_DIM) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = torch.zeros((len(rows), feature_dim), dtype=torch.float32)
    illegal_masks = torch.ones((len(rows), NUM_CARDS), dtype=torch.bool)
    actions = torch.empty((len(rows),), dtype=torch.long)
    for i, row in enumerate(rows):
        vector, illegal = encode_state_from_row(row, feature_dim=feature_dim)
        features[i] = torch.tensor(vector, dtype=torch.float32)
        illegal_masks[i] = torch.tensor(illegal, dtype=torch.bool)
        actions[i] = card_from_text(row["action"]).index
    return features, illegal_masks, actions


def encode_state_from_row(row: dict[str, str], feature_dim: int = FEATURE_DIM) -> tuple[list[float], list[bool]]:
    player = int(row["player"])
    trick = int(row["trick"])
    team = int(row["team"])
    hand = parse_cards(row.get("hand", ""))
    current_trick = parse_trick_cards(row.get("current_trick", ""))
    legal = parse_cards(row.get("legal_actions", ""))

    if feature_dim == LEGACY_FEATURE_DIM:
        return build_legacy_features(
            player=player,
            trick=trick,
            team=team,
            hand=hand,
            current_trick=current_trick,
            legal=legal,
        )

    played_cards = parse_cards(row.get("played_cards", ""))
    void_suits = parse_void_suits(row.get("void_suits", ""))
    current_winner = parse_optional_int(row.get("current_winner"))
    current_player = parse_optional_int(row.get("current_player"))
    return build_modern_features(
        player=player,
        trick=trick,
        team=team,
        hand=hand,
        played_cards=played_cards,
        current_trick=current_trick,
        legal=legal,
        void_suits=void_suits,
        current_winner=current_winner,
        current_player=current_player,
    )


def encode_state_from_game_state(
    state: GameState,
    player: int,
    feature_dim: int = FEATURE_DIM,
) -> tuple[list[float], list[bool]]:
    if feature_dim == LEGACY_FEATURE_DIM:
        return build_legacy_features(
            player=player,
            trick=state.trick_index,
            team=team_for(player),
            hand=state.hands[player],
            current_trick=[card for _, card in state.current_trick],
            legal=state.legal_actions(player),
        )
    if feature_dim == FEATURE_DIM_HOKM:
        return build_hokm_features(state, player)
    if feature_dim == FEATURE_DIM_ATTR:
        return build_attributed_features(state, player)
    return build_modern_features(
        player=player,
        trick=state.trick_index,
        team=team_for(player),
        hand=state.hands[player],
        played_cards=state.played_cards,
        current_trick=[card for _, card in state.current_trick],
        legal=state.legal_actions(player),
        void_suits=infer_void_suits(state),
        current_winner=state.current_winner(),
        current_player=state.current_player,
    )


def build_history_block(state: GameState, player: int) -> list[float]:
    """52 x 7 attributed play history from ``player``'s point of view.

    Covers every played card (completed tricks AND the current trick, matching
    ``state.played_cards``). Seats are encoded relative to the observer so the
    pattern is seat-invariant: 0=me, 1=left opponent, 2=partner, 3=right opponent.
    """
    block = [0.0] * (NUM_CARDS * HISTORY_DIMS_PER_CARD)
    tricks: list[tuple[int, tuple[tuple[int, Card], ...]]] = [
        (index, trick.plays) for index, trick in enumerate(state.completed_tricks)
    ]
    if state.current_trick:
        tricks.append((state.trick_index, tuple(state.current_trick)))
    for trick_index, plays in tricks:
        led_suit = plays[0][1].suit
        for position, (seat, card) in enumerate(plays):
            base = card.index * HISTORY_DIMS_PER_CARD
            block[base + (seat - player) % NUM_PLAYERS] = 1.0
            block[base + 4] = 1.0 if position == 0 else 0.0
            block[base + 5] = 1.0 if card.suit == led_suit else 0.0
            block[base + 6] = trick_index / (NUM_TRICKS - 1)
    return block


def build_attributed_features(state: GameState, player: int) -> tuple[list[float], list[bool]]:
    """The modern 260 vector with its played-cards block replaced by the
    attributed history block. Built by splicing into ``build_modern_features``
    output, so ``attributed_to_modern`` recovers the modern vector exactly."""
    modern, illegal = build_modern_features(
        player=player,
        trick=state.trick_index,
        team=team_for(player),
        hand=state.hands[player],
        played_cards=state.played_cards,
        current_trick=[card for _, card in state.current_trick],
        legal=state.legal_actions(player),
        void_suits=infer_void_suits(state),
        current_winner=state.current_winner(),
        current_player=state.current_player,
    )
    history = build_history_block(state, player)
    vector = modern[:NUM_CARDS] + history + modern[2 * NUM_CARDS :]
    return vector, illegal


def build_trump_block(state: GameState, player: int) -> list[float]:
    """The 8 Hokm dims documented at ``TRUMP_BLOCK_DIM``. All zeros when there is no
    trump, so a Hokm-width net fed a no-trump state degrades to the 572 behaviour."""
    block = [0.0] * TRUMP_BLOCK_DIM
    trump = state.trump
    if trump is None:
        return block
    block[SUIT_ORDER.index(trump)] = 1.0
    if state.current_trick:
        led_suit = state.current_trick[0][1].suit
        block[4] = 1.0 if led_suit == trump else 0.0
        # "Ruffed" means a trump is winning a trick it did not lead — the situation
        # where following suit can no longer win and the decision changes completely.
        if led_suit != trump and any(card.suit == trump for _, card in state.current_trick):
            block[5] = 1.0
    mine = sum(1 for card in state.hands[player] if card.suit == trump)
    block[6] = mine / float(CARDS_PER_SUIT)
    # play_card adds to played_cards as each card hits the table, so this already
    # counts the cards in the current (incomplete) trick.
    seen = sum(1 for card in state.played_cards if card.suit == trump)
    block[7] = max(CARDS_PER_SUIT - mine - seen, 0) / float(CARDS_PER_SUIT)
    return block


def build_hokm_features(state: GameState, player: int) -> tuple[list[float], list[bool]]:
    vector, illegal = build_attributed_features(state, player)
    return vector + build_trump_block(state, player), illegal


def attributed_to_modern(features: torch.Tensor) -> torch.Tensor:
    """Project (N, 572) attributed features to the exact (N, 260) modern vector.

    The unattributed played-card bit is the max over the 4 relative-seat
    channels; all other blocks are shared verbatim. Lets one attributed dataset
    train BOTH arms of a with/without-attribution comparison on the same games.
    """
    if features.shape[1] != FEATURE_DIM_ATTR:
        raise ValueError(f"expected dim {FEATURE_DIM_ATTR}, got {features.shape[1]}")
    hand = features[:, :NUM_CARDS]
    history = features[:, NUM_CARDS : NUM_CARDS + NUM_CARDS * HISTORY_DIMS_PER_CARD]
    played = history.reshape(-1, NUM_CARDS, HISTORY_DIMS_PER_CARD)[:, :, :NUM_PLAYERS].amax(dim=2)
    tail = features[:, NUM_CARDS + NUM_CARDS * HISTORY_DIMS_PER_CARD :]
    return torch.cat([hand, played, tail], dim=1)


def build_legacy_features(
    player: int,
    trick: int,
    team: int,
    hand: list[Card],
    current_trick: list[Card],
    legal: list[Card],
) -> tuple[list[float], list[bool]]:
    vector = [0.0] * LEGACY_FEATURE_DIM
    offset = 0
    for card in hand:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for card in current_trick:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    legal_indexes = {card.index for card in legal}
    for index in legal_indexes:
        vector[offset + index] = 1.0
    offset += NUM_CARDS
    vector[offset + player] = 1.0
    offset += 4
    vector[offset + max(0, min(12, trick))] = 1.0
    offset += 13
    vector[offset + team] = 1.0
    offset += 2
    vector[offset + min(3, len(current_trick))] = 1.0
    illegal = [index not in legal_indexes for index in range(NUM_CARDS)]
    return vector, illegal


def build_modern_features(
    player: int,
    trick: int,
    team: int,
    hand: list[Card],
    played_cards: set[Card] | list[Card],
    current_trick: list[Card],
    legal: list[Card],
    void_suits: list[set[str]] | None = None,
    current_winner: int | None = None,
    current_player: int | None = None,
) -> tuple[list[float], list[bool]]:
    vector = [0.0] * FEATURE_DIM
    played_set = set(played_cards)
    legal_indexes = {card.index for card in legal}
    trick_cards = list(current_trick)
    current_penalties = [card for card in trick_cards if card.is_penalty_jack]
    if void_suits is None:
        void_suits = [set() for _ in range(NUM_PLAYERS)]

    offset = 0
    for card in hand:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for card in played_set:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for card in trick_cards:
        vector[offset + card.index] = 1.0
    offset += NUM_CARDS
    for index in legal_indexes:
        vector[offset + index] = 1.0
    offset += NUM_CARDS
    for card in unseen_penalty_jacks_from_played(played_set):
        vector[offset + card.index] = 1.0
    offset += NUM_SUITS
    for seat in range(NUM_PLAYERS):
        for suit_index, suit in enumerate(SUIT_ORDER):
            vector[offset + seat * NUM_SUITS + suit_index] = 1.0 if suit in void_suits[seat] else 0.0
    offset += NUM_PLAYERS * NUM_SUITS
    relation = 0
    if current_winner is not None:
        if current_winner == player:
            relation = 1
        elif current_winner == partner_of(player):
            relation = 2
        else:
            relation = 3
    vector[offset + relation] = 1.0
    offset += 4
    if current_player is None:
        current_player = player
    vector[offset + current_player] = 1.0
    offset += NUM_PLAYERS
    vector[offset + team] = 1.0
    offset += 2
    vector[offset + min(NUM_TRICKS - 1, trick)] = 1.0
    offset += NUM_TRICKS
    vector[offset + min(NUM_PLAYERS - 1, len(trick_cards))] = 1.0
    offset += NUM_PLAYERS
    led_suit = trick_cards[0].suit if trick_cards else None
    if led_suit is not None:
        vector[offset + SUIT_ORDER.index(led_suit)] = 1.0
    offset += NUM_SUITS
    vector[offset] = sum(card.penalty_severity for card in current_penalties) / 10.0

    illegal = [index not in legal_indexes for index in range(NUM_CARDS)]
    return vector, illegal


def parse_cards(text: str) -> list[Card]:
    text = text.strip()
    if not text:
        return []
    return [card_from_text(token) for token in text.split()]


def parse_trick_cards(text: str) -> list[Card]:
    text = text.strip()
    if not text:
        return []
    cards = []
    for token in text.split():
        if ":" in token:
            _, card_text = token.split(":", 1)
        else:
            card_text = token
        cards.append(card_from_text(card_text))
    return cards


def parse_optional_int(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    return int(text)


def parse_void_suits(text: str | None) -> list[set[str]]:
    voids = [set() for _ in range(NUM_PLAYERS)]
    if not text:
        return voids
    for seat_token in text.split("|"):
        if not seat_token:
            continue
        seat_text, suits_text = seat_token.split(":", 1)
        seat = int(seat_text.lstrip("P"))
        voids[seat] = {suit for suit in suits_text.split(",") if suit}
    return voids


def unseen_penalty_jacks_from_played(played_cards: set[Card]) -> list[Card]:
    jacks = [Card("C", 11), Card("D", 11), Card("H", 11), Card("S", 11)]
    return [card for card in jacks if card not in played_cards]
