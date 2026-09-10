from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bc import FEATURE_DIM, NUM_CARDS, encode_state_from_game_state, resolve_device
from .cards import Card, partner_of, team_for
from .engine import GameState, new_round, play_card
from .history import trick_penalty_cards, unseen_penalty_jacks
from .players import HeuristicPlayer, RandomPlayer, SearchPlayer, would_win_current_trick

# Action-context features appended to the Q input alongside the action one-hot.
# These expose the same causal signals the search/heuristic players reason about,
# so DMC does not have to rediscover them indirectly from sparse returns.
ACTION_CONTEXT_DIM = 16

DMC_INPUT_DIM = FEATURE_DIM + NUM_CARDS + ACTION_CONTEXT_DIM


def action_context_features(state: GameState, player: int, card: Card) -> list[float]:
    """Per-(state, action) causal features for the DMC Q input.

    All values are in roughly [0, 1] so they sit on the same scale as the
    existing one-hot/state features. Computed from the state *before* ``card``
    is played, so it is valid both at data-generation time (the taken action)
    and at inference time (each legal candidate).
    """
    leading = not state.current_trick
    led_suit = state.current_trick[0][1].suit if not leading else None
    winner = state.current_winner()
    partner = partner_of(player)
    my_team = team_for(player)

    winner_is_self = winner == player
    winner_is_partner = winner == partner
    winner_is_opponent = winner is not None and team_for(winner) != my_team

    follows_suit = (not leading) and card.suit == led_suit
    would_win = (not leading) and would_win_current_trick(state, card)

    trick_penalties = trick_penalty_cards(state)
    penalty_load = sum(c.penalty_severity for c in trick_penalties)
    unseen_jacks = unseen_penalty_jacks(state)
    suit_jack = Card(card.suit, 11)
    led_jack = Card(led_suit, 11) if led_suit is not None else None
    is_penalty_jack = card.is_penalty_jack

    return [
        (card.rank - 2) / 12.0,                                              # action_rank_normalized
        1.0 if is_penalty_jack else 0.0,                                     # action_is_penalty_jack
        card.penalty_severity / 4.0,                                         # action_penalty_severity
        1.0 if follows_suit else 0.0,                                        # action_follows_suit
        1.0 if would_win else 0.0,                                           # action_would_win_current_trick
        1.0 if (would_win and winner_is_partner) else 0.0,                   # action_would_overtake_partner
        1.0 if (would_win and winner_is_opponent) else 0.0,                  # action_would_overtake_opponent
        1.0 if winner_is_self else 0.0,                                      # current_winner_is_self
        1.0 if winner_is_partner else 0.0,                                   # current_winner_is_partner
        1.0 if winner_is_opponent else 0.0,                                  # current_winner_is_opponent
        1.0 if trick_penalties else 0.0,                                     # current_trick_has_penalty_jack
        penalty_load / 10.0,                                                 # current_trick_penalty_load
        1.0 if (is_penalty_jack and winner_is_partner and not would_win) else 0.0,   # action_dumps_penalty_on_partner
        1.0 if (is_penalty_jack and winner_is_opponent and not would_win) else 0.0,  # action_dumps_penalty_on_opponent
        1.0 if (leading and suit_jack in unseen_jacks) else 0.0,             # action_leads_suit_with_unseen_jack
        1.0 if (led_jack is not None and led_jack in unseen_jacks) else 0.0, # led_suit_jack_unseen
    ]


class DMCQNet(nn.Module):
    def __init__(self, input_dim: int = DMC_INPUT_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DMCExample:
    features: list[float]
    action_index: int
    player: int
    action_context: list[float]
    trick: int


@dataclass
class DMCBatch:
    features: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    action_context: torch.Tensor


@dataclass
class TrainDMCResult:
    train_examples: int
    best_loss: float
    output_path: str


class DMCPlayer:
    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        # Checkpoints predating action-context features lack this key -> 0 (legacy input).
        self.action_context_dim = int(checkpoint.get("action_context_dim", 0))
        input_dim = self.feature_dim + NUM_CARDS + self.action_context_dim
        self.model = DMCQNet(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        return choose_q_card(
            self.model, self.device, state, player, rng,
            epsilon=0.0, action_context_dim=self.action_context_dim,
        )


# A lead is "dangerous" if leading it tends to pull a penalty Jack onto your own
# team. The oracle diagnostic showed DMC's worst EV losses are exactly these leads
# (e.g. leading KC with the club Jack live -> self-captures it ~96% of the time).
HIGH_LEAD_RANK = 12  # Q, K, A


def is_dangerous_lead(state: GameState, player: int, card: Card, high_lead_rank: int = HIGH_LEAD_RANK) -> bool:
    """True if leading ``card`` risks dragging a penalty Jack onto the leader's team.

    Two cases: (1) leading a bare penalty Jack, or (2) leading a high card in a suit
    whose Jack is still held by *another* player (unseen and not in the leader's own
    hand), which invites opponents to dump that Jack on the trick the leader wins.
    """
    if card.is_penalty_jack:
        return True
    suit_jack = Card(card.suit, 11)
    jack_held_elsewhere = suit_jack not in state.played_cards and suit_jack not in state.hands[player]
    return jack_held_elsewhere and card.rank >= high_lead_rank


class GuardedDMCPlayer:
    """DMC policy with a narrow guard on lead decisions only.

    Follow/discard play is left entirely to DMC (where the diagnostic showed it is
    sound). On leads, the candidate set is restricted to non-dangerous cards before
    DMC ranks them, so its strong play among safe leads is preserved while the
    Jack-pulling blunders are removed. If every legal lead is dangerous, the guard
    yields and DMC chooses from the full set.
    """

    def __init__(self, checkpoint_path: str, device: str | None = None, high_lead_rank: int = HIGH_LEAD_RANK) -> None:
        self.dmc = DMCPlayer(checkpoint_path, device)
        self.high_lead_rank = high_lead_rank

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return legal[0]
        if not state.current_trick:  # leading
            safe = [c for c in legal if not is_dangerous_lead(state, player, c, self.high_lead_rank)]
            if safe and len(safe) < len(legal):
                return choose_q_card(
                    self.dmc.model, self.dmc.device, state, player, rng,
                    epsilon=0.0, action_context_dim=self.dmc.action_context_dim, candidates=safe,
                )
        return self.dmc.choose_card(state, player, rng)


class DMCTrainingPlayer:
    def __init__(self, model: DMCQNet, device: torch.device, epsilon: float) -> None:
        self.model = model
        self.device = device
        self.epsilon = epsilon

    def choose_card(self, state: GameState, player: int, rng: Random) -> Card:
        return choose_q_card(self.model, self.device, state, player, rng, epsilon=self.epsilon)


def choose_q_card(
    model: DMCQNet,
    device: torch.device,
    state: GameState,
    player: int,
    rng: Random,
    epsilon: float,
    action_context_dim: int = ACTION_CONTEXT_DIM,
    candidates: list[Card] | None = None,
) -> Card:
    legal = candidates if candidates is not None else state.legal_actions(player)
    if len(legal) == 1:
        return legal[0]
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice(legal)
    features, _ = encode_state_from_game_state(state, player)
    use_context = action_context_dim > 0
    rows = [
        build_q_input(features, card.index, action_context_features(state, player, card) if use_context else None)
        for card in legal
    ]
    with torch.no_grad():
        x = torch.tensor(rows, dtype=torch.float32, device=device)
        q_values = model(x).cpu()
    best = int(torch.argmax(q_values).item())
    return legal[best]


def train_dmc(
    output_path: str,
    iterations: int = 100,
    games_per_iter: int = 200,
    epochs_per_iter: int = 1,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    replay_size: int = 200000,
    seed: int = 1,
    device: str | None = None,
    opponent: str = "random",
    focus_team: int | None = 0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    save_every: int = 10,
    rollouts: int = 24,
    lambda_clubs: float = 2.5,
    lambda_hearts: float = 0.25,
    shape_clubs: float = 4.0,
    shape_hearts: float = 2.0,
    shape_spades: float = 1.0,
    shape_diamonds: float = 0.0,
    init_checkpoint: str | None = None,
) -> TrainDMCResult:
    device_obj = resolve_device(device)
    torch.manual_seed(seed)
    rng = Random(seed)
    model = DMCQNet(hidden_dim=hidden_dim).to(device_obj)
    loaded_from = None
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device_obj)
        checkpoint_hidden = int(checkpoint.get("hidden_dim", hidden_dim))
        checkpoint_feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
        if checkpoint_hidden != hidden_dim:
            raise ValueError(
                f"init checkpoint hidden_dim={checkpoint_hidden} does not match --hidden-dim={hidden_dim}"
            )
        if checkpoint_feature_dim != FEATURE_DIM:
            raise ValueError(
                f"init checkpoint feature_dim={checkpoint_feature_dim} does not match current FEATURE_DIM={FEATURE_DIM}"
            )
        checkpoint_action_context = int(checkpoint.get("action_context_dim", 0))
        if checkpoint_action_context != ACTION_CONTEXT_DIM:
            raise ValueError(
                f"init checkpoint action_context_dim={checkpoint_action_context} does not match current "
                f"ACTION_CONTEXT_DIM={ACTION_CONTEXT_DIM}; this checkpoint predates action-context features. "
                f"Train fresh from random (the clean path) instead of initializing from it."
            )
        model.load_state_dict(checkpoint["model_state"])
        loaded_from = init_checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    replay: list[DMCBatch] = []
    best_loss = float("inf")
    best_state = None
    total_examples = 0

    for iteration in range(1, iterations + 1):
        epsilon = interpolate_epsilon(iteration, iterations, epsilon_start, epsilon_end)
        fresh = generate_dmc_batch(
            model=model,
            device=device_obj,
            games=games_per_iter,
            seed=rng.randrange(10**9),
            epsilon=epsilon,
            opponent=opponent,
            focus_team=focus_team,
            rollouts=rollouts,
            lambda_clubs=lambda_clubs,
            lambda_hearts=lambda_hearts,
            shape_clubs=shape_clubs,
            shape_hearts=shape_hearts,
            shape_spades=shape_spades,
            shape_diamonds=shape_diamonds,
        )
        replay.append(fresh)
        replay = trim_replay(replay, replay_size)
        combined = concat_batches(replay)
        total_examples += int(fresh.features.shape[0])
        loader = DataLoader(
            TensorDataset(combined.features, combined.actions, combined.action_context, combined.returns),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(1, epochs_per_iter + 1):
            train_loss = train_q_epoch(model, loader, optimizer, loss_fn, device_obj)
            print(
                f"iter={iteration} epoch={epoch} "
                f"epsilon={epsilon:.3f} "
                f"examples={combined.features.shape[0]} "
                f"mse={train_loss:.4f}"
            )
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        if save_every > 0 and iteration % save_every == 0:
            save_dmc_checkpoint(
                Path(output_path).with_suffix(f".iter{iteration:03d}.pt"),
                best_state,
                hidden_dim,
                best_loss,
                extra={
                    "iteration": iteration,
                    "training_mode": "dmc",
                    "opponent": opponent,
                    "focus_team": focus_team,
                    "epsilon": epsilon,
                    "replay_size": replay_size,
                    "shape_clubs": shape_clubs,
                    "shape_hearts": shape_hearts,
                    "shape_spades": shape_spades,
                    "shape_diamonds": shape_diamonds,
                    "init_checkpoint": loaded_from,
                },
            )

    save_dmc_checkpoint(
        Path(output_path),
        best_state,
        hidden_dim,
        best_loss,
        extra={
            "training_mode": "dmc",
            "opponent": opponent,
            "focus_team": focus_team,
            "iterations": iterations,
            "games_per_iter": games_per_iter,
            "replay_size": replay_size,
            "shape_clubs": shape_clubs,
            "shape_hearts": shape_hearts,
            "shape_spades": shape_spades,
            "shape_diamonds": shape_diamonds,
            "init_checkpoint": loaded_from,
        },
    )
    return TrainDMCResult(train_examples=total_examples, best_loss=best_loss, output_path=output_path)




def generate_dmc_batch(
    model: DMCQNet,
    device: torch.device,
    games: int,
    seed: int,
    epsilon: float,
    opponent: str,
    focus_team: int | None,
    rollouts: int,
    lambda_clubs: float,
    lambda_hearts: float,
    shape_clubs: float,
    shape_hearts: float,
    shape_spades: float,
    shape_diamonds: float,
) -> DMCBatch:
    rng = Random(seed)
    examples: list[tuple[DMCExample, float]] = []
    for _ in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        training_player = DMCTrainingPlayer(model, device, epsilon)
        opponent_player = build_opponent(opponent, rollouts, lambda_clubs, lambda_hearts)
        players = [training_player, opponent_player, training_player, opponent_player]
        round_examples: list[DMCExample] = []
        while not state.round_over:
            actor = state.current_player
            features, _ = encode_state_from_game_state(state, actor)
            action = players[actor].choose_card(state.clone(), actor, rng)
            if focus_team is None or team_for(actor) == focus_team:
                round_examples.append(
                    DMCExample(
                        features=features,
                        action_index=action.index,
                        player=actor,
                        action_context=action_context_features(state, actor, action),
                        trick=state.trick_index,
                    )
                )
            play_card(state, actor, action)
        rewards_to_go = shaped_rewards_to_go(
            state,
            shape_by_suit={
                "C": shape_clubs,
                "H": shape_hearts,
                "S": shape_spades,
                "D": shape_diamonds,
            },
        )
        # Credit each action with the shaped reward from its own trick forward
        # (reward-to-go), not the whole-round return, so penalty-Jack outcomes are
        # attributed to the decisions around the trick that captured them.
        examples.extend(
            (example, rewards_to_go[example.player][example.trick]) for example in round_examples
        )
    return dmc_examples_to_batch(examples)


def shaped_rewards_to_go(state: GameState, shape_by_suit: dict[str, float]) -> dict[int, list[float]]:
    """Per-player reward-to-go indexed by trick.

    ``result[player][t]`` is the shaped reward player ``t`` accumulates from trick
    ``t`` through the end of the round, from that player's team perspective. The
    suffix sum at ``t == 0`` equals the whole-round value in
    :func:`shaped_returns_by_player` (verified in tests).
    """
    num_tricks = len(state.completed_tricks)
    per_trick = {player: [0.0] * num_tricks for player in range(4)}
    for index, result in enumerate(state.completed_tricks):
        winning_team = team_for(result.winner)
        for player in range(4):
            captures = team_for(player) == winning_team
            reward = result.points if captures else -result.points
            for card in result.penalties:
                extra_penalty = shape_by_suit.get(card.suit, 0.0)
                if extra_penalty <= 0:
                    continue
                reward += -extra_penalty if captures else extra_penalty
            per_trick[player][index] = float(reward)
    rewards_to_go = {player: [0.0] * num_tricks for player in range(4)}
    for player in range(4):
        running = 0.0
        for index in range(num_tricks - 1, -1, -1):
            running += per_trick[player][index]
            rewards_to_go[player][index] = running
    return rewards_to_go


def shaped_returns_by_player(state: GameState, shape_by_suit: dict[str, float]) -> dict[int, float]:
    returns = {player: float(state.team_diff_for(player)) for player in range(4)}
    for result in state.completed_tricks:
        if not result.penalties:
            continue
        capturing_team = team_for(result.winner)
        for card in result.penalties:
            extra_penalty = shape_by_suit.get(card.suit, 0.0)
            if extra_penalty <= 0:
                continue
            for player in range(4):
                if team_for(player) == capturing_team:
                    returns[player] -= extra_penalty
                else:
                    returns[player] += extra_penalty
    return returns


def dmc_examples_to_batch(examples: list[tuple[DMCExample, float]]) -> DMCBatch:
    features = torch.zeros((len(examples), FEATURE_DIM), dtype=torch.float32)
    actions = torch.empty((len(examples),), dtype=torch.long)
    returns = torch.empty((len(examples),), dtype=torch.float32)
    action_context = torch.zeros((len(examples), ACTION_CONTEXT_DIM), dtype=torch.float32)
    for i, (example, return_value) in enumerate(examples):
        features[i] = torch.tensor(example.features, dtype=torch.float32)
        actions[i] = example.action_index
        returns[i] = float(return_value)
        action_context[i] = torch.tensor(example.action_context, dtype=torch.float32)
    return DMCBatch(features=features, actions=actions, returns=returns, action_context=action_context)


def train_q_epoch(
    model: DMCQNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_total = 0.0
    seen = 0
    for features, actions, action_context, returns in loader:
        features = features.to(device)
        actions = actions.to(device)
        action_context = action_context.to(device)
        returns = returns.to(device)
        x = build_q_inputs_tensor(features, actions, action_context)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = loss_fn(preds, returns)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item()) * features.shape[0]
        seen += features.shape[0]
    return loss_total / max(1, seen)


def build_q_input(
    features: list[float],
    action_index: int,
    action_context: list[float] | None = None,
) -> list[float]:
    row = list(features) + [0.0] * NUM_CARDS
    row[FEATURE_DIM + action_index] = 1.0
    if action_context is not None:
        row.extend(action_context)
    return row


def build_q_inputs_tensor(
    features: torch.Tensor,
    actions: torch.Tensor,
    action_context: torch.Tensor | None = None,
) -> torch.Tensor:
    action_one_hot = torch.zeros((features.shape[0], NUM_CARDS), dtype=features.dtype, device=features.device)
    action_one_hot.scatter_(1, actions.view(-1, 1), 1.0)
    parts = [features, action_one_hot]
    if action_context is not None:
        parts.append(action_context)
    return torch.cat(parts, dim=1)


def build_opponent(name: str, rollouts: int, lambda_clubs: float, lambda_hearts: float):
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
    raise ValueError(f"Unknown DMC opponent: {name}")


def interpolate_epsilon(iteration: int, iterations: int, start: float, end: float) -> float:
    if iterations <= 1:
        return end
    t = (iteration - 1) / (iterations - 1)
    return start + (end - start) * t


def concat_batches(batches: list[DMCBatch]) -> DMCBatch:
    return DMCBatch(
        features=torch.cat([batch.features for batch in batches], dim=0),
        actions=torch.cat([batch.actions for batch in batches], dim=0),
        returns=torch.cat([batch.returns for batch in batches], dim=0),
        action_context=torch.cat([batch.action_context for batch in batches], dim=0),
    )


def trim_replay(replay: list[DMCBatch], max_examples: int) -> list[DMCBatch]:
    if max_examples <= 0:
        return replay
    trimmed = list(replay)
    total = sum(int(batch.features.shape[0]) for batch in trimmed)
    while trimmed and total > max_examples:
        total -= int(trimmed[0].features.shape[0])
        trimmed.pop(0)
    return trimmed


def save_dmc_checkpoint(
    path: Path,
    model_state: dict[str, torch.Tensor] | None,
    hidden_dim: int,
    best_loss: float,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "model_state": model_state,
        "feature_dim": FEATURE_DIM,
        "action_context_dim": ACTION_CONTEXT_DIM,
        "credit_assignment": "reward_to_go",
        "hidden_dim": hidden_dim,
        "best_loss": best_loss,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
