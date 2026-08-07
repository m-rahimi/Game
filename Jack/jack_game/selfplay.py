"""Parallel ISMCTS self-play for the ExIt loop.

Each decision is played by net-guided ISMCTS and recorded as:
  base features (260) | value_context (9) | MCTS visit distribution (52) | outcome

The visit distribution is the improved-policy target (better than the raw policy,
because search refines it); the outcome is the value target. Early moves are
sampled from the visit counts for exploration; later moves are greedy. Forced
(single-legal) moves carry no training signal and are skipped.
"""
from __future__ import annotations

from multiprocessing import Pool
from random import Random

import torch

from .bc import NUM_CARDS, encode_state_from_game_state
from .engine import new_round, play_card
from .mcts import ISMCTSPlayer
from .policy import PolicyFunction
from .value import ValueFunction, value_context_features


def _build_player(policy_ckpt, value_ckpt, sims, c_puct):
    policy = PolicyFunction(policy_ckpt, device="cpu")
    value = ValueFunction(value_ckpt, device="cpu")
    return ISMCTSPlayer(value_fn=value, policy_fn=policy, simulations=sims, c_puct=c_puct, time_limit_seconds=None)


def _sample_from_visits(legal, visit_counts, rng):
    weights = [visit_counts.get(c.index, 0) for c in legal]
    total = sum(weights)
    if total <= 0:
        return rng.choice(legal)
    threshold = rng.random() * total
    acc = 0.0
    for card, w in zip(legal, weights):
        acc += w
        if threshold <= acc:
            return card
    return legal[-1]


def _selfplay_chunk(arg):
    games, seed, policy_ckpt, value_ckpt, sims, c_puct, temp_moves = arg
    torch.set_num_threads(1)  # avoid thread oversubscription across many workers
    rng = Random(seed)
    player = _build_player(policy_ckpt, value_ckpt, sims, c_puct)
    feats, vctx, visits, legals, outcomes = [], [], [], [], []
    for _ in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        pending = []  # (row_index, actor)
        move_no = 0
        while not state.round_over:
            actor = state.current_player
            legal = state.legal_actions(actor)
            if len(legal) == 1:
                play_card(state, actor, legal[0])  # forced; no training signal
                continue
            card, vc = player.choose_with_visits(state.clone(), actor, rng)
            features, illegal = encode_state_from_game_state(state, actor)
            dist = [0.0] * NUM_CARDS
            total = sum(vc.values())
            for idx, n in vc.items():
                dist[idx] = n / total
            feats.append(features)
            vctx.append(value_context_features(state, actor))
            visits.append(dist)
            legals.append([not x for x in illegal])
            pending.append((len(outcomes), actor))
            outcomes.append(0.0)
            chosen = _sample_from_visits(legal, vc, rng) if move_no < temp_moves else card
            play_card(state, actor, chosen)
            move_no += 1
        for row_index, actor in pending:
            outcomes[row_index] = float(state.team_diff_for(actor))
    return feats, vctx, visits, legals, outcomes


def generate_selfplay(policy_ckpt, value_ckpt, games, workers, sims, c_puct, temp_moves, seed):
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    tasks = [(g, seed + 1000 * i, policy_ckpt, value_ckpt, sims, c_puct, temp_moves)
             for i, g in enumerate(per) if g > 0]
    feats, vctx, visits, legals, outcomes = [], [], [], [], []
    with Pool(processes=len(tasks)) as pool:
        for f, v, vi, lg, o in pool.imap_unordered(_selfplay_chunk, tasks):
            feats.extend(f); vctx.extend(v); visits.extend(vi); legals.extend(lg); outcomes.extend(o)
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(vctx, dtype=torch.float32),
        torch.tensor(visits, dtype=torch.float32),
        torch.tensor(legals, dtype=torch.bool),
        torch.tensor(outcomes, dtype=torch.float32),
    )


def _distill_chunk(arg):
    """Distillation recorder (2026-08): like _selfplay_chunk, plus
    - configurable feature encoding (572 attributed for the new-model line),
    - root Q per action (margin/damage-control signal, from choose_with_stats),
    - epsilon-random executed moves (DAgger-style coverage: the label is still the
      teacher's distribution for the state; only the EXECUTED move is randomised,
      so later states go off-distribution and get expert labels),
    - per-row game ids (rows/game vary because forced moves are skipped, so
      game-level splits need explicit ids, not 52-blocks).
    """
    (games, seed, game_base, policy_ckpt, value_ckpt, sims, c_puct,
     temp_moves, epsilon, feature_dim) = arg
    torch.set_num_threads(1)  # avoid thread oversubscription across many workers
    rng = Random(seed)
    player = _build_player(policy_ckpt, value_ckpt, sims, c_puct)
    feats, vctx, visits, qs, legals, outcomes, game_ids = [], [], [], [], [], [], []
    for g in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        pending = []  # (row_index, actor)
        move_no = 0
        while not state.round_over:
            actor = state.current_player
            legal = state.legal_actions(actor)
            if len(legal) == 1:
                play_card(state, actor, legal[0])  # forced; no training signal
                continue
            card, vc, q = player.choose_with_stats(state.clone(), actor, rng)
            features, illegal = encode_state_from_game_state(state, actor, feature_dim=feature_dim)
            dist = [0.0] * NUM_CARDS
            total = sum(vc.values())
            for idx, n in vc.items():
                dist[idx] = n / total
            qrow = [0.0] * NUM_CARDS
            for idx, val in q.items():
                qrow[idx] = val  # acting player's perspective, team-diff units
            feats.append(features)
            vctx.append(value_context_features(state, actor))
            visits.append(dist)
            qs.append(qrow)
            legals.append([not x for x in illegal])
            game_ids.append(game_base + g)
            pending.append((len(outcomes), actor))
            outcomes.append(0.0)
            if rng.random() < epsilon:
                chosen = rng.choice(legal)  # exploration: execute random, keep expert label
            elif move_no < temp_moves:
                chosen = _sample_from_visits(legal, vc, rng)
            else:
                chosen = card
            play_card(state, actor, chosen)
            move_no += 1
        for row_index, actor in pending:
            outcomes[row_index] = float(state.team_diff_for(actor))
    return feats, vctx, visits, qs, legals, outcomes, game_ids


def generate_distill_selfplay(policy_ckpt, value_ckpt, games, workers, sims, c_puct,
                              temp_moves, seed, epsilon=0.05, feature_dim=None):
    """Run the distillation recorder across workers. Returns a dict of tensors."""
    from .bc import FEATURE_DIM

    feature_dim = feature_dim or FEATURE_DIM
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    tasks = []
    base = 0
    for i, g in enumerate(per):
        if g > 0:
            tasks.append((g, seed + 1000 * i, base, policy_ckpt, value_ckpt, sims,
                          c_puct, temp_moves, epsilon, feature_dim))
            base += g
    feats, vctx, visits, qs, legals, outcomes, game_ids = [], [], [], [], [], [], []
    with Pool(processes=len(tasks)) as pool:
        for f, v, vi, q, lg, o, gi in pool.imap_unordered(_distill_chunk, tasks):
            feats.extend(f); vctx.extend(v); visits.extend(vi); qs.extend(q)
            legals.extend(lg); outcomes.extend(o); game_ids.extend(gi)
    return {
        "features": torch.tensor(feats, dtype=torch.float32),
        "value_context": torch.tensor(vctx, dtype=torch.float32),
        "visits": torch.tensor(visits, dtype=torch.float32),
        "q": torch.tensor(qs, dtype=torch.float32),
        "legal": torch.tensor(legals, dtype=torch.bool),
        "outcomes": torch.tensor(outcomes, dtype=torch.float32),
        "game_id": torch.tensor(game_ids, dtype=torch.long),
    }
