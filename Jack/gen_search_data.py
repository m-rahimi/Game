"""Parallel search self-play data generator (ExIt seed data / scaling-curve input).

Plays games with the search player and records, per decision:
  features (260)  | chosen action index | legal mask (52) | final team-diff outcome

One pass yields targets for BOTH the policy net (action) and the value net
(outcome). Search rollouts are CPU-bound, so games fan out across processes.

Saves a .pt with tensors: features, actions, legal, outcomes (+ metadata).
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from random import Random

import torch

from jack_game.bc import FEATURE_DIM, FEATURE_DIM_ATTR, encode_state_from_game_state
from jack_game.engine import new_round, play_card
from jack_game.players import SearchPlayer
from jack_game.value import value_context_features


def card_locations(state, actor) -> list[int]:
    """Per card: 0 = seen by actor (own hand or already played), else the RELATIVE
    seat holding it (1=left opponent, 2=partner, 3=right opponent). Belief-net target."""
    loc = [0] * 52
    for seat in range(4):
        if seat == actor:
            continue
        for card in state.hands[seat]:
            loc[card.index] = (seat - actor) % 4
    return loc


def generate_chunk(arg):
    games, seed, rollouts, time_limit, feature_dim, record_hands = arg
    rng = Random(seed)
    teacher = SearchPlayer(rollouts_per_action=rollouts, time_limit_seconds=time_limit)
    feats: list[list[float]] = []
    actions: list[int] = []
    legals: list[list[bool]] = []
    vctx: list[list[float]] = []
    outcomes: list[float] = []
    locs: list[list[int]] = []
    for _ in range(games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        pending = []  # (row_index, actor)
        while not state.round_over:
            actor = state.current_player
            features, illegal = encode_state_from_game_state(state, actor, feature_dim=feature_dim)
            card = teacher.choose_card(state.clone(), actor, rng)
            feats.append(features)
            actions.append(card.index)
            legals.append([not f for f in illegal])
            vctx.append(value_context_features(state, actor))  # sunk-score context for the value net
            if record_hands:
                locs.append(card_locations(state, actor))  # true hidden-card locations (belief target)
            pending.append((len(outcomes), actor))
            outcomes.append(0.0)  # placeholder, filled after the round
            play_card(state, actor, card)
        for row_index, actor in pending:
            outcomes[row_index] = float(state.team_diff_for(actor))
    return feats, actions, legals, vctx, outcomes, locs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--time-limit", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--features", choices=["modern", "attr"], default="modern",
                    help="attr = 572-dim attributed play history (who/lead/followed/when per card)")
    ap.add_argument("--record-hands", action="store_true",
                    help="record true hidden-card locations per decision (belief-net target)")
    args = ap.parse_args()

    feature_dim = FEATURE_DIM_ATTR if args.features == "attr" else FEATURE_DIM
    per = [args.games // args.workers] * args.workers
    for i in range(args.games - sum(per)):
        per[i] += 1
    tasks = [(g, args.seed + 1000 * i, args.rollouts, args.time_limit, feature_dim, args.record_hands)
             for i, g in enumerate(per) if g > 0]

    feats: list[list[float]] = []
    actions: list[int] = []
    legals: list[list[bool]] = []
    vctx: list[list[float]] = []
    outcomes: list[float] = []
    locs: list[list[int]] = []
    with Pool(processes=len(tasks)) as pool:
        for f, a, l, v, o, c in pool.imap_unordered(generate_chunk, tasks):
            feats.extend(f)
            actions.extend(a)
            legals.extend(l)
            vctx.extend(v)
            outcomes.extend(o)
            locs.extend(c)

    payload = {
        "features": torch.tensor(feats, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "legal": torch.tensor(legals, dtype=torch.bool),
        "value_context": torch.tensor(vctx, dtype=torch.float32),
        "outcomes": torch.tensor(outcomes, dtype=torch.float32),
        "games": args.games,
        "teacher": "search",
        "rollouts": args.rollouts,
        "feature_dim": feature_dim,
    }
    if args.record_hands:
        payload["card_loc"] = torch.tensor(locs, dtype=torch.int8)
    torch.save(payload, args.out)
    print(f"games={args.games} examples={len(actions)} workers={len(tasks)} "
          f"rollouts={args.rollouts} -> wrote {args.out}")


if __name__ == "__main__":
    main()
