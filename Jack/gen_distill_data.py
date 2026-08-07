"""Generate distillation data from the STRONGEST player (Suggestion 2, 2026-08).

Teacher = net-guided ISMCTS at high simulations (defaults: the 572@196k nets,
800 sims). Records per decision (forced moves skipped):
  features (572 attributed) | value_context (9) | root visit distribution (52)
  | root Q per action (52) | legal mask | final outcome | game id

The visit distribution is the training target for the next policy: it is the
summary of everything the 800 simulations computed, so a policy trained on it
(KL) amortises the search into a single forward pass. Exploration: temperature
sampling on early decisions + epsilon-random EXECUTED moves (labels stay
expert) for off-distribution coverage.

Run (CPU-bound; workers use CPU inference):
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY gen_distill_data.py --out data/distill_ismcts800_100k.pt \
     --games 100000 --workers 56 --sims 800
"""
from __future__ import annotations

import argparse
from time import monotonic

import torch

from jack_game.bc import FEATURE_DIM_ATTR
from jack_game.selfplay import generate_distill_selfplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-moves", type=int, default=8,
                    help="sample-from-visits for the first N choice moves")
    ap.add_argument("--epsilon", type=float, default=0.05,
                    help="prob of executing a random legal move (label stays expert)")
    ap.add_argument("--policy-ckpt", default="checkpoints/policy_attr572_g196000.pt")
    ap.add_argument("--value-ckpt", default="checkpoints/value_attr572_ctx_g196000.pt")
    ap.add_argument("--feature-dim", type=int, default=FEATURE_DIM_ATTR)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    t0 = monotonic()
    data = generate_distill_selfplay(
        policy_ckpt=args.policy_ckpt, value_ckpt=args.value_ckpt,
        games=args.games, workers=args.workers, sims=args.sims, c_puct=args.c_puct,
        temp_moves=args.temp_moves, seed=args.seed, epsilon=args.epsilon,
        feature_dim=args.feature_dim,
    )
    data.update({
        "games": args.games, "sims": args.sims, "c_puct": args.c_puct,
        "temp_moves": args.temp_moves, "epsilon": args.epsilon,
        "teacher_policy": args.policy_ckpt, "teacher_value": args.value_ckpt,
        "feature_dim": args.feature_dim, "seed": args.seed,
    })
    torch.save(data, args.out)
    dt = monotonic() - t0
    n = data["features"].shape[0]
    print(f"games={args.games} rows={n} ({n/args.games:.1f} choice-moves/game) "
          f"sims={args.sims} epsilon={args.epsilon} wall={dt/3600:.2f}h "
          f"({args.games/max(dt,1e-9):.2f} games/s) -> wrote {args.out}")


if __name__ == "__main__":
    main()
