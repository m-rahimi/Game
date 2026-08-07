"""Train the belief net and run the Gate A evaluation.

Gate A: on a GAME-LEVEL holdout, the net's per-hidden-card log-loss must
clearly beat the no-inference dealer baseline (capacity-proportional + voids —
what uniform determinization implicitly assumes). Also reports top-1 placement
accuracy and a per-trick-phase breakdown (the fixable blunders live in tricks
3+, so mid/late-game reading is what matters).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import FEATURE_DIM_ATTR, resolve_device
from jack_game.belief import (
    BELIEF_CLASSES,
    BeliefNet,
    baseline_log_probs,
    belief_loss,
    save_belief_checkpoint,
)

TRICK_START = 550  # trick-number one-hot block in the 572 layout
PHASES = [("tricks 1-3", 0, 3), ("tricks 4-9", 3, 9), ("tricks 10-13", 9, 13)]


def eval_holdout(model, features, card_loc, device, bs=8192):
    """Returns dict of (net_logloss, base_logloss, net_top1, base_top1) overall
    and per trick phase."""
    model.eval()
    tricks = features[:, TRICK_START:TRICK_START + 13].argmax(dim=1)
    sums = {}  # key -> [net_ll, base_ll, net_hit, base_hit, n_cards]
    with torch.no_grad():
        for start in range(0, features.shape[0], bs):
            xb = features[start:start + bs].to(device)
            yb = card_loc[start:start + bs].to(device)
            tb = tricks[start:start + bs]
            hidden = yb > 0
            target = (yb.long().clamp(min=1) - 1)
            net_lp = torch.log_softmax(model(xb), dim=2)
            base_lp = baseline_log_probs(features[start:start + bs]).to(device)
            net_ll = -net_lp.gather(2, target.unsqueeze(2)).squeeze(2)
            base_ll = -base_lp.gather(2, target.unsqueeze(2)).squeeze(2)
            net_hit = (net_lp.argmax(dim=2) == target).float()
            base_hit = (base_lp.argmax(dim=2) == target).float()
            for name, lo, hi in [("all", 0, 13)] + PHASES:
                rows = (tb >= lo) & (tb < hi)
                mask = hidden & rows.to(device).unsqueeze(1)
                if not mask.any():
                    continue
                bucket = sums.setdefault(name, [0.0] * 5)
                bucket[0] += float((net_ll * mask).sum())
                bucket[1] += float((base_ll * mask).sum())
                bucket[2] += float((net_hit * mask).sum())
                bucket[3] += float((base_hit * mask).sum())
                bucket[4] += float(mask.sum())
    return {name: (v[0] / v[4], v[1] / v[4], v[2] / v[4], v[3] / v[4]) for name, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_beliefs_40k.pt")
    ap.add_argument("--out", default="checkpoints/belief_search40k.pt")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, loc = data["features"], data["card_loc"]
    assert feats.shape[1] == FEATURE_DIM_ATTR and "card_loc" in data

    # game-level split (52 contiguous rows per game) — row splits leak, always.
    torch.manual_seed(args.seed)
    assert feats.shape[0] % 52 == 0
    game_perm = torch.randperm(feats.shape[0] // 52)
    perm = (game_perm[:, None] * 52 + torch.arange(52)[None, :]).reshape(-1)
    feats, loc = feats[perm], loc[perm]
    nh = min(max(20000, feats.shape[0] // 10), feats.shape[0] // 2)
    nh -= nh % 52
    x_ho, y_ho = feats[:nh], loc[:nh]
    x_tr, y_tr = feats[nh:], loc[nh:]
    print(f"train={x_tr.shape[0]} holdout={nh} (game-level split) device={device}", flush=True)

    model = BeliefNet(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    print(f"BeliefNet {args.hidden_dim}x{args.num_layers}, "
          f"params={sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)

    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            belief_loss(model(xb), yb).backward()
            opt.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = eval_holdout(model, x_ho, y_ho, device)
            net_ll, base_ll, net_t1, base_t1 = metrics["all"]
            print(f"epoch={epoch:2d} holdout logloss net={net_ll:.4f} baseline={base_ll:.4f} "
                  f"top1 net={net_t1:.3f} baseline={base_t1:.3f}", flush=True)
            if net_ll < best:
                best = net_ll
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    metrics = eval_holdout(model, x_ho, y_ho, device)
    print("\n=== Gate A report (game-level holdout, per hidden card) ===")
    print(f"{'phase':>13} {'net logloss':>12} {'base logloss':>13} {'net top1':>9} {'base top1':>10}")
    for name in ["all"] + [p[0] for p in PHASES]:
        if name in metrics:
            n_ll, b_ll, n_t1, b_t1 = metrics[name]
            print(f"{name:>13} {n_ll:>12.4f} {b_ll:>13.4f} {n_t1:>9.3f} {b_t1:>10.3f}")
    n_ll, b_ll, _, _ = metrics["all"]
    print(f"\nuniform-1/3 reference logloss: {torch.log(torch.tensor(3.0)):.4f}")
    print(f"GATE A: net beats no-inference dealer by {b_ll - n_ll:+.4f} nats/card "
          f"({'PASS' if n_ll < b_ll - 0.02 else 'FAIL'})")

    save_belief_checkpoint(Path(args.out), best_state, args.hidden_dim, args.num_layers, best,
                           extra={"data": args.data, "holdout_logloss": n_ll, "baseline_logloss": b_ll})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
