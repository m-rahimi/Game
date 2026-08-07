"""Data-scaling curve for the NEW 572-dim model, with a FIXED holdout.

Answers the 2026-07-25 question: was the 572-dim model just data-starved at 40k
games? Trains the policy (572) and value (572+9) at increasing training-set
sizes, all evaluated on ONE fixed game-level holdout, so the only thing that
changes across points is how many training games the net saw.

Compare the value holdout MSE against the 260-baseline game-split number 13.07:
  - falling toward / below 13.07 as games grow  -> was data-starved; scale up.
  - flat above 13.07                             -> attribution doesn't generalize.

Saves the best-scale checkpoints for the head-to-head:
  checkpoints/policy_attr572_g{G}.pt   checkpoints/value_attr572_ctx_g{G}.pt

Run (GPU):
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY scale_attr_curve.py --data data/search_attr_200k.pt --depth 2 \
     --train-games 40000 100000 196000 --holdout-games 4000 --epochs 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import FEATURE_DIM_ATTR, resolve_device
from jack_game.policy import PolicyNet, save_policy_checkpoint
from jack_game.value import VALUE_CONTEXT_DIM, ValueNet, save_value_checkpoint


def train_value(inputs, targets, device, hidden, num_layers, epochs, bs, lr, seed, tag):
    torch.manual_seed(seed)
    model = ValueNet(input_dim=inputs.shape[1], hidden_dim=hidden, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=bs, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        tot = seen = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * xb.shape[0]; seen += xb.shape[0]
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    value  {tag} epoch={epoch:>2} train_mse={tot/seen:.4f}", flush=True)
    return model


def train_policy(inputs, actions, legal, device, hidden, num_layers, epochs, bs, lr, seed, tag):
    torch.manual_seed(seed)
    model = PolicyNet(input_dim=inputs.shape[1], hidden_dim=hidden, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(inputs, actions, legal), batch_size=bs, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        tot = seen = 0.0
        for xb, ab, lb in loader:
            xb, ab, lb = xb.to(device), ab.to(device), lb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).masked_fill(~lb, float("-inf"))
            loss = nn.functional.cross_entropy(logits, ab)
            loss.backward(); opt.step()
            tot += float(loss) * xb.shape[0]; seen += xb.shape[0]
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    policy {tag} epoch={epoch:>2} train_ce={tot/seen:.4f}", flush=True)
    return model


def value_holdout_mse(model, inputs, targets, device, bs=16384):
    model.eval(); total = 0.0
    with torch.no_grad():
        for s in range(0, inputs.shape[0], bs):
            total += float(nn.functional.mse_loss(
                model(inputs[s:s+bs].to(device)), targets[s:s+bs].to(device), reduction="sum"))
    return total / inputs.shape[0]


def policy_holdout(model, inputs, actions, legal, device, bs=16384):
    model.eval(); correct = 0; ce = 0.0
    with torch.no_grad():
        for s in range(0, inputs.shape[0], bs):
            xb = inputs[s:s+bs].to(device); ab = actions[s:s+bs].to(device); lb = legal[s:s+bs].to(device)
            logits = model(xb).masked_fill(~lb, float("-inf"))
            correct += int((logits.argmax(1) == ab).sum())
            ce += float(nn.functional.cross_entropy(logits, ab, reduction="sum"))
    return correct / inputs.shape[0], ce / inputs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_attr_200k.pt")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--train-games", type=int, nargs="+", default=[40000, 100000, 196000])
    ap.add_argument("--holdout-games", type=int, default=4000)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--baseline-mse", type=float, default=13.07, help="260+ctx game-split reference")
    ap.add_argument("--skip-policy", action="store_true")
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, acts, legal = data["features"], data["actions"], data["legal"]
    vctx, outs = data["value_context"], data["outcomes"]
    assert feats.shape[1] == FEATURE_DIM_ATTR
    n_games = feats.shape[0] // 52
    print(f"data={args.data} games={n_games} depth=L{args.depth} device={device}", flush=True)

    # One game-level shuffle; holdout = FIRST holdout-games; training pool = the rest.
    g = torch.Generator().manual_seed(args.seed)
    gperm = torch.randperm(n_games, generator=g)
    row_perm = (gperm[:, None] * 52 + torch.arange(52)[None, :]).reshape(-1)
    feats, acts, legal, vctx, outs = feats[row_perm], acts[row_perm], legal[row_perm], vctx[row_perm], outs[row_perm]

    nh = args.holdout_games * 52
    pool_games = n_games - args.holdout_games
    val_in = torch.cat([feats, vctx], dim=1)  # 581-dim value input
    # fixed holdout
    vho, yho = val_in[:nh], outs[:nh]
    pho, aho, lho = feats[:nh], acts[:nh], legal[:nh]
    print(f"fixed holdout = {args.holdout_games} games ({nh} rows); pool = {pool_games} games", flush=True)

    rows = []
    for G in args.train_games:
        assert G <= pool_games, f"train games {G} > pool {pool_games}"
        a, b = nh, nh + G * 52  # training slice from the pool, after the holdout
        tag = f"g{G}"
        print(f"\n=== train games={G} (rows {a}..{b}) ===", flush=True)
        mv = train_value(val_in[a:b], outs[a:b], device, args.hidden_dim, args.depth,
                         args.epochs, args.batch_size, args.lr, args.seed, tag)
        mse = value_holdout_mse(mv, vho, yho, device)
        vpath = Path(f"checkpoints/value_attr572_ctx_{tag}.pt")
        save_value_checkpoint(vpath, {k: v.cpu() for k, v in mv.state_dict().items()},
                              args.hidden_dim, mse, value_context_dim=VALUE_CONTEXT_DIM,
                              feature_dim=FEATURE_DIM_ATTR, num_layers=args.depth,
                              extra={"data": args.data, "split": "game", "train_games": G})
        acc = ce = float("nan")
        if not args.skip_policy:
            mp = train_policy(feats[a:b], acts[a:b], legal[a:b], device, args.hidden_dim, args.depth,
                              args.epochs, args.batch_size, args.lr, args.seed, tag)
            acc, ce = policy_holdout(mp, pho, aho, lho, device)
            ppath = Path(f"checkpoints/policy_attr572_{tag}.pt")
            save_policy_checkpoint(ppath, {k: v.cpu() for k, v in mp.state_dict().items()},
                                   args.hidden_dim, ce, feature_dim=FEATURE_DIM_ATTR,
                                   num_layers=args.depth, extra={"data": args.data, "split": "game", "train_games": G})
        print(f"  games={G:>7}  value_MSE={mse:.3f}  policy_top1={acc:.4f}", flush=True)
        rows.append((G, mse, acc, ce))

    print("\n=== DATA-SCALING CURVE (fixed holdout, 572-dim, L{}) ===".format(args.depth), flush=True)
    print(f"260+ctx baseline (game-split) = {args.baseline_mse}")
    print(f"{'games':>8} {'value_MSE':>10} {'vs_base':>9} {'policy_top1':>12}")
    for G, mse, acc, ce in rows:
        print(f"{G:>8} {mse:>10.3f} {mse-args.baseline_mse:>+9.3f} {acc:>12.4f}")
    if len(rows) >= 2:
        d = rows[-1][1] - rows[0][1]
        print(f"\nvalue MSE trend {rows[0][0]}->{rows[-1][0]} games: {d:+.3f} "
              f"({'FALLING (was data-starved)' if d < -0.2 else 'FLAT/RISING (does not generalize)'})")


if __name__ == "__main__":
    main()
