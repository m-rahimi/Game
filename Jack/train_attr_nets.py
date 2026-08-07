"""Train the NEW richer-input model: policy + value on the 572-dim attributed
features (the same encoding the belief net uses), swept over network depth.

This is the "more complex input" model requested on 2026-07-24:
  policy : 572            -> [hidden x N] -> 52
  value  : 572 + 9 (ctx)  -> [hidden x N] -> 1     (keeps the 9 value-context dims)

Depth N is swept (default 2,3,4) and every arm is trained + evaluated on an
IDENTICAL game-level holdout so depths are directly comparable. Game-level split
is mandatory here: each game is 52 correlated rows sharing one outcome, and the
attributed block nearly fingerprints the game, so a row-level split leaks (this
is the documented trap that made the old 572 value net look good and play badly
-- see BEATING_SEARCH_PROGRESS.md). Pick the winner by holdout AND by the
head-to-head in eval_new_vs_best.py; a value net with negative sim-scaling is
broken regardless of its MSE.

Checkpoints are written per depth:
  checkpoints/policy_attr572_l{N}.pt
  checkpoints/value_attr572_ctx_l{N}.pt
with feature_dim=572, value_context_dim=9, num_layers=N so PolicyFunction /
ValueFunction load them with no extra flags.

Run (GPU):
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY train_attr_nets.py --data data/search_attr_40k.pt --depths 2 3 4 \
     --hidden-dim 256 --epochs 30 --batch-size 2048
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


def game_split_perm(n_rows: int, seed: int) -> torch.Tensor:
    """Permutation that keeps each game's 52 contiguous rows together, then
    shuffles at the game level. Returns a row-index permutation."""
    assert n_rows % 52 == 0, f"rows {n_rows} not a multiple of 52"
    g = torch.Generator().manual_seed(seed)
    game_perm = torch.randperm(n_rows // 52, generator=g)
    return (game_perm[:, None] * 52 + torch.arange(52)[None, :]).reshape(-1)


def train_value(inputs, targets, device, hidden, num_layers, epochs, bs, lr, seed):
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
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
            seen += xb.shape[0]
        print(f"    value  L{num_layers} epoch={epoch:>2} mse={tot/seen:.4f}")
    return model


def train_policy(inputs, actions, legal, device, hidden, num_layers, epochs, bs, lr, seed):
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
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
            seen += xb.shape[0]
        print(f"    policy L{num_layers} epoch={epoch:>2} ce={tot/seen:.4f}")
    return model


def value_holdout_mse(model, inputs, targets, device, bs=8192):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for s in range(0, inputs.shape[0], bs):
            xb = inputs[s:s + bs].to(device)
            yb = targets[s:s + bs].to(device)
            total += float(nn.functional.mse_loss(model(xb), yb, reduction="sum"))
    return total / inputs.shape[0]


def policy_holdout_metrics(model, inputs, actions, legal, device, bs=8192):
    model.eval()
    correct = 0
    ce_total = 0.0
    with torch.no_grad():
        for s in range(0, inputs.shape[0], bs):
            xb = inputs[s:s + bs].to(device)
            ab = actions[s:s + bs].to(device)
            lb = legal[s:s + bs].to(device)
            logits = model(xb).masked_fill(~lb, float("-inf"))
            correct += int((logits.argmax(dim=1) == ab).sum())
            ce_total += float(nn.functional.cross_entropy(logits, ab, reduction="sum"))
    return correct / inputs.shape[0], ce_total / inputs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_attr_40k.pt")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4],
                    help="hidden-layer counts to sweep")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout-games", type=int, default=4000,
                    help="games (52-row blocks) held out for evaluation")
    ap.add_argument("--out-prefix", default="checkpoints/")
    ap.add_argument("--skip-policy", action="store_true")
    ap.add_argument("--skip-value", action="store_true")
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="debug: use only the first N rows (must be multiple of 52)")
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, acts, legal = data["features"], data["actions"], data["legal"]
    vctx, outs = data["value_context"], data["outcomes"]
    assert feats.shape[1] == FEATURE_DIM_ATTR, (feats.shape, FEATURE_DIM_ATTR)
    assert vctx.shape[1] == VALUE_CONTEXT_DIM, (vctx.shape, VALUE_CONTEXT_DIM)

    if args.limit_rows:
        k = args.limit_rows
        feats, acts, legal, vctx, outs = feats[:k], acts[:k], legal[:k], vctx[:k], outs[:k]

    # Game-level split: identical holdout for every depth so they compare fairly.
    perm = game_split_perm(feats.shape[0], args.seed)
    feats, acts, legal, vctx, outs = feats[perm], acts[perm], legal[perm], vctx[perm], outs[perm]
    nh = args.holdout_games * 52
    assert nh < feats.shape[0], "holdout larger than dataset"
    print(f"data={args.data} rows={feats.shape[0]} holdout_rows={nh} "
          f"({args.holdout_games} games) depths={args.depths} device={device}")

    # value input = attributed 572 + 9 context = 581
    val_in = torch.cat([feats, vctx], dim=1)
    v_tr, v_ho = val_in[nh:], val_in[:nh]
    y_tr, y_ho = outs[nh:], outs[:nh]
    p_tr, p_ho = feats[nh:], feats[:nh]
    a_tr, a_ho = acts[nh:], acts[:nh]
    l_tr, l_ho = legal[nh:], legal[:nh]

    results = []
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True) if "/" in args.out_prefix else None
    for d in args.depths:
        print(f"\n=== depth L{d} (hidden={args.hidden_dim}) ===")
        mse = acc = ce = float("nan")
        if not args.skip_value:
            mv = train_value(v_tr, y_tr, device, args.hidden_dim, d, args.epochs,
                             args.batch_size, args.lr, args.seed)
            mse = value_holdout_mse(mv, v_ho, y_ho, device)
            vpath = Path(f"{args.out_prefix}value_attr572_ctx_l{d}.pt")
            save_value_checkpoint(vpath, {k: v.cpu() for k, v in mv.state_dict().items()},
                                  args.hidden_dim, mse, value_context_dim=VALUE_CONTEXT_DIM,
                                  feature_dim=FEATURE_DIM_ATTR, num_layers=d,
                                  extra={"data": args.data, "split": "game"})
            print(f"  [value ] L{d} holdout MSE = {mse:.3f}  -> {vpath}")
        if not args.skip_policy:
            mp = train_policy(p_tr, a_tr, l_tr, device, args.hidden_dim, d, args.epochs,
                              args.batch_size, args.lr, args.seed)
            acc, ce = policy_holdout_metrics(mp, p_ho, a_ho, l_ho, device)
            ppath = Path(f"{args.out_prefix}policy_attr572_l{d}.pt")
            save_policy_checkpoint(ppath, {k: v.cpu() for k, v in mp.state_dict().items()},
                                   args.hidden_dim, ce, feature_dim=FEATURE_DIM_ATTR,
                                   num_layers=d, extra={"data": args.data, "split": "game"})
            print(f"  [policy] L{d} holdout top1 = {acc:.4f}  ce = {ce:.4f}  -> {ppath}")
        results.append((d, mse, acc, ce))

    print("\n=== SWEEP SUMMARY (game-split holdout) ===")
    print(f"{'depth':>6} {'value_MSE':>10} {'policy_top1':>12} {'policy_ce':>10}")
    for d, mse, acc, ce in results:
        print(f"L{d:<5} {mse:>10.3f} {acc:>12.4f} {ce:>10.4f}")
    if not args.skip_value:
        best = min((r for r in results), key=lambda r: r[1])
        print(f"\nlowest holdout value MSE: L{best[0]} ({best[1]:.3f}) "
              f"-- confirm with eval_new_vs_best.py before trusting it.")


if __name__ == "__main__":
    main()
