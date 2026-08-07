"""Attributed-history feature experiment.

Question: does knowing WHO played each card, and in what context (lead / forced
follow / free discard, when), improve the nets over the unattributed 260-dim
encoding? The 260 baseline is derived from the same 572-dim dataset
(``attributed_to_modern``), so the comparison is exactly paired: same games,
same targets, only the encoding differs.

Trains four nets on one search dataset generated with --features attr:
  value  WITH attribution (572 + 9 ctx)  vs  value  baseline (260 + 9 ctx)
  policy WITH attribution (572)          vs  policy baseline (260)
Reports held-out value MSE and policy top-1 accuracy for both arms and saves
the attributed checkpoints for the ISMCTS eval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import FEATURE_DIM_ATTR, attributed_to_modern, resolve_device
from jack_game.policy import PolicyNet, save_policy_checkpoint
from jack_game.value import VALUE_CONTEXT_DIM, ValueNet, save_value_checkpoint


def train_value(inputs, targets, device, hidden, epochs, bs, lr):
    model = ValueNet(input_dim=inputs.shape[1], hidden_dim=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()
    return model


def train_policy(inputs, actions, legal, device, hidden, epochs, bs, lr):
    model = PolicyNet(input_dim=inputs.shape[1], hidden_dim=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(inputs, actions, legal), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, ab, lb in loader:
            xb, ab, lb = xb.to(device), ab.to(device), lb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).masked_fill(~lb, float("-inf"))
            nn.functional.cross_entropy(logits, ab).backward()
            opt.step()
    return model


def value_holdout_mse(model, inputs, targets, bs=8192):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], bs):
            xb, yb = inputs[start : start + bs], targets[start : start + bs]
            total += float(nn.functional.mse_loss(model(xb), yb, reduction="sum"))
    return total / inputs.shape[0]


def policy_holdout_metrics(model, inputs, actions, legal, bs=8192):
    model.eval()
    correct = 0
    ce_total = 0.0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], bs):
            xb = inputs[start : start + bs]
            ab = actions[start : start + bs]
            lb = legal[start : start + bs]
            logits = model(xb).masked_fill(~lb, float("-inf"))
            correct += int((logits.argmax(dim=1) == ab).sum())
            ce_total += float(nn.functional.cross_entropy(logits, ab, reduction="sum"))
    return correct / inputs.shape[0], ce_total / inputs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_attr_40k.pt")
    ap.add_argument("--policy-out", default="checkpoints/policy_attr.pt")
    ap.add_argument("--value-out", default="checkpoints/value_attr_ctx.pt")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", choices=["row", "game"], default="row",
                    help="game = keep all 52 rows of a game on one side (no sibling leakage)")
    ap.add_argument("--skip-policy", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, acts, legal = data["features"], data["actions"], data["legal"]
    vctx, outs = data["value_context"], data["outcomes"]
    assert feats.shape[1] == FEATURE_DIM_ATTR, (feats.shape, FEATURE_DIM_ATTR)
    assert vctx.shape[1] == VALUE_CONTEXT_DIM, (vctx.shape, VALUE_CONTEXT_DIM)

    base_feats = attributed_to_modern(feats)  # exact 260-dim baseline, same games
    if args.split == "game":
        # Rows are stored as contiguous 52-row blocks, one per game.
        assert feats.shape[0] % 52 == 0
        game_perm = torch.randperm(feats.shape[0] // 52)
        perm = (game_perm[:, None] * 52 + torch.arange(52)[None, :]).reshape(-1)
    else:
        perm = torch.randperm(feats.shape[0])
    feats, base_feats = feats[perm], base_feats[perm]
    acts, legal, vctx, outs = acts[perm], legal[perm], vctx[perm], outs[perm]
    nh = min(max(20000, feats.shape[0] // 10), feats.shape[0] // 2)
    print(f"examples={feats.shape[0]} holdout={nh} attr_dim={feats.shape[1]} "
          f"base_dim={base_feats.shape[1]} device={device}")

    # --- value arms (both get the 9 context features; attribution is the only delta)
    attr_v = torch.cat([feats, vctx], dim=1)
    base_v = torch.cat([base_feats, vctx], dim=1)
    y_tr, y_ho = outs[nh:], outs[:nh].to(device)

    m_vbase = train_value(base_v[nh:], y_tr, device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
    mse_base = value_holdout_mse(m_vbase, base_v[:nh].to(device), y_ho)
    print(f"[value ] holdout MSE  baseline 260+ctx : {mse_base:.3f}")

    m_vattr = train_value(attr_v[nh:], y_tr, device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
    mse_attr = value_holdout_mse(m_vattr, attr_v[:nh].to(device), y_ho)
    print(f"[value ] holdout MSE  attributed 572+ctx: {mse_attr:.3f} "
          f"({100 * (mse_base - mse_attr) / mse_base:+.1f}% vs baseline)")

    # --- policy arms
    if args.skip_policy:
        print("(policy arms skipped)")
        return
    a_tr, a_ho = acts[nh:], acts[:nh].to(device)
    l_tr, l_ho = legal[nh:], legal[:nh].to(device)

    m_pbase = train_policy(base_feats[nh:], a_tr, l_tr, device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
    acc_base, ce_base = policy_holdout_metrics(m_pbase, base_feats[:nh].to(device), a_ho, l_ho)
    print(f"[policy] holdout  baseline 260 : top1={acc_base:.4f} ce={ce_base:.4f}")

    m_pattr = train_policy(feats[nh:], a_tr, l_tr, device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
    acc_attr, ce_attr = policy_holdout_metrics(m_pattr, feats[:nh].to(device), a_ho, l_ho)
    print(f"[policy] holdout  attributed 572: top1={acc_attr:.4f} ce={ce_attr:.4f} "
          f"({100 * (acc_attr - acc_base):+.2f} pts top1 vs baseline)")

    save_value_checkpoint(Path(args.value_out), {k: v.cpu() for k, v in m_vattr.state_dict().items()},
                          args.hidden_dim, mse_attr, value_context_dim=VALUE_CONTEXT_DIM,
                          feature_dim=FEATURE_DIM_ATTR, extra={"data": args.data})
    save_policy_checkpoint(Path(args.policy_out), {k: v.cpu() for k, v in m_pattr.state_dict().items()},
                           args.hidden_dim, ce_attr, feature_dim=FEATURE_DIM_ATTR, extra={"data": args.data})
    print(f"wrote {args.value_out} and {args.policy_out}")


if __name__ == "__main__":
    main()
