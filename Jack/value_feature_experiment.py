"""Value-feature fix experiment.

Trains the value net WITHOUT and WITH the sunk-score context features on the same
search dataset, reports held-out MSE for both, and saves the with-context net for
the ISMCTS sim-scaling re-test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import resolve_device
from jack_game.value import ValueNet, save_value_checkpoint, VALUE_CONTEXT_DIM


def train(inputs, targets, device, hidden, epochs, bs, lr):
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


def holdout_mse(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(inputs), targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_ctx_40k.pt")
    ap.add_argument("--value-out", default="checkpoints/value_search_ctx.pt")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, vctx, outs = data["features"], data["value_context"], data["outcomes"]
    assert vctx.shape[1] == VALUE_CONTEXT_DIM, (vctx.shape, VALUE_CONTEXT_DIM)

    perm = torch.randperm(feats.shape[0])
    feats, vctx, outs = feats[perm], vctx[perm], outs[perm]
    nh = max(20000, feats.shape[0] // 10)
    print(f"examples={feats.shape[0]} holdout={nh} value_context_dim={VALUE_CONTEXT_DIM} device={device}")

    base_tr, base_ho = feats[nh:], feats[:nh].to(device)
    ctx_tr = torch.cat([feats, vctx], dim=1)[nh:]
    ctx_ho = torch.cat([feats, vctx], dim=1)[:nh].to(device)
    y_tr, y_ho = outs[nh:], outs[:nh].to(device)

    m_base = train(base_tr, y_tr, device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
    mse_base = holdout_mse(m_base, base_ho, y_ho)

    m_ctx = train(ctx_tr, y_tr, device, args.hidden_dim, args.epochs, args.batch_size, args.lr)
    mse_ctx = holdout_mse(m_ctx, ctx_ho, y_ho)

    save_value_checkpoint(Path(args.value_out), {k: v.cpu() for k, v in m_ctx.state_dict().items()},
                          args.hidden_dim, mse_ctx, value_context_dim=VALUE_CONTEXT_DIM)

    print(f"holdout value MSE  WITHOUT context: {mse_base:.3f}")
    print(f"holdout value MSE  WITH    context: {mse_ctx:.3f}")
    print(f"improvement: {100*(mse_base-mse_ctx)/mse_base:.1f}%   wrote {args.value_out}")


if __name__ == "__main__":
    main()
