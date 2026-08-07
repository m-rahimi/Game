"""Train policy + value nets on the full search dataset and save both checkpoints.

Produces the strong search-distilled priors used as the ISMCTS policy prior and
value leaf evaluator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import resolve_device
from jack_game.policy import PolicyNet, save_policy_checkpoint
from jack_game.value import ValueNet, save_value_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_24k.pt")
    ap.add_argument("--policy-out", default="checkpoints/policy_search.pt")
    ap.add_argument("--value-out", default="checkpoints/value_search.pt")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, acts, legal, outs = data["features"], data["actions"], data["legal"], data["outcomes"]
    print(f"examples={feats.shape[0]} device={device}")

    # Policy
    pmodel = PolicyNet(hidden_dim=args.hidden_dim).to(device)
    popt = torch.optim.AdamW(pmodel.parameters(), lr=args.lr, weight_decay=1e-4)
    ploader = DataLoader(TensorDataset(feats, acts, legal), batch_size=args.batch_size, shuffle=True)
    for ep in range(1, args.epochs + 1):
        pmodel.train(); tot = 0.0; seen = 0
        for xb, ab, lb in ploader:
            xb, ab, lb = xb.to(device), ab.to(device), lb.to(device)
            popt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(pmodel(xb).masked_fill(~lb, float("-inf")), ab)
            loss.backward(); popt.step()
            tot += float(loss) * xb.shape[0]; seen += xb.shape[0]
        if ep % 5 == 0 or ep == args.epochs:
            print(f"[policy] epoch={ep} ce={tot/seen:.4f}")
    save_policy_checkpoint(Path(args.policy_out), {k: v.cpu() for k, v in pmodel.state_dict().items()},
                           args.hidden_dim, tot / seen)

    # Value
    vmodel = ValueNet(hidden_dim=args.hidden_dim).to(device)
    vopt = torch.optim.AdamW(vmodel.parameters(), lr=args.lr, weight_decay=1e-4)
    vloader = DataLoader(TensorDataset(feats, outs), batch_size=args.batch_size, shuffle=True)
    for ep in range(1, args.epochs + 1):
        vmodel.train(); tot = 0.0; seen = 0
        for xb, yb in vloader:
            xb, yb = xb.to(device), yb.to(device)
            vopt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(vmodel(xb), yb)
            loss.backward(); vopt.step()
            tot += float(loss) * xb.shape[0]; seen += xb.shape[0]
        if ep % 5 == 0 or ep == args.epochs:
            print(f"[value] epoch={ep} mse={tot/seen:.4f}")
    save_value_checkpoint(Path(args.value_out), {k: v.cpu() for k, v in vmodel.state_dict().items()},
                          args.hidden_dim, tot / seen)
    print(f"wrote {args.policy_out} and {args.value_out}")


if __name__ == "__main__":
    main()
