"""Capacity/data scaling sweep for the attributed (572+9) value net.

Question: is holdout MSE 9.38 (256x2, 1.87M examples) limited by model capacity
or by data — or is it near the irreducible floor?

Grid (all on data/search_attr_40k.pt, one fixed 208k holdout):
  capacity axis @ full 1.87M train:  256x2  512x2  512x3  1024x3
  data axis     @ 256x2 and 1024x3:  0.5M   1.0M   (full = shared cells)

Same recipe as attr_feature_experiment.py (AdamW lr 1e-3 wd 1e-4, bs 2048,
30 epochs); holdout MSE evaluated every 5 epochs, best reported (guards
against late-epoch overfitting in the bigger nets). Saves the winner as a
loadable checkpoint if it beats the 256x2 full-data cell.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import FEATURE_DIM_ATTR, resolve_device
from jack_game.value import VALUE_CONTEXT_DIM, ValueNet, save_value_checkpoint


def holdout_mse(model, inputs, targets, bs=8192):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], bs):
            total += float(nn.functional.mse_loss(
                model(inputs[start:start + bs]), targets[start:start + bs], reduction="sum"))
    return total / inputs.shape[0]


def run_cell(x_tr, y_tr, x_ho, y_ho, hidden, layers, device, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    model = ValueNet(input_dim=x_tr.shape[1], hidden_dim=hidden, num_layers=layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True)
    best = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()
        if epoch % 5 == 0 or epoch == epochs:
            mse = holdout_mse(model, x_ho, y_ho)
            print(f"    epoch={epoch:2d} holdout_mse={mse:.3f}", flush=True)
            if mse < best:
                best = mse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return best, best_state, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_attr_40k.pt")
    ap.add_argument("--winner-out", default="checkpoints/value_attr_deep.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", choices=["row", "game"], default="row",
                    help="game = keep all 52 rows of a game on one side (no sibling leakage)")
    ap.add_argument("--cells", default=None,
                    help="comma list to filter, e.g. '256x2@full,1024x3@full'")
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, vctx, outs = data["features"], data["value_context"], data["outcomes"]
    assert feats.shape[1] == FEATURE_DIM_ATTR

    inputs = torch.cat([feats, vctx], dim=1)
    torch.manual_seed(args.seed)
    if args.split == "game":
        # Rows are stored as contiguous 52-row blocks, one per game.
        assert inputs.shape[0] % 52 == 0
        n_games = inputs.shape[0] // 52
        game_perm = torch.randperm(n_games)
        perm = (game_perm[:, None] * 52 + torch.arange(52)[None, :]).reshape(-1)
    else:
        perm = torch.randperm(inputs.shape[0])
    inputs, outs = inputs[perm], outs[perm]
    nh = max(20000, inputs.shape[0] // 10)
    x_ho, y_ho = inputs[:nh].to(device), outs[:nh].to(device)
    x_full, y_full = inputs[nh:], outs[nh:]
    print(f"train={x_full.shape[0]} holdout={nh} input_dim={inputs.shape[1]} device={device}", flush=True)

    full = x_full.shape[0]
    cells = [
        # (hidden, layers, train_size)
        (256, 2, 500_000), (256, 2, 1_000_000), (256, 2, full),   # data axis, current arch
        (512, 2, full), (512, 3, full),                            # capacity axis
        (1024, 3, 500_000), (1024, 3, 1_000_000), (1024, 3, full), # data axis, big arch
    ]
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",")}
        cells = [(h, l, n) for h, l, n in cells
                 if f"{h}x{l}@{'full' if n == full else f'{n/1e6:.1f}M'}" in wanted]
        assert cells, f"no cells matched {args.cells}"

    results = []
    baseline_best_state = None
    for hidden, layers, n in cells:
        label = f"{hidden}x{layers} @ {n/1e6:.1f}M"
        print(f"[cell] {label}", flush=True)
        best, state, n_params = run_cell(
            x_full[:n], y_full[:n], x_ho, y_ho, hidden, layers, device,
            args.epochs, args.batch_size, args.lr, args.seed)
        results.append((best, label, hidden, layers, n, state, n_params))
        print(f"[cell] {label}  params={n_params/1e3:.0f}k  BEST holdout_mse={best:.3f}", flush=True)

    print("\n=== summary (sorted by holdout MSE) ===", flush=True)
    print(f"{'arch':>8} {'train':>7} {'params':>8} {'holdout MSE':>12}")
    for best, label, hidden, layers, n, _, n_params in sorted(results):
        print(f"{hidden:>5}x{layers} {n/1e6:>6.1f}M {n_params/1e3:>7.0f}k {best:>12.3f}")

    ref = next(r for r in results if r[2] == 256 and r[3] == 2 and r[4] == full)
    winner = min(results)
    if winner[0] < ref[0]:
        _, label, hidden, layers, n, state, _ = winner
        save_value_checkpoint(Path(args.winner_out), state, hidden, winner[0],
                              value_context_dim=VALUE_CONTEXT_DIM, feature_dim=FEATURE_DIM_ATTR,
                              num_layers=layers, extra={"data": args.data, "train_examples": n})
        print(f"\nwinner {label} beats 256x2-full ({winner[0]:.3f} < {ref[0]:.3f}) -> wrote {args.winner_out}")
    else:
        print(f"\nno cell beat 256x2-full ({ref[0]:.3f}); nothing saved")


if __name__ == "__main__":
    main()
