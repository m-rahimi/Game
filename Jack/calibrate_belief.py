"""Temperature-calibrate the belief net on its game-split holdout.

Overconfident belief probs make the weighted determinizer overcommit to wrong
reads (measured as the plain-blunder growth in the human games). A single
softmax temperature T (probs = softmax(logits / T)) fixes the confidence scale
without retraining. Writes a calibrated copy of the checkpoint with the fitted
``temperature`` plus per-trick top-1 accuracies (used to derive λ schedules).
"""
from __future__ import annotations

import argparse

import torch

from jack_game.bc import resolve_device
from jack_game.belief import BeliefNet
from train_belief import TRICK_START


def masked_nll(logits, targets, hidden, T):
    lp = torch.log_softmax(logits / T, dim=2)
    nll = -lp.gather(2, targets.unsqueeze(2)).squeeze(2)
    return float((nll * hidden).sum() / hidden.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/search_beliefs_40k.pt")
    ap.add_argument("--checkpoint", default="checkpoints/belief_search40k.pt")
    ap.add_argument("--out", default="checkpoints/belief_search40k_cal.pt")
    ap.add_argument("--seed", type=int, default=1, help="must match train_belief.py split seed")
    args = ap.parse_args()

    device = resolve_device(None)
    data = torch.load(args.data, weights_only=False)
    feats, loc = data["features"], data["card_loc"]

    # recreate the exact game-level holdout used by train_belief.py
    torch.manual_seed(args.seed)
    game_perm = torch.randperm(feats.shape[0] // 52)
    perm = (game_perm[:, None] * 52 + torch.arange(52)[None, :]).reshape(-1)
    feats, loc = feats[perm], loc[perm]
    nh = min(max(20000, feats.shape[0] // 10), feats.shape[0] // 2)
    nh -= nh % 52
    x_ho, y_ho = feats[:nh], loc[:nh]

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BeliefNet(hidden_dim=int(checkpoint["hidden_dim"]),
                      num_layers=int(checkpoint["num_layers"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # collect holdout logits once
    logits_all, tgt_all, hid_all, trick_all = [], [], [], []
    with torch.no_grad():
        for start in range(0, nh, 8192):
            xb = x_ho[start:start + 8192].to(device)
            yb = y_ho[start:start + 8192].to(device)
            logits_all.append(model(xb))
            tgt_all.append(yb.long().clamp(min=1) - 1)
            hid_all.append(yb > 0)
            trick_all.append(x_ho[start:start + 8192, TRICK_START:TRICK_START + 13].argmax(dim=1))
    logits = torch.cat(logits_all)
    targets = torch.cat(tgt_all)
    hidden = torch.cat(hid_all)
    tricks = torch.cat(trick_all).to(device)

    # fit T: coarse grid then refine
    grid = [x / 20 for x in range(10, 61)]  # 0.5 .. 3.0
    best_T = min(grid, key=lambda T: masked_nll(logits, targets, hidden, T))
    fine = [best_T + d / 100 for d in range(-4, 5)]
    best_T = min(fine, key=lambda T: masked_nll(logits, targets, hidden, T))

    nll_raw = masked_nll(logits, targets, hidden, 1.0)
    nll_cal = masked_nll(logits, targets, hidden, best_T)
    print(f"holdout NLL raw (T=1): {nll_raw:.4f}   calibrated (T={best_T:.2f}): {nll_cal:.4f}")

    # per-trick: top-1 accuracy and mean confidence (raw vs calibrated)
    probs_cal = torch.softmax(logits / best_T, dim=2)
    probs_raw = torch.softmax(logits, dim=2)
    hit = (logits.argmax(dim=2) == targets).float()
    trick_top1 = []
    print(f"{'trick':>6} {'top1':>6} {'conf(raw)':>10} {'conf(cal)':>10}")
    for t in range(13):
        mask = hidden & (tricks == t).unsqueeze(1)
        if not mask.any():
            trick_top1.append(1 / 3)
            continue
        acc = float((hit * mask).sum() / mask.sum())
        conf_raw = float((probs_raw.max(dim=2).values * mask).sum() / mask.sum())
        conf_cal = float((probs_cal.max(dim=2).values * mask).sum() / mask.sum())
        trick_top1.append(acc)
        print(f"{t + 1:>6} {acc:>6.3f} {conf_raw:>10.3f} {conf_cal:>10.3f}")

    payload = dict(checkpoint)
    payload["temperature"] = best_T
    payload["trick_top1"] = trick_top1
    torch.save(payload, args.out)
    print(f"wrote {args.out} (temperature={best_T:.2f})")


if __name__ == "__main__":
    main()
