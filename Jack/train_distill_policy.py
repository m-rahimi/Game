"""Train the distilled policy: KL to ISMCTS visit distributions (Suggestion 2).

Target = the root visit distribution of the 800-sim teacher — the full output of
the search, not its argmax. Loss = soft cross-entropy over legal moves
(equivalent to KL up to the target's constant entropy):

    loss = -(visits * log_softmax(masked logits)).sum(dim=1)

Split is by game_id (rows per game vary — forced moves are skipped — so 52-block
splitting does NOT apply here). Reports on the holdout: soft-CE, KL, and top-1
agreement with the teacher's most-visited move.

Run (GPU):
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY train_distill_policy.py --data data/distill_ismcts800_100k.pt \
     --depths 2 3 --epochs 30 --out-prefix checkpoints/policy_distill800_
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jack_game.bc import resolve_device
from jack_game.policy import PolicyNet, save_policy_checkpoint


def soft_ce(logits, legal, target):
    logits = logits.masked_fill(~legal, float("-inf"))
    logp = torch.log_softmax(logits, dim=1)
    # target is 0 on illegal moves, so masked -inf never multiplies a nonzero weight
    return -(target * logp.masked_fill(~legal, 0.0)).sum(dim=1)


def holdout_metrics(model, feats, legal, visits, device, bs=16384):
    model.eval()
    ce_total = kl_total = 0.0
    agree = 0
    with torch.no_grad():
        for s in range(0, feats.shape[0], bs):
            xb = feats[s:s + bs].to(device)
            lb = legal[s:s + bs].to(device)
            tb = visits[s:s + bs].to(device)
            logits = model(xb)
            ce = soft_ce(logits, lb, tb)
            ce_total += float(ce.sum())
            # KL(target || pred) = soft_ce - target entropy
            ent = -(tb.clamp_min(1e-9).log() * tb).sum(dim=1)
            kl_total += float((ce - ent).sum())
            pred = logits.masked_fill(~lb, float("-inf")).argmax(dim=1)
            agree += int((pred == tb.argmax(dim=1)).sum())
    n = feats.shape[0]
    return ce_total / n, kl_total / n, agree / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill_ismcts800_100k.pt")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout-frac", type=float, default=0.05,
                    help="fraction of GAMES held out")
    ap.add_argument("--init-from", default=None,
                    help="optional warm-start policy checkpoint (must match dims)")
    ap.add_argument("--out-prefix", default="checkpoints/policy_distill800_")
    args = ap.parse_args()

    device = resolve_device(None)
    torch.manual_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    feats, visits, legal, gid = data["features"], data["visits"], data["legal"], data["game_id"]
    feature_dim = feats.shape[1]
    n_games = int(gid.max()) + 1
    print(f"data={args.data} rows={feats.shape[0]} games={n_games} "
          f"feature_dim={feature_dim} device={device}")

    # Game-level split by game_id.
    g = torch.Generator().manual_seed(args.seed)
    gperm = torch.randperm(n_games, generator=g)
    n_ho_games = max(1, int(n_games * args.holdout_frac))
    ho_games = torch.zeros(n_games, dtype=torch.bool)
    ho_games[gperm[:n_ho_games]] = True
    ho_mask = ho_games[gid]
    tr = ~ho_mask
    print(f"train rows={int(tr.sum())} holdout rows={int(ho_mask.sum())} "
          f"({n_ho_games} games held out)")

    f_tr, l_tr, v_tr = feats[tr], legal[tr], visits[tr]
    f_ho, l_ho, v_ho = feats[ho_mask], legal[ho_mask], visits[ho_mask]

    results = []
    for d in args.depths:
        torch.manual_seed(args.seed)
        model = PolicyNet(input_dim=feature_dim, hidden_dim=args.hidden_dim, num_layers=d).to(device)
        if args.init_from:
            ck = torch.load(args.init_from, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            print(f"  warm-started L{d} from {args.init_from}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(f_tr, l_tr, v_tr),
                            batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            tot = seen = 0.0
            for xb, lb, tb in loader:
                xb, lb, tb = xb.to(device), lb.to(device), tb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = soft_ce(model(xb), lb, tb).mean()
                loss.backward()
                opt.step()
                tot += float(loss) * xb.shape[0]
                seen += xb.shape[0]
            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"    L{d} epoch={epoch:>2} train_soft_ce={tot/seen:.4f}", flush=True)
        ce, kl, agree = holdout_metrics(model, f_ho, l_ho, v_ho, device)
        path = Path(f"{args.out_prefix}l{d}.pt")
        save_policy_checkpoint(path, {k: v.cpu() for k, v in model.state_dict().items()},
                               args.hidden_dim, ce, feature_dim=feature_dim, num_layers=d,
                               extra={"data": args.data, "target": "ismcts_visits",
                                      "split": "game_id", "init_from": args.init_from})
        print(f"  [L{d}] holdout soft_ce={ce:.4f} KL={kl:.4f} "
              f"top1-agree-with-teacher={agree:.4f} -> {path}", flush=True)
        results.append((d, ce, kl, agree))

    print("\n=== DISTILL SUMMARY (game-split holdout) ===")
    print(f"{'depth':>6} {'soft_ce':>9} {'KL':>8} {'top1_agree':>11}")
    for d, ce, kl, agree in results:
        print(f"L{d:<5} {ce:>9.4f} {kl:>8.4f} {agree:>11.4f}")
    print("\nNext: evaluate GREEDY play, e.g.")
    print("  eval_greedy_h2h.py --policy-a <new> --policy-b checkpoints/policy_attr572_g196000.pt")
    print("  eval_greedy_vs_ismcts.py --greedy-policy <new> --simulations 600")


if __name__ == "__main__":
    main()
