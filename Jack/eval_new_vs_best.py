"""Head-to-head: the NEW model (Team 0) vs the current BEST model (Team 1).

Both sides are net-guided ISMCTS; only the policy/value checkpoints differ, so
this isolates the effect of the richer 572-dim input + depth. Same seeded deals,
identical simulation budget per side. Sweeps the simulation count so you can see
sim-scaling: a healthy value net's win rate should hold or rise with more sims;
NEGATIVE sim-scaling means the new value net is broken (biased signal amplified
by deeper search) regardless of its holdout MSE.

Default BEST = ISMCTS(policy_search100k + value_search_ctx) -- the production best.
Default NEW  = ISMCTS(policy_attr572_l{depth} + value_attr572_ctx_l{depth}).

Team assignment is swapped every other game to cancel any seat advantage.

Run (GPU):
  PY=/Adhoc/Software/conda_envs/mrenv/bin/python
  $PY eval_new_vs_best.py --new-depth 3 --games 200 --sims 200 600 1000
  # or point at explicit checkpoints:
  $PY eval_new_vs_best.py \
     --new-policy checkpoints/policy_attr572_l3.pt \
     --new-value  checkpoints/value_attr572_ctx_l3.pt \
     --games 200 --sims 200 600
"""
from __future__ import annotations

import argparse
from random import Random
from time import monotonic

from jack_game.cards import team_for
from jack_game.engine import new_round, play_card
from jack_game.mcts import ISMCTSPlayer
from jack_game.policy import PolicyFunction
from jack_game.value import ValueFunction


def build_ismcts(policy_fn, value_fn, sims, c_puct):
    return ISMCTSPlayer(value_fn=value_fn, policy_fn=policy_fn,
                        simulations=sims, c_puct=c_puct)


def play_matchup(new_pol, new_val, best_pol, best_val, games, sims, c_puct, seed):
    """Return dict of metrics for NEW (as the tracked side) vs BEST at one sim count."""
    rng = Random(seed)
    new_wins = new_diff = 0
    jacks_new = jacks_best = 0
    tie = 0
    t_new = t_best = 0.0
    n_new = n_best = 0
    for g in range(games):
        new_p = build_ismcts(new_pol, new_val, sims, c_puct)
        best_p = build_ismcts(best_pol, best_val, sims, c_puct)
        # Swap seats every other game so NEW isn't always Team 0.
        new_is_team0 = (g % 2 == 0)
        if new_is_team0:
            players = [new_p, best_p, new_p, best_p]
        else:
            players = [best_p, new_p, best_p, new_p]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            t0 = monotonic()
            card = players[actor].choose_card(state.clone(), actor, rng)
            dt = monotonic() - t0
            is_new_actor = (team_for(actor) == 0) == new_is_team0
            if is_new_actor:
                t_new += dt; n_new += 1
            else:
                t_best += dt; n_best += 1
            play_card(state, actor, card)
        new_team = 0 if new_is_team0 else 1
        s_new = state.team_score(new_team)
        s_best = state.team_score(1 - new_team)
        new_wins += int(s_new > s_best)
        tie += int(s_new == s_best)
        new_diff += s_new - s_best
        for r in state.completed_tricks:
            for c in r.penalties:
                if (team_for(r.winner) == 0) == new_is_team0:
                    jacks_new += 1
                else:
                    jacks_best += 1
    return {
        "sims": sims, "games": games,
        "new_win_rate": new_wins / games,
        "tie_rate": tie / games,
        "new_score_diff": new_diff / games,
        "jacks_new": jacks_new / games, "jacks_best": jacks_best / games,
        "ms_new": 1000 * t_new / max(1, n_new), "ms_best": 1000 * t_best / max(1, n_best),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-depth", type=int, default=None,
                    help="if set, use checkpoints/{policy,value}_attr572*_l{depth}")
    ap.add_argument("--new-policy", default=None)
    ap.add_argument("--new-value", default=None)
    ap.add_argument("--best-policy", default="checkpoints/policy_search100k.pt")
    ap.add_argument("--best-value", default="checkpoints/value_search_ctx.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, nargs="+", default=[200, 600])
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.new_policy is None or args.new_value is None:
        if args.new_depth is None:
            ap.error("provide --new-depth OR both --new-policy and --new-value")
        args.new_policy = args.new_policy or f"checkpoints/policy_attr572_l{args.new_depth}.pt"
        args.new_value = args.new_value or f"checkpoints/value_attr572_ctx_l{args.new_depth}.pt"

    print(f"NEW : policy={args.new_policy}  value={args.new_value}")
    print(f"BEST: policy={args.best_policy}  value={args.best_value}")
    new_pol = PolicyFunction(args.new_policy)
    new_val = ValueFunction(args.new_value)
    best_pol = PolicyFunction(args.best_policy)
    best_val = ValueFunction(args.best_value)
    print(f"NEW feature_dim: policy={new_pol.feature_dim} value={new_val.feature_dim} "
          f"(+ctx {new_val.value_context_dim})")

    rows = []
    for sims in args.sims:
        # Same seed for every sim count -> same deals, so differences are pure sim effect.
        m = play_matchup(new_pol, new_val, best_pol, best_val, args.games, sims, args.c_puct, args.seed)
        rows.append(m)
        print(f"\nsims={sims:>4} ({args.games} games)  NEW vs BEST")
        print(f"  NEW win rate : {m['new_win_rate']:.3f}   (tie {m['tie_rate']:.3f})")
        print(f"  NEW score diff: {m['new_score_diff']:+.3f} / game")
        print(f"  penalty Jacks/game: NEW {m['jacks_new']:.2f}  BEST {m['jacks_best']:.2f}")
        print(f"  avg decision: NEW {m['ms_new']:.0f}ms  BEST {m['ms_best']:.0f}ms")

    print("\n=== SIM-SCALING SUMMARY (NEW vs BEST) ===")
    print(f"{'sims':>6} {'NEW_win':>9} {'score_diff':>11}")
    for m in rows:
        print(f"{m['sims']:>6} {m['new_win_rate']:>9.3f} {m['new_score_diff']:>+11.3f}")
    if len(rows) >= 2:
        trend = rows[-1]["new_win_rate"] - rows[0]["new_win_rate"]
        verdict = "POSITIVE" if trend > 0.01 else ("FLAT" if trend > -0.01 else "NEGATIVE")
        print(f"\nsim-scaling ({rows[0]['sims']}->{rows[-1]['sims']} sims): {trend:+.3f} win rate = {verdict}")
        print("(NEGATIVE sim-scaling = the new value net is broken; do not ship it as best.)")


if __name__ == "__main__":
    main()
