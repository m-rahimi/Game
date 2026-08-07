"""Diagnose whether DMC's Jack-dilemma choices that diverge from the heuristic
are genuine mistakes or just aggressive-but-fine.

For every DMC decision in a Jack-relevant spot where DMC != heuristic, score BOTH
candidate cards with paired Monte-Carlo rollouts (shared sampled information sets,
heuristic playout) and compare expected team-score differential. Higher EV = better.
"""
from __future__ import annotations

import argparse
from random import Random
from statistics import mean

from jack_game.cards import Card, team_for
from jack_game.cli import make_ai
from jack_game.engine import new_round, play_card
from jack_game.history import trick_penalty_cards, unseen_penalty_jacks
from jack_game.players import HeuristicPlayer, rollout_round, sample_information_set


def evaluate_action(state, player, card, deals, heuristic, rng):
    """Mean team-diff EV and club-Jack self-capture rate for `card`, over shared deals."""
    diffs = []
    club_self_capture = []
    for sampled in deals:
        rollout_state = sampled.clone()
        play_card(rollout_state, player, card)
        rollout_round(rollout_state, heuristic, rng)
        diffs.append(rollout_state.team_diff_for(player))
        captured = any(
            Card("C", 11) in r.penalties and team_for(r.winner) == team_for(player)
            for r in rollout_state.completed_tricks
        )
        club_self_capture.append(1.0 if captured else 0.0)
    return mean(diffs), mean(club_self_capture)


def is_jack_dilemma(state, player) -> bool:
    legal = state.legal_actions(player)
    if trick_penalty_cards(state):              # a penalty Jack sits in the current trick
        return True
    if any(c.is_penalty_jack for c in legal):   # we hold a penalty Jack and may play it
        return True
    # leading a suit whose Jack is still live (could pull a penalty out)
    if not state.current_trick:
        unseen = unseen_penalty_jacks(state)
        if any(Card(c.suit, 11) in unseen for c in legal):
            return True
    return False


def describe(state, player) -> str:
    trick = " ".join(f"P{p}:{c}" for p, c in state.current_trick) or "(leading)"
    hand = " ".join(str(c) for c in state.legal_actions(player))
    return f"trick={state.trick_index} P{player} table=[{trick}] legal=[{hand}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dmc_rtg_shaped_heuristic_from_random.pt")
    ap.add_argument("--policy", default="dmc", choices=["dmc", "dmc_guarded"])
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--deals", type=int, default=80, help="rollouts per candidate (paired)")
    ap.add_argument("--eps", type=float, default=0.25, help="EV gap to call a winner")
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    dmc = make_ai(args.policy, checkpoint_path=args.checkpoint)
    heuristic = HeuristicPlayer()
    rng = Random(args.seed)

    dmc_better = dmc_worse = tie = 0
    deltas = []
    dmc_capture_deltas = []
    examples = []

    for _ in range(args.games):
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            is_dmc_seat = team_for(actor) == 0
            if is_dmc_seat and len(state.legal_actions(actor)) > 1 and is_jack_dilemma(state, actor):
                dmc_card = dmc.choose_card(state.clone(), actor, rng)
                heur_card = heuristic.choose_card(state.clone(), actor, rng)
                if dmc_card != heur_card:
                    deals = [sample_information_set(state, actor, rng) for _ in range(args.deals)]
                    dmc_ev, dmc_cap = evaluate_action(state, actor, dmc_card, deals, heuristic, rng)
                    heur_ev, heur_cap = evaluate_action(state, actor, heur_card, deals, heuristic, rng)
                    delta = dmc_ev - heur_ev
                    deltas.append(delta)
                    dmc_capture_deltas.append(dmc_cap - heur_cap)
                    if delta > args.eps:
                        dmc_better += 1
                        verdict = "DMC better"
                    elif delta < -args.eps:
                        dmc_worse += 1
                        verdict = "DMC WORSE"
                    else:
                        tie += 1
                        verdict = "~tie"
                    if len(examples) < args.max_examples and abs(delta) > args.eps:
                        examples.append(
                            f"  [{verdict}] {describe(state, actor)}\n"
                            f"      DMC plays {dmc_card} (EV {dmc_ev:+.2f}, clubJ self-cap {dmc_cap:.2f}) | "
                            f"heur plays {heur_card} (EV {heur_ev:+.2f}, clubJ self-cap {heur_cap:.2f}) | "
                            f"delta {delta:+.2f}"
                        )
            actor = state.current_player
            card = (dmc if team_for(actor) == 0 else heuristic).choose_card(state.clone(), actor, rng)
            play_card(state, actor, card)

    total = dmc_better + dmc_worse + tie
    print(f"checkpoint: {args.checkpoint}")
    print(f"Jack-dilemma decisions where DMC diverged from heuristic: {total}")
    if total == 0:
        return
    print(f"  DMC choice better (EV gap > {args.eps}): {dmc_better}  ({dmc_better/total:.0%})")
    print(f"  DMC choice WORSE  (EV gap > {args.eps}): {dmc_worse}  ({dmc_worse/total:.0%})")
    print(f"  ~tie                                  : {tie}  ({tie/total:.0%})")
    print(f"  mean EV delta (DMC - heuristic): {mean(deltas):+.3f}")
    print(f"  mean club-Jack self-capture delta (DMC - heuristic): {mean(dmc_capture_deltas):+.3f}")
    print("\nExample divergences:")
    for ex in examples:
        print(ex)


if __name__ == "__main__":
    main()
