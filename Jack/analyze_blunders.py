"""Blunder review for logged human-vs-model games (Step 1 of hand-reading work).

For every AI decision in a game log (recorded via `python -m jack_game play --log ...`),
score each legal candidate two ways with paired heuristic rollouts:

  trueEV   — rollouts from the TRUE deal (we logged it): what was actually best.
  beliefEV — rollouts over uniform determinizations (voids respected): what a
             hand-reading-blind player sees. This is the information the model
             (and PIMC search) actually has.

Verdicts per decision, with regret = bestEV - chosenEV:
  HAND-READING blunder — bad vs the true hands (true regret >= eps) but fine
      under uniform belief (belief regret < eps/2): the move was only wrong
      because the model cannot infer hidden hands. Fixable by belief modeling.
  PLAIN blunder — bad under BOTH: a search/eval mistake, not an information one.
  ok — everything else.

Human-flagged moves ('flag' during play) are always reported with a verdict.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from random import Random
from statistics import mean

from jack_game.cards import card_from_text
from jack_game.engine import play_card
from jack_game.gamelog import replay_decisions
from jack_game.players import HeuristicPlayer, rollout_round, sample_information_set


def candidate_evs(state, actor, legal, deals_per_candidate: int, true_deal: bool, seed: int) -> dict[str, float]:
    """Mean team-diff EV per legal card, paired across candidates (shared randomness)."""
    heuristic = HeuristicPlayer()
    evs: dict[str, list[float]] = {str(card): [] for card in legal}
    for k in range(deals_per_candidate):
        rng = Random(seed * 10_000 + k)
        base = state.clone() if true_deal else sample_information_set(state, actor, rng)
        for card in legal:
            rollout = base.clone()
            rollout_rng = Random(seed * 10_000 + k)  # same continuation noise per candidate
            play_card(rollout, actor, card)
            rollout_round(rollout, heuristic, rollout_rng)
            evs[str(card)].append(rollout.team_diff_for(actor))
    return {card: mean(values) for card, values in evs.items()}


def decision_type(state, card) -> str:
    if not state.current_trick:
        return "lead"
    led = state.current_trick[0][1].suit
    return "follow" if card.suit == led else "discard"


def analyze_decision(job):
    game_index, move_index, game, deals, seed = job
    for index, state, move in replay_decisions(game):
        if index != move_index:
            continue
        actor = int(move["seat"])
        legal = state.legal_actions(actor)
        chosen = move["card"]
        true_ev = candidate_evs(state, actor, legal, deals, true_deal=True, seed=seed)
        belief_ev = candidate_evs(state, actor, legal, deals, true_deal=False, seed=seed + 1)
        best_true = max(true_ev, key=true_ev.get)
        regret_true = true_ev[best_true] - true_ev[chosen]
        regret_belief = max(belief_ev.values()) - belief_ev[chosen]
        table = " ".join(f"P{p}:{c}" for p, c in state.current_trick) or "(leading)"
        return {
            "game": game_index,
            "move": move_index,
            "trick": state.trick_index,
            "seat": actor,
            "type": decision_type(state, card_from_text(chosen)),
            "chosen": chosen,
            "best_true": best_true,
            "regret_true": regret_true,
            "regret_belief": regret_belief,
            "true_ev": true_ev,
            "belief_ev": belief_ev,
            "table": table,
            "legal": " ".join(str(c) for c in legal),
            "visits": move.get("visits"),
            "q": move.get("q"),
            "flagged": move_index in game.get("flags", []),
        }
    raise ValueError(f"move {move_index} not found in game {game_index}")


def verdict(result, eps: float) -> str:
    if result["regret_true"] >= eps and result["regret_belief"] < eps / 2:
        return "HAND-READING"
    if result["regret_true"] >= eps and result["regret_belief"] >= eps / 2:
        return "PLAIN"
    return "ok"


def describe(result, eps: float) -> str:
    flag = " [FLAGGED BY YOU]" if result["flagged"] else ""
    lines = [
        f"game {result['game']} move {result['move']}  trick {result['trick'] + 1}  "
        f"P{result['seat']} ({result['type']})  table=[{result['table']}]{flag}",
        f"  legal: {result['legal']}",
        f"  played {result['chosen']:>4}  trueEV {result['true_ev'][result['chosen']]:+.2f}   "
        f"best was {result['best_true']} (trueEV {result['true_ev'][result['best_true']]:+.2f})   "
        f"true regret {result['regret_true']:+.2f}, belief regret {result['regret_belief']:+.2f}",
        f"  verdict: {verdict(result, eps)}"
        + (" — only wrong because hidden hands were unreadable" if verdict(result, eps) == "HAND-READING" else ""),
    ]
    if result["visits"]:
        cards = sorted(result["visits"], key=result["visits"].get, reverse=True)[:4]
        from jack_game.mcts import index_to_card

        stats = "  ".join(
            f"{index_to_card(int(i))}:N={result['visits'][i]}"
            + (f",Q={result['q'][i]:+.2f}" if result.get("q") and i in result["q"] else "")
            for i in cards
        )
        lines.append(f"  mcts root: {stats}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--deals", type=int, default=48, help="paired rollouts per candidate per arm")
    ap.add_argument("--eps", type=float, default=0.5, help="regret (score-diff units) to call a blunder")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--include-human", action="store_true", help="also analyze the human's moves")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    games = json.loads(open(args.log).read())["games"]
    jobs = []
    for game_index, game in enumerate(games):
        for index, state, move in replay_decisions(game):
            if move["human"] and not args.include_human:
                continue
            if len(state.legal_actions(int(move["seat"]))) < 2:
                continue
            jobs.append((game_index, index, game, args.deals, args.seed + 97 * len(jobs)))
    print(f"games={len(games)} decisions to analyze={len(jobs)} "
          f"(deals={args.deals}/candidate/arm, eps={args.eps})")

    with Pool(processes=min(args.workers, max(1, len(jobs)))) as pool:
        results = pool.map(analyze_decision, jobs)

    blunders = [r for r in results if verdict(r, args.eps) != "ok"]
    hand_reading = [r for r in blunders if verdict(r, args.eps) == "HAND-READING"]
    plain = [r for r in blunders if verdict(r, args.eps) == "PLAIN"]

    print("\n=== summary ===")
    print(f"decisions: {len(results)}   blunders (true regret >= {args.eps}): {len(blunders)} "
          f"({len(blunders) / max(1, len(results)):.0%})")
    print(f"  HAND-READING (fine under belief, bad vs true hands): {len(hand_reading)}")
    print(f"  PLAIN (bad under both):                              {len(plain)}")
    print(f"mean true regret: {mean(r['regret_true'] for r in results):+.3f}")
    for kind in ("lead", "follow", "discard"):
        sub = [r for r in results if r["type"] == kind]
        if sub:
            n_bl = sum(1 for r in sub if verdict(r, args.eps) != "ok")
            print(f"  {kind:>7}: {len(sub):3d} decisions, {n_bl:2d} blunders, "
                  f"mean true regret {mean(r['regret_true'] for r in sub):+.3f}")

    flagged = [r for r in results if r["flagged"]]
    if flagged:
        print("\n=== moves you flagged ===")
        for r in flagged:
            print(describe(r, args.eps) + "\n")

    print(f"\n=== top {args.top} by true regret ===")
    for r in sorted(results, key=lambda r: -r["regret_true"])[: args.top]:
        print(describe(r, args.eps) + "\n")


if __name__ == "__main__":
    main()
