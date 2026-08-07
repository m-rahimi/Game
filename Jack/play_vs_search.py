"""Run guarded-DMC (Team 0) head-to-head against the search player (Team 1).

Prints a full play-by-play of one game so you can judge the model's decisions,
then plays N silent games for an aggregate win rate / Jack-capture readout.
"""
from __future__ import annotations

import argparse
from random import Random

from jack_game.cards import team_for
from jack_game.dmc import GuardedDMCPlayer
from jack_game.engine import new_round, play_card
from jack_game.players import SearchPlayer


def label(p: int) -> str:
    return f"P{p}(DMC)" if team_for(p) == 0 else f"P{p}(search)"


def play_one(players, seed: int, trace: bool):
    rng = Random(seed)
    state = new_round(seed=seed, leader=0)
    while not state.round_over:
        actor = state.current_player
        card = players[actor].choose_card(state.clone(), actor, rng)
        result = play_card(state, actor, card)
        if trace and result is not None:
            plays = " ".join(f"{label(p)}:{c}" for p, c in result.plays)
            jacks = " ".join(str(c) for c in result.penalties) or "-"
            print(
                f"  trick {state.trick_index:2d}: {plays}  -> {label(result.winner)} "
                f"wins {result.points:+d}  (penalty Jacks: {jacks})"
            )
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dmc_rtg_shaped_heuristic_from_random.pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    def make_players():
        dmc = GuardedDMCPlayer(args.checkpoint)
        search = SearchPlayer(rollouts_per_action=args.rollouts, time_limit_seconds=args.time_limit)
        return [dmc, search, dmc, search]  # seats 0/2 = DMC, 1/3 = search

    rng = Random(args.seed)

    print("=== ONE TRACED GAME: Team 0 = guarded-DMC, Team 1 = search ===")
    state = play_one(make_players(), seed=rng.randrange(10**9), trace=True)
    print(f"  FINAL: DMC(team0)={state.team_score(0)}  search(team1)={state.team_score(1)}  "
          f"-> {'DMC wins' if state.team_score(0) > state.team_score(1) else 'search wins' if state.team_score(1) > state.team_score(0) else 'tie'}")

    print(f"\n=== AGGREGATE over {args.games} games (search: {args.rollouts} rollouts, {args.time_limit}s/move) ===")
    wins = diff = 0
    jacks0 = jacks1 = 0
    for _ in range(args.games):
        state = play_one(make_players(), seed=rng.randrange(10**9), trace=False)
        wins += int(state.team_score(0) > state.team_score(1))
        diff += state.team_score(0) - state.team_score(1)
        for r in state.completed_tricks:
            for c in r.penalties:
                if team_for(r.winner) == 0:
                    jacks0 += 1
                else:
                    jacks1 += 1
    n = args.games
    print(f"  DMC win rate vs search:  {wins/n:.3f}")
    print(f"  DMC score diff vs search: {diff/n:+.3f}")
    print(f"  penalty Jacks captured/game:  DMC {jacks0/n:.2f}   search {jacks1/n:.2f}")


if __name__ == "__main__":
    main()
