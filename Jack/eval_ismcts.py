"""Head-to-head: ISMCTS (policy prior + value net), Team 0, vs plain search, Team 1.

Reports win rate, score diff, Jack capture, and average decision time per side.
"""
from __future__ import annotations

import argparse
from random import Random
from time import monotonic

from jack_game.cards import team_for
from jack_game.engine import new_round, play_card
from jack_game.mcts import ISMCTSPlayer
from jack_game.players import SearchPlayer
from jack_game.policy import PolicyFunction
from jack_game.value import ValueFunction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-checkpoint", default="checkpoints/policy_heuristic.pt")
    ap.add_argument("--value-checkpoint", default="checkpoints/value_heuristic_big.pt")
    ap.add_argument("--belief-checkpoint", default=None, help="optional belief net for weighted determinization")
    ap.add_argument("--belief-lambda", type=float, default=0.5, help="belief vs neutral dealer mix (1.0 = pure belief)")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--simulations", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--search-rollouts", type=int, default=10)
    ap.add_argument("--search-time-limit", type=float, default=0.5)
    ap.add_argument("--opponent", choices=["search", "heuristic"], default="search")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from jack_game.players import HeuristicPlayer

    policy_fn = PolicyFunction(args.policy_checkpoint)
    value_fn = ValueFunction(args.value_checkpoint)
    belief_fn = None
    if args.belief_checkpoint:
        from jack_game.belief import BeliefFunction

        belief_fn = BeliefFunction(args.belief_checkpoint)
    rng = Random(args.seed)

    wins = diff = 0
    jacks0 = jacks1 = 0
    move_time = {0: 0.0, 1: 0.0}
    move_count = {0: 0, 1: 0}

    for _ in range(args.games):
        ismcts = ISMCTSPlayer(
            value_fn=value_fn, policy_fn=policy_fn,
            simulations=args.simulations, c_puct=args.c_puct,
            belief_fn=belief_fn, belief_lambda=args.belief_lambda,
        )
        if args.opponent == "heuristic":
            opponent = HeuristicPlayer()
        else:
            opponent = SearchPlayer(
                rollouts_per_action=args.search_rollouts, time_limit_seconds=args.search_time_limit,
            )
        players = [ismcts, opponent, ismcts, opponent]
        state = new_round(seed=rng.randrange(10**9), leader=0)
        while not state.round_over:
            actor = state.current_player
            t0 = monotonic()
            card = players[actor].choose_card(state.clone(), actor, rng)
            move_time[team_for(actor)] += monotonic() - t0
            move_count[team_for(actor)] += 1
            play_card(state, actor, card)
        wins += int(state.team_score(0) > state.team_score(1))
        diff += state.team_score(0) - state.team_score(1)
        for r in state.completed_tricks:
            for c in r.penalties:
                if team_for(r.winner) == 0:
                    jacks0 += 1
                else:
                    jacks1 += 1

    n = args.games
    print(f"policy: {args.policy_checkpoint}  value: {args.value_checkpoint}  belief: {args.belief_checkpoint}")
    print(f"settings: {n} games, sims={args.simulations}, c_puct={args.c_puct}, "
          f"search_rollouts={args.search_rollouts}, search_time={args.search_time_limit}s")
    print(f"ISMCTS (team0) win rate vs {args.opponent}: {wins/n:.3f}")
    print(f"ISMCTS score diff vs {args.opponent}:       {diff/n:+.3f}")
    print(f"penalty Jacks captured/game: ISMCTS {jacks0/n:.2f}  {args.opponent} {jacks1/n:.2f}")
    print(f"avg decision time: ISMCTS {1000*move_time[0]/max(1,move_count[0]):.1f}ms  "
          f"plain {1000*move_time[1]/max(1,move_count[1]):.1f}ms")


if __name__ == "__main__":
    main()
