from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random

from .cards import Card, SUIT_NAMES, card_from_text, card_sort_key, team_for
from .engine import GameState, new_round, play_card, play_round
from .players import HeuristicPlayer, RandomPlayer, SearchPlayer
from .session import MATCH_TARGET, MatchState


@dataclass(frozen=True)
class MatchupMetrics:
    games: int
    team0_win_rate: float
    team0_score_diff: float
    team0_jack_capture_rate: dict[str, float]
    team1_jack_capture_rate: dict[str, float]


def make_ai(
    name: str,
    rollouts: int = 24,
    lambda_clubs: float = 2.5,
    lambda_hearts: float = 0.25,
    time_limit: float | None = 2.0,
    checkpoint_path: str | None = None,
    policy_checkpoint: str | None = None,
    value_checkpoint: str | None = None,
    simulations: int = 200,
    belief_checkpoint: str | None = None,
):
    if name == "random":
        return RandomPlayer()
    if name == "search":
        return SearchPlayer(
            rollouts_per_action=rollouts,
            lambda_clubs=lambda_clubs,
            lambda_hearts=lambda_hearts,
            time_limit_seconds=time_limit,
        )
    if name == "search_value":
        if checkpoint_path is None:
            raise ValueError("search_value player requires --checkpoint (a value-net checkpoint)")
        from .value import ValueFunction

        return SearchPlayer(
            rollouts_per_action=rollouts,
            time_limit_seconds=time_limit,
            value_fn=ValueFunction(checkpoint_path),
        )
    if name == "bc":
        if checkpoint_path is None:
            raise ValueError("bc player requires --checkpoint")
        from .bc import BCPlayer

        return BCPlayer(checkpoint_path)
    if name == "dmc":
        if checkpoint_path is None:
            raise ValueError("dmc player requires --checkpoint")
        from .dmc import DMCPlayer

        return DMCPlayer(checkpoint_path)
    if name == "dmc_guarded":
        if checkpoint_path is None:
            raise ValueError("dmc_guarded player requires --checkpoint")
        from .dmc import GuardedDMCPlayer

        return GuardedDMCPlayer(checkpoint_path)
    if name == "policy":
        ckpt = policy_checkpoint or checkpoint_path
        if ckpt is None:
            raise ValueError("policy player requires --policy-checkpoint (or --checkpoint)")
        if str(ckpt).endswith(".onnx"):
            from .onnx_policy import get_onnx_policy_player

            return get_onnx_policy_player(ckpt)
        from .policy import GreedyPolicyPlayer

        return GreedyPolicyPlayer(ckpt)
    if name == "ismcts":
        if policy_checkpoint is None or value_checkpoint is None:
            raise ValueError("ismcts player requires --policy-checkpoint and --value-checkpoint")
        from .mcts import ISMCTSPlayer
        from .policy import PolicyFunction
        from .value import ValueFunction

        belief_fn = None
        if belief_checkpoint is not None:
            from .belief import BeliefFunction

            belief_fn = BeliefFunction(belief_checkpoint)
        return ISMCTSPlayer(
            value_fn=ValueFunction(value_checkpoint),
            policy_fn=PolicyFunction(policy_checkpoint),
            simulations=simulations,
            time_limit_seconds=time_limit,
            belief_fn=belief_fn,
        )
    return HeuristicPlayer()


def play_human_game(seed: int | None = None, ai: str = "heuristic", human_player: int = 0, checkpoint_path: str | None = None,
                    policy_checkpoint: str | None = None, value_checkpoint: str | None = None, simulations: int = 400,
                    log_path: str | None = None, belief_checkpoint: str | None = None) -> None:
    rng = Random(seed)
    match = MatchState.new(rng)
    bots = [
        make_ai(ai, checkpoint_path=checkpoint_path, policy_checkpoint=policy_checkpoint,
                value_checkpoint=value_checkpoint, simulations=simulations, belief_checkpoint=belief_checkpoint)
        for _ in range(4)
    ]
    log = None
    if log_path:
        from .gamelog import GameLog

        log = GameLog(log_path, meta={
            "ai": ai, "human_player": human_player, "simulations": simulations,
            "checkpoint": checkpoint_path, "policy_checkpoint": policy_checkpoint,
            "value_checkpoint": value_checkpoint, "belief_checkpoint": belief_checkpoint, "seed": seed,
            "match_target": MATCH_TARGET,
        })

    print("You are player", human_player, "on Team", team_for(human_player), "(players 0 and 2 vs 1 and 3).")
    print(f"First to {MATCH_TARGET} points wins the match.")
    print("Enter cards like JC, 10H, AS. Type 'hand' to reprint your hand.")
    if log:
        print("Logging to", log_path, "— type 'flag' on your turn to mark the last AI move as a suspected blunder.")

    while True:
        match.begin_round()
        round_seed = rng.randrange(10**9)
        state = new_round(seed=round_seed, leader=match.next_leader)

        if match.tiebreak:
            print(f"\n=== Tie-break round (both teams at {MATCH_TARGET}) ===")
        else:
            print(f"\n=== Round {match.round_number} ===")
            print(f"Match totals: Team 0={match.team_totals[0]}  Team 1={match.team_totals[1]}")
        print(f"P{match.next_leader} leads this round.")

        if log:
            round_meta = {
                "round": match.round_number,
                "tiebreak": match.tiebreak,
                "match_totals_before": match.team_totals.copy(),
                "leader": match.next_leader,
            }
            log.start_game(state, extra_meta=round_meta)

        _play_round_interactive(state, human_player, bots, rng, log)

        round_t0 = state.team_score(0)
        round_t1 = state.team_score(1)
        print("\nRound over.")
        print(f"Round scores: Team 0={round_t0}  Team 1={round_t1}")
        if not match.tiebreak:
            print(f"Match totals: Team 0={match.team_totals[0] + round_t0}  Team 1={match.team_totals[1] + round_t1}")

        if log:
            log.end_game(state)

        result = match.apply_round(state)
        if result is None:
            if match.tiebreak:
                print("Tie-break round tied — playing another tie-break round.")
            match.rotate_leader()
            continue

        if result == "tiebreak":
            print(f"\nBoth teams reached exactly {MATCH_TARGET} — tie-break round next.")
            match.tiebreak = True
            match.rotate_leader()
            continue

        winner = int(result)
        print(f"\n=== Match over ===")
        if not match.tiebreak:
            print(f"Final match totals: Team 0={match.team_totals[0]}  Team 1={match.team_totals[1]}")
        else:
            print(f"Tie-break round winner: Team {winner}")
        if team_for(human_player) == winner:
            print("Your team won the match!")
        else:
            print("Your team lost the match.")
        if log:
            print(f"Match logged ({len(log.games)} round(s) in {log_path}). "
                  f"Review with: python analyze_blunders.py --log {log_path}")
        break


def _play_round_interactive(
    state: GameState,
    human_player: int,
    bots: list,
    rng: Random,
    log,
) -> None:
    while not state.round_over:
        actor = state.current_player
        if not state.current_trick:
            print(f"\nTrick {state.trick_index + 1}, leader P{actor}")
        else:
            print("Current trick:", format_trick(state.current_trick))

        if actor == human_player:
            card = prompt_for_card(state, actor, log=log)
            if log:
                log.record_move(actor, card, human=True)
        else:
            bot = bots[actor]
            if log and hasattr(bot, "choose_with_stats"):
                card, visits, q = bot.choose_with_stats(state.clone(), actor, rng)
                log.record_move(actor, card, human=False, visits=visits, q=q)
            else:
                card = bot.choose_card(state.clone(), actor, rng)
                if log:
                    log.record_move(actor, card, human=False)
            print(f"P{actor} plays {card}")
        result = play_card(state, actor, card)
        if result is not None:
            print("Completed trick:", format_trick(result.plays))
            print(f"P{result.winner} wins {result.points:+d} point(s)")
            print(f"Scores: Team 0={state.team_score(0)} Team 1={state.team_score(1)}")


def prompt_for_card(state: GameState, player: int, log=None) -> Card:
    while True:
        legal = state.legal_actions(player)
        print("Hand:", " ".join(str(card) for card in sorted(state.hands[player], key=card_sort_key)))
        print("Legal:", " ".join(str(card) for card in legal))
        raw = input("Play card> ").strip()
        if raw.lower() == "hand":
            continue
        if raw.lower() == "flag":
            if log is None:
                print("Logging is off (--log not set); nothing to flag.")
            else:
                flagged = log.flag_last_ai_move()
                print(f"Flagged last AI move: {flagged}." if flagged else "No AI move to flag yet.")
            continue
        try:
            card = card_from_text(raw)
        except ValueError:
            print("Unknown card.")
            continue
        if card not in legal:
            print("Illegal play. You must follow suit if possible.")
            continue
        return card


def generate_data(path: str, games: int = 1000, seed: int = 1, teacher: str = "heuristic") -> None:
    rng = Random(seed)
    policy = make_ai(teacher)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game",
                "player",
                "trick",
                "team",
                "hand",
                "current_trick",
                "legal_actions",
                "action",
            ],
        )
        writer.writeheader()
        for game in range(games):
            state = new_round(seed=rng.randrange(10**9), leader=0)
            while not state.round_over:
                actor = state.current_player
                action = policy.choose_card(state.clone(), actor, rng)
                writer.writerow(
                    {
                        "game": game,
                        "player": actor,
                        "trick": state.trick_index,
                        "team": team_for(actor),
                        "hand": encode_cards(state.hands[actor]),
                        "current_trick": encode_trick(state.current_trick),
                        "legal_actions": encode_cards(state.legal_actions(actor)),
                        "action": str(action),
                    }
                )
                play_card(state, actor, action)


def compare(policy_name: str, games: int, seed: int, rollouts: int = 24, lambda_clubs: float = 2.5, lambda_hearts: float = 0.25, checkpoint_path: str | None = None) -> tuple[float, float]:
    policy = make_ai(policy_name, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts, checkpoint_path=checkpoint_path)
    players = [policy, RandomPlayer(), policy, RandomPlayer()]
    rng = Random(seed)
    wins = 0
    diff = 0
    for _ in range(games):
        state = play_round(players, seed=rng.randrange(10**9))
        wins += int(state.team_score(0) > state.team_score(1))
        diff += state.team_score(0) - state.team_score(1)
    return wins / games, diff / games


def run_matchup(
    team0: str,
    team1: str,
    games: int,
    seed: int,
    rollouts: int = 24,
    lambda_clubs: float = 2.5,
    lambda_hearts: float = 0.25,
    team0_checkpoint: str | None = None,
    team1_checkpoint: str | None = None,
) -> MatchupMetrics:
    rng = Random(seed)
    team0_jacks = {suit: 0 for suit in SUIT_NAMES}
    team1_jacks = {suit: 0 for suit in SUIT_NAMES}
    wins = 0
    diff = 0
    for _ in range(games):
        players = [
            make_ai(team0, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts, checkpoint_path=team0_checkpoint),
            make_ai(team1, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts, checkpoint_path=team1_checkpoint),
            make_ai(team0, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts, checkpoint_path=team0_checkpoint),
            make_ai(team1, rollouts=rollouts, lambda_clubs=lambda_clubs, lambda_hearts=lambda_hearts, checkpoint_path=team1_checkpoint),
        ]
        state = play_round(players, seed=rng.randrange(10**9))
        team0_score = state.team_score(0)
        team1_score = state.team_score(1)
        wins += int(team0_score > team1_score)
        diff += team0_score - team1_score
        for result in state.completed_tricks:
            for card in result.penalties:
                if team_for(result.winner) == 0:
                    team0_jacks[card.suit] += 1
                else:
                    team1_jacks[card.suit] += 1
    return MatchupMetrics(
        games=games,
        team0_win_rate=wins / games,
        team0_score_diff=diff / games,
        team0_jack_capture_rate={SUIT_NAMES[suit]: team0_jacks[suit] / games for suit in team0_jacks},
        team1_jack_capture_rate={SUIT_NAMES[suit]: team1_jacks[suit] / games for suit in team1_jacks},
    )


def sweep_search(games: int, seed: int, rollouts_values: list[int], club_values: list[float], heart_values: list[float]) -> list[tuple[float, float, int, float, float]]:
    rows = []
    for rollouts in rollouts_values:
        for lambda_clubs in club_values:
            for lambda_hearts in heart_values:
                metrics = run_matchup(
                    "search",
                    "heuristic",
                    games=games,
                    seed=seed,
                    rollouts=rollouts,
                    lambda_clubs=lambda_clubs,
                    lambda_hearts=lambda_hearts,
                )
                rows.append((metrics.team0_win_rate, metrics.team0_score_diff, rollouts, lambda_clubs, lambda_hearts))
    return sorted(rows, reverse=True)


def print_matchup(metrics: MatchupMetrics, team0: str, team1: str) -> None:
    print(f"games: {metrics.games}")
    print(f"team0_{team0}_win_rate_vs_team1_{team1}: {metrics.team0_win_rate:.3f}")
    print(f"team0_{team0}_score_diff_vs_team1_{team1}: {metrics.team0_score_diff:.3f}")
    for suit, rate in metrics.team0_jack_capture_rate.items():
        print(f"team0_{team0}_{suit}_jack_capture_rate: {rate:.3f}")
    for suit, rate in metrics.team1_jack_capture_rate.items():
        print(f"team1_{team1}_{suit}_jack_capture_rate: {rate:.3f}")


def encode_cards(cards: list[Card] | tuple[Card, ...]) -> str:
    ordered = sorted(cards, key=card_sort_key)
    return " ".join(str(card) for card in ordered)


def encode_trick(plays: list[tuple[int, Card]] | tuple[tuple[int, Card], ...]) -> str:
    return " ".join(f"P{player}:{card}" for player, card in plays)


def format_trick(plays: list[tuple[int, Card]] | tuple[tuple[int, Card], ...]) -> str:
    return encode_trick(plays) if plays else "(empty)"
