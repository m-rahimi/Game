from __future__ import annotations

import argparse

from .bc import train_bc, train_bc_online
from .dmc import train_dmc
from .value import train_value
from .policy import train_policy
from .cli import compare, generate_data, play_human_game, print_matchup, run_matchup, sweep_search
from .diagnostics import print_report

POLICIES = ["heuristic", "search", "random", "bc", "dmc", "dmc_guarded", "search_value", "ismcts", "policy"]


def float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item]


def int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Jack game tools.")
    subparsers = parser.add_subparsers(dest="command")

    diagnose = subparsers.add_parser("diagnose", help="Run heuristic Jack-danger diagnostics.")
    diagnose.add_argument("--games", type=int, default=1000)
    diagnose.add_argument("--seed", type=int, default=1)

    play = subparsers.add_parser("play", help="Play a match (first to 21) from the terminal.")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--ai", choices=POLICIES, default="heuristic")
    play.add_argument("--checkpoint", default=None, help="Checkpoint path when --ai bc/dmc/dmc_guarded is used.")
    play.add_argument("--policy-checkpoint", default=None, help="Policy net when --ai ismcts is used.")
    play.add_argument("--value-checkpoint", default=None, help="Value net when --ai ismcts is used.")
    play.add_argument("--belief-checkpoint", default=None, help="Belief net for weighted determinization (ismcts).")
    play.add_argument("--simulations", type=int, default=400, help="MCTS simulations when --ai ismcts is used.")
    play.add_argument("--player", type=int, choices=[0, 1, 2, 3], default=0)
    play.add_argument("--log", default=None, help="JSON path to log the game for blunder review (analyze_blunders.py).")

    data = subparsers.add_parser("generate-data", help="Generate teacher state/action CSV data.")
    data.add_argument("--out", required=True)
    data.add_argument("--games", type=int, default=1000)
    data.add_argument("--seed", type=int, default=1)
    data.add_argument("--teacher", choices=["heuristic", "search", "random"], default="heuristic")

    train = subparsers.add_parser("train-bc", help="Train a behavior-cloned policy from teacher CSV data.")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--device", default=None, help="cpu, cuda, or omit for auto.")

    online = subparsers.add_parser("train-bc-online", help="Train BC while generating fresh teacher games each iteration.")
    online.add_argument("--out", required=True)
    online.add_argument("--iterations", type=int, default=5)
    online.add_argument("--games-per-iter", type=int, default=100)
    online.add_argument("--epochs-per-iter", type=int, default=1)
    online.add_argument("--batch-size", type=int, default=512)
    online.add_argument("--lr", type=float, default=1e-3)
    online.add_argument("--hidden-dim", type=int, default=256)
    online.add_argument("--val-fraction", type=float, default=0.1)
    online.add_argument("--seed", type=int, default=1)
    online.add_argument("--device", default=None, help="cpu, cuda, or omit for auto.")
    online.add_argument("--teacher-team0", choices=["heuristic", "search", "random"], default="search")
    online.add_argument("--teacher-team1", choices=["heuristic", "search", "random"], default="heuristic")
    online.add_argument("--focus-team", choices=["0", "1", "both"], default="0")
    online.add_argument("--rollouts", type=int, default=24)
    online.add_argument("--lambda-clubs", type=float, default=2.5)
    online.add_argument("--lambda-hearts", type=float, default=0.25)
    online.add_argument("--replay-size", type=int, default=50000)
    online.add_argument("--danger-multiplier", type=int, default=4)
    online.add_argument("--save-every", type=int, default=1)

    dmc = subparsers.add_parser("train-dmc", help="Train a Deep Monte Carlo Q policy from full-round returns.")
    dmc.add_argument("--out", required=True)
    dmc.add_argument("--init-checkpoint", default=None, help="Optional DMC checkpoint to initialize from.")
    dmc.add_argument("--iterations", type=int, default=100)
    dmc.add_argument("--games-per-iter", type=int, default=200)
    dmc.add_argument("--epochs-per-iter", type=int, default=1)
    dmc.add_argument("--batch-size", type=int, default=512)
    dmc.add_argument("--lr", type=float, default=1e-3)
    dmc.add_argument("--hidden-dim", type=int, default=256)
    dmc.add_argument("--replay-size", type=int, default=200000)
    dmc.add_argument("--seed", type=int, default=1)
    dmc.add_argument("--device", default=None, help="cpu, cuda, or omit for auto.")
    dmc.add_argument("--opponent", choices=["random", "heuristic", "search"], default="random")
    dmc.add_argument("--focus-team", choices=["0", "1", "both"], default="0")
    dmc.add_argument("--epsilon-start", type=float, default=1.0)
    dmc.add_argument("--epsilon-end", type=float, default=0.05)
    dmc.add_argument("--save-every", type=int, default=10)
    dmc.add_argument("--rollouts", type=int, default=24)
    dmc.add_argument("--lambda-clubs", type=float, default=2.5)
    dmc.add_argument("--lambda-hearts", type=float, default=0.25)
    dmc.add_argument("--shape-clubs", type=float, default=4.0)
    dmc.add_argument("--shape-hearts", type=float, default=2.0)
    dmc.add_argument("--shape-spades", type=float, default=1.0)
    dmc.add_argument("--shape-diamonds", type=float, default=0.0)

    value = subparsers.add_parser("train-value", help="Train a state-value network for net-guided search.")
    value.add_argument("--out", required=True)
    value.add_argument("--games", type=int, default=2000)
    value.add_argument("--epochs", type=int, default=10)
    value.add_argument("--batch-size", type=int, default=512)
    value.add_argument("--lr", type=float, default=1e-3)
    value.add_argument("--hidden-dim", type=int, default=256)
    value.add_argument("--teacher", choices=["heuristic", "search", "random", "dmc", "dmc_guarded"], default="heuristic")
    value.add_argument("--teacher-checkpoint", default=None, help="checkpoint for dmc/dmc_guarded teacher")
    value.add_argument("--rollouts", type=int, default=24, help="search teacher rollouts (if --teacher search)")
    value.add_argument("--seed", type=int, default=1)
    value.add_argument("--device", default=None, help="cpu, cuda, or omit for auto.")

    policy = subparsers.add_parser("train-policy", help="Train a policy-prior network for net-guided search.")
    policy.add_argument("--out", required=True)
    policy.add_argument("--games", type=int, default=2000)
    policy.add_argument("--epochs", type=int, default=10)
    policy.add_argument("--batch-size", type=int, default=512)
    policy.add_argument("--lr", type=float, default=1e-3)
    policy.add_argument("--hidden-dim", type=int, default=256)
    policy.add_argument("--teacher", choices=["heuristic", "search", "random", "dmc", "dmc_guarded"], default="heuristic")
    policy.add_argument("--teacher-checkpoint", default=None, help="checkpoint for dmc/dmc_guarded teacher")
    policy.add_argument("--rollouts", type=int, default=24, help="search teacher rollouts (if --teacher search)")
    policy.add_argument("--seed", type=int, default=1)
    policy.add_argument("--device", default=None, help="cpu, cuda, or omit for auto.")

    versus = subparsers.add_parser("compare", help="Compare a team policy against random opponents.")
    versus.add_argument("--policy", choices=POLICIES, default="heuristic")
    versus.add_argument("--checkpoint", default=None, help="Checkpoint path when --policy bc is used.")
    versus.add_argument("--games", type=int, default=1000)
    versus.add_argument("--seed", type=int, default=1)
    versus.add_argument("--rollouts", type=int, default=24)
    versus.add_argument("--lambda-clubs", type=float, default=2.5)
    versus.add_argument("--lambda-hearts", type=float, default=0.25)

    matchup = subparsers.add_parser("matchup", help="Run Team 0 policy against Team 1 policy.")
    matchup.add_argument("--team0", choices=POLICIES, default="search")
    matchup.add_argument("--team1", choices=POLICIES, default="heuristic")
    matchup.add_argument("--team0-checkpoint", default=None, help="Checkpoint path when --team0 bc is used.")
    matchup.add_argument("--team1-checkpoint", default=None, help="Checkpoint path when --team1 bc is used.")
    matchup.add_argument("--games", type=int, default=100)
    matchup.add_argument("--seed", type=int, default=1)
    matchup.add_argument("--rollouts", type=int, default=24)
    matchup.add_argument("--lambda-clubs", type=float, default=2.5)
    matchup.add_argument("--lambda-hearts", type=float, default=0.25)

    sweep = subparsers.add_parser("sweep-search", help="Sweep search parameters against heuristic.")
    sweep.add_argument("--games", type=int, default=50)
    sweep.add_argument("--seed", type=int, default=1)
    sweep.add_argument("--rollouts", default="12,24,48")
    sweep.add_argument("--lambda-clubs", default="0.5,1.0,1.5")
    sweep.add_argument("--lambda-hearts", default="0.25,0.5,1.0")

    args = parser.parse_args()
    if args.command in (None, "diagnose"):
        print_report(games=args.games, seed=args.seed)
    elif args.command == "play":
        play_human_game(seed=args.seed, ai=args.ai, human_player=args.player, checkpoint_path=args.checkpoint,
                        policy_checkpoint=args.policy_checkpoint, value_checkpoint=args.value_checkpoint,
                        simulations=args.simulations, log_path=args.log, belief_checkpoint=args.belief_checkpoint)
    elif args.command == "generate-data":
        generate_data(args.out, games=args.games, seed=args.seed, teacher=args.teacher)
        print(f"Wrote {args.out}")
    elif args.command == "train-bc":
        result = train_bc(
            data_path=args.data,
            output_path=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
        )
        print(f"train_examples: {result.train_examples}")
        print(f"val_examples: {result.val_examples}")
        print(f"best_val_accuracy: {result.best_val_accuracy:.4f}")
        print(f"wrote: {result.output_path}")
    elif args.command == "train-bc-online":
        focus_team = None if args.focus_team == "both" else int(args.focus_team)
        result = train_bc_online(
            output_path=args.out,
            iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            epochs_per_iter=args.epochs_per_iter,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            teacher_team0=args.teacher_team0,
            teacher_team1=args.teacher_team1,
            focus_team=focus_team,
            rollouts=args.rollouts,
            lambda_clubs=args.lambda_clubs,
            lambda_hearts=args.lambda_hearts,
            replay_size=args.replay_size,
            danger_multiplier=args.danger_multiplier,
            save_every=args.save_every,
        )
        print(f"train_examples: {result.train_examples}")
        print(f"val_examples: {result.val_examples}")
        print(f"best_val_accuracy: {result.best_val_accuracy:.4f}")
        print(f"wrote: {result.output_path}")
    elif args.command == "train-dmc":
        focus_team = None if args.focus_team == "both" else int(args.focus_team)
        result = train_dmc(
            output_path=args.out,
            iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            epochs_per_iter=args.epochs_per_iter,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            replay_size=args.replay_size,
            seed=args.seed,
            device=args.device,
            opponent=args.opponent,
            focus_team=focus_team,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            save_every=args.save_every,
            rollouts=args.rollouts,
            lambda_clubs=args.lambda_clubs,
            lambda_hearts=args.lambda_hearts,
            shape_clubs=args.shape_clubs,
            shape_hearts=args.shape_hearts,
            shape_spades=args.shape_spades,
            shape_diamonds=args.shape_diamonds,
            init_checkpoint=args.init_checkpoint,
        )
        print(f"train_examples: {result.train_examples}")
        print(f"best_loss: {result.best_loss:.4f}")
        print(f"wrote: {result.output_path}")
    elif args.command == "train-value":
        result = train_value(
            output_path=args.out,
            games=args.games,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            teacher=args.teacher,
            rollouts=args.rollouts,
            teacher_checkpoint=args.teacher_checkpoint,
            seed=args.seed,
            device=args.device,
        )
        print(f"train_examples: {result.train_examples}")
        print(f"best_loss: {result.best_loss:.4f}")
        print(f"wrote: {result.output_path}")
    elif args.command == "train-policy":
        result = train_policy(
            output_path=args.out,
            games=args.games,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            teacher=args.teacher,
            rollouts=args.rollouts,
            teacher_checkpoint=args.teacher_checkpoint,
            seed=args.seed,
            device=args.device,
        )
        print(f"train_examples: {result.train_examples}")
        print(f"best_loss: {result.best_loss:.4f}")
        print(f"wrote: {result.output_path}")
    elif args.command == "compare":
        win_rate, score_diff = compare(
            args.policy,
            args.games,
            args.seed,
            rollouts=args.rollouts,
            lambda_clubs=args.lambda_clubs,
            lambda_hearts=args.lambda_hearts,
            checkpoint_path=args.checkpoint,
        )
        print(f"{args.policy}_team_win_rate_vs_random: {win_rate:.3f}")
        print(f"{args.policy}_team_score_diff_vs_random: {score_diff:.3f}")
    elif args.command == "matchup":
        metrics = run_matchup(
            args.team0,
            args.team1,
            args.games,
            args.seed,
            rollouts=args.rollouts,
            lambda_clubs=args.lambda_clubs,
            lambda_hearts=args.lambda_hearts,
            team0_checkpoint=args.team0_checkpoint,
            team1_checkpoint=args.team1_checkpoint,
        )
        print_matchup(metrics, args.team0, args.team1)
    elif args.command == "sweep-search":
        rows = sweep_search(
            games=args.games,
            seed=args.seed,
            rollouts_values=int_list(args.rollouts),
            club_values=float_list(args.lambda_clubs),
            heart_values=float_list(args.lambda_hearts),
        )
        print("win_rate score_diff rollouts lambda_clubs lambda_hearts")
        for win_rate, score_diff, rollouts, lambda_clubs, lambda_hearts in rows:
            print(f"{win_rate:.3f} {score_diff:.3f} {rollouts} {lambda_clubs:.2f} {lambda_hearts:.2f}")


if __name__ == "__main__":
    main()
