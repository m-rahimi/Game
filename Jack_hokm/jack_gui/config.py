from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

POLICIES = [
    "heuristic", "search", "random", "bc", "dmc", "dmc_guarded",
    "search_value", "ismcts", "policy",
]

JACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = JACK_ROOT / "checkpoints/student_ens_l3.pt"
DEFAULT_HOKM_POLICY = JACK_ROOT / "checkpoints/student_hokm_iimc_sT035.pt"
DEFAULT_VALUE = JACK_ROOT / "checkpoints/value_search_ctx.pt"
DEFAULT_BELIEF = JACK_ROOT / "checkpoints/belief_search40k_cal.pt"


@dataclass(frozen=True)
class GuiConfig:
    ai: str = "policy"
    human_seat: int = 0
    seed: int | None = None
    checkpoint: str | None = None
    policy_checkpoint: str | None = None
    value_checkpoint: str | None = None
    belief_checkpoint: str | None = None
    hokm_policy_checkpoint: str | None = None
    hokm_value_checkpoint: str | None = None
    simulations: int = 400
    rollouts: int = 24
    lambda_clubs: float = 2.5
    lambda_hearts: float = 0.25
    time_limit: float | None = 2.0
    debug_ai: bool = False
    log_path: str | None = None
    hokm: bool | None = None
    ask_hokm: bool = True


def effective_policy_checkpoint(config: GuiConfig, hokm: bool) -> str | None:
    if hokm and config.hokm_policy_checkpoint:
        return config.hokm_policy_checkpoint
    return config.policy_checkpoint


def effective_value_checkpoint(config: GuiConfig, hokm: bool) -> str | None:
    if hokm and config.hokm_value_checkpoint:
        return config.hokm_value_checkpoint
    return config.value_checkpoint


def with_hokm(config: GuiConfig, hokm: bool) -> GuiConfig:
    return replace(config, hokm=hokm)


def parse_args(argv: list[str] | None = None) -> GuiConfig:
    parser = argparse.ArgumentParser(description="Jack card game (Kivy GUI).")
    parser.add_argument("--ai", choices=POLICIES, default="policy")
    parser.add_argument("--player", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for bc/dmc/dmc_guarded/search_value.")
    parser.add_argument("--policy-checkpoint", default=None, help="Policy net for classic (no-trump) play.")
    parser.add_argument("--value-checkpoint", default=None, help="Value net for ismcts.")
    parser.add_argument("--belief-checkpoint", default=None, help="Belief net for ismcts.")
    parser.add_argument("--hokm-policy-checkpoint", default=None, help="Policy net when Hokm is selected.")
    parser.add_argument("--hokm-value-checkpoint", default=None, help="Value net for Hokm ismcts.")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations for ismcts.")
    parser.add_argument("--rollouts", type=int, default=24, help="Rollouts for search/search_value.")
    parser.add_argument("--lambda-clubs", type=float, default=2.5)
    parser.add_argument("--lambda-hearts", type=float, default=0.25)
    parser.add_argument("--time-limit", type=float, default=2.0, help="Seconds per AI move (search/ismcts).")
    parser.add_argument("--hokm", dest="hokm", action="store_true", default=None,
                        help="Play the Hokm (trump) variant.")
    parser.add_argument("--no-hokm", dest="hokm", action="store_false",
                        help="Play the classic no-trump game.")
    parser.add_argument("--ask-hokm", action="store_true", default=False,
                        help="Show the Hokm yes/no dialog at startup (default in play_gui.sh).")
    parser.add_argument(
        "--log",
        default=None,
        help="JSON path to log rounds for blunder review.",
    )
    args = parser.parse_args(argv)

    policy = args.policy_checkpoint
    value = args.value_checkpoint
    belief = args.belief_checkpoint
    checkpoint = args.checkpoint
    hokm_policy = args.hokm_policy_checkpoint
    hokm_value = args.hokm_value_checkpoint

    if args.ai == "policy":
        policy = policy or str(DEFAULT_POLICY)
        hokm_policy = hokm_policy or str(DEFAULT_HOKM_POLICY)
    elif args.ai == "ismcts":
        policy = policy or str(DEFAULT_POLICY)
        hokm_policy = hokm_policy or str(DEFAULT_HOKM_POLICY)
        value = value or str(DEFAULT_VALUE)
        hokm_value = hokm_value or value
        belief = belief or str(DEFAULT_BELIEF)

    ask_hokm = args.ask_hokm or args.hokm is None

    return GuiConfig(
        ai=args.ai,
        human_seat=args.player,
        seed=args.seed,
        checkpoint=checkpoint,
        policy_checkpoint=policy,
        value_checkpoint=value,
        belief_checkpoint=belief,
        hokm_policy_checkpoint=hokm_policy,
        hokm_value_checkpoint=hokm_value,
        simulations=args.simulations,
        rollouts=args.rollouts,
        lambda_clubs=args.lambda_clubs,
        lambda_hearts=args.lambda_hearts,
        time_limit=args.time_limit,
        log_path=args.log,
        hokm=args.hokm,
        ask_hokm=ask_hokm,
    )
