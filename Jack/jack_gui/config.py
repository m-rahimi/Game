from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

POLICIES = [
    "heuristic", "search", "random", "bc", "dmc", "dmc_guarded",
    "search_value", "ismcts", "policy",
]

JACK5_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = JACK5_ROOT / "checkpoints/policy_distill800_l3.pt"
DEFAULT_VALUE = JACK5_ROOT / "checkpoints/value_search_ctx.pt"
DEFAULT_BELIEF = JACK5_ROOT / "checkpoints/belief_search40k_cal.pt"


@dataclass(frozen=True)
class GuiConfig:
    ai: str = "policy"
    human_seat: int = 0
    seed: int | None = None
    checkpoint: str | None = None
    policy_checkpoint: str | None = None
    value_checkpoint: str | None = None
    belief_checkpoint: str | None = None
    simulations: int = 400
    rollouts: int = 24
    lambda_clubs: float = 2.5
    lambda_hearts: float = 0.25
    time_limit: float | None = 2.0
    debug_ai: bool = False


def parse_args(argv: list[str] | None = None) -> GuiConfig:
    parser = argparse.ArgumentParser(description="Jack card game (Kivy GUI).")
    parser.add_argument("--ai", choices=POLICIES, default="policy")
    parser.add_argument("--player", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for bc/dmc/dmc_guarded/search_value.")
    parser.add_argument("--policy-checkpoint", default=None, help="Policy net for policy/ismcts.")
    parser.add_argument("--value-checkpoint", default=None, help="Value net for ismcts.")
    parser.add_argument("--belief-checkpoint", default=None, help="Belief net for ismcts.")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations for ismcts.")
    parser.add_argument("--rollouts", type=int, default=24, help="Rollouts for search/search_value.")
    parser.add_argument("--lambda-clubs", type=float, default=2.5)
    parser.add_argument("--lambda-hearts", type=float, default=0.25)
    parser.add_argument("--time-limit", type=float, default=2.0, help="Seconds per AI move (search/ismcts).")
    args = parser.parse_args(argv)

    policy = args.policy_checkpoint
    value = args.value_checkpoint
    belief = args.belief_checkpoint
    checkpoint = args.checkpoint

    if args.ai == "policy":
        policy = policy or str(DEFAULT_POLICY)
    elif args.ai == "ismcts":
        policy = policy or str(DEFAULT_POLICY if DEFAULT_POLICY.exists() else JACK5_ROOT / "checkpoints/policy_search100k.pt")
        value = value or str(DEFAULT_VALUE)
        belief = belief or str(DEFAULT_BELIEF)

    return GuiConfig(
        ai=args.ai,
        human_seat=args.player,
        seed=args.seed,
        checkpoint=checkpoint,
        policy_checkpoint=policy,
        value_checkpoint=value,
        belief_checkpoint=belief,
        simulations=args.simulations,
        rollouts=args.rollouts,
        lambda_clubs=args.lambda_clubs,
        lambda_hearts=args.lambda_hearts,
        time_limit=args.time_limit,
    )
