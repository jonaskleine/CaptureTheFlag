from __future__ import annotations

import argparse

from .agents import (
    AggressiveRuleBasedAgent,
    BalancedRuleBasedAgent,
    DefensiveRuleBasedAgent,
    RandomAgent,
)
from .agent_factory import build_team
from .core.engine import GameEngine
from .core.map_templates import MAPS, get_map
from .gui import SimulationConfig, SimulationWindow

AGENT_FACTORIES = {
    "random": lambda: RandomAgent(),
    "balanced": lambda: BalancedRuleBasedAgent(),
    "aggressive": lambda: AggressiveRuleBasedAgent(),
    "defensive": lambda: DefensiveRuleBasedAgent(),
    "rule": lambda: BalancedRuleBasedAgent(),
}

STRATEGY_CHOICES = ["balanced", "aggressive", "defensive", "trained"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Capture the Flag simulation.")
    parser.add_argument(
        "--map",
        default="open_field",
        choices=sorted(MAPS),
        help="Map to use",
    )
    parser.add_argument("--turns", type=int, default=50, help="Maximum number of turns")
    parser.add_argument(
        "--team0",
        default="balanced",
        choices=STRATEGY_CHOICES,
        help="Controller for team 0",
    )
    parser.add_argument(
        "--team0-support",
        default=None,
        choices=STRATEGY_CHOICES,
        help="Second agent on team 0",
    )
    parser.add_argument(
        "--team0-policy",
        default=None,
        help="Path to a saved Q-table for a trained team 0 agent",
    )
    parser.add_argument(
        "--team1",
        default="defensive",
        choices=STRATEGY_CHOICES,
        help="Controller for team 1",
    )
    parser.add_argument(
        "--team1-support",
        default=None,
        choices=STRATEGY_CHOICES,
        help="Second agent on team 1",
    )
    parser.add_argument(
        "--team1-policy",
        default=None,
        help="Path to a saved Q-table for a trained team 1 agent",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a popout window instead of printing ASCII frames",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=450,
        help="Animation delay in milliseconds for the GUI mode",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    map_template = get_map(args.map)

    if args.gui:
        window = SimulationWindow(
            SimulationConfig(
                map_template=map_template,
                team0_agent=args.team0,
                team0_support_agent=args.team0_support,
                team0_policy_path=args.team0_policy,
                team1_agent=args.team1,
                team1_support_agent=args.team1_support,
                team1_policy_path=args.team1_policy,
                max_turns=args.turns,
                step_delay_ms=args.delay_ms,
            )
        )
        window.run()
        return

    engine = GameEngine(map_template)
    engine.config.max_turns = args.turns
    team0 = build_team(
        args.team0,
        args.team0_support,
        primary_policy_path=args.team0_policy,
    )
    team1 = build_team(
        args.team1,
        args.team1_support,
        primary_policy_path=args.team1_policy,
    )

    print(f"Loaded map: {args.map}")
    print(engine.render_ascii())
    print()

    for _ in range(args.turns):
        state = engine.step((team0, team1))
        print(f"Turn {state.turn}")
        print(engine.render_ascii())
        if state.last_events:
            for event in state.last_events:
                print(f"- {event}")
        print()
        if state.winner is not None:
            break

    if engine.state.winner is None:
        print("Simulation ended without a winner.")
    else:
        print(f"Winner: Team {engine.state.winner}")


if __name__ == "__main__":
    main()
