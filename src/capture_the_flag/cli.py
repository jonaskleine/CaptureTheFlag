from __future__ import annotations

import argparse

from .agents import RandomAgent, RuleBasedAgent
from .core.engine import GameEngine
from .core.map_templates import MAPS, get_map
from .gui import SimulationConfig, SimulationWindow

AGENT_FACTORIES = {
    "random": lambda: RandomAgent(),
    "rule": lambda: RuleBasedAgent(),
}


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
        default="rule",
        choices=sorted(AGENT_FACTORIES),
        help="Controller for team 0",
    )
    parser.add_argument(
        "--team1",
        default="random",
        choices=sorted(AGENT_FACTORIES),
        help="Controller for team 1",
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
                team1_agent=args.team1,
                max_turns=args.turns,
                step_delay_ms=args.delay_ms,
            )
        )
        window.run()
        return

    engine = GameEngine(map_template)
    engine.config.max_turns = args.turns
    team0 = [AGENT_FACTORIES[args.team0](), AGENT_FACTORIES[args.team0]()]
    team1 = [AGENT_FACTORIES[args.team1](), AGENT_FACTORIES[args.team1]()]

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
