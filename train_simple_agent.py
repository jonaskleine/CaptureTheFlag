from __future__ import annotations

import argparse
from pathlib import Path

from capture_the_flag.training.simple_q_learning import (
    evaluate_policy,
    load_policy,
    save_policy,
    train_simple_agent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simple tabular Q-learning agent against the rule-based teams."
    )
    parser.add_argument("--episodes", type=int, default=1500, help="Training episodes")
    parser.add_argument("--map", default="open_field", help="Map to train on")
    parser.add_argument(
        "--max-turns", type=int, default=120, help="Maximum turns per episode"
    )
    parser.add_argument(
        "--teammate",
        default="balanced",
        choices=["balanced", "aggressive", "defensive"],
        help="Fixed teammate used alongside the learning agent",
    )
    parser.add_argument(
        "--opponent0",
        default="aggressive",
        choices=["balanced", "aggressive", "defensive"],
        help="First opposing agent",
    )
    parser.add_argument(
        "--opponent1",
        default="defensive",
        choices=["balanced", "aggressive", "defensive"],
        help="Second opposing agent",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for exploration"
    )
    parser.add_argument(
        "--bootstrap-episodes",
        type=int,
        default=300,
        help="Teacher rollout episodes used to warm-start the Q-table",
    )
    parser.add_argument(
        "--bootstrap-teacher",
        default="balanced",
        choices=["balanced", "aggressive", "defensive"],
        help="Rule-based teacher used during warm-start rollouts",
    )
    parser.add_argument(
        "--save",
        default="q_table.json",
        help="Where to save the learned Q-table",
    )
    parser.add_argument(
        "--evaluate-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes after training",
    )
    parser.add_argument(
        "--load",
        default="",
        help="Load an existing Q-table before training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.load:
        policy = load_policy(args.load)
    else:
        policy = None

    result = train_simple_agent(
        episodes=args.episodes,
        map_name=args.map,
        max_turns=args.max_turns,
        teammate=args.teammate,
        opponents=(args.opponent0, args.opponent1),
        seed=args.seed,
        bootstrap_episodes=args.bootstrap_episodes,
        bootstrap_teacher=args.bootstrap_teacher,
        policy=policy,
    )
    save_policy(result.policy, args.save)
    stats = evaluate_policy(
        result.policy,
        episodes=args.evaluate_episodes,
        map_name=args.map,
        max_turns=args.max_turns,
        teammate=args.teammate,
        opponents=(args.opponent0, args.opponent1),
    )

    print(f"Saved policy to {Path(args.save).resolve()}")
    print(f"Final epsilon: {result.policy.epsilon:.3f}")
    print(
        "Evaluation: "
        f"win_rate={stats['win_rate']:.3f}, "
        f"avg_team_score={stats['avg_team_score']:.2f}, "
        f"avg_opponent_score={stats['avg_opponent_score']:.2f}"
    )


if __name__ == "__main__":
    main()
