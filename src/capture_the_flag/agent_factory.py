from __future__ import annotations

from pathlib import Path

from .agents import (
    AggressiveRuleBasedAgent,
    BalancedRuleBasedAgent,
    DefensiveRuleBasedAgent,
    QLearningAgent,
    RandomAgent,
    TabularQLearningPolicy,
)

AGENT_NAMES = {"random", "balanced", "aggressive", "defensive", "trained"}


def _build_rule_based_agent(name: str):
    if name == "random":
        return RandomAgent()
    if name == "balanced":
        return BalancedRuleBasedAgent()
    if name == "aggressive":
        return AggressiveRuleBasedAgent()
    if name == "defensive":
        return DefensiveRuleBasedAgent()
    raise KeyError(
        f"Unknown agent '{name}'. Available agents: balanced, defensive, aggressive, random, trained"
    )


def build_agent(
    name: str,
    policy_path: str | Path | None = None,
    *,
    explore: bool = False,
):
    if name == "trained":
        if policy_path is None:
            raise ValueError("A policy path is required when using the trained agent.")
        policy = TabularQLearningPolicy.load(policy_path)
        return QLearningAgent(policy=policy, explore=explore)
    return _build_rule_based_agent(name)


def build_team(
    primary_agent: str,
    support_agent: str | None = None,
    *,
    primary_policy_path: str | Path | None = None,
    support_policy_path: str | Path | None = None,
    explore: bool = False,
) -> list[object]:
    if primary_agent not in AGENT_NAMES:
        raise KeyError(
            f"Unknown agent '{primary_agent}'. Available agents: {', '.join(sorted(AGENT_NAMES))}"
        )

    resolved_support = support_agent
    if resolved_support is None:
        resolved_support = "balanced" if primary_agent == "trained" else primary_agent
    if resolved_support not in AGENT_NAMES:
        raise KeyError(
            f"Unknown agent '{resolved_support}'. Available agents: {', '.join(sorted(AGENT_NAMES))}"
        )

    resolved_support_policy = support_policy_path or primary_policy_path
    return [
        build_agent(primary_agent, primary_policy_path, explore=explore),
        build_agent(
            resolved_support,
            resolved_support_policy if resolved_support == "trained" else None,
            explore=explore,
        ),
    ]
