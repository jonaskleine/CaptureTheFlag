from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..agents import (
    AggressiveRuleBasedAgent,
    BalancedRuleBasedAgent,
    DefensiveRuleBasedAgent,
    RandomAgent,
    QLearningAgent,
    TabularQLearningPolicy,
)
from ..agents.q_learning import encode_state
from ..core.engine import GameEngine
from ..core.map_templates import get_map
from ..core.models import ACTION_DELTAS, Action, GameState, Position

AgentFactory = Callable[[], object]

AGENT_FACTORIES: dict[str, AgentFactory] = {
    "random": lambda: RandomAgent(),
    "balanced": lambda: BalancedRuleBasedAgent(),
    "aggressive": lambda: AggressiveRuleBasedAgent(),
    "defensive": lambda: DefensiveRuleBasedAgent(),
}

CURRICULUM: tuple[tuple[float, str, tuple[str, str]], ...] = (
    (0.25, "open_field", ("random", "random")),
    (0.55, "open_field", ("balanced", "balanced")),
    (0.80, "lanes", ("aggressive", "balanced")),
    (1.00, "central_wall", ("aggressive", "defensive")),
)


@dataclass(slots=True)
class TrainingResult:
    policy: TabularQLearningPolicy
    episode_rewards: list[float]
    episode_scores: list[tuple[int, int]]


def _factory(name: str) -> AgentFactory:
    try:
        return AGENT_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown agent '{name}'. Available agents: {', '.join(sorted(AGENT_FACTORIES))}"
        ) from exc


def _episode_plan(
    episode_index: int,
    total_episodes: int,
    requested_map: str,
    requested_opponents: tuple[str, str],
) -> tuple[str, tuple[str, str]]:
    if total_episodes <= 1:
        return requested_map, requested_opponents

    progress = (episode_index + 1) / total_episodes
    for cutoff, map_name, opponents in CURRICULUM:
        if progress <= cutoff:
            if cutoff < 1.0:
                return map_name, opponents
            return requested_map, requested_opponents
    return requested_map, requested_opponents


def _action_from_positions(previous: Position, current: Position) -> Action | None:
    delta = (current.x - previous.x, current.y - previous.y)
    for action, action_delta in ACTION_DELTAS.items():
        if action_delta == delta:
            return action
    return None


def _bootstrap_policy_from_teacher(
    policy: TabularQLearningPolicy,
    episodes: int,
    map_name: str,
    max_turns: int,
    teammate: str,
    opponents: tuple[str, str],
    teacher_agent: str,
) -> None:
    if episodes <= 0:
        return

    for episode_index in range(episodes):
        episode_map_name, episode_opponents = _episode_plan(
            episode_index,
            episodes,
            map_name,
            opponents,
        )
        engine = GameEngine(get_map(episode_map_name))
        engine.config.max_turns = max_turns
        teacher = _factory(teacher_agent)()
        team0 = [teacher, _factory(teammate)()]
        team1 = [_factory(episode_opponents[0])(), _factory(episode_opponents[1])()]

        while engine.state.turn < max_turns:
            previous_state = engine.state.clone_shallow()
            previous_position = previous_state.agents["t0_a0"].position
            engine.step((team0, team1))
            current_position = engine.state.agents["t0_a0"].position
            action = _action_from_positions(previous_position, current_position)
            if action is None:
                if engine.state.turn >= max_turns:
                    break
                continue
            state_key = encode_state(previous_state, 0, "t0_a0")
            next_state_key = encode_state(engine.state, 0, "t0_a0")
            reward = compute_reward(previous_state, engine.state, 0, "t0_a0")
            policy.update(
                state_key,
                action,
                reward,
                next_state_key,
                engine.state.turn >= max_turns,
            )
            if engine.state.turn >= max_turns:
                break


def _distance_reward(
    previous_state: GameState,
    current_state: GameState,
    team_id: int,
    agent_id: str,
) -> float:
    previous_agent = previous_state.agents[agent_id]
    current_agent = current_state.agents[agent_id]
    if current_agent.captured:
        return -0.1

    reward = 0.0
    enemy_team = 1 - team_id

    enemy_flag = current_state.teams[enemy_team].flag_position
    home_base = current_state.teams[team_id].flag_spawn

    if current_agent.carrying_flag_team == enemy_team:
        previous_distance = previous_agent.position.manhattan_distance(home_base)
        current_distance = current_agent.position.manhattan_distance(home_base)
        reward += 0.45 * (previous_distance - current_distance)
    elif current_state.flag_carriers[enemy_team] is None:
        previous_distance = previous_agent.position.manhattan_distance(enemy_flag)
        current_distance = current_agent.position.manhattan_distance(enemy_flag)
        reward += 0.18 * (previous_distance - current_distance)

    teammate_id = next(
        candidate
        for candidate in current_state.team_agent_ids[team_id]
        if candidate != agent_id
    )
    teammate = current_state.agents[teammate_id]
    teammate_previous = previous_state.agents[teammate_id]

    if teammate.carrying_flag_team == enemy_team:
        previous_distance = previous_agent.position.manhattan_distance(
            teammate_previous.position
        )
        current_distance = current_agent.position.manhattan_distance(teammate.position)
        reward += 0.08 * (previous_distance - current_distance)
        if current_agent.position.manhattan_distance(teammate.position) == 1:
            reward += 0.08

    enemy_carrier_id = current_state.flag_carriers[team_id]
    if enemy_carrier_id is not None:
        enemy_carrier = current_state.agents[enemy_carrier_id]
        previous_distance = previous_agent.position.manhattan_distance(
            previous_state.agents[enemy_carrier_id].position
        )
        current_distance = current_agent.position.manhattan_distance(
            enemy_carrier.position
        )
        reward += 0.1 * (previous_distance - current_distance)
        if current_agent.position.manhattan_distance(enemy_carrier.position) == 1:
            reward += 0.08

    if not current_state.map_template.is_territory(team_id, current_agent.position):
        reward += 0.03
    if current_agent.carrying_flag_team is not None:
        reward += 0.02

    return reward


def compute_reward(
    previous_state: GameState,
    current_state: GameState,
    team_id: int,
    agent_id: str,
) -> float:
    previous_agent = previous_state.agents[agent_id]
    current_agent = current_state.agents[agent_id]
    reward = -0.01

    reward += 100.0 * (
        current_state.teams[team_id].score - previous_state.teams[team_id].score
    )
    reward -= 100.0 * (
        current_state.teams[1 - team_id].score - previous_state.teams[1 - team_id].score
    )

    if current_agent.captured and not previous_agent.captured:
        reward -= 8.0

    if (
        current_state.flag_carriers[1 - team_id] == agent_id
        and previous_state.flag_carriers[1 - team_id] != agent_id
    ):
        reward += 6.0

    if (
        previous_state.flag_carriers[team_id] is None
        and current_state.flag_carriers[team_id] is not None
    ):
        reward += 1.0

    if (
        previous_state.flag_carriers[1 - team_id] is None
        and current_state.flag_carriers[1 - team_id] is not None
    ):
        reward += 1.5

    if (
        current_state.flag_carriers[1 - team_id] == agent_id
        and not previous_agent.captured
    ):
        reward += 0.5

    reward += _distance_reward(previous_state, current_state, team_id, agent_id)
    return reward


def train_simple_agent(
    episodes: int = 1500,
    map_name: str = "open_field",
    max_turns: int = 120,
    teammate: str = "balanced",
    opponents: tuple[str, str] = ("aggressive", "defensive"),
    seed: int | None = 7,
    bootstrap_episodes: int | None = None,
    bootstrap_teacher: str = "balanced",
    policy: TabularQLearningPolicy | None = None,
) -> TrainingResult:
    policy = policy or TabularQLearningPolicy(seed=seed)
    warmup_episodes = bootstrap_episodes
    if warmup_episodes is None:
        warmup_episodes = max(100, episodes // 4)

    _bootstrap_policy_from_teacher(
        policy=policy,
        episodes=warmup_episodes,
        map_name=map_name,
        max_turns=max_turns,
        teammate=teammate,
        opponents=opponents,
        teacher_agent=bootstrap_teacher,
    )

    episode_rewards: list[float] = []
    episode_scores: list[tuple[int, int]] = []

    for episode_index in range(episodes):
        episode_map_name, episode_opponents = _episode_plan(
            episode_index,
            episodes,
            map_name,
            opponents,
        )
        engine = GameEngine(get_map(episode_map_name))
        engine.config.max_turns = max_turns
        engine.reset()
        engine.config.max_turns = max_turns
        learner = QLearningAgent(policy=policy, explore=True)
        team0 = [learner, _factory(teammate)()]
        team1 = [_factory(episode_opponents[0])(), _factory(episode_opponents[1])()]

        total_reward = 0.0
        while engine.state.turn < max_turns:
            previous_state = engine.state.clone_shallow()
            engine.step((team0, team1))
            reward = compute_reward(previous_state, engine.state, 0, "t0_a0")
            total_reward += reward
            if learner.last_state_key is not None and learner.last_action is not None:
                done = engine.state.turn >= max_turns
                next_state_key = encode_state(engine.state, 0, "t0_a0")
                policy.update(
                    learner.last_state_key,
                    learner.last_action,
                    reward,
                    next_state_key,
                    done,
                )
            if engine.state.turn >= max_turns:
                break

        policy.decay()
        episode_rewards.append(total_reward)
        episode_scores.append(
            (engine.state.teams[0].score, engine.state.teams[1].score)
        )

    return TrainingResult(
        policy=policy,
        episode_rewards=episode_rewards,
        episode_scores=episode_scores,
    )


def evaluate_policy(
    policy: TabularQLearningPolicy,
    episodes: int = 100,
    map_name: str = "open_field",
    max_turns: int = 120,
    teammate: str = "balanced",
    opponents: tuple[str, str] = ("aggressive", "defensive"),
    seed: int | None = 21,
) -> dict[str, float]:
    wins = 0
    total_scores = 0.0
    total_opponent_scores = 0.0

    for _ in range(episodes):
        engine = GameEngine(get_map(map_name))
        engine.config.max_turns = max_turns
        engine.reset()
        engine.config.max_turns = max_turns
        team0 = [QLearningAgent(policy=policy, explore=False), _factory(teammate)()]
        team1 = [_factory(opponents[0])(), _factory(opponents[1])()]

        while engine.state.turn < max_turns:
            engine.step((team0, team1))
            if engine.state.turn >= max_turns:
                break

        total_scores += engine.state.teams[0].score
        total_opponent_scores += engine.state.teams[1].score
        if engine.state.teams[0].score > engine.state.teams[1].score:
            wins += 1

    return {
        "win_rate": wins / episodes,
        "avg_team_score": total_scores / episodes,
        "avg_opponent_score": total_opponent_scores / episodes,
    }


def save_policy(policy: TabularQLearningPolicy, path: str | Path) -> None:
    policy.save(path)


def load_policy(path: str | Path) -> TabularQLearningPolicy:
    return TabularQLearningPolicy.load(path)
