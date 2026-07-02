from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..core.models import Action, GameState, Position
from .base import BaseAgent

TRAINING_ACTIONS: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
)


def _clamp(value: int, limit: int = 4) -> int:
    return max(-limit, min(limit, value))


def _bucket_distance(value: int, limit: int = 6) -> int:
    return max(0, min(limit, value))


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _relative_features(
    origin: Position, target: Position | None
) -> tuple[int, int, int]:
    if target is None:
        return (0, 0, 0)
    return (
        _sign(target.x - origin.x),
        _sign(target.y - origin.y),
        _bucket_distance(origin.manhattan_distance(target)),
    )


def _nearest_enemy(state: GameState, team_id: int, agent_id: str) -> Position | None:
    agent = state.agents[agent_id]
    enemies = [enemy for enemy in state.agents.values() if enemy.team_id != team_id]
    if not enemies:
        return None
    nearest = min(
        enemies,
        key=lambda enemy: agent.position.manhattan_distance(enemy.position),
    )
    return nearest.position


def encode_state(state: GameState, team_id: int, agent_id: str) -> tuple[int, ...]:
    agent = state.agents[agent_id]
    teammate_id = next(
        candidate
        for candidate in state.team_agent_ids[team_id]
        if candidate != agent_id
    )
    teammate = state.agents[teammate_id]
    own_team = state.teams[team_id]
    enemy_team = state.teams[1 - team_id]
    nearest_enemy = _nearest_enemy(state, team_id, agent_id)

    own_base_dx, own_base_dy, own_base_distance = _relative_features(
        agent.position, own_team.flag_spawn
    )
    own_flag_dx, own_flag_dy, own_flag_distance = _relative_features(
        agent.position, own_team.flag_position
    )
    enemy_flag_dx, enemy_flag_dy, enemy_flag_distance = _relative_features(
        agent.position, enemy_team.flag_position
    )
    teammate_dx, teammate_dy, teammate_distance = _relative_features(
        agent.position, teammate.position
    )
    enemy_dx, enemy_dy, enemy_distance = _relative_features(
        agent.position, nearest_enemy
    )
    teammate_target_dx, teammate_target_dy, teammate_target_distance = (
        _relative_features(agent.position, teammate.broadcast_target)
    )

    return (
        int(state.map_template.is_territory(team_id, agent.position)),
        int(agent.position.x >= state.map_template.width // 2),
        int(agent.carrying_flag_team is not None),
        int(agent.captured),
        int(state.flag_carriers[team_id] is not None),
        int(state.flag_carriers[1 - team_id] is not None),
        own_base_dx,
        own_base_dy,
        own_base_distance,
        own_flag_dx,
        own_flag_dy,
        own_flag_distance,
        enemy_flag_dx,
        enemy_flag_dy,
        enemy_flag_distance,
        teammate_dx,
        teammate_dy,
        teammate_distance,
        enemy_dx,
        enemy_dy,
        enemy_distance,
        teammate_target_dx,
        teammate_target_dy,
        teammate_target_distance,
        _sign(own_team.flag_position.x - own_team.flag_spawn.x),
        _sign(own_team.flag_position.y - own_team.flag_spawn.y),
    )


def legal_actions(state: GameState, team_id: int, agent_id: str) -> list[Action]:
    agent = state.agents[agent_id]
    actions = [
        action
        for action in TRAINING_ACTIONS
        if state.map_template.in_bounds(agent.position.moved(action))
        and not state.map_template.is_wall(agent.position.moved(action))
    ]
    return actions or [Action.STAY]


@dataclass
class TabularQLearningPolicy:
    alpha: float = 0.15
    gamma: float = 0.98
    epsilon: float = 0.55
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.05
    seed: int | None = None
    q_table: dict[str, dict[str, float]] = field(default_factory=dict)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _state_key(self, state_key: tuple[int, ...]) -> str:
        return json.dumps(list(state_key), separators=(",", ":"))

    def _action_values(self, state_key: tuple[int, ...]) -> dict[str, float]:
        key = self._state_key(state_key)
        values = self.q_table.setdefault(
            key,
            {action.value: 0.0 for action in TRAINING_ACTIONS},
        )
        for action in TRAINING_ACTIONS:
            values.setdefault(action.value, 0.0)
        return values

    def select_action(
        self,
        state_key: tuple[int, ...],
        allowed_actions: Iterable[Action],
        explore: bool = True,
    ) -> Action:
        legal_actions = list(allowed_actions)
        if not legal_actions:
            return Action.STAY

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        values = self._action_values(state_key)
        best_score = max(values[action.value] for action in legal_actions)
        best_actions = [
            action for action in legal_actions if values[action.value] == best_score
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        state_key: tuple[int, ...],
        action: Action,
        reward: float,
        next_state_key: tuple[int, ...],
        done: bool,
    ) -> None:
        values = self._action_values(state_key)
        current_value = values[action.value]
        if done:
            target = reward
        else:
            next_values = self._action_values(next_state_key)
            target = reward + self.gamma * max(
                next_values[future_action.value] for future_action in TRAINING_ACTIONS
            )
        values[action.value] = current_value + self.alpha * (target - current_value)

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "q_table": self.q_table,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularQLearningPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
        )
        policy.q_table = {
            key: {action: float(value) for action, value in action_values.items()}
            for key, action_values in payload.get("q_table", {}).items()
        }
        return policy


@dataclass
class QLearningAgent(BaseAgent):
    policy: TabularQLearningPolicy
    explore: bool = True
    last_state_key: tuple[int, ...] | None = None
    last_action: Action | None = None

    def choose_action(self, state: GameState, team_id: int, agent_id: str) -> Action:
        state_key = encode_state(state, team_id, agent_id)
        actions = legal_actions(state, team_id, agent_id)
        action = self.policy.select_action(state_key, actions, explore=self.explore)
        self.last_state_key = state_key
        self.last_action = action
        return action
