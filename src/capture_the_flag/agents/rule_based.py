from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from ..core.models import Action, AgentState, GameState, Position
from .base import BaseAgent

StrategyName = Literal["balanced", "aggressive", "defensive"]
MissionName = Literal[
    "return_home",
    "recover_flag",
    "escort_flag",
    "intercept_carrier",
    "defend_home",
    "advance",
    "patrol",
]


@dataclass
class _TargetMemory:
    target: Position | None = None
    mission: MissionName = "patrol"
    lock_key: tuple[object, ...] = ()
    patrol_index: int = 0


@dataclass
class RuleBasedAgent(BaseAgent):
    strategy: StrategyName = "balanced"
    rng: random.Random = field(default_factory=random.Random, repr=False)
    _memory: dict[str, _TargetMemory] = field(default_factory=dict, repr=False)

    def choose_target(
        self, state: GameState, team_id: int, agent_id: str
    ) -> Position | None:
        memory = self._memory.setdefault(agent_id, _TargetMemory())
        teammate_id = self._teammate_id(state, team_id, agent_id)
        teammate_intent = state.agents[teammate_id].broadcast_target

        if self.strategy == "defensive":
            mission, target, lock_key = self._defensive_objective(
                state, team_id, agent_id, memory
            )
        else:
            mission, target, lock_key = self._generic_objective(
                state, team_id, agent_id, teammate_intent, memory
            )

        if memory.target is not None and memory.lock_key == lock_key:
            return memory.target

        memory.target = target
        memory.mission = mission
        memory.lock_key = lock_key
        return target

    def choose_action(self, state: GameState, team_id: int, agent_id: str) -> Action:
        agent = state.agents[agent_id]
        target = self.choose_target(state, team_id, agent_id)
        if target is None:
            target = self._center_target(state, team_id)

        legal_actions = [
            action
            for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
            if self._is_legal(state, agent.position.moved(action))
        ]
        if not legal_actions:
            return Action.UP

        if self.strategy == "defensive":
            own_flag_tile = state.teams[team_id].flag_position
            legal_actions = [
                action
                for action in legal_actions
                if agent.position.moved(action) != own_flag_tile
            ] or legal_actions
        elif self.strategy == "aggressive":
            preferred_steps = set(
                state.map_template.shortest_step_choices(agent.position, target)
            )
            if preferred_steps:
                progressing_actions = [
                    action
                    for action in legal_actions
                    if agent.position.moved(action) in preferred_steps
                ]
                if progressing_actions:
                    legal_actions = progressing_actions

        scored_actions: list[tuple[float, Action]] = []
        for action in legal_actions:
            candidate = agent.position.moved(action)
            score = -candidate.manhattan_distance(target)
            if candidate == target:
                score += 10
            if candidate == agent.previous_position:
                score -= 2
            if self.strategy == "defensive":
                if state.map_template.is_territory(team_id, candidate):
                    score += 2
                else:
                    score -= 50
            elif self.strategy == "aggressive":
                if not state.map_template.is_territory(team_id, candidate):
                    score += 2
            scored_actions.append((score, action))

        best_score = max(score for score, _ in scored_actions)
        best_actions = [
            action for score, action in scored_actions if score == best_score
        ]
        return self.rng.choice(best_actions)

    def _generic_objective(
        self,
        state: GameState,
        team_id: int,
        agent_id: str,
        teammate_intent: Position | None,
        memory: _TargetMemory,
    ) -> tuple[MissionName, Position, tuple[object, ...]]:
        team = state.teams[team_id]
        enemy_team = 1 - team_id
        enemy_team_state = state.teams[enemy_team]

        if self.strategy == "aggressive":
            if state.agents[agent_id].carrying_flag_team is not None:
                carrier = state.agents[agent_id]
                return (
                    "return_home",
                    team.flag_spawn,
                    ("return_home", team_id, carrier.agent_id),
                )
            target = enemy_team_state.flag_position
            return (
                "advance",
                target,
                ("advance", team_id, target.x, target.y),
            )

        if state.flag_carriers[enemy_team] is not None:
            carrier = state.agents[state.flag_carriers[enemy_team]]
            return (
                "intercept_carrier",
                carrier.position,
                ("intercept_carrier", carrier.agent_id),
            )

        if enemy_team_state.flag_position != enemy_team_state.flag_spawn:
            if teammate_intent == enemy_team_state.flag_position:
                target = self._support_target(
                    state, team_id, enemy_team_state.flag_position
                )
                return (
                    "escort_flag",
                    target,
                    (
                        "escort_flag",
                        enemy_team_state.flag_position.x,
                        enemy_team_state.flag_position.y,
                    ),
                )
            return (
                "recover_flag",
                enemy_team_state.flag_position,
                (
                    "recover_flag",
                    enemy_team_state.flag_position.x,
                    enemy_team_state.flag_position.y,
                ),
            )

        target = self._balanced_target(state, team_id, memory)
        return "patrol", target, ("patrol", team_id, target.x, target.y)

    def _defensive_objective(
        self,
        state: GameState,
        team_id: int,
        agent_id: str,
        memory: _TargetMemory,
    ) -> tuple[MissionName, Position, tuple[object, ...]]:
        team = state.teams[team_id]
        enemy_team = 1 - team_id
        agent_index = state.team_agent_ids[team_id].index(agent_id)

        intruders = [
            enemy
            for enemy in state.agents.values()
            if enemy.team_id == enemy_team
            and state.map_template.is_territory(team_id, enemy.position)
        ]
        guard_points = self._defensive_guard_points(state, team_id)
        if intruders:
            closest_enemy = min(
                intruders,
                key=lambda enemy: enemy.position.manhattan_distance(team.flag_spawn),
            )
            if agent_index == 0:
                target = self._block_point(state, team_id, closest_enemy.position)
                return (
                    "defend_home",
                    target,
                    (
                        "block",
                        closest_enemy.agent_id,
                        target.x,
                        target.y,
                    ),
                )

            if guard_points:
                target = guard_points[1] if len(guard_points) > 1 else guard_points[0]
            else:
                target = team.flag_spawn
            return (
                "defend_home",
                target,
                (
                    "support",
                    closest_enemy.agent_id,
                    target.x,
                    target.y,
                ),
            )

        if agent_index == 0:
            target = guard_points[0] if guard_points else team.flag_spawn
            return "defend_home", target, ("anchor", target.x, target.y)

        if guard_points:
            target = guard_points[1] if len(guard_points) > 1 else guard_points[0]
        else:
            target = team.flag_spawn
        return "defend_home", target, ("support", target.x, target.y)

    def _balanced_target(
        self, state: GameState, team_id: int, memory: _TargetMemory
    ) -> Position:
        control_points = self._control_points(state, team_id)
        if not control_points:
            return self._center_target(state, team_id)
        target = control_points[memory.patrol_index % len(control_points)]
        memory.patrol_index = (memory.patrol_index + 1) % len(control_points)
        return target

    def _advance_target(
        self, state: GameState, team_id: int, memory: _TargetMemory
    ) -> Position:
        control_points = self._control_points(state, team_id)
        if not control_points:
            return self._center_target(state, team_id)
        target = control_points[memory.patrol_index % len(control_points)]
        memory.patrol_index = (memory.patrol_index + 1) % len(control_points)
        return target

    def _screen_point(self, state: GameState, team_id: int) -> Position | None:
        points = self._screen_points(state, team_id)
        return points[0] if points else None

    def _defensive_guard_points(self, state: GameState, team_id: int) -> list[Position]:
        spawn = state.teams[team_id].flag_spawn
        candidates = [
            Position(max(1, spawn.x - 1), spawn.y),
            Position(spawn.x, max(1, spawn.y - 1)),
            Position(spawn.x, min(state.map_template.height - 2, spawn.y + 1)),
            Position(min(state.map_template.width - 2, spawn.x + 1), spawn.y),
        ]
        points = [
            point
            for point in candidates
            if state.map_template.in_bounds(point)
            and not state.map_template.is_wall(point)
            and state.map_template.is_territory(team_id, point)
            and point != state.teams[team_id].flag_position
        ]
        points.sort(
            key=lambda point: (
                point.manhattan_distance(spawn),
                abs(point.y - spawn.y),
                point.y,
                point.x,
            )
        )
        return points

    def _screen_points(self, state: GameState, team_id: int) -> list[Position]:
        midpoint = state.map_template.width // 2
        spawn = state.teams[team_id].flag_spawn
        if team_id == 0:
            x_candidates = [midpoint + 1, midpoint + 2, midpoint + 3]
        else:
            x_candidates = [midpoint - 2, midpoint - 3, midpoint - 4]

        candidates = [
            Position(x, y)
            for x in x_candidates
            for y in (spawn.y, spawn.y - 1, spawn.y + 1)
        ]
        points = [
            point
            for point in candidates
            if state.map_template.in_bounds(point)
            and not state.map_template.is_wall(point)
            and state.map_template.is_territory(team_id, point)
            and point != state.teams[team_id].flag_position
        ]
        points.sort(
            key=lambda point: (
                abs(point.x - midpoint),
                point.manhattan_distance(spawn),
                abs(point.y - spawn.y),
            )
        )
        return points

    def _guard_point(self, state: GameState, team_id: int) -> Position:
        spawn = state.teams[team_id].flag_spawn
        candidates = [
            Position(max(1, spawn.x - 1), spawn.y),
            Position(spawn.x, max(1, spawn.y - 1)),
            Position(spawn.x, min(state.map_template.height - 2, spawn.y + 1)),
            Position(min(state.map_template.width - 2, spawn.x + 1), spawn.y),
        ]
        points = [
            point
            for point in candidates
            if state.map_template.in_bounds(point)
            and not state.map_template.is_wall(point)
            and state.map_template.is_territory(team_id, point)
            and point != state.teams[team_id].flag_position
        ]
        if points:
            points.sort(
                key=lambda point: (
                    point.manhattan_distance(spawn),
                    abs(point.y - spawn.y),
                    point.y,
                    point.x,
                )
            )
            return points[0]
        return spawn

    def _block_point(
        self, state: GameState, team_id: int, intruder_position: Position
    ) -> Position:
        spawn = state.teams[team_id].flag_spawn
        dx = (
            1
            if spawn.x > intruder_position.x
            else -1 if spawn.x < intruder_position.x else 0
        )
        dy = (
            1
            if spawn.y > intruder_position.y
            else -1 if spawn.y < intruder_position.y else 0
        )
        candidates = [
            Position(spawn.x + dx, spawn.y + dy),
            Position(spawn.x + dx, spawn.y),
            Position(spawn.x, spawn.y + dy),
            Position(spawn.x + dx * 2, spawn.y),
            Position(spawn.x, spawn.y + dy * 2),
        ]
        points = [
            point
            for point in candidates
            if state.map_template.in_bounds(point)
            and not state.map_template.is_wall(point)
            and state.map_template.is_territory(team_id, point)
            and point != state.teams[team_id].flag_position
        ]
        if points:
            points.sort(
                key=lambda point: (
                    point.manhattan_distance(spawn),
                    point.manhattan_distance(intruder_position),
                )
            )
            return points[0]
        return spawn

    def _support_target(
        self, state: GameState, team_id: int, threatened_position: Position
    ) -> Position:
        closest_enemy = self._nearest_enemy(state, team_id, threatened_position)
        if closest_enemy is None:
            return threatened_position
        return self._between_target(threatened_position, closest_enemy.position, state)

    def _between_target(
        self, start: Position, enemy: Position, state: GameState
    ) -> Position:
        dx = 1 if enemy.x > start.x else -1 if enemy.x < start.x else 0
        dy = 1 if enemy.y > start.y else -1 if enemy.y < start.y else 0
        candidate = Position(start.x + dx, start.y + dy)
        if state.map_template.in_bounds(candidate) and not state.map_template.is_wall(
            candidate
        ):
            return candidate
        return start

    def _nearest_enemy(
        self, state: GameState, team_id: int, origin: Position
    ) -> AgentState | None:
        enemies = [
            agent
            for agent in state.agents.values()
            if agent.team_id != team_id and not agent.captured
        ]
        if not enemies:
            return None
        return min(enemies, key=lambda enemy: enemy.position.manhattan_distance(origin))

    def _control_points(self, state: GameState, team_id: int) -> list[Position]:
        mid_x = state.map_template.width // 2
        center_y = state.map_template.height // 2
        x = mid_x - 1 if team_id == 0 else mid_x
        candidates = [
            Position(x, center_y),
            Position(x, max(1, center_y - 3)),
            Position(x, min(state.map_template.height - 2, center_y + 3)),
        ]
        return [
            point
            for point in candidates
            if state.map_template.in_bounds(point)
            and not state.map_template.is_wall(point)
        ]

    def _center_target(self, state: GameState, team_id: int) -> Position:
        mid_x = state.map_template.width // 2
        center_y = state.map_template.height // 2
        x = mid_x - 1 if team_id == 0 else mid_x
        return Position(x, center_y)

    def _is_legal(self, state: GameState, position: Position) -> bool:
        return state.map_template.in_bounds(
            position
        ) and not state.map_template.is_wall(position)

    def _teammate_id(self, state: GameState, team_id: int, agent_id: str) -> str:
        return next(
            candidate
            for candidate in state.team_agent_ids[team_id]
            if candidate != agent_id
        )


@dataclass
class BalancedRuleBasedAgent(RuleBasedAgent):
    strategy: StrategyName = "balanced"


@dataclass
class AggressiveRuleBasedAgent(RuleBasedAgent):
    strategy: StrategyName = "aggressive"


@dataclass
class DefensiveRuleBasedAgent(RuleBasedAgent):
    strategy: StrategyName = "defensive"


class HeuristicAgent(RuleBasedAgent):
    """Backward-compatible alias for the balanced strategy."""
