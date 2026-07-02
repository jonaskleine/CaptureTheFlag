from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping


class Action(str, Enum):
    STAY = "stay"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


ACTION_DELTAS: Mapping[Action, tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int

    def moved(self, action: Action) -> "Position":
        dx, dy = ACTION_DELTAS[action]
        return Position(self.x + dx, self.y + dy)

    def manhattan_distance(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass(slots=True)
class AgentState:
    agent_id: str
    team_id: int
    spawn: Position
    position: Position
    previous_position: Position | None = None
    carrying_flag_team: int | None = None
    captured: bool = False
    broadcast_target: Position | None = None


@dataclass(slots=True)
class TeamState:
    team_id: int
    name: str
    flag_spawn: Position
    flag_position: Position
    score: int = 0


@dataclass(frozen=True, slots=True)
class MapTemplate:
    name: str
    width: int
    height: int
    walls: frozenset[Position]
    walkable_tiles: frozenset[Position]
    team_spawns: tuple[tuple[Position, ...], tuple[Position, ...]]
    team_flags: tuple[Position, Position]
    next_step_matrix: dict[Position, dict[Position, Position]] = field(
        default_factory=dict
    )
    next_step_choices: dict[Position, dict[Position, tuple[Position, ...]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_ascii(
        cls,
        name: str,
        layout: Iterable[str],
        team_spawns: tuple[tuple[Position, ...], tuple[Position, ...]],
        team_flags: tuple[Position, Position],
    ) -> "MapTemplate":
        rows = [row.rstrip("\n") for row in layout]
        if not rows:
            raise ValueError("layout must not be empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("layout must be rectangular")
        walls = {
            Position(x, y)
            for y, row in enumerate(rows)
            for x, cell in enumerate(row)
            if cell == "#"
        }
        return cls(
            name=name,
            width=width,
            height=len(rows),
            walls=frozenset(walls),
            team_spawns=team_spawns,
            team_flags=team_flags,
        )

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def is_wall(self, position: Position) -> bool:
        return position in self.walls

    def is_territory(self, team_id: int, position: Position) -> bool:
        midpoint = self.width // 2
        if team_id == 0:
            return position.x >= midpoint
        return position.x < midpoint

    def home_side(self, team_id: int) -> str:
        return "right" if team_id == 0 else "left"

    def shortest_step(self, start: Position, target: Position) -> Position | None:
        choices = self.shortest_step_choices(start, target)
        return choices[0] if choices else None

    def shortest_step_choices(
        self, start: Position, target: Position
    ) -> tuple[Position, ...]:
        cached_choices = self.next_step_choices.get(start)
        if cached_choices is not None and target in cached_choices:
            return cached_choices[target]

        from .map_templates import _a_star_shortest_steps

        choices = _a_star_shortest_steps(self, start, target)
        self.next_step_choices.setdefault(start, {})[target] = choices
        if choices:
            self.next_step_matrix.setdefault(start, {})[target] = choices[0]
        return choices


@dataclass(slots=True)
class GameConfig:
    max_turns: int = 200
    score_to_win: int = 0


@dataclass(slots=True)
class GameState:
    turn: int
    map_template: MapTemplate
    teams: tuple[TeamState, TeamState]
    agents: dict[str, AgentState] = field(default_factory=dict)
    team_agent_ids: tuple[tuple[str, ...], tuple[str, ...]] = field(
        default_factory=lambda: ((), ())
    )
    team_planned_positions: tuple[dict[str, Position], dict[str, Position]] = field(
        default_factory=lambda: ({}, {})
    )
    flag_carriers: dict[int, str | None] = field(
        default_factory=lambda: {0: None, 1: None}
    )
    last_events: list[str] = field(default_factory=list)
    winner: int | None = None

    def agent_states_for_team(self, team_id: int) -> list[AgentState]:
        return [agent for agent in self.agents.values() if agent.team_id == team_id]

    def clone_shallow(self) -> "GameState":
        return GameState(
            turn=self.turn,
            map_template=self.map_template,
            teams=self.teams,
            agents={
                agent_id: replace(agent) for agent_id, agent in self.agents.items()
            },
            team_agent_ids=self.team_agent_ids,
            team_planned_positions=(
                dict(self.team_planned_positions[0]),
                dict(self.team_planned_positions[1]),
            ),
            flag_carriers=dict(self.flag_carriers),
            last_events=list(self.last_events),
            winner=self.winner,
        )
