from __future__ import annotations

from typing import Protocol

from .models import (
    ACTION_DELTAS,
    Action,
    AgentState,
    GameConfig,
    GameState,
    MapTemplate,
    Position,
    TeamState,
)


class Agent(Protocol):
    def choose_action(
        self, state: GameState, team_id: int, agent_id: str
    ) -> Action: ...


class GameEngine:
    def __init__(
        self, map_template: MapTemplate, config: GameConfig | None = None
    ) -> None:
        self.map_template = map_template
        self.config = config or GameConfig()
        self.state = self._create_initial_state(map_template)

    def _create_initial_state(self, map_template: MapTemplate) -> GameState:
        teams = (
            TeamState(
                team_id=0,
                name="Team A",
                flag_spawn=map_template.team_flags[0],
                flag_position=map_template.team_flags[0],
            ),
            TeamState(
                team_id=1,
                name="Team B",
                flag_spawn=map_template.team_flags[1],
                flag_position=map_template.team_flags[1],
            ),
        )
        agents: dict[str, AgentState] = {}
        team_agent_ids: list[list[str]] = [[], []]
        for team_id, spawn_positions in enumerate(map_template.team_spawns):
            ids_for_team: list[str] = []
            for index, spawn in enumerate(spawn_positions):
                agent_id = f"t{team_id}_a{index}"
                agents[agent_id] = AgentState(
                    agent_id=agent_id, team_id=team_id, spawn=spawn, position=spawn
                )
                ids_for_team.append(agent_id)
            team_agent_ids[team_id] = ids_for_team
        return GameState(
            turn=0,
            map_template=map_template,
            teams=teams,
            agents=agents,
            team_agent_ids=(tuple(team_agent_ids[0]), tuple(team_agent_ids[1])),
        )

    def reset(self) -> GameState:
        self.state = self._create_initial_state(self.map_template)
        return self.state

    def step(self, agents_by_team: tuple[list[Agent], list[Agent]]) -> GameState:
        if self.state.winner is not None:
            return self.state

        self.state.last_events = []
        self._reactivate_captured_agents()
        for agent in self.state.agents.values():
            agent.previous_position = agent.position
        proposed_positions: dict[str, Position] = {}
        team_plans: list[dict[str, Position]] = [{}, {}]
        next_broadcast_targets: dict[str, Position | None] = {}
        for team_id in (0, 1):
            planned_positions = team_plans[team_id]
            for agent_id in self.state.team_agent_ids[team_id]:
                agent = self.state.agents[agent_id]
                if agent.captured:
                    proposed_positions[agent.agent_id] = agent.position
                    planned_positions[agent.agent_id] = agent.position
                    next_broadcast_targets[agent_id] = None
                    continue
                planning_state = self.state.clone_shallow()
                planning_state.team_planned_positions = (
                    dict(team_plans[0]),
                    dict(team_plans[1]),
                )
                agent_index = self.state.team_agent_ids[agent.team_id].index(
                    agent.agent_id
                )
                controller = agents_by_team[agent.team_id][agent_index]
                target_selector = getattr(controller, "choose_target", None)
                chosen_target = (
                    target_selector(planning_state, agent.team_id, agent.agent_id)
                    if callable(target_selector)
                    else None
                )
                next_broadcast_targets[agent_id] = chosen_target
                action = controller.choose_action(
                    planning_state, agent.team_id, agent.agent_id
                )
                proposed_position = self._choose_non_conflicting_destination(
                    agent.position,
                    action,
                    set(planned_positions.values()),
                )
                proposed_positions[agent.agent_id] = proposed_position
                planned_positions[agent.agent_id] = proposed_position

        self.state.team_planned_positions = (
            dict(team_plans[0]),
            dict(team_plans[1]),
        )
        for agent_id, target in next_broadcast_targets.items():
            self.state.agents[agent_id].broadcast_target = target
        self._resolve_moves(proposed_positions)
        self._resolve_flags()
        self._check_score_limit()
        self.state.turn += 1
        if self.state.turn >= self.config.max_turns and self.state.winner is None:
            self._resolve_winner_by_score()
            self.state.last_events.append("Reached max turns.")
        return self.state

    def _reactivate_captured_agents(self) -> None:
        for agent in self.state.agents.values():
            if not agent.captured:
                continue
            agent.captured = False
            agent.position = agent.spawn
            agent.previous_position = agent.spawn
            agent.carrying_flag_team = None
            agent.broadcast_target = None
            self.state.last_events.append(f"{agent.agent_id} respawned.")

    def _choose_non_conflicting_destination(
        self,
        position: Position,
        action: Action,
        reserved_positions: set[Position],
    ) -> Position:
        if action is Action.STAY:
            return position

        preferred_order = [action, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        seen: set[Action] = set()
        for candidate_action in preferred_order:
            if candidate_action in seen:
                continue
            seen.add(candidate_action)
            if candidate_action is Action.STAY:
                continue
            candidate = position.moved(candidate_action)
            if (
                self._is_legal_destination(candidate)
                and candidate not in reserved_positions
            ):
                return candidate

        for candidate_action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            candidate = position.moved(candidate_action)
            if self._is_legal_destination(candidate):
                return candidate

        return position

    def _is_legal_destination(self, position: Position) -> bool:
        return self.map_template.in_bounds(position) and not self.map_template.is_wall(
            position
        )

    def _resolve_moves(self, proposed_positions: dict[str, Position]) -> None:
        self.state.last_events.append("Resolved simultaneous movement.")
        for agent_id, position in proposed_positions.items():
            self.state.agents[agent_id].position = position

        agents_by_position: dict[Position, list[AgentState]] = {}
        for agent in self.state.agents.values():
            if agent.captured:
                continue
            agents_by_position.setdefault(agent.position, []).append(agent)

        for position, occupants in agents_by_position.items():
            if len(occupants) < 2:
                continue
            if not any(
                occupant.team_id != other.team_id
                for index, occupant in enumerate(occupants)
                for other in occupants[index + 1 :]
            ):
                continue

            for agent in occupants:
                if not any(other.team_id != agent.team_id for other in occupants):
                    continue
                if (
                    agent.carrying_flag_team is not None
                    or self.map_template.is_territory(agent.team_id, position)
                ):
                    self._capture_agent(agent)

    def _capture_agent(self, agent: AgentState) -> None:
        caught_position = agent.position
        agent.captured = True
        if agent.carrying_flag_team is not None:
            carried_team = agent.carrying_flag_team
            self._drop_flag(carried_team, caught_position)
            agent.carrying_flag_team = None
        agent.position = agent.spawn
        self.state.last_events.append(f"{agent.agent_id} was captured.")

    def _resolve_flags(self) -> None:
        for agent in self.state.agents.values():
            if agent.captured:
                continue
            if agent.carrying_flag_team is None:
                pickup_team = self._flag_team_at_position(agent.position)
                if pickup_team is None:
                    continue
                if self.state.flag_carriers[pickup_team] is not None:
                    continue
                if (
                    pickup_team == agent.team_id
                    and agent.position == self.state.teams[pickup_team].flag_spawn
                ):
                    continue
                if pickup_team == agent.team_id:
                    self._return_flag(pickup_team)
                    self.state.last_events.append(
                        f"{agent.agent_id} returned their flag to base."
                    )
                    continue
                agent.carrying_flag_team = pickup_team
                self.state.flag_carriers[pickup_team] = agent.agent_id
                self.state.last_events.append(
                    f"{agent.agent_id} picked up the enemy flag."
                )
                continue

            carried_team = agent.carrying_flag_team
            if carried_team == agent.team_id and agent.position == agent.spawn:
                self._return_flag(carried_team)
                agent.carrying_flag_team = None
                self.state.last_events.append(f"{agent.agent_id} returned their flag.")
            elif (
                carried_team != agent.team_id
                and agent.position == self.state.teams[agent.team_id].flag_spawn
            ):
                team = self.state.teams[agent.team_id]
                team.score += 1
                self._return_flag(carried_team)
                agent.carrying_flag_team = None
                self.state.last_events.append(f"{team.name} scored.")

    def _flag_team_at_position(self, position: Position) -> int | None:
        for team in self.state.teams:
            if team.flag_position == position:
                return team.team_id
        return None

    def _drop_flag(self, team_id: int, position: Position) -> None:
        self.state.flag_carriers[team_id] = None
        self.state.teams[team_id].flag_position = position

    def _return_flag(self, team_id: int) -> None:
        self.state.flag_carriers[team_id] = None
        self.state.teams[team_id].flag_position = self.state.teams[team_id].flag_spawn

    def _resolve_winner_by_score(self) -> None:
        score0 = self.state.teams[0].score
        score1 = self.state.teams[1].score
        if score0 > score1:
            self.state.winner = 0
            self.state.last_events.append("Team A won the game.")
        elif score1 > score0:
            self.state.winner = 1
            self.state.last_events.append("Team B won the game.")

    def _check_score_limit(self) -> None:
        if self.config.score_to_win <= 0 or self.state.winner is not None:
            return
        if any(team.score >= self.config.score_to_win for team in self.state.teams):
            self._resolve_winner_by_score()
            self.state.last_events.append("Reached score limit.")

    def build_observation(self, agent_id: str) -> dict[str, object]:
        agent = self.state.agents[agent_id]
        team_id = agent.team_id
        teammate_id = next(
            candidate
            for candidate in self.state.team_agent_ids[team_id]
            if candidate != agent_id
        )
        enemy_team = 1 - team_id
        teammate = self.state.agents[teammate_id]

        return {
            "turn": self.state.turn,
            "agent_id": agent_id,
            "team_id": team_id,
            "self": {
                "position": agent.position,
                "previous_position": agent.previous_position,
                "spawn": agent.spawn,
                "carrying_flag_team": agent.carrying_flag_team,
                "captured": agent.captured,
                "broadcast_target": agent.broadcast_target,
            },
            "teammate": {
                "position": teammate.position,
                "previous_position": teammate.previous_position,
                "spawn": teammate.spawn,
                "carrying_flag_team": teammate.carrying_flag_team,
                "captured": teammate.captured,
                "previous_target": teammate.broadcast_target,
            },
            "enemies": [
                {
                    "agent_id": enemy.agent_id,
                    "position": enemy.position,
                    "previous_position": enemy.previous_position,
                    "spawn": enemy.spawn,
                    "carrying_flag_team": enemy.carrying_flag_team,
                    "captured": enemy.captured,
                }
                for enemy in self.state.agents.values()
                if enemy.team_id == enemy_team
            ],
            "flags": {
                "own": {
                    "position": self.state.teams[team_id].flag_position,
                    "spawn": self.state.teams[team_id].flag_spawn,
                    "carrier": self.state.flag_carriers[team_id],
                },
                "enemy": {
                    "position": self.state.teams[enemy_team].flag_position,
                    "spawn": self.state.teams[enemy_team].flag_spawn,
                    "carrier": self.state.flag_carriers[enemy_team],
                },
            },
            "map": {
                "width": self.map_template.width,
                "height": self.map_template.height,
                "walls": list(self.map_template.walls),
                "territory": [
                    [
                        self.map_template.is_territory(team_id, Position(x, y))
                        for x in range(self.map_template.width)
                    ]
                    for y in range(self.map_template.height)
                ],
                "static_grid": self._static_grid_rows(),
                "shortest_step_matrix": self.map_template.next_step_matrix,
                "shortest_step_choices": self.map_template.next_step_choices,
            },
            "team_previous_target": teammate.broadcast_target,
        }

    def _static_grid_rows(self) -> list[str]:
        template = self.state.map_template
        rows = [["." for _ in range(template.width)] for _ in range(template.height)]
        for wall in template.walls:
            rows[wall.y][wall.x] = "#"
        return ["".join(row) for row in rows]

    def render_ascii(self) -> str:
        template = self.state.map_template
        grid = [["." for _ in range(template.width)] for _ in range(template.height)]
        mid_x = template.width // 2
        for y in range(template.height):
            grid[y][mid_x] = "|"
        for wall in template.walls:
            grid[wall.y][wall.x] = "#"
        for team in self.state.teams:
            if self.state.flag_carriers[team.team_id] is not None:
                continue
            grid[team.flag_position.y][team.flag_position.x] = (
                "F" if team.team_id == 0 else "f"
            )
        for agent in self.state.agents.values():
            if agent.captured:
                char = "x" if agent.team_id == 0 else "y"
            else:
                char = (
                    "a"
                    if agent.carrying_flag_team is not None and agent.team_id == 0
                    else "A"
                )
                if agent.team_id == 1:
                    char = "b" if agent.carrying_flag_team is not None else "B"
            grid[agent.position.y][agent.position.x] = char
        return "\n".join("".join(row) for row in grid)
