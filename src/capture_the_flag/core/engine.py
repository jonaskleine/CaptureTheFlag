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
        proposed_positions: dict[str, Position] = {}
        team_plans: list[dict[str, Position]] = [{}, {}]
        for team_id in (0, 1):
            planned_positions = team_plans[team_id]
            for agent_id in self.state.team_agent_ids[team_id]:
                agent = self.state.agents[agent_id]
                if agent.captured:
                    proposed_positions[agent.agent_id] = agent.position
                    planned_positions[agent.agent_id] = agent.position
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
        self._resolve_moves(proposed_positions)
        self._resolve_flags()
        self._check_winner()
        self.state.turn += 1
        if self.state.turn >= self.config.max_turns and self.state.winner is None:
            self.state.last_events.append("Reached max turns.")
        return self.state

    def _reactivate_captured_agents(self) -> None:
        for agent in self.state.agents.values():
            if not agent.captured:
                continue
            agent.captured = False
            agent.position = agent.spawn
            agent.carrying_flag_team = None
            self.state.last_events.append(f"{agent.agent_id} respawned.")

    def _choose_non_conflicting_destination(
        self,
        position: Position,
        action: Action,
        reserved_positions: set[Position],
    ) -> Position:
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

        for agent_id, agent in self.state.agents.items():
            if agent.captured:
                continue
            opponents = [
                other
                for other in self.state.agents.values()
                if other.team_id != agent.team_id
                and other.position == agent.position
                and not other.captured
            ]
            if opponents and self.map_template.is_territory(
                agent.team_id, agent.position
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
                agent.carrying_flag_team = pickup_team
                self.state.flag_carriers[pickup_team] = agent.agent_id
                if pickup_team == agent.team_id:
                    self.state.last_events.append(
                        f"{agent.agent_id} picked up their own flag."
                    )
                else:
                    self.state.last_events.append(
                        f"{agent.agent_id} picked up the enemy flag."
                    )
                continue

            carried_team = agent.carrying_flag_team
            if carried_team == agent.team_id and agent.position == agent.spawn:
                self._return_flag(carried_team)
                agent.carrying_flag_team = None
                self.state.last_events.append(f"{agent.agent_id} returned their flag.")
            elif carried_team != agent.team_id and agent.position == agent.spawn:
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

    def _check_winner(self) -> None:
        for team in self.state.teams:
            if team.score >= self.config.score_to_win:
                self.state.winner = team.team_id
                self.state.last_events.append(f"{team.name} won the game.")
                return

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
