from __future__ import annotations

from dataclasses import dataclass

from ..core.models import Action, GameState, Position
from .base import BaseAgent


@dataclass
class RuleBasedAgent(BaseAgent):
    """Scaffold for hand-authored or student-written rules.

    Override score_action or target_position to create more advanced strategies.
    """

    def choose_action(self, state: GameState, team_id: int, agent_id: str) -> Action:
        agent = state.agents[agent_id]
        enemy_flag = state.teams[1 - team_id].flag_position
        return self._move_toward(agent.position, enemy_flag)

    def _move_toward(self, current: Position, target: Position) -> Action:
        dx = target.x - current.x
        dy = target.y - current.y
        if abs(dx) > abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        if dy != 0:
            return Action.DOWN if dy > 0 else Action.UP
        return Action.STAY


class HeuristicAgent(RuleBasedAgent):
    """Convenience alias for rule-based agents with custom heuristics."""

    def choose_action(self, state: GameState, team_id: int, agent_id: str) -> Action:
        return super().choose_action(state, team_id, agent_id)
