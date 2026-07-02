from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.engine import Agent
from ..core.models import Action, GameState, Position


class BaseAgent(ABC):
    """Base class for all agents.

    Subclasses only need to implement choose_action.
    """

    @abstractmethod
    def choose_action(self, state: GameState, team_id: int, agent_id: str) -> Action:
        raise NotImplementedError

    def choose_target(
        self, state: GameState, team_id: int, agent_id: str
    ) -> Position | None:
        return None


class AgentAdapter(BaseAgent, Agent):
    pass
