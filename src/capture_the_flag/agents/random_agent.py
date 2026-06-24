from __future__ import annotations

import random

from ..core.models import Action, GameState
from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def choose_action(self, state: GameState, team_id: int, agent_id: str) -> Action:
        return self.random.choice(list(Action))
