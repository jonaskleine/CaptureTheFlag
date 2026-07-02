from __future__ import annotations

from dataclasses import dataclass

try:
    import tkinter as tk
except (
    ImportError
) as exc:  # pragma: no cover - tkinter is standard on Windows, but keep a clear error.
    tk = None
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None

from .agents import (
    AggressiveRuleBasedAgent,
    BalancedRuleBasedAgent,
    DefensiveRuleBasedAgent,
    RandomAgent,
)
from .agent_factory import build_team
from .core.engine import GameEngine
from .core.map_templates import MapTemplate
from .core.models import Position

AGENT_FACTORIES = {
    "random": lambda: RandomAgent(),
    "balanced": lambda: BalancedRuleBasedAgent(),
    "aggressive": lambda: AggressiveRuleBasedAgent(),
    "defensive": lambda: DefensiveRuleBasedAgent(),
    "rule": lambda: BalancedRuleBasedAgent(),
}

STRATEGY_CHOICES = ["balanced", "aggressive", "defensive"]


@dataclass(slots=True)
class SimulationConfig:
    map_template: MapTemplate
    team0_agent: str = "balanced"
    team0_support_agent: str | None = None
    team0_policy_path: str | None = None
    team1_agent: str = "defensive"
    team1_support_agent: str | None = None
    team1_policy_path: str | None = None
    max_turns: int = 50
    step_delay_ms: int = 450


class SimulationWindow:
    def __init__(self, config: SimulationConfig) -> None:
        if tk is None:
            raise RuntimeError("tkinter is not available") from _TK_IMPORT_ERROR

        self.config = config
        self.engine = GameEngine(config.map_template)
        self.engine.config.max_turns = config.max_turns
        self.team0 = build_team(
            config.team0_agent,
            config.team0_support_agent,
            primary_policy_path=config.team0_policy_path,
        )
        self.team1 = build_team(
            config.team1_agent,
            config.team1_support_agent,
            primary_policy_path=config.team1_policy_path,
        )
        self.cell_size = 56
        self.margin = 24
        self.side_panel_width = 280
        self.root = tk.Tk()
        self.root.title(f"CaptureTheFlag - {config.map_template.name}")
        self.root.resizable(False, False)
        width = (
            self.margin * 2
            + config.map_template.width * self.cell_size
            + self.side_panel_width
        )
        height = self.margin * 2 + config.map_template.height * self.cell_size
        self.root.geometry(f"{width}x{height}")

        self.canvas = tk.Canvas(
            self.root, width=width, height=height, bg="#10141a", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.status_text = tk.StringVar()
        self.status_text.set("Starting simulation...")
        self._ended = False

    def run(self) -> None:
        self._render()
        self.root.after(self.config.step_delay_ms, self._advance)
        self.root.mainloop()

    def _advance(self) -> None:
        if self._ended:
            return

        if (
            self.engine.state.winner is not None
            or self.engine.state.turn >= self.config.max_turns
        ):
            self._ended = True
            self._render()
            return

        self.engine.step((self.team0, self.team1))
        self._render()

        if (
            self.engine.state.winner is None
            and self.engine.state.turn < self.config.max_turns
        ):
            self.root.after(self.config.step_delay_ms, self._advance)
        else:
            self._ended = True

    def _render(self) -> None:
        self.canvas.delete("all")
        template = self.engine.state.map_template

        self._draw_board_frame(template)
        self._draw_midline(template)

        mid_x = template.width // 2
        for y in range(template.height):
            for x in range(template.width):
                left = self.margin + x * self.cell_size
                top = self.margin + y * self.cell_size
                right = left + self.cell_size
                bottom = top + self.cell_size
                if template.is_wall(Position(x, y)):
                    fill = "#2a3140"
                else:
                    fill = "#cfe8ff" if x < mid_x else "#ffd6d6"
                self.canvas.create_rectangle(
                    left, top, right, bottom, fill=fill, outline="#8a93a6"
                )

        for team in self.engine.state.teams:
            if self.engine.state.flag_carriers[team.team_id] is None:
                self._draw_flag(team.team_id)

        for agent in self.engine.state.agents.values():
            self._draw_agent(
                agent.team_id, agent.position.x, agent.position.y, agent.agent_id
            )

        self._draw_side_panel()

    def _draw_board_frame(self, template: MapTemplate) -> None:
        board_width = template.width * self.cell_size
        board_height = template.height * self.cell_size
        self.canvas.create_rectangle(
            self.margin - 4,
            self.margin - 4,
            self.margin + board_width + 4,
            self.margin + board_height + 4,
            fill="#18202b",
            outline="#334155",
            width=2,
        )

    def _draw_territories(self, template: MapTemplate) -> None:
        mid_x = template.width // 2
        left_width = mid_x * self.cell_size
        right_width = template.width * self.cell_size - left_width
        board_top = self.margin
        board_bottom = self.margin + template.height * self.cell_size

        self.canvas.create_rectangle(
            self.margin,
            board_top,
            self.margin + left_width,
            board_bottom,
            fill="#cfe8ff",
            outline="",
        )
        self.canvas.create_rectangle(
            self.margin + left_width,
            board_top,
            self.margin + left_width + right_width,
            board_bottom,
            fill="#ffd6d6",
            outline="",
        )

    def _draw_midline(self, template: MapTemplate) -> None:
        mid_x = template.width // 2
        x = self.margin + mid_x * self.cell_size
        top = self.margin
        bottom = self.margin + template.height * self.cell_size
        self.canvas.create_line(
            x,
            top,
            x,
            bottom,
            fill="#f8fafc",
            width=2,
            dash=(8, 6),
        )

    def _draw_flag(self, team_id: int) -> None:
        flag = self.engine.state.teams[team_id].flag_position
        left = self.margin + flag.x * self.cell_size
        top = self.margin + flag.y * self.cell_size
        right = left + self.cell_size
        bottom = top + self.cell_size
        fill = "#ffb703" if team_id == 0 else "#8ecae6"
        text = "F" if team_id == 0 else "f"
        self.canvas.create_oval(
            left + 10,
            top + 10,
            right - 10,
            bottom - 10,
            fill=fill,
            outline="#1f2937",
            width=2,
        )
        self.canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text=text,
            fill="#111827",
            font=("Segoe UI", 14, "bold"),
        )

    def _draw_agent(self, team_id: int, x: int, y: int, agent_id: str) -> None:
        left = self.margin + x * self.cell_size
        top = self.margin + y * self.cell_size
        right = left + self.cell_size
        bottom = top + self.cell_size
        agent = self.engine.state.agents[agent_id]
        if agent.captured:
            fill = "#94a3b8" if team_id == 0 else "#fca5a5"
        else:
            fill = "#2563eb" if team_id == 0 else "#dc2626"
        outline = "#0f172a"
        label = agent_id.split("_")[-1].upper()
        if agent.carrying_flag_team is not None:
            label = f"{label}{'F' if agent.carrying_flag_team != team_id else 'H'}"
            badge_fill = "#fbbf24" if agent.carrying_flag_team != team_id else "#34d399"
            badge_text = "E" if agent.carrying_flag_team != team_id else "O"
            self.canvas.create_rectangle(
                right - 18,
                top + 8,
                right - 8,
                top + 22,
                fill=badge_fill,
                outline="#78350f",
                width=1,
            )
            self.canvas.create_text(
                right - 13,
                top + 15,
                text=badge_text,
                fill="#111827",
                font=("Segoe UI", 8, "bold"),
            )
        elif agent.captured:
            label = f"{label}R"
        self.canvas.create_oval(
            left + 8,
            top + 8,
            right - 8,
            bottom - 8,
            fill=fill,
            outline=outline,
            width=2,
        )
        self.canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text=label,
            fill="white",
            font=("Segoe UI", 13, "bold"),
        )

        carrier = self.engine.state.flag_carriers[1 - team_id]
        if carrier == agent_id:
            self.canvas.create_rectangle(
                right - 18,
                top + 8,
                right - 8,
                top + 22,
                fill="#fbbf24",
                outline="#78350f",
                width=1,
            )
            self.canvas.create_text(
                right - 13,
                top + 15,
                text="F",
                fill="#111827",
                font=("Segoe UI", 8, "bold"),
            )

    def _draw_side_panel(self) -> None:
        template = self.engine.state.map_template
        panel_left = self.margin + template.width * self.cell_size + 18
        top = self.margin
        self.canvas.create_text(
            panel_left,
            top,
            anchor="nw",
            text="CaptureTheFlag",
            fill="#e5e7eb",
            font=("Segoe UI", 20, "bold"),
        )
        top += 38
        lines = [
            f"Map: {template.name}",
            f"Turn: {self.engine.state.turn}",
            f"Team A score: {self.engine.state.teams[0].score}",
            f"Team B score: {self.engine.state.teams[1].score}",
            f"Winner: {self.engine.state.winner if self.engine.state.winner is not None else 'None'}",
            f"Team A agents: {sum(1 for agent in self.engine.state.agents.values() if agent.team_id == 0)}",
            f"Team B agents: {sum(1 for agent in self.engine.state.agents.values() if agent.team_id == 1)}",
            "Flags: 2",
            "Blue half: Team B can be caught here",
            "Red half: Team A can be caught here",
        ]
        for line in lines:
            self.canvas.create_text(
                panel_left,
                top,
                anchor="nw",
                text=line,
                fill="#cbd5e1",
                font=("Segoe UI", 11),
            )
            top += 24

        top += 8
        self.canvas.create_text(
            panel_left,
            top,
            anchor="nw",
            text="Recent events",
            fill="#e5e7eb",
            font=("Segoe UI", 12, "bold"),
        )
        top += 24
        events = self.engine.state.last_events[-6:] or ["No events yet."]
        for event in events:
            self.canvas.create_text(
                panel_left,
                top,
                anchor="nw",
                text=f"- {event}",
                fill="#cbd5e1",
                font=("Segoe UI", 10),
                width=self.side_panel_width - 24,
            )
            top += 34
