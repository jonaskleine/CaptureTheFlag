# CaptureTheFlag

A lightweight, student-friendly Capture the Flag framework for testing agents against each other.

## What is included

- A small simultaneous-turn game engine for 2v2 play.
- One flag per team with simple capture and scoring rules.
- Five symmetrical grid maps to experiment with.
- A clean agent interface for learned, scripted, and rule-based agents.
- A rule-based agent scaffold that students can extend later.

## Project layout

- `src/capture_the_flag/core/` contains the engine, state models, and map templates.
- `src/capture_the_flag/agents/` contains the agent base class plus example agents.
- `src/capture_the_flag/cli.py` runs a quick simulation from the command line.

## Quick start

Install the package in editable mode and run a sample match:

```bash
pip install -e .
python -m capture_the_flag.cli --gui --map open_field --team0 rule --team1 random --turns 50
```

Use `--gui` to open the popout simulation window. Omit it if you want the ASCII console output instead.

If you prefer the shorter console script, add your Python user Scripts directory to PATH first, then `ctf-sim` will work as well.

If you do not install the package, run it with `PYTHONPATH=src` set in your environment.

## Extending the framework

- Add a new agent by subclassing `BaseAgent` in `src/capture_the_flag/agents/base.py`.
- Add rule logic by extending `RuleBasedAgent` in `src/capture_the_flag/agents/rule_based.py`.
- Add or tweak maps in `src/capture_the_flag/core/map_templates.py`.
- Adjust game rules in `src/capture_the_flag/core/engine.py`.

## Notes

- The current implementation is a scaffold, not a finished game.
- The code is intentionally compact so students can modify the core loop without wading through framework overhead.