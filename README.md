# CaptureTheFlag

A lightweight, student-friendly Capture the Flag framework for testing agents against each other.

## What is included

- A small simultaneous-turn game engine for 2v2 play.
- One flag per team with simple capture and scoring rules.
- A dropped flag returns home immediately when a friendly defender reaches it.
- Matches run until the turn limit by default, with an optional score limit if you want early termination.
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
python -m capture_the_flag.cli --gui --map open_field --team0 aggressive --team1 defensive --turns 50
```

Use `--gui` to open the popout simulation window. Omit it if you want the ASCII console output instead.

To load a trained agent into the GUI, point the primary team at a saved Q-table:

```bash
python -m capture_the_flag.cli --gui --map open_field --team0 trained --team0-policy q_table.json --team0-support balanced --team1 defensive --turns 50
```

To train a simple tabular Q-learning agent against the rule-based opponents, run:

```bash
python train_simple_agent.py --episodes 1500 --bootstrap-episodes 300 --save q_table.json
```

The script warm-starts the policy from rule-based teacher rollouts, then fine-tunes it with Q-learning and evaluates the saved policy against the selected rule-based team setup.

If you prefer the shorter console script, add your Python user Scripts directory to PATH first, then `ctf-sim` will work as well.

If you do not install the package, run it with `PYTHONPATH=src` set in your environment.

## Extending the framework

- Add a new agent by subclassing `BaseAgent` in `src/capture_the_flag/agents/base.py`.
- Add rule logic by extending `RuleBasedAgent` in `src/capture_the_flag/agents/rule_based.py`.
- Add or tweak maps in `src/capture_the_flag/core/map_templates.py`.
- Adjust game rules in `src/capture_the_flag/core/engine.py`.

The built-in rule-based team strategies are `balanced`, `aggressive`, and `defensive`.
Use them from the CLI or GUI to quickly compare different team behaviors.

For reinforcement learning, call `GameEngine.build_observation(agent_id)` to get a
dictionary containing self, teammate, enemies, flags, territory, the delayed teammate
intent signal, and on-demand shortest-step lookup data.

## Notes

- The current implementation is a scaffold, not a finished game.
- The code is intentionally compact so students can modify the core loop without wading through framework overhead.