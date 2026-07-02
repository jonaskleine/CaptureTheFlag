from __future__ import annotations

import heapq

from .models import Action, MapTemplate, Position


def _p(x: int, y: int) -> Position:
    return Position(x, y)


def _row(left_half: str) -> str:
    if len(left_half) != 8:
        raise ValueError("symmetrical rows must use an 8-character half")
    return left_half + left_half[::-1]


def _rot180(top_rows: list[str]) -> list[str]:
    if len(top_rows) != 8:
        raise ValueError("top half must contain 8 rows")
    return top_rows + [row[::-1] for row in reversed(top_rows)]


def _build_next_step_matrix(
    layout: list[str],
) -> tuple[
    dict[Position, dict[Position, Position]],
    dict[Position, dict[Position, tuple[Position, ...]]],
]:
    height = len(layout)
    width = len(layout[0])
    walls = {
        Position(x, y)
        for y, row in enumerate(layout)
        for x, cell in enumerate(row)
        if cell == "#"
    }

    def in_bounds(position: Position) -> bool:
        return 0 <= position.x < width and 0 <= position.y < height

    matrix: dict[Position, dict[Position, Position]] = {}
    choice_matrix: dict[Position, dict[Position, tuple[Position, ...]]] = {}

    for source_y, source_row in enumerate(layout):
        for source_x, _source_cell in enumerate(source_row):
            source = Position(source_x, source_y)
            if source in walls:
                continue

            next_steps: dict[Position, Position] = {source: source}
            next_choices: dict[Position, tuple[Position, ...]] = {source: (source,)}

            for target_y, target_row in enumerate(layout):
                for target_x, _target_cell in enumerate(target_row):
                    target = Position(target_x, target_y)
                    if target == source or target in walls:
                        continue

                    distances: dict[Position, int] = {source: 0}
                    parents: dict[Position, set[Position]] = {source: set()}
                    frontier: list[tuple[int, int, Position]] = [
                        (source.manhattan_distance(target), 0, source)
                    ]
                    counter = 1

                    while frontier:
                        _, _, current = heapq.heappop(frontier)
                        current_distance = distances[current]
                        for action in (
                            Action.UP,
                            Action.DOWN,
                            Action.LEFT,
                            Action.RIGHT,
                        ):
                            candidate = current.moved(action)
                            if not in_bounds(candidate) or candidate in walls:
                                continue

                            new_distance = current_distance + 1
                            if (
                                candidate not in distances
                                or new_distance < distances[candidate]
                            ):
                                distances[candidate] = new_distance
                                parents[candidate] = {current}
                                priority = new_distance + candidate.manhattan_distance(
                                    target
                                )
                                heapq.heappush(frontier, (priority, counter, candidate))
                                counter += 1
                            elif distances[candidate] == new_distance:
                                parents.setdefault(candidate, set()).add(current)

                    if target not in distances:
                        continue

                    first_steps: set[Position] = set()
                    frontier_nodes = {target}
                    while frontier_nodes:
                        next_frontier: set[Position] = set()
                        for node in frontier_nodes:
                            for parent in parents.get(node, set()):
                                if parent == source:
                                    first_steps.add(node)
                                else:
                                    next_frontier.add(parent)
                        frontier_nodes = next_frontier

                    if not first_steps:
                        continue

                    ordered_steps = tuple(
                        sorted(first_steps, key=lambda pos: (pos.y, pos.x))
                    )
                    next_steps[target] = ordered_steps[0]
                    next_choices[target] = ordered_steps

            matrix[source] = next_steps
            choice_matrix[source] = next_choices

    return matrix, choice_matrix


def _a_star_shortest_steps(
    template: MapTemplate,
    source: Position,
    target: Position,
) -> tuple[Position, ...]:
    width = template.width
    height = template.height
    walls = template.walls

    def in_bounds(position: Position) -> bool:
        return 0 <= position.x < width and 0 <= position.y < height

    distances: dict[Position, int] = {source: 0}
    parents: dict[Position, set[Position]] = {source: set()}
    frontier: list[tuple[int, int, Position]] = [
        (source.manhattan_distance(target), 0, source)
    ]
    counter = 1

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == target:
            break
        current_distance = distances[current]
        for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            candidate = current.moved(action)
            if not in_bounds(candidate) or candidate in walls:
                continue
            new_distance = current_distance + 1
            if candidate not in distances or new_distance < distances[candidate]:
                distances[candidate] = new_distance
                parents[candidate] = {current}
                priority = new_distance + candidate.manhattan_distance(target)
                heapq.heappush(frontier, (priority, counter, candidate))
                counter += 1
            elif distances[candidate] == new_distance:
                parents.setdefault(candidate, set()).add(current)

    if target not in distances:
        return ()

    first_steps: set[Position] = set()
    frontier_nodes = {target}
    while frontier_nodes:
        next_frontier: set[Position] = set()
        for node in frontier_nodes:
            for parent in parents.get(node, set()):
                if parent == source:
                    first_steps.add(node)
                else:
                    next_frontier.add(parent)
        frontier_nodes = next_frontier

    return tuple(sorted(first_steps, key=lambda pos: (pos.y, pos.x)))


def _make_template(
    name: str,
    top_rows: list[str],
    team_spawns: tuple[tuple[Position, ...], tuple[Position, ...]],
    team_flags: tuple[Position, Position],
) -> MapTemplate:
    layout = _rot180([_row(row) for row in top_rows])
    walls = {
        Position(x, y)
        for y, row in enumerate(layout)
        for x, cell in enumerate(row)
        if cell == "#"
    }
    walkable_tiles = {
        Position(x, y)
        for y, row in enumerate(layout)
        for x, cell in enumerate(row)
        if cell != "#"
    }
    return MapTemplate(
        name=name,
        width=16,
        height=16,
        walls=frozenset(walls),
        walkable_tiles=frozenset(walkable_tiles),
        team_spawns=team_spawns,
        team_flags=team_flags,
    )


MAPS: dict[str, MapTemplate] = {
    "open_field": _make_template(
        "open_field",
        [
            "." * 8,
            "." * 8,
            "." * 8,
            "." * 8,
            "." * 8,
            "." * 8,
            "." * 8,
            "." * 8,
        ],
        team_spawns=((_p(1, 2), _p(1, 13)), (_p(14, 13), _p(14, 2))),
        team_flags=(_p(3, 7), _p(12, 8)),
    ),
    "central_wall": _make_template(
        "central_wall",
        [
            "........",
            "...##...",
            "..#..#..",
            "........",
            "...##...",
            "........",
            "..#..#..",
            "...##...",
        ],
        team_spawns=((_p(1, 2), _p(1, 13)), (_p(14, 13), _p(14, 2))),
        team_flags=(_p(3, 7), _p(12, 8)),
    ),
    "lanes": _make_template(
        "lanes",
        [
            "........",
            ".#..#...",
            ".##..#..",
            "........",
            "...##...",
            "........",
            ".##..#..",
            ".#..#...",
        ],
        team_spawns=((_p(1, 2), _p(1, 13)), (_p(14, 13), _p(14, 2))),
        team_flags=(_p(3, 7), _p(12, 8)),
    ),
    "islands": _make_template(
        "islands",
        [
            "........",
            "..#...#.",
            "........",
            ".#.#....",
            "...##...",
            ".#.#....",
            "........",
            "..#...#.",
        ],
        team_spawns=((_p(1, 2), _p(1, 13)), (_p(14, 13), _p(14, 2))),
        team_flags=(_p(3, 7), _p(12, 8)),
    ),
    "maze": _make_template(
        "maze",
        [
            "........",
            ".###.#..",
            "...#..#.",
            ".#..#..#",
            "..##....",
            ".#..#..#",
            "...#..#.",
            ".###.#..",
        ],
        team_spawns=((_p(1, 2), _p(1, 13)), (_p(14, 13), _p(14, 2))),
        team_flags=(_p(3, 7), _p(12, 8)),
    ),
}


def get_map(name: str) -> MapTemplate:
    try:
        return MAPS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown map '{name}'. Available maps: {', '.join(sorted(MAPS))}"
        ) from exc
