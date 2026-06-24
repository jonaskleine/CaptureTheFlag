from __future__ import annotations

from .models import MapTemplate, Position


def _p(x: int, y: int) -> Position:
    return Position(x, y)


def _sym(left_half: str) -> str:
    if len(left_half) != 7:
        raise ValueError("symmetrical rows must use a 7-character left half")
    return left_half + left_half[::-1]


MAPS: dict[str, MapTemplate] = {
    "open_field": MapTemplate.from_ascii(
        "open_field",
        [
            _sym("......."),
            _sym("......."),
            _sym("......."),
            _sym("......."),
            _sym("......."),
            _sym("......."),
            _sym("......."),
            _sym("......."),
            _sym("......."),
        ],
        team_spawns=((_p(1, 1), _p(1, 7)), (_p(12, 1), _p(12, 7))),
        team_flags=(_p(2, 4), _p(11, 4)),
    ),
    "central_wall": MapTemplate.from_ascii(
        "central_wall",
        [
            _sym("......."),
            _sym("...#.#."),
            _sym("..#...#"),
            _sym("......."),
            _sym(".##...#"),
            _sym("......."),
            _sym("..#...#"),
            _sym("...#.#."),
            _sym("......."),
        ],
        team_spawns=((_p(1, 1), _p(1, 7)), (_p(12, 1), _p(12, 7))),
        team_flags=(_p(2, 4), _p(11, 4)),
    ),
    "lanes": MapTemplate.from_ascii(
        "lanes",
        [
            _sym("......."),
            _sym(".#..#.."),
            _sym(".#..##."),
            _sym("......."),
            _sym("...##.."),
            _sym("......."),
            _sym(".#..##."),
            _sym(".#..#.."),
            _sym("......."),
        ],
        team_spawns=((_p(1, 1), _p(1, 7)), (_p(12, 1), _p(12, 7))),
        team_flags=(_p(2, 4), _p(11, 4)),
    ),
    "islands": MapTemplate.from_ascii(
        "islands",
        [
            _sym("......."),
            _sym("..#...#"),
            _sym("......."),
            _sym(".#.#..."),
            _sym("...##.."),
            _sym(".#.#..."),
            _sym("......."),
            _sym("..#...#"),
            _sym("......."),
        ],
        team_spawns=((_p(1, 1), _p(1, 7)), (_p(12, 1), _p(12, 7))),
        team_flags=(_p(2, 4), _p(11, 4)),
    ),
    "maze": MapTemplate.from_ascii(
        "maze",
        [
            _sym("......."),
            _sym(".###.#."),
            _sym("...#..#"),
            _sym(".#..#.."),
            _sym("...##.."),
            _sym(".#..#.."),
            _sym("...#..#"),
            _sym(".###.#."),
            _sym("......."),
        ],
        team_spawns=((_p(1, 1), _p(1, 7)), (_p(12, 1), _p(12, 7))),
        team_flags=(_p(2, 4), _p(11, 4)),
    ),
}


def get_map(name: str) -> MapTemplate:
    try:
        return MAPS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown map '{name}'. Available maps: {', '.join(sorted(MAPS))}"
        ) from exc
