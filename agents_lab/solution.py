from collections import deque
import json
from pathlib import Path

WAREHOUSE = (
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def endpoints(grid):
    if not grid or any(set(row) - set("# .SG") for row in grid):
        raise ValueError("Invalid warehouse symbols")
    positions = {symbol: [(r, c) for r, row in enumerate(grid)
                          for c, value in enumerate(row) if value == symbol]
                 for symbol in "SG"}
    if any(len(value) != 1 for value in positions.values()):
        raise ValueError("Expected one S and one G")
    return positions["S"][0], positions["G"][0]


def neighbours(grid, position):
    for dr, dc in DIRECTIONS:
        r, c = position[0] + dr, position[1] + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]) and grid[r][c] in ".SG":
            yield r, c


def navigate(grid):
    start, goal = endpoints(grid)
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        current = frontier.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return {"path": path[::-1], "length": len(path) - 1,
                    "expanded": expanded, "found": True}
        expanded += 1
        for successor in neighbours(grid, current):
            if successor not in parent:
                parent[successor] = current
                frontier.append(successor)
    return {"path": [], "length": None, "expanded": expanded, "found": False}


def validate(grid, result):
    if not result["found"]:
        assert not result["path"] and result["length"] is None
        return
    start, goal = endpoints(grid)
    path = result["path"]
    assert path[0] == start and path[-1] == goal
    assert result["length"] == len(path) - 1
    assert all(b in tuple(neighbours(grid, a)) for a, b in zip(path, path[1:]))


def main():
    cases = {
        "original": WAREHOUSE,
        "adjacent": ("#####", "#SG##", "#####"),
        "blocked": ("#####", "#S#G#", "#####"),
        "alternative_paths": ("#######", "#S....#", "#.....#", "#....G#", "#######"),
    }
    results = {name: navigate(grid) for name, grid in cases.items()}
    for name, grid in cases.items():
        validate(grid, results[name])
    assert results["original"]["found"]
    assert results["adjacent"]["length"] == 1
    assert not results["blocked"]["found"]
    assert results["alternative_paths"]["length"] == 6
    original = results["original"]
    overlay = [list(row) for row in WAREHOUSE]
    for r, c in original["path"][1:-1]:
        overlay[r][c] = "*"
    folder = Path(__file__).resolve().parent
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (folder / "path.txt").write_text("\n".join("".join(row) for row in overlay) + "\n")
    print(json.dumps(results, indent=2))
    print("All four navigation tests passed.")


if __name__ == "__main__":
    main()
