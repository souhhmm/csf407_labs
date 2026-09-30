from collections import deque
import heapq
from itertools import count
import json
import math
from pathlib import Path

WAREHOUSE = (
    "#################", "#S....#.........#", "#.###.#.#######.#",
    "#...#.#.......#.#", "###.#.#######.#.#", "#...#.........#.#",
    "#.###########.#.#", "#.............#G#", "#################",
)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def endpoints(grid):
    if not grid or any(not row or set(row) - set("#.SG") for row in grid):
        raise ValueError("Invalid map")
    positions = {s: [(r, c) for r, row in enumerate(grid)
                     for c, value in enumerate(row) if value == s] for s in "SG"}
    if any(len(p) != 1 for p in positions.values()):
        raise ValueError("Expected exactly one start and goal")
    return positions["S"][0], positions["G"][0]


def neighbours(grid, current):
    for dr, dc in DIRECTIONS:
        r, c = current[0] + dr, current[1] + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]) and grid[r][c] in ".SG":
            yield r, c


def finish(parent, current, expanded):
    path = []
    while current is not None:
        path.append(current)
        current = parent[current]
    return {"found": True, "path": path[::-1], "length": len(path) - 1,
            "expanded": expanded}


def failure(expanded):
    return {"found": False, "path": [], "length": None, "expanded": expanded}


def bfs(grid):
    start, goal = endpoints(grid)
    frontier, parent = deque([start]), {start: None}
    expanded = 0
    while frontier:
        current = frontier.popleft()
        if current == goal:
            return finish(parent, current, expanded)
        expanded += 1
        for successor in neighbours(grid, current):
            if successor not in parent:
                parent[successor] = current
                frontier.append(successor)
    return failure(expanded)


def heuristic(position, goal, kind):
    dr, dc = abs(position[0] - goal[0]), abs(position[1] - goal[1])
    return {"zero": 0, "manhattan": dr + dc,
            "euclidean": math.hypot(dr, dc), "twice_manhattan": 2 * (dr + dc)}[kind]


def astar(grid, kind="manhattan"):
    start, goal = endpoints(grid)
    serial = count()
    frontier = [(heuristic(start, goal, kind), next(serial), 0, start)]
    distance, parent = {start: 0}, {start: None}
    expanded = 0
    while frontier:
        _, _, g, current = heapq.heappop(frontier)
        if g != distance[current]:
            continue  # a better route has replaced this heap entry.
        if current == goal:
            return finish(parent, current, expanded)
        expanded += 1
        for successor in neighbours(grid, current):
            candidate = g + 1
            if candidate < distance.get(successor, math.inf):
                distance[successor] = candidate
                parent[successor] = current
                priority = candidate + heuristic(successor, goal, kind)
                heapq.heappush(frontier, (priority, next(serial), candidate, successor))
    return failure(expanded)


def validate(grid, result):
    if result["found"]:
        start, goal = endpoints(grid)
        path = result["path"]
        assert (path[0], path[-1]) == (start, goal)
        assert len(path) - 1 == result["length"]
        assert all(b in tuple(neighbours(grid, a)) for a, b in zip(path, path[1:]))
    else:
        assert not result["path"] and result["length"] is None


def main():
    cases = {
        "original": WAREHOUSE,
        "adjacent": ("#####", "#SG##", "#####"),
        "blocked": ("#######", "#S....#", "###.###", "#...#G#", "#######"),
        "alternative_paths": ("#######", "#S....#", "#.###.#", "#....G#", "#######"),
    }
    algorithms = {"BFS": bfs, **{kind: (lambda grid, kind=kind: astar(grid, kind))
                  for kind in ("manhattan", "zero", "euclidean", "twice_manhattan")}}
    results = {}
    for name, grid in cases.items():
        results[name] = {label: algorithm(grid) for label, algorithm in algorithms.items()}
        reference = results[name]["BFS"]
        for label, result in results[name].items():
            validate(grid, result)
            assert result["found"] == reference["found"]
            if label != "twice_manhattan":
                assert result["length"] == reference["length"]
    assert results["original"]["BFS"]["found"]
    assert results["adjacent"]["manhattan"]["length"] == 1
    assert not results["blocked"]["manhattan"]["found"]
    assert results["alternative_paths"]["manhattan"]["length"] == 6
    folder = Path(__file__).resolve().parent
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    overlay = [list(row) for row in WAREHOUSE]
    for r, c in results["original"]["manhattan"]["path"][1:-1]:
        overlay[r][c] = "*"
    (folder / "path.txt").write_text("\n".join("".join(row) for row in overlay) + "\n")
    for name, row in results.items():
        print(name, {key: {k: v for k, v in value.items() if k != "path"}
                     for key, value in row.items()})
    print("All navigation, reachability, and shortest-path checks passed.")


if __name__ == "__main__":
    main()
