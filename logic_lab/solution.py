from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Action:
    name: str
    positive: frozenset[str]
    negative: frozenset[str]
    add: frozenset[str]
    delete: frozenset[str]

    def applicable(self, state):
        return self.positive <= state and not (self.negative & state)

    def apply(self, state):
        if not self.applicable(state):
            raise ValueError(f"Preconditions not satisfied: {self.name}")
        return frozenset((state - self.delete) | self.add)


def fact(entity, location):
    return f"At({entity},{location})"


def warehouse_actions():
    actions = []
    for source, target in (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")):
        actions.append(Action(f"Move({source},{target})", frozenset([fact("Robot", source)]),
                              frozenset(), frozenset([fact("Robot", target)]),
                              frozenset([fact("Robot", source)])))
    for location in "ABC":
        actions.append(Action(f"PickUp(Package,{location})",
                              frozenset([fact("Robot", location), fact("Package", location)]),
                              frozenset(["Holding(Package)"]), frozenset(["Holding(Package)"]),
                              frozenset([fact("Package", location)])))
        actions.append(Action(f"Drop(Package,{location})",
                              frozenset([fact("Robot", location), "Holding(Package)"]),
                              frozenset(), frozenset([fact("Package", location)]),
                              frozenset(["Holding(Package)"])))
    return tuple(actions)


def plan(initial, actions, goal):
    initial = frozenset(initial)
    frontier, parent = deque([initial]), {initial: None}
    while frontier:
        current = frontier.popleft()
        if goal <= current:
            answer = []
            while parent[current] is not None:
                previous, action = parent[current]
                answer.append(action)
                current = previous
            return answer[::-1]
        for action in actions:
            if action.applicable(current):
                successor = action.apply(current)
                if successor not in parent:
                    parent[successor] = (current, action)
                    frontier.append(successor)
    return None


def verify(initial, actions, goal):
    state = frozenset(initial)
    states = [sorted(state)]
    for action in actions:
        assert action.applicable(state)
        state = action.apply(state)
        assert sum(fact("Robot", loc) in state for loc in "ABC") == 1
        assert sum(fact("Package", loc) in state for loc in "ABC") + ("Holding(Package)" in state) == 1
        states.append(sorted(state))
    assert goal <= state
    return states


def main():
    initial = frozenset([fact("Robot", "A"), fact("Package", "A")])
    goal = frozenset([fact("Package", "C")])
    actions = warehouse_actions()
    manual_names = ("PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)")
    manual = [next(a for a in actions if a.name == name) for name in manual_names]
    manual_states = verify(initial, manual, goal)
    cases = {
        "original": actions,
        "no_pickup": tuple(a for a in actions if not a.name.startswith("PickUp")),
        "irrelevant_action": actions + (Action("Wait", frozenset(), frozenset(),
                                             frozenset(), frozenset()),),
    }
    results = {}
    for name, available in cases.items():
        answer = plan(initial, available, goal)
        results[name] = {"initial": sorted(initial), "goal": sorted(goal),
                         "found": answer is not None,
                         "plan": [a.name for a in answer] if answer is not None else [],
                         "states": verify(initial, answer, goal) if answer is not None else []}
    assert len(results["original"]["plan"]) == 4
    assert not results["no_pickup"]["found"]
    assert len(results["irrelevant_action"]["plan"]) == 4
    robot_only = [a for a in actions if a.name.startswith("Move")]
    assert plan(initial, robot_only, goal) is None
    assert plan(initial, actions, initial) == []
    forbidden = Action("Forbidden", frozenset(), frozenset([fact("Robot", "A")]),
                       frozenset(["Invalid"]), frozenset())
    assert not forbidden.applicable(initial)
    try:
        forbidden.apply(initial)
    except ValueError:
        pass
    else:
        raise AssertionError("Inapplicable action was executed")
    results["manual_plan"] = {"plan": list(manual_names), "states": manual_states}
    results["initially_applicable"] = [a.name for a in actions if a.applicable(initial)]
    folder = Path(__file__).resolve().parent
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("All planning, replay, invariant, and precondition checks passed.")


if __name__ == "__main__":
    main()
