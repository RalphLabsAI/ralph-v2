"""Deterministic key-and-door gridworld — a multi-turn env for the exposure-bias test.

Why this env exists: the exposure-bias / self_state slice needs a MULTI-TURN environment
where (a) each turn is a genuine action (native step boundary, so re-prompting continues
rather than restarts), and (b) the correct next action depends on ACCUMULATED state, so a
wrong early action puts the agent in a state where later optimal actions diverge — drift
compounds with horizon. That is exactly the regime const's design targets and the CoT pile
could not provide.

It is PURE and DETERMINISTIC: `step(state, action)` is a function, seeded from the round
commit; no code execution, no sandbox, no network. The oracle (BFS over the key/door state
space) gives the optimal next action from ANY state, so the "did the student do what the
teacher would do here" check is deterministic — no LLM judge needed in the crown path.

The grid is a tuple of tuples of CELL TOKENS: '.' floor, '#' wall, 'K<c>' key of colour c,
'D<c>' door of colour c (passable only while holding key c), 'G' goal. Actions:
north|south|east|west|take. Observation after each action = a compact view (position,
cell, 4 neighbours, inventory).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

ACTIONS = ("north", "south", "east", "west", "take")
_DELTA = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


@dataclass(frozen=True)
class GridState:
    grid: tuple           # tuple of tuples of cell-token strings (immutable)
    pos: tuple            # (x, y)
    keys: frozenset       # colours held
    goal: tuple           # (x, y) of 'G'
    steps: int = 0

    def cell(self, x: int, y: int) -> str:
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[y]):
            return self.grid[y][x]
        return "#"

    @property
    def solved(self) -> bool:
        return self.pos == self.goal


def _passable(state: GridState, x: int, y: int) -> bool:
    c = state.cell(x, y)
    if c == "#":
        return False
    if c.startswith("D"):
        return c[1] in state.keys   # door passable only while holding its key colour
    return True


def _set_cell(grid: tuple, pos: tuple, token: str) -> tuple:
    x, y = pos
    row = grid[y]
    return grid[:y] + (row[:x] + (token,) + row[x + 1:],) + grid[y + 1:]


def step(state: GridState, action: str) -> tuple[GridState, str, bool]:
    """(state, action) -> (state', observation, done). Illegal/blocked moves are no-ops
    with an explanatory observation (so the agent must reason, not brute-force)."""
    a = (action or "").strip().lower().split()[0] if action and action.strip() else ""
    if a not in ACTIONS:
        return replace(state, steps=state.steps + 1), f"unknown action '{action}'. valid: {', '.join(ACTIONS)}", False
    if a == "take":
        c = state.cell(*state.pos)
        if c.startswith("K"):
            ns = replace(state, keys=state.keys | {c[1]},
                         grid=_set_cell(state.grid, state.pos, "."), steps=state.steps + 1)
            return ns, _obs(ns, f"took key {c[1]}"), ns.solved
        return replace(state, steps=state.steps + 1), _obs(state, "nothing to take here"), False
    dx, dy = _DELTA[a]
    nx, ny = state.pos[0] + dx, state.pos[1] + dy
    if not _passable(state, nx, ny):
        blocker = state.cell(nx, ny)
        why = "a wall" if blocker == "#" else f"locked door {blocker[1:]}"
        return replace(state, steps=state.steps + 1), _obs(state, f"blocked by {why}"), False
    ns = replace(state, pos=(nx, ny), steps=state.steps + 1)
    return ns, _obs(ns, f"moved {a}"), ns.solved


def _obs(state: GridState, note: str) -> str:
    x, y = state.pos
    nb = {d: state.cell(x + _DELTA[d][0], y + _DELTA[d][1]) for d in ("north", "south", "east", "west")}
    inv = "".join(sorted(state.keys)) or "none"
    here = state.cell(x, y)
    return (f"{note}. pos=({x},{y}) on '{here}'. "
            f"north='{nb['north']}' south='{nb['south']}' east='{nb['east']}' west='{nb['west']}'. "
            f"keys={inv}. goal at ({state.goal[0]},{state.goal[1]}).")


def observe(state: GridState) -> str:
    return _obs(state, "current")


def parse_action(text: str) -> str:
    """Extract the first valid action word from a model's turn (tolerant of chatter)."""
    low = (text or "").lower()
    # earliest-occurring valid action token wins
    best, best_at = "", len(low) + 1
    for a in ACTIONS:
        i = low.find(a)
        if 0 <= i < best_at:
            best, best_at = a, i
    return best  # "" if none found -> env treats as unknown action (a no-op)


# -------------------------------------------------------------------------
# Oracle — optimal next action from ANY state (BFS over (pos, keys)).
# -------------------------------------------------------------------------

def oracle_action(state: GridState) -> str | None:
    """The action a perfect agent takes next. None if already solved / unsolvable."""
    if state.solved:
        return None
    start = (state.pos, state.keys)
    seen = {start}
    q = deque([(state, [])])
    while q:
        s, path = q.popleft()
        if s.solved:
            return path[0] if path else None
        for a in ACTIONS:
            ns, _, _ = step(s, a)
            key = (ns.pos, ns.keys)
            if key in seen:
                continue
            seen.add(key)
            q.append((ns, path + [a]))
    return None


def oracle_distance(state: GridState) -> int | None:
    """Fewest steps to the goal from this state (BFS over (pos, keys)). None if stuck."""
    if state.solved:
        return 0
    seen = {(state.pos, state.keys)}
    q = deque([(state, 0)])
    while q:
        s, d = q.popleft()
        for a in ACTIONS:
            ns, _, done = step(s, a)
            if done:
                return d + 1
            key = (ns.pos, ns.keys)
            if key in seen:
                continue
            seen.add(key)
            q.append((ns, d + 1))
    return None


def oracle_optimal_actions(state: GridState) -> set[str]:
    """The SET of actions that strictly reduce distance-to-goal — the deterministic
    'what the teacher would do here' checker, tolerant of optimal ties (several shortest
    paths). An action that stalls or backtracks is not optimal."""
    d0 = oracle_distance(state)
    if not d0:
        return set()
    best = set()
    for a in ACTIONS:
        ns, _, done = step(state, a)
        if done:
            best.add(a)
            continue
        dn = oracle_distance(ns)
        if dn is not None and dn < d0:
            best.add(a)
    return best


def oracle_solution(state: GridState, max_steps: int = 60) -> list[str]:
    """Full optimal action sequence to the goal (for pile authoring / a perfect agent)."""
    acts, s = [], state
    for _ in range(max_steps):
        a = oracle_action(s)
        if a is None:
            break
        s, _, done = step(s, a)
        acts.append(a)
        if done:
            break
    return acts


# -------------------------------------------------------------------------
# Task generation — seeded, with a guaranteed key->door->goal dependency so
# early drift compounds. Difficulty scales the grid + number of locked doors.
# -------------------------------------------------------------------------

def make_task(seed: int, difficulty: int = 1, _tries: int = 0) -> tuple[GridState, str]:
    """Return (start_state, task_text). Deterministic in seed. Grids are genuinely DIVERSE
    (random size within difficulty, random internal walls, random key/door/goal/start
    placement) so different seeds give different oracle paths and state sequences — a
    templated pile collapses teacher_state variance. A key->door->goal dependency is
    guaranteed (door wall splits key-side from goal-side) so early drift compounds.
    Difficulty 1..3 grows the grid, wall density, and colour count."""
    import random
    rng = random.Random(seed * 100003 + _tries)
    w = rng.randint(6 + difficulty, 8 + 2 * difficulty)
    h = rng.randint(3 + difficulty, 4 + difficulty)
    colours = ["r", "b", "g"][:difficulty]

    grid = [["#"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            grid[y][x] = "."

    start = (1, rng.randint(1, h - 2))
    goal = (w - 2, rng.randint(1, h - 2))
    if goal == start:
        return make_task(seed, difficulty, _tries + 1)
    grid[goal[1]][goal[0]] = "G"

    # one door wall per colour, left-to-right, each with its key on the LEFT (start) side
    door_xs = sorted(rng.sample(range(2, w - 2), len(colours))) if w - 4 >= len(colours) else [w // 2]
    for col, dx in zip(colours, door_xs):
        for y in range(1, h - 1):
            if grid[y][dx] == ".":
                grid[y][dx] = "#"
        door_y = rng.randint(1, h - 2)
        grid[door_y][dx] = "D" + col
        # key strictly left of this door and left of the goal
        for _ in range(80):
            kx, ky = rng.randint(1, dx - 1), rng.randint(1, h - 2)
            if (kx, ky) != start and grid[ky][kx] == ".":
                grid[ky][kx] = "K" + col
                break
        else:
            return make_task(seed, difficulty, _tries + 1)

    # a few random internal walls for path variety (never on start/goal/keys/doors)
    for _ in range(rng.randint(0, difficulty + 1)):
        wx, wy = rng.randint(1, w - 2), rng.randint(1, h - 2)
        if grid[wy][wx] == "." and (wx, wy) not in (start, goal):
            grid[wy][wx] = "#"

    gs = GridState(grid=tuple(tuple(r) for r in grid), pos=start, keys=frozenset(), goal=goal)
    sol = oracle_solution(gs)
    if not sol or len(sol) < 4 + 2 * difficulty:   # require a non-trivial path
        if _tries < 40:
            return make_task(seed, difficulty, _tries + 1)
        return make_task(seed + 1, difficulty)
    task = (f"You are in a gridworld. Reach the goal 'G'. Doors 'D<c>' are locked until "
            f"you stand on the matching key 'K<c>' and take it. One action per turn from: "
            f"{', '.join(ACTIONS)}. Reply with ONLY the single next action word.\n"
            f"{observe(gs)}")
    return gs, task
