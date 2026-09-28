"""
tactics.py: space-time reasoning for Bomberman.

Pure lookup module: answers "is tile (x, y) lethal at time t", "is there a
safe trajectory from here", that kind of thing. No action selection here,
that's the learned Q-function in callbacks.py, which just treats these
functions as features.

Time convention:
    t = 0   game_state as given, before our action
    t = k   world after our k-th action and the environment step has
            resolved (bomb timers down, blasts applied)

For a bomb in game_state['bombs'] given as ((x, y), c):
    t_detonate = c + TIMING.fuse_offset
and it's lethal for TIMING.blast_steps slices starting at t_detonate.

A bomb we place at t=0 is observed at t=1 with countdown
TIMING.fresh_countdown, so it detonates at fresh_countdown + fuse_offset.

game_state['explosion_map'][x, y] == n means the tile is lethal for
t in [1, n]: shifted by one from the usual convention because the
engine checks kills after agents move, so a currently-1 tile is safe now
but fatal next step. In practice this only ever seems to be 0 or 1
(explosion_map stores explosion.timer - 1, EXPLOSION_TIMER = 2).
"""

from __future__ import annotations

from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np

# settings resolved at import time rather than hardcoded since the
# organisers can still edit settings.py before the deadline

_DEFAULTS = dict(
    COLS=17, ROWS=17,
    BOMB_POWER=3, BOMB_TIMER=4, EXPLOSION_TIMER=2,
    MAX_STEPS=400,
    REWARD_KILL=5, REWARD_COIN=1,
)


def _load_settings():
    # falls back to _DEFAULTS if settings.py isn't importable (e.g. tests
    # run outside the framework)
    cfg = dict(_DEFAULTS)
    try:
        import settings as s  # only importable from inside the framework
        for key in cfg:
            if hasattr(s, key):
                cfg[key] = getattr(s, key)
    except Exception:
        pass
    return cfg


S = _load_settings()

# used to weight kill-vs-coin trade-offs in reward shaping elsewhere
KILL_COIN_RATIO = float(S["REWARD_KILL"]) / max(1.0, float(S["REWARD_COIN"]))

# order must match the framework's ACTIONS list: indices double as
# array indices in callbacks.py
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVES = ["UP", "RIGHT", "DOWN", "LEFT"]
STEP_ACTIONS = MOVES + ["WAIT"]  # subset that can appear as a BFS transition

# x increases rightward, y increases downward, so UP decrements y - easy
# to get backwards, check against field.shape if unsure
DELTA = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}


@dataclass
class Timing:
    """Engine timing offsets."""
    fuse_offset: int = 1
    blast_steps: int = int(S["EXPLOSION_TIMER"])
    expl_offset: int = 0
    # a bomb placed now has countdown = BOMB_TIMER on the *next*
    # observation, not decremented until the end of the step it was
    # placed in a bomb placed at step 1 detonates at step 5
    fresh_countdown: int = int(S["BOMB_TIMER"])

    @property
    def horizon(self) -> int:
        """Time slices needed to resolve every currently-live bomb."""
        return int(S["BOMB_TIMER"]) + self.fuse_offset + self.blast_steps + 1


TIMING = Timing()

Escape = namedtuple(
    "Escape",
    ["safe", "min_steps", "safe_first", "n_safe_tiles", "first_steps"],
)


def blast_coords(field, x, y, power=None):
    """Tiles covered by a bomb placed at (x, y).

    Matches the engine's get_blast_coords: stopped by stone walls
    (field == -1), propagates through crates (field == 1).
    """
    if power is None:
        power = int(S["BOMB_POWER"])
    w, h = field.shape
    out = [(x, y)]
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if field[nx, ny] == -1:
                break
            out.append((nx, ny))
    return out


def danger_map(field, bombs=(), explosion_map=None, extra_bombs=(),
               horizon=None, timing=None):
    """Space-time lethality over the given horizon.

    Returns a (horizon, W*H) uint8 array where [t, p] == 1 means tile p is
    lethal at time t. Flattened as x * H + y, matching field's layout.

    extra_bombs is for counterfactuals ("what if I bomb here right now")
    without touching game_state, used by can_kill below. Entries are
    ((x, y), countdown).
    """
    tm = timing or TIMING
    horizon = horizon or tm.horizon
    w, h = field.shape
    n = w * h
    lethal = np.zeros((horizon, n), dtype=np.uint8)

    for (bx, by), countdown in list(bombs) + list(extra_bombs):
        t0 = int(countdown) + tm.fuse_offset
        if t0 >= horizon:
            continue  # detonates outside our window
        coords = blast_coords(field, bx, by)
        flat = [x * h + y for (x, y) in coords]
        for t in range(t0, min(t0 + tm.blast_steps, horizon)):
            lethal[t, flat] = 1

    if explosion_map is not None:
        # residual blasts already in progress, given directly rather than
        # derived from a countdown
        for x, y in np.argwhere(np.asarray(explosion_map) > 0):
            until = int(explosion_map[x, y]) + tm.expl_offset
            if until > 0:
                lethal[1:min(until + 1, horizon), x * h + y] = 1

    return lethal


def walkable_map(field, bombs=(), blocked=()):
    """Occupancy mask of tiles a body could currently stand on.

    Bomb tiles and other agents' bodies are excluded on top of the
    static field geometry.
    """
    w, h = field.shape
    free = bytearray((np.asarray(field) == 0).astype(np.uint8).tobytes())
    for (bx, by), _ in bombs:
        free[bx * h + by] = 0
    for (bx, by) in blocked:
        free[bx * h + by] = 0
    return free


def _clean_from(lethal):
    """For each tile, earliest time after which it's lethal-free for the
    rest of the horizon (0 if safe throughout)."""
    horizon = lethal.shape[0]
    ever = lethal.any(axis=0)
    last = horizon - 1 - np.argmax(lethal[::-1], axis=0)
    return np.where(ever, last + 1, 0).astype(np.int16)


def survivable(start, lethal, free, field_shape, timing=None):
    """Can a body at `start` reach a tile it can occupy indefinitely
    (a permanently safe tile) under the given danger map?

    Implemented as one BFS over (x, y, t, first action) instead of four
    separate searches, so results share the visited set and total work
    stays O(W*H*horizon).

    Returns an Escape namedtuple:
        safe          bool: at least one action leads to survival
        min_steps     int: steps to nearest permanently-safe tile
                      (0 if start already qualifies), or -1
        safe_first    dict: action -> bool, keyed by first action taken
        n_safe_tiles  int: distinct safe tiles reachable
        first_steps   dict: action -> steps to safety via that action, or -1
    """
    tm = timing or TIMING
    w, h = field_shape
    n = w * h
    horizon = lethal.shape[0]
    clean = _clean_from(lethal)

    sx, sy = start
    p0 = sx * h + sy

    # standing still on the start tile is always legal, even if `free`
    # marks it occupied (e.g. by our own just-dropped bomb)
    free = bytearray(free)
    free[p0] = 1

    rows = [lethal[t].tobytes() for t in range(horizon)]

    safe_first = {a: False for a in STEP_ACTIONS}
    first_steps = {a: -1 for a in STEP_ACTIONS}
    safe_tiles = set()
    min_steps = -1

    if clean[p0] == 0:                      # start tile already safe forever
        min_steps = 0
        safe_tiles.add(p0)

    # visited indexed by (first action, tile, time) so trajectories with
    # different first moves are tracked independently
    visited = bytearray(len(STEP_ACTIONS) * n * horizon)
    queue = deque()

    for ai, action in enumerate(STEP_ACTIONS):
        dx, dy = DELTA[action]
        nx, ny = sx + dx, sy + dy
        if not (0 <= nx < w and 0 <= ny < h):
            continue
        p = nx * h + ny
        if not free[p]:
            continue
        if horizon > 1 and rows[1][p]:
            continue
        key = (ai * n + p) * horizon + 1
        if visited[key]:
            continue
        visited[key] = 1
        queue.append((ai, p, 1))
        if clean[p] <= 1:
            safe_first[action] = True
            first_steps[action] = 1
            safe_tiles.add(p)
            if min_steps < 0:
                min_steps = 1

    while queue:
        ai, p, t = queue.popleft()
        if t + 1 >= horizon:
            continue  # nothing left in the horizon to extend into
        x, y = divmod(p, h)
        for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            q = nx * h + ny
            if not free[q] or rows[t + 1][q]:
                continue
            key = (ai * n + q) * horizon + (t + 1)
            if visited[key]:
                continue
            visited[key] = 1
            queue.append((ai, q, t + 1))
            if clean[q] <= t + 1:
                action = STEP_ACTIONS[ai]
                safe_tiles.add(q)
                if not safe_first[action]:
                    safe_first[action] = True
                    first_steps[action] = t + 1
                if min_steps < 0:
                    min_steps = t + 1

    return Escape(
        safe=any(safe_first.values()) or min_steps == 0,
        min_steps=min_steps,
        safe_first=safe_first,
        n_safe_tiles=len(safe_tiles),
        first_steps=first_steps,
    )


THREAT_INF = 99


def threat_map(field, others, bombs=(), reach=3, timing=None):
    """Earliest time each tile could be turned lethal by an opponent
    placing a new bomb.

    For each opponent still holding a bomb, flood-fill tiles reachable
    within `reach` steps (worst-case bound, not a claim they'd actually
    survive getting there), assume detonation on arrival. Taking the
    min over opponents gives a conditional threat: a dead-end corridor
    is harmless while empty and a trap once someone's close enough to
    seal it.

    Returns int16 array shaped like `field`; THREAT_INF marks tiles with
    no credible threat within `reach`.
    """
    tm = timing or TIMING
    w, h = field.shape
    out = np.full(w * h, THREAT_INF, dtype=np.int16)
    if not others:
        return out.reshape(w, h)

    free = walkable_map(field, bombs=bombs)
    detonation_delay = tm.fresh_countdown + tm.fuse_offset

    for other in others:
        _, _, bomb_available, (ox, oy) = other
        if not bomb_available:
            continue
        # plain BFS, ignoring danger to the opponent upper bound on
        # their reach, not a realistic plan for them
        dist = {ox * h + oy: 0}
        queue = deque([(ox, oy, 0)])
        while queue:
            x, y, d = queue.popleft()
            if d >= reach:
                continue
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                p = nx * h + ny
                if not free[p] or p in dist:
                    continue
                dist[p] = d + 1
                queue.append((nx, ny, d + 1))

        for p, d in dist.items():
            x, y = divmod(p, h)
            t_lethal = d + detonation_delay
            for (bx, by) in blast_coords(field, x, y):
                q = bx * h + by
                if t_lethal < out[q]:
                    out[q] = t_lethal  # earliest threat across opponents

    return out.reshape(w, h)


def can_kill(field, my_pos, opp_pos, bombs=(), explosion_map=None,
             other_bodies=(), timing=None):
    """Would placing a bomb at my_pos guarantee opp_pos has no escape?

    Opponent modelled with unrestricted movement, subject only to blast
    geometry: returns True only when every trajectory they have is
    eventually lethal (worst case, sound but can be over-cautious).
    """
    tm = timing or TIMING
    lethal = danger_map(
        field, bombs, explosion_map,
        extra_bombs=[(tuple(my_pos), tm.fresh_countdown)], timing=tm,
    )
    blockers = [p for p in other_bodies if tuple(p) != tuple(opp_pos)]
    blockers.append(tuple(my_pos))       # our body plus the bomb we'd place
    free = walkable_map(field, bombs=bombs, blocked=blockers)
    esc = survivable(opp_pos, lethal, free, field.shape, timing=tm)
    return not esc.safe


BFS_INF = 255


def bfs_distances(free, sources, field_shape):
    """Multi-source, unweighted BFS over walkable tiles.

    Returns a (W, H) uint8 array of shortest-path distances from the
    nearest source; BFS_INF marks unreachable tiles.
    """
    w, h = field_shape
    dist = np.full(w * h, BFS_INF, dtype=np.uint8)
    queue = deque()
    for (x, y) in sources:
        if not (0 <= x < w and 0 <= y < h):
            continue
        p = x * h + y
        if dist[p] == 0:
            continue
        dist[p] = 0
        queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        d = dist[x * h + y]
        if d >= BFS_INF - 1:
            continue  # would overflow uint8 on the next step
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            q = nx * h + ny
            if not free[q] or dist[q] != BFS_INF:
                continue
            dist[q] = d + 1
            queue.append((nx, ny))
    return dist.reshape(w, h)


def crate_seed_tiles(field):
    """Free tiles next to at least one crate: candidates for a
    crate-clearing bomb."""
    w, h = field.shape
    seeds = []
    for x in range(w):
        for y in range(h):
            if field[x, y] != 0:
                continue
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 1:
                    seeds.append((x, y))
                    break
    return seeds


def tile_degree(field, x, y):
    """Count of free orthogonal neighbours (1 = dead end / corridor
    terminus)."""
    w, h = field.shape
    deg = 0
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 0:
            deg += 1
    return deg
