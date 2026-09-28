"""
tactics.py -- space-time reasoning primitives for Bomberman.

Everything here is *descriptive*: it answers questions about the board
("could this tile kill me at time t?", "does an escape exist?").  Nothing here
ever picks an action.  Action selection happens exclusively as an argmax over a
learned Q-function in callbacks.py.

--------------------------------------------------------------------------
TIME CONVENTION  (read this before touching anything)
--------------------------------------------------------------------------
t = 0   the current game_state, before our action has been executed
t = k   the world after our k-th action has been executed *and* the
        environment has resolved that step (bombs ticked, explosions applied)

A bomb reported in game_state['bombs'] as ((x, y), c) detonates at

        t_detonate = c + TIMING.fuse_offset

and its blast tiles are lethal for `TIMING.blast_steps` consecutive time
slices starting at t_detonate.

A bomb *we drop now* enters the world with countdown `TIMING.fresh_countdown`
as seen in the next observation, so it detonates at
`TIMING.fresh_countdown + TIMING.fuse_offset`.

game_state['explosion_map'][x, y] == n marks a tile as lethal for
t in [1, n].  Note [1, n], NOT [0, n-1]: the engine evaluates kills *after*
the agents have moved, so a tile reported with value 1 is harmless to be
looking at and fatal to step onto.  In practice the engine only ever reports
0 or 1, because explosion_map stores `explosion.timer - 1` and the timer is
EXPLOSION_TIMER = 2 at creation.

>>> These offsets are MEASURED, not assumed.  All four were verified against
>>> the real engine on 22.08.2026; see tests/engine_timing_evidence.md for the
>>> probe transcripts.  Re-verify if the organisers change settings.py.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Settings are read at runtime.  The brief allows the organisers to change
# settings.py up to seven days before the deadline, so nothing is hardcoded.
# --------------------------------------------------------------------------

_DEFAULTS = dict(
    COLS=17, ROWS=17,
    BOMB_POWER=3, BOMB_TIMER=4, EXPLOSION_TIMER=2,
    MAX_STEPS=400,
    REWARD_KILL=5, REWARD_COIN=1,
)


def _load_settings():
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

KILL_COIN_RATIO = float(S["REWARD_KILL"]) / max(1.0, float(S["REWARD_COIN"]))

# --------------------------------------------------------------------------
# Actions.  Order matches the framework's ACTIONS list.
# --------------------------------------------------------------------------

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVES = ["UP", "RIGHT", "DOWN", "LEFT"]
STEP_ACTIONS = MOVES + ["WAIT"]  # actions that move a body during the escape BFS

# image coordinates: x to the right, y downwards -> UP is y - 1
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
    """Engine timing offsets.  Calibrate with agent_code/probe_timing/."""
    fuse_offset: int = 1
    blast_steps: int = int(S["EXPLOSION_TIMER"])
    expl_offset: int = 0
    # A bomb we drop now is created with timer = BOMB_TIMER and is only
    # decremented at the END of the step it was dropped in, so it behaves like
    # an observed bomb of countdown BOMB_TIMER, not BOMB_TIMER - 1.  Measured:
    # dropped on step 1, detonates on step 5, giving us exactly four moves.
    fresh_countdown: int = int(S["BOMB_TIMER"])

    @property
    def horizon(self) -> int:
        """Time slices we need to look ahead to resolve every live bomb."""
        return int(S["BOMB_TIMER"]) + self.fuse_offset + self.blast_steps + 1


TIMING = Timing()

Escape = namedtuple(
    "Escape",
    ["safe", "min_steps", "safe_first", "n_safe_tiles", "first_steps"],
)


# --------------------------------------------------------------------------
# 1. Blast geometry
# --------------------------------------------------------------------------

def blast_coords(field, x, y, power=None):
    """Tiles covered by a bomb at (x, y).

    The blast stops at stone walls (field == -1) but passes *through* crates
    (field == 1), matching the engine's get_blast_coords.
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


# --------------------------------------------------------------------------
# 2. Danger map:  lethal[t][x * H + y]
# --------------------------------------------------------------------------

def danger_map(field, bombs=(), explosion_map=None, extra_bombs=(),
               horizon=None, timing=None):
    """Space-time lethality.

    Returns a (horizon, W*H) uint8 array; entry 1 means "standing here at
    time t kills you".

    extra_bombs lets us ask counterfactual questions ("what if I bomb now?")
    without mutating the game state.  Each entry is ((x, y), countdown).
    """
    tm = timing or TIMING
    horizon = horizon or tm.horizon
    w, h = field.shape
    n = w * h
    lethal = np.zeros((horizon, n), dtype=np.uint8)

    for (bx, by), countdown in list(bombs) + list(extra_bombs):
        t0 = int(countdown) + tm.fuse_offset
        if t0 >= horizon:
            continue
        coords = blast_coords(field, bx, by)
        flat = [x * h + y for (x, y) in coords]
        for t in range(t0, min(t0 + tm.blast_steps, horizon)):
            lethal[t, flat] = 1

    if explosion_map is not None:
        for x, y in np.argwhere(np.asarray(explosion_map) > 0):
            until = int(explosion_map[x, y]) + tm.expl_offset
            if until > 0:
                lethal[1:min(until + 1, horizon), x * h + y] = 1

    return lethal


def walkable_map(field, bombs=(), blocked=()):
    """Tiles a body may occupy.  Bombs and other agents block movement."""
    w, h = field.shape
    free = bytearray((np.asarray(field) == 0).astype(np.uint8).tobytes())
    for (bx, by), _ in bombs:
        free[bx * h + by] = 0
    for (bx, by) in blocked:
        free[bx * h + by] = 0
    return free


def _clean_from(lethal):
    """clean[p] = earliest time from which tile p is never lethal again."""
    horizon = lethal.shape[0]
    ever = lethal.any(axis=0)
    last = horizon - 1 - np.argmax(lethal[::-1], axis=0)
    return np.where(ever, last + 1, 0).astype(np.int16)


# --------------------------------------------------------------------------
# 3. Escape BFS over (x, y, t)
# --------------------------------------------------------------------------

def survivable(start, lethal, free, field_shape, timing=None):
    """Can a body at `start` reach a tile it can permanently stand on?

    BFS over (x, y, t), pruning any state the danger map calls lethal.  Each
    search node carries the first action that produced it, so one search
    answers "am I safe?" and "is direction d safe?" simultaneously.

    Returns an Escape namedtuple:
        safe          bool   -- some action leads to survival
        min_steps     int    -- steps to the nearest permanently-safe tile
                                (0 if we may simply stand still), or -1
        safe_first    dict   -- action -> bool, per first action
        n_safe_tiles  int    -- distinct safe tiles reachable
        first_steps   dict   -- action -> steps to safety via that action, or -1
    """
    tm = timing or TIMING
    w, h = field_shape
    n = w * h
    horizon = lethal.shape[0]
    clean = _clean_from(lethal)

    sx, sy = start
    p0 = sx * h + sy

    # We may always remain where we already are, even if our own bomb sits
    # under us and makes the tile formally unwalkable.
    free = bytearray(free)
    free[p0] = 1

    rows = [lethal[t].tobytes() for t in range(horizon)]

    safe_first = {a: False for a in STEP_ACTIONS}
    first_steps = {a: -1 for a in STEP_ACTIONS}
    safe_tiles = set()
    min_steps = -1

    if clean[p0] == 0:                      # already permanently safe
        min_steps = 0
        safe_tiles.add(p0)

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
            continue
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


# --------------------------------------------------------------------------
# 4. Threat map -- what the *opponents* can do to us
# --------------------------------------------------------------------------

THREAT_INF = 99


def threat_map(field, others, bombs=(), reach=3, timing=None):
    """Earliest time each tile could be made lethal by an opponent's new bomb.

    For every opponent that still holds a bomb we flood-fill the tiles they can
    reach within `reach` steps, and for each such tile assume they bomb on
    arrival.  The union tells us which tiles are *conditionally* deadly: a
    dead-end corridor is harmless when nobody is near and a coffin when an
    opponent is three tiles from its mouth.

    Returns an int16 array of shape field.shape, THREAT_INF where unthreatened.
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
        # flood-fill reachable tiles, ignoring danger (worst case for us)
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
                    out[q] = t_lethal

    return out.reshape(w, h)


# --------------------------------------------------------------------------
# 5. can_kill -- the five-point question
# --------------------------------------------------------------------------

def can_kill(field, my_pos, opp_pos, bombs=(), explosion_map=None,
             other_bodies=(), timing=None):
    """Would a bomb dropped at my_pos leave `opp_pos` with no escape at all?

    The opponent is given full freedom of movement, so this is the pessimistic
    (correct) test: we only claim a kill when *every* opponent trajectory dies.
    """
    tm = timing or TIMING
    lethal = danger_map(
        field, bombs, explosion_map,
        extra_bombs=[(tuple(my_pos), tm.fresh_countdown)], timing=tm,
    )
    blockers = [p for p in other_bodies if tuple(p) != tuple(opp_pos)]
    blockers.append(tuple(my_pos))       # our body plus the fresh bomb
    free = walkable_map(field, bombs=bombs, blocked=blockers)
    esc = survivable(opp_pos, lethal, free, field.shape, timing=tm)
    return not esc.safe


# --------------------------------------------------------------------------
# 6. Plain BFS distances (no time dimension)
# --------------------------------------------------------------------------

BFS_INF = 255


def bfs_distances(free, sources, field_shape):
    """Multi-source BFS over walkable tiles.  Returns a (W, H) uint8 array."""
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
            continue
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
    """Free tiles from which a crate is adjacent -- the useful bombing spots."""
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
    """Number of free neighbours -- a cheap dead-end detector."""
    w, h = field.shape
    deg = 0
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 0:
            deg += 1
    return deg
