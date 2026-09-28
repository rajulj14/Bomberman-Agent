"""
features.py -- action-conditional features  phi(s, a).

We deliberately do NOT use the usual phi(s) with one weight vector per action.
Instead Q(s, a) = w . phi(s, a) with a single shared w, where phi describes
what would happen if action a were taken.

Two consequences we care about:

  * The D4 symmetry of the board becomes structural rather than statistical.
    All four moves are scored by the same weights on the same descriptors, so
    the model physically cannot acquire a directional bias and no rotation /
    mirror data augmentation is needed for movement.
  * The parameter count drops from roughly 200 to 29, which matters a lot for
    a linear model trained on the few hundred thousand transitions we can
    realistically collect.

COMPLIANCE
    Features describe *properties of candidate actions* (does an escape exist
    if I bomb here, is this opponent trapped if I bomb here).  No feature ever
    names a best action, and nothing here is consulted at selection time
    except through the learned weights.  The model still has to learn how to
    trade these quantities off against each other -- and, as the weight tables
    in the report show, its answers are not the obvious ones.
"""

from __future__ import annotations

import numpy as np

from . import tactics as T
from .tactics import ACTIONS, BFS_INF, DELTA, MOVES, THREAT_INF

# --------------------------------------------------------------------------
# Feature layout.  Blocks are disjoint: a MOVE only writes into the move
# block, so bomb weights are never updated by a movement transition.
# --------------------------------------------------------------------------

# Hunting features can be switched off without changing the length of the
# feature vector.  That keeps the with/without comparison honest: both
# variants have identical dimensionality, identical everything else, and
# differ only in whether these entries are ever non-zero.
USE_HUNTING_FEATURES = True

GLOBAL_FEATURES = [
    "g_bomb_available",
    "g_opponents_alive",
    "g_coins_visible",
    "g_episode_progress",
    "g_currently_in_danger",
    "g_prey_cornered",
    "g_last_alive",
    "g_crates_remaining",
]

TYPE_FEATURES = ["t_is_move", "t_is_bomb", "t_is_wait"]

MOVE_FEATURES = [
    "m_legal",
    "m_lethal_next_step",
    "m_survivable",
    "m_steps_to_safety",
    "m_safe_tiles_reachable",
    "m_threatened_by_opponent",
    "m_closer_to_coin",
    "m_closer_to_crate",
    "m_closer_to_opponent",
    "m_tile_degree",
    "m_blast_ttl",
    "m_onto_coin",
    "m_closer_to_kill_spot",
    "m_cuts_prey_escape",
    "m_closer_to_crate_spot",
    "m_escape_room",
]

BOMB_FEATURES = [
    "b_available",
    "b_escape_exists",
    "b_steps_to_safety",
    "b_crates_hit",
    "b_opponents_trapped",
    "b_opponent_proximity",
    "b_prey_in_blast",
    "b_prey_cornered",
    "b_crates_x_safe",
    "b_cornered_x_escape",
]

WAIT_FEATURES = [
    "w_tile_permanently_safe",
    "w_lethal_next_step",
    "w_threatened_by_opponent",
]

FEATURE_NAMES = (
    GLOBAL_FEATURES + TYPE_FEATURES + MOVE_FEATURES + BOMB_FEATURES + WAIT_FEATURES
)
N_FEATURES = len(FEATURE_NAMES)
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# normalisation constants -- keep every entry roughly in [0, 1] so a single
# learning rate works for the whole vector
_MAX_STEPS = float(T.S["MAX_STEPS"])
_SAFETY_SCALE = 5.0
_TILE_SCALE = 10.0
_CRATE_SCALE = 3.0


class Context:
    """Everything derivable from one game_state, computed once per step.

    Building this costs roughly a millisecond; the six phi() calls that follow
    are then almost free because they only read from it.
    """

    def __init__(self, game_state, timing=None):
        self.timing = timing or T.TIMING
        gs = game_state
        self.field = np.asarray(gs["field"])
        self.w, self.h = self.field.shape
        self.bombs = list(gs["bombs"])
        self.explosion_map = gs["explosion_map"]
        self.coins = list(gs["coins"])
        self.others = list(gs["others"])
        self.step = gs["step"]

        _, self.score, self.bomb_available, self.pos = gs["self"]
        self.x, self.y = self.pos
        self.other_positions = [o[3] for o in self.others]
        self.coin_set = set(self.coins)

        # --- space-time lethality, current world -------------------------
        self.lethal = T.danger_map(
            self.field, self.bombs, self.explosion_map, timing=self.timing
        )
        self.clean_from = T._clean_from(self.lethal)
        self.free = T.walkable_map(
            self.field, bombs=self.bombs, blocked=self.other_positions
        )
        self.escape = T.survivable(
            self.pos, self.lethal, self.free, self.field.shape, timing=self.timing
        )

        # --- counterfactual: what if I bomb right now? -------------------
        if self.bomb_available:
            lethal_b = T.danger_map(
                self.field, self.bombs, self.explosion_map,
                extra_bombs=[(self.pos, self.timing.fresh_countdown)],
                timing=self.timing,
            )
            free_b = T.walkable_map(
                self.field,
                bombs=self.bombs + [(self.pos, self.timing.fresh_countdown)],
                blocked=self.other_positions,
            )
            self.escape_after_bomb = T.survivable(
                self.pos, lethal_b, free_b, self.field.shape, timing=self.timing
            )
            self.bomb_blast = T.blast_coords(self.field, self.x, self.y)
            self.crates_hit = sum(
                1 for (bx, by) in self.bomb_blast if self.field[bx, by] == 1
            )
            self.opponents_trapped = sum(
                1 for p in self.other_positions
                if T.can_kill(self.field, self.pos, p, self.bombs,
                              self.explosion_map, self.other_positions,
                              timing=self.timing)
            )
        else:
            self.escape_after_bomb = None
            self.bomb_blast = []
            self.crates_hit = 0
            self.opponents_trapped = 0

        # --- threat: what the opponents can do to us ---------------------
        self.threat = T.threat_map(
            self.field, self.others, bombs=self.bombs, timing=self.timing
        )

        # --- navigation distance fields ----------------------------------
        # BFS is run from the targets, so d[tile] answers "how far is the
        # nearest coin from here" for every tile at once.  A move is then
        # "closer" iff d[target] < d[me], which is exactly the potential
        # difference our reward shaping uses.
        nav_free = T.walkable_map(self.field, bombs=self.bombs)
        self.d_coin = T.bfs_distances(nav_free, self.coins, self.field.shape)
        self.d_crate = T.bfs_distances(
            nav_free, T.crate_seed_tiles(self.field), self.field.shape
        )
        self.d_opponent = T.bfs_distances(
            nav_free, self.other_positions, self.field.shape
        )

        # --- hunting -----------------------------------------------------
        # Kills are worth five coins each and there are fifteen points of
        # them on the board against nine points of coins, so the ability to
        # *create* a kill matters more than the ability to take one that
        # happens to appear.  b_opponents_trapped only fires at the moment
        # the shot is already available; these features describe the approach.
        self._nav_free = nav_free
        self.d_from_me = T.bfs_distances(nav_free, [self.pos], self.field.shape)
        self.prey = None
        self.prey_escape_tiles = 0
        self.prey_in_blast = False
        self.d_killspot = None

        if USE_HUNTING_FEATURES and self.other_positions:
            reachable = [
                (int(self.d_from_me[px, py]), (px, py))
                for (px, py) in self.other_positions
                if int(self.d_from_me[px, py]) < BFS_INF
            ]
            if reachable:
                _, self.prey = min(reachable, key=lambda t: t[0])

        if self.prey is not None:
            self.prey_escape_tiles = self._prey_freedom(blocked_extra=[self.pos])
            if self.bomb_available:
                self.prey_in_blast = tuple(self.prey) in set(self.bomb_blast)
                self._find_kill_spots()

        # --- farming ------------------------------------------------------
        # The mirror of kill spots.  b_crates_hit only says how good bombing
        # *here* would be; nothing told the agent how to walk somewhere worth
        # bombing.  Measured farming rate was half the rule-based agent's
        # despite surviving 40% longer, which is what this is aimed at.
        self.d_cratespot = None
        if self.bomb_available:
            self._find_crate_spots()

        self.crates_remaining = int((self.field == 1).sum())

    # ------------------------------------------------------------------
    def _prey_freedom(self, blocked_extra=()):
        """How many tiles the hunted opponent can still safely stand on.

        A low count means cornered.  This is the same escape BFS we use for
        ourselves, just run from their position -- which is why the hunting
        features cost almost nothing to add.
        """
        blocked = [p for p in self.other_positions if tuple(p) != tuple(self.prey)]
        blocked += list(blocked_extra)
        free = T.walkable_map(self.field, bombs=self.bombs, blocked=blocked)
        esc = T.survivable(
            self.prey, self.lethal, free, self.field.shape, timing=self.timing
        )
        return esc.n_safe_tiles

    def _find_crate_spots(self, radius=8, min_crates=2):
        """Reachable tiles where a bomb hits >= min_crates and leaves an escape.

        Escape is required, so this cannot lure the agent into a suicide spot.
        Only tiles adjacent to a crate are considered, which keeps the scan
        small; the escape BFS is the expensive part and runs at most a few
        dozen times.
        """
        spots = []
        for (x, y) in T.crate_seed_tiles(self.field):
            d = int(self.d_from_me[x, y])
            if d > radius or d >= BFS_INF:
                continue
            blast = T.blast_coords(self.field, x, y)
            hits = sum(1 for (bx, by) in blast if self.field[bx, by] == 1)
            if hits < min_crates:
                continue
            lethal_b = T.danger_map(
                self.field, self.bombs, self.explosion_map,
                extra_bombs=[((x, y), self.timing.fresh_countdown)],
                timing=self.timing,
            )
            free_b = T.walkable_map(
                self.field,
                bombs=self.bombs + [((x, y), self.timing.fresh_countdown)],
                blocked=self.other_positions,
            )
            if T.survivable((x, y), lethal_b, free_b, self.field.shape,
                            timing=self.timing).safe:
                spots.append((x, y))
        if spots:
            self.d_cratespot = T.bfs_distances(
                self._nav_free, spots, self.field.shape
            )

    def _find_kill_spots(self, radius=6):
        """Tiles within reach from which a bomb would leave the prey no escape.

        The candidate set is exactly blast_coords(prey): a blast is symmetric,
        so tile t covers the prey if and only if the prey's own blast covers t
        (same power, same walls, same line of sight).  That turns a 289-tile
        scan into at most 13 candidates and keeps the expensive can_kill calls
        in single figures.

        This misses the rarer case of sealing a corridor from outside blast
        range -- an approximation, noted rather than silently ignored.
        """
        prey = tuple(self.prey)
        spots = []
        for (x, y) in T.blast_coords(self.field, prey[0], prey[1]):
            if self.field[x, y] != 0:
                continue
            d = int(self.d_from_me[x, y])
            if d > radius or d >= BFS_INF:
                continue
            if T.can_kill(self.field, (x, y), prey, self.bombs,
                          self.explosion_map, self.other_positions,
                          timing=self.timing):
                spots.append((x, y))
        if spots:
            self.d_killspot = T.bfs_distances(
                self._nav_free, spots, self.field.shape
            )

    # ------------------------------------------------------------------
    def in_danger(self) -> bool:
        return bool(self.clean_from[self.x * self.h + self.y] > 0)

    def blast_ttl(self, x, y) -> int:
        """Steps until this tile first becomes lethal (horizon if never)."""
        col = self.lethal[:, x * self.h + y]
        hits = np.flatnonzero(col)
        return int(hits[0]) if hits.size else self.lethal.shape[0]

    def dist_to_nearest_coin(self) -> int:
        return int(self.d_coin[self.x, self.y])

    def dist_to_nearest_crate(self) -> int:
        return int(self.d_crate[self.x, self.y])


# --------------------------------------------------------------------------
def phi(ctx: Context, action: str) -> np.ndarray:
    """Feature vector for taking `action` in the state `ctx` describes."""
    v = np.zeros(N_FEATURES, dtype=np.float32)

    # ---- global block, identical for every action ---------------------
    v[IDX["g_bomb_available"]] = 1.0 if ctx.bomb_available else 0.0
    v[IDX["g_opponents_alive"]] = len(ctx.others) / 3.0
    v[IDX["g_coins_visible"]] = min(len(ctx.coins), 9) / 9.0
    v[IDX["g_episode_progress"]] = min(ctx.step, _MAX_STEPS) / _MAX_STEPS
    v[IDX["g_currently_in_danger"]] = 1.0 if ctx.in_danger() else 0.0
    v[IDX["g_prey_cornered"]] = _cornered(ctx.prey_escape_tiles) if ctx.prey else 0.0
    # Being last alive unlocks ~250 steps of uncontested farming.  The agent
    # survives roughly half its rounds and had no way to know the game had
    # changed underneath it.
    v[IDX["g_last_alive"]] = 1.0 if not ctx.others else 0.0
    v[IDX["g_crates_remaining"]] = min(ctx.crates_remaining, 120) / 120.0

    if action in MOVES:
        v[IDX["t_is_move"]] = 1.0
        _fill_move(v, ctx, action)
    elif action == "BOMB":
        v[IDX["t_is_bomb"]] = 1.0
        _fill_bomb(v, ctx)
    else:
        v[IDX["t_is_wait"]] = 1.0
        _fill_wait(v, ctx)

    return v


def _fill_move(v, ctx: Context, action: str):
    dx, dy = DELTA[action]
    nx, ny = ctx.x + dx, ctx.y + dy

    in_bounds = 0 <= nx < ctx.w and 0 <= ny < ctx.h
    legal = bool(in_bounds and ctx.free[nx * ctx.h + ny])
    v[IDX["m_legal"]] = 1.0 if legal else 0.0
    if not legal:
        # Everything else stays zero.  The agent learns to avoid illegal moves
        # from the m_legal weight plus the INVALID_ACTION penalty; we do not
        # mask them out, because masking would be an action override.
        return

    p = nx * ctx.h + ny
    v[IDX["m_lethal_next_step"]] = float(ctx.lethal[1, p]) if ctx.lethal.shape[0] > 1 else 0.0
    v[IDX["m_survivable"]] = 1.0 if ctx.escape.safe_first[action] else 0.0

    steps = ctx.escape.first_steps[action]
    if steps >= 0:
        v[IDX["m_steps_to_safety"]] = min(steps, _SAFETY_SCALE) / _SAFETY_SCALE
    else:
        v[IDX["m_steps_to_safety"]] = 1.0

    v[IDX["m_safe_tiles_reachable"]] = min(
        ctx.escape.n_safe_tiles, _TILE_SCALE
    ) / _TILE_SCALE

    threat_t = int(ctx.threat[nx, ny])
    if threat_t < THREAT_INF:
        # sooner threat -> larger value
        v[IDX["m_threatened_by_opponent"]] = max(0.0, 1.0 - threat_t / 10.0)

    here = (ctx.x, ctx.y)
    v[IDX["m_closer_to_coin"]] = _closer(ctx.d_coin, here, (nx, ny))
    v[IDX["m_closer_to_crate"]] = _closer(ctx.d_crate, here, (nx, ny))
    v[IDX["m_closer_to_opponent"]] = _closer(ctx.d_opponent, here, (nx, ny))

    v[IDX["m_tile_degree"]] = T.tile_degree(ctx.field, nx, ny) / 4.0
    v[IDX["m_blast_ttl"]] = min(ctx.blast_ttl(nx, ny), 6) / 6.0
    v[IDX["m_onto_coin"]] = 1.0 if (nx, ny) in ctx.coin_set else 0.0

    if ctx.d_killspot is not None:
        v[IDX["m_closer_to_kill_spot"]] = _closer(ctx.d_killspot, here, (nx, ny))
    if ctx.d_cratespot is not None:
        v[IDX["m_closer_to_crate_spot"]] = _closer(ctx.d_cratespot, here, (nx, ny))

    # How much room the target tile leaves us -- the escape BFS already counts
    # reachable safe tiles from there, and a low count is a corridor we may
    # not get out of if someone bombs the mouth.  m_tile_degree only sees one
    # step; this sees the pocket.
    v[IDX["m_escape_room"]] = min(ctx.escape.n_safe_tiles, 12) / 12.0
    if ctx.prey is not None and (nx, ny) != tuple(ctx.prey):
        # Standing in the right doorway removes an escape route without
        # spending a bomb.  Cheap to check: rerun the prey's escape BFS with
        # our body on the target tile instead of our current one.
        after = ctx._prey_freedom(blocked_extra=[(nx, ny)])
        if after < ctx.prey_escape_tiles:
            v[IDX["m_cuts_prey_escape"]] = 1.0


def _fill_bomb(v, ctx: Context):
    v[IDX["b_available"]] = 1.0 if ctx.bomb_available else 0.0
    if not ctx.bomb_available:
        return

    esc = ctx.escape_after_bomb
    v[IDX["b_escape_exists"]] = 1.0 if esc.safe else 0.0
    if esc.min_steps >= 0:
        v[IDX["b_steps_to_safety"]] = min(esc.min_steps, _SAFETY_SCALE) / _SAFETY_SCALE
    else:
        v[IDX["b_steps_to_safety"]] = 1.0

    v[IDX["b_crates_hit"]] = min(ctx.crates_hit, _CRATE_SCALE) / _CRATE_SCALE
    v[IDX["b_opponents_trapped"]] = min(ctx.opponents_trapped, 3) / 3.0

    d = int(ctx.d_opponent[ctx.x, ctx.y])
    if d < BFS_INF:
        v[IDX["b_opponent_proximity"]] = 1.0 / (1.0 + d)

    v[IDX["b_prey_in_blast"]] = 1.0 if ctx.prey_in_blast else 0.0
    cornered = _cornered(ctx.prey_escape_tiles) if ctx.prey is not None else 0.0
    if ctx.prey is not None:
        v[IDX["b_prey_cornered"]] = cornered

    # Explicit interaction terms.  A linear model has one weight per feature,
    # so it cannot learn "crates are worth bombing WHEN nobody is hunting me"
    # or "take the shot WHEN they are cornered AND I can get out".  Handing it
    # the products is the cheapest possible test of whether the missing
    # interactions are what caps performance.
    threat_here = int(ctx.threat[ctx.x, ctx.y])
    safe_now = 1.0 if threat_here >= THREAT_INF else 0.0
    v[IDX["b_crates_x_safe"]] = (min(ctx.crates_hit, _CRATE_SCALE)
                                 / _CRATE_SCALE) * safe_now
    v[IDX["b_cornered_x_escape"]] = cornered * (1.0 if esc.safe else 0.0)


def _fill_wait(v, ctx: Context):
    p = ctx.x * ctx.h + ctx.y
    v[IDX["w_tile_permanently_safe"]] = 1.0 if ctx.clean_from[p] == 0 else 0.0
    if ctx.lethal.shape[0] > 1:
        v[IDX["w_lethal_next_step"]] = float(ctx.lethal[1, p])
    threat_t = int(ctx.threat[ctx.x, ctx.y])
    if threat_t < THREAT_INF:
        v[IDX["w_threatened_by_opponent"]] = max(0.0, 1.0 - threat_t / 10.0)


_FREEDOM_SCALE = 12.0


def _cornered(escape_tiles) -> float:
    """1.0 when the prey has nowhere to go, 0.0 when it is free."""
    return max(0.0, 1.0 - min(escape_tiles, _FREEDOM_SCALE) / _FREEDOM_SCALE)


def _closer(dist_field, here, there) -> float:
    d_here = int(dist_field[here[0], here[1]])
    d_there = int(dist_field[there[0], there[1]])
    if d_here >= BFS_INF or d_there >= BFS_INF:
        return 0.0
    return 1.0 if d_there < d_here else 0.0


# --------------------------------------------------------------------------
def phi_all(ctx: Context) -> np.ndarray:
    """(6, N_FEATURES) matrix, one row per action, in ACTIONS order."""
    return np.stack([phi(ctx, a) for a in ACTIONS])


def survivable_actions(ctx: Context) -> list[str]:
    """Actions the escape BFS believes we can survive.

    TRAINING ONLY.  Used to bias exploratory sampling away from immediate
    suicide, annealed to nothing over training.  Never consulted by act() at
    test time -- see the note in train.py.
    """
    out = [a for a in MOVES if ctx.escape.safe_first.get(a, False)]
    if ctx.escape.safe_first.get("WAIT", False):
        out.append("WAIT")
    if ctx.bomb_available and ctx.escape_after_bomb is not None \
            and ctx.escape_after_bomb.safe:
        out.append("BOMB")
    return out
