"""
features.py: action-conditional features phi(s, a).

Instead of the usual phi(s) with a separate weight per action, Q(s, a) is
computed from a single shared feature vector phi(s, a) that describes what
would happen if action a were taken, evaluated once per candidate action.

This gets us two things: the four move directions are scored against the
same descriptors (so the model can't pick up a directional bias, no
rotation/mirror augmentation needed), and the feature count drops from
~200 to 47, which matters given how few transitions we can collect.

Features only describe properties of candidate actions (is there an
escape if I bomb here, is the opponent trapped if I bomb here): nothing
here picks an action itself, the model still has to learn how to weigh
these against each other.
"""

from __future__ import annotations

import numpy as np

from . import tactics as T
from . import opponent_model as OM
from .tactics import ACTIONS, BFS_INF, DELTA, MOVES, THREAT_INF

# Feature layout. Blocks are disjoint: a MOVE only writes into the move
# block, so bomb weights never get touched by a movement transition.

# hunting features can be toggled without changing vector length -- both
# variants have the same dimensionality, just differ in whether these
# entries are ever non-zero
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
    "g_prey_predictability",
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
    "m_predicted_threat",
    "m_step_to_coin",
    "m_step_to_crate",
    "m_step_to_opponent",
    "m_revisit",
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
    "b_prey_kill_prob",
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

# normalisation constants: keep every entry roughly in [0, 1] so one
# learning rate works for the whole vector
_MAX_STEPS = float(T.S["MAX_STEPS"])
_SAFETY_SCALE = 5.0
_TILE_SCALE = 10.0
_CRATE_SCALE = 3.0


_OPP_MODEL = None


def _opponent_model():
    global _OPP_MODEL
    if _OPP_MODEL is None:
        _OPP_MODEL = OM.OpponentModel.load()
    return _OPP_MODEL


class Context:
    """Everything derivable from one game_state, computed once per step.

    Building this costs about a millisecond; the six phi() calls that
    follow are then cheap since they only read from it.
    """

    def __init__(self, game_state, timing=None, coordinate_history=None):
        self.timing = timing or T.TIMING
        # recent own positions, most recent last: lets us catch
        # oscillation that a stateless feature would miss
        self.history = coordinate_history
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

        # space-time lethality, current world
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

        # counterfactual: what if I bomb right now?
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

        # threat: what the opponents can do to us
        self.threat = T.threat_map(
            self.field, self.others, bombs=self.bombs, timing=self.timing
        )

        # navigation distance fields, run from the targets so d[tile] gives
        # "distance from here to the nearest X" for every tile at once
        nav_free = T.walkable_map(self.field, bombs=self.bombs)
        self.d_coin = T.bfs_distances(nav_free, self.coins, self.field.shape)
        self.d_crate = T.bfs_distances(
            nav_free, T.crate_seed_tiles(self.field), self.field.shape
        )
        self.d_opponent = T.bfs_distances(
            nav_free, self.other_positions, self.field.shape
        )

        # hunting: track the nearest reachable opponent as prey
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

        # farming: where to go bomb crates
        self.d_cratespot = None
        if self.bomb_available:
            self._find_crate_spots()

        self.crates_remaining = int((self.field == 1).sum())

        # opponent behaviour model: replaces the worst-case can_kill
        # (which only fires when every opponent trajectory dies) with a
        # graded "will they actually escape"
        self.opp_probs = []
        self.prey_kill_prob = 0.0
        self.prey_predictability = 0.0
        self.pred_threat = None

        model = _opponent_model()
        if model.ready and self.other_positions:
            shared = OM.shared_maps(self.field, self.bombs, self.explosion_map,
                                    self.coins, self.pos, timing=self.timing)
            for other in self.others:
                _, _, ob, opos = other
                feats = OM.opponent_features(
                    self.field, self.bombs, self.explosion_map, self.coins,
                    opos, ob, self.other_positions, self.pos,
                    timing=self.timing, shared=shared)
                self.opp_probs.append(model.predict_proba(feats))
            self.pred_threat = OM.predicted_threat(
                self.field, self.bombs, self.others, self.opp_probs,
                timing=self.timing)
            if self.prey is not None:
                idx = self.other_positions.index(tuple(self.prey)) \
                    if tuple(self.prey) in [tuple(p) for p in self.other_positions] else 0
                probs = self.opp_probs[idx]
                self.prey_predictability = max(probs.values())
                if self.bomb_available:
                    self.prey_kill_prob = OM.kill_probability(
                        self.field, self.bombs, self.explosion_map, self.pos,
                        self.prey, self.other_positions, probs,
                        timing=self.timing)

    def _prey_freedom(self, blocked_extra=()):
        """How many tiles the hunted opponent can still safely stand on.

        Same escape BFS we use for ourselves, just run from their
        position: low count means cornered.
        """
        blocked = [p for p in self.other_positions if tuple(p) != tuple(self.prey)]
        blocked += list(blocked_extra)
        free = T.walkable_map(self.field, bombs=self.bombs, blocked=blocked)
        esc = T.survivable(
            self.prey, self.lethal, free, self.field.shape, timing=self.timing
        )
        return esc.n_safe_tiles

    def _find_crate_spots(self, radius=8, min_crates=2):
        """Reachable tiles where a bomb hits >= min_crates and still leaves
        an escape. Only tiles adjacent to a crate are scanned, since the
        escape BFS is the expensive part.
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
        """Tiles within reach from which a bomb would leave the prey no
        escape. Candidates are exactly blast_coords(prey) since blast is
        symmetric, so this stays cheap (~13 candidates instead of the
        whole board). Doesn't catch sealing a corridor from outside blast
        range, which is fine, just not covered.
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


def phi(ctx: Context, action: str) -> np.ndarray:
    """Feature vector for taking `action` in the state `ctx` describes."""
    v = np.zeros(N_FEATURES, dtype=np.float32)

    # global block, same for every action
    v[IDX["g_bomb_available"]] = 1.0 if ctx.bomb_available else 0.0
    v[IDX["g_opponents_alive"]] = len(ctx.others) / 3.0
    v[IDX["g_coins_visible"]] = min(len(ctx.coins), 9) / 9.0
    v[IDX["g_episode_progress"]] = min(ctx.step, _MAX_STEPS) / _MAX_STEPS
    v[IDX["g_currently_in_danger"]] = 1.0 if ctx.in_danger() else 0.0
    v[IDX["g_prey_cornered"]] = _cornered(ctx.prey_escape_tiles) if ctx.prey else 0.0
    # last-alive unlocks uncontested farming; the model can't infer this
    # from the state otherwise
    v[IDX["g_last_alive"]] = 1.0 if not ctx.others else 0.0
    v[IDX["g_crates_remaining"]] = min(ctx.crates_remaining, 120) / 120.0
    v[IDX["g_prey_predictability"]] = ctx.prey_predictability

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
        # rest stays zero: illegal moves get penalised via m_legal plus
        # the INVALID_ACTION reward, not masked out here
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

    # room the target tile leaves us: m_tile_degree only sees one step,
    # this catches a corridor whose mouth could get sealed
    v[IDX["m_escape_room"]] = min(ctx.escape.n_safe_tiles, 12) / 12.0

    # "is this ON the shortest path", vs m_closer_* which only says the
    # distance goes down -- disambiguates between several reducing moves
    v[IDX["m_step_to_coin"]] = _on_path(ctx.d_coin, here, (nx, ny))
    if ctx.d_cratespot is not None:
        v[IDX["m_step_to_crate"]] = _on_path(ctx.d_cratespot, here, (nx, ny))
    else:
        v[IDX["m_step_to_crate"]] = _on_path(ctx.d_crate, here, (nx, ny))
    v[IDX["m_step_to_opponent"]] = _on_path(ctx.d_opponent, here, (nx, ny))

    # revisits recently, capped at 3 so one re-crossing (often legit in a
    # corridor) doesn't look like oscillation
    if ctx.history:
        v[IDX["m_revisit"]] = min(ctx.history.count((nx, ny)), 3) / 3.0
    if ctx.pred_threat is not None:
        v[IDX["m_predicted_threat"]] = float(ctx.pred_threat[nx, ny])
    if ctx.prey is not None and (nx, ny) != tuple(ctx.prey):
        # standing in the doorway can remove an escape route without
        # spending a bomb: rerun the prey's escape BFS with our body on
        # the target tile
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

    # interaction terms (crates_hit x safety, cornered x escape) fed in
    # directly instead of making the model learn them itself
    threat_here = int(ctx.threat[ctx.x, ctx.y])
    safe_now = 1.0 if threat_here >= THREAT_INF else 0.0
    v[IDX["b_crates_x_safe"]] = (min(ctx.crates_hit, _CRATE_SCALE)
                                 / _CRATE_SCALE) * safe_now
    v[IDX["b_cornered_x_escape"]] = cornered * (1.0 if esc.safe else 0.0)
    v[IDX["b_prey_kill_prob"]] = ctx.prey_kill_prob


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
    """1.0 when the prey has nowhere to go, 0.0 when it's free."""
    return max(0.0, 1.0 - min(escape_tiles, _FREEDOM_SCALE) / _FREEDOM_SCALE)


def _on_path(dist_field, here, there) -> float:
    """Is `there` the next step on a shortest path from `here` to the target?

    1.0 on the path, 0.5 reachable but off it, 0.0 unreachable. More
    informative than _closer(): on an open board several moves reduce the
    distance, this pins down which one is actually shortest. Ties both
    count as 1.0.
    """
    d_here = int(dist_field[here[0], here[1]])
    d_there = int(dist_field[there[0], there[1]])
    if d_here >= BFS_INF or d_there >= BFS_INF:
        return 0.0
    return 1.0 if d_there == d_here - 1 else 0.5


def _closer(dist_field, here, there) -> float:
    d_here = int(dist_field[here[0], here[1]])
    d_there = int(dist_field[there[0], there[1]])
    if d_here >= BFS_INF or d_there >= BFS_INF:
        return 0.0
    return 1.0 if d_there < d_here else 0.0


def phi_all(ctx: Context) -> np.ndarray:
    """(6, N_FEATURES) matrix, one row per action, in ACTIONS order."""
    return np.stack([phi(ctx, a) for a in ACTIONS])


def survivable_actions(ctx: Context) -> list[str]:
    """Actions the escape BFS thinks we can survive.

    Training only, biases exploratory sampling away from immediate
    suicide, annealed to nothing. Never used by act() at test time.
    """
    out = [a for a in MOVES if ctx.escape.safe_first.get(a, False)]
    if ctx.escape.safe_first.get("WAIT", False):
        out.append("WAIT")
    if ctx.bomb_available and ctx.escape_after_bomb is not None \
            and ctx.escape_after_bomb.safe:
        out.append("BOMB")
    return out
