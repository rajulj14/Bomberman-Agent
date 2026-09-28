"""
opponent_model.py: learn what opponents actually do, not what they could do.

threat_map and can_kill in tactics.py are worst-case: can_kill only fires
when EVERY opponent trajectory dies, so a shot gets refused the moment
there's one theoretical escape, even if the opponent is visibly walking
the other way toward a coin. Since kills are worth more than most other
points combined, refusing marginal shots is expensive. This module swaps
"could they escape" for "will they escape".

This model predicts an opponent's action distribution: it does not pick
our action. Its output only enters phi(s, a) as extra feature entries,
same as everything else.

Opponent actions aren't given to us directly, so they're inferred by
comparing consecutive observations: a position change is the move, an
unchanged position with a new bomb under it is BOMB, otherwise WAIT. That
only uses information we'd also have at test time.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from . import tactics as T
from .tactics import ACTIONS, BFS_INF, DELTA, THREAT_INF

MODEL_FILE = os.path.join(os.path.dirname(__file__), "opponent_model.pkl")

OPP_FEATURE_NAMES = [
    "in_danger_now",
    "danger_up", "danger_right", "danger_down", "danger_left",
    "free_up", "free_right", "free_down", "free_left",
    "toward_coin_up", "toward_coin_right", "toward_coin_down", "toward_coin_left",
    "coin_reachable",
    "adjacent_crate",
    "bomb_available",
    "escape_exists_if_bomb",
    "dist_to_me",
    "tile_degree",
    "steps_to_safety",
]
N_OPP_FEATURES = len(OPP_FEATURE_NAMES)


def shared_maps(field, bombs, explosion_map, coins, my_pos, timing=None):
    """Per-step quantities every opponent's feature vector reuses, so we
    don't recompute them once per opponent."""
    tm = timing or T.TIMING
    lethal = T.danger_map(field, bombs, explosion_map, timing=tm)
    nav_free = T.walkable_map(field, bombs=bombs)
    return {
        "lethal": lethal,
        "clean": T._clean_from(lethal),
        "nav_free": nav_free,
        "d_coin": T.bfs_distances(nav_free, list(coins), field.shape),
        "d_me": T.bfs_distances(nav_free, [tuple(my_pos)], field.shape),
    }


def opponent_features(field, bombs, explosion_map, coins, opp_pos,
                      opp_bomb_available, other_bodies, my_pos, timing=None,
                      shared=None):
    """Describe one opponent's situation, from what we can observe."""
    tm = timing or T.TIMING
    w, h = field.shape
    ox, oy = opp_pos
    v = np.zeros(N_OPP_FEATURES, dtype=np.float32)
    if shared is None:
        shared = shared_maps(field, bombs, explosion_map, coins, my_pos, tm)
    lethal = shared["lethal"]
    clean = shared["clean"]
    blockers = [tuple(p) for p in other_bodies if tuple(p) != tuple(opp_pos)]
    blockers.append(tuple(my_pos))
    free = T.walkable_map(field, bombs=bombs, blocked=blockers)
    v[0] = 1.0 if clean[ox * h + oy] > 0 else 0.0
    nav_free = shared["nav_free"]
    d_coin = shared["d_coin"]
    here_coin = int(d_coin[ox, oy])
    for i, a in enumerate(["UP", "RIGHT", "DOWN", "LEFT"]):
        dx, dy = DELTA[a]
        nx, ny = ox + dx, oy + dy
        if not (0 <= nx < w and 0 <= ny < h):
            continue
        p = nx * h + ny
        if lethal.shape[0] > 1:
            v[1 + i] = float(lethal[1, p])
        v[5 + i] = 1.0 if free[p] else 0.0
        if here_coin < BFS_INF and int(d_coin[nx, ny]) < here_coin:
            v[9 + i] = 1.0
    v[13] = 1.0 if here_coin < BFS_INF else 0.0
    v[14] = 1.0 if any(
        0 <= ox + dx < w and 0 <= oy + dy < h and field[ox + dx, oy + dy] == 1
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
    ) else 0.0
    v[15] = 1.0 if opp_bomb_available else 0.0
    if opp_bomb_available:
        lethal_b = T.danger_map(
            field, bombs, explosion_map,
            extra_bombs=[(tuple(opp_pos), tm.fresh_countdown)], timing=tm)
        free_b = T.walkable_map(
            field, bombs=list(bombs) + [(tuple(opp_pos), tm.fresh_countdown)],
            blocked=blockers)
        v[16] = 1.0 if T.survivable(opp_pos, lethal_b, free_b, field.shape,
                                    timing=tm).safe else 0.0
    dm = int(shared["d_me"][ox, oy])
    v[17] = 1.0 / (1.0 + dm) if dm < BFS_INF else 0.0
    v[18] = T.tile_degree(field, ox, oy) / 4.0
    esc = T.survivable(opp_pos, lethal, free, field.shape, timing=tm)
    v[19] = min(max(esc.min_steps, 0), 5) / 5.0
    return v


def infer_action(old_pos, new_pos, old_bombs, new_bombs):
    """Recover an opponent's action from two consecutive observations.

    Returns an action string, or None if the opponent vanished (died) and
    we can't infer anything.
    """
    if old_pos is None or new_pos is None:
        return None
    ox, oy = old_pos
    nx, ny = new_pos
    dx, dy = nx - ox, ny - oy
    for a in ["UP", "RIGHT", "DOWN", "LEFT"]:
        if DELTA[a] == (dx, dy):
            return a
    if (dx, dy) != (0, 0):
        return None                      # shouldn't happen - would be a teleport
    old_set = {tuple(p) for p, _ in old_bombs}
    new_set = {tuple(p) for p, _ in new_bombs}
    if (ox, oy) in new_set and (ox, oy) not in old_set:
        return "BOMB"
    return "WAIT"


class OpponentModel:
    """Wraps a scikit-learn classifier over OPP_FEATURE_NAMES."""

    def __init__(self, clf=None, classes=None):
        self.clf = clf
        self.classes = classes or list(ACTIONS)

    @classmethod
    def load(cls, path=MODEL_FILE):
        if not os.path.isfile(path):
            return cls(None)
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("n_features") != N_OPP_FEATURES:
                return cls(None)
            return cls(blob["clf"], blob["classes"])
        except Exception:
            return cls(None)

    def save(self, path=MODEL_FILE):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"clf": self.clf, "classes": self.classes,
                         "n_features": N_OPP_FEATURES}, fh)
        os.replace(tmp, path)

    @property
    def ready(self):
        return self.clf is not None

    def predict_proba(self, feats):
        """P(action) as a dict. Uniform when no model is loaded."""
        if self.clf is None:
            return {a: 1.0 / len(ACTIONS) for a in ACTIONS}
        p = self.clf.predict_proba(feats.reshape(1, -1))[0]
        out = {a: 0.0 for a in ACTIONS}
        for cls_name, prob in zip(self.clf.classes_, p):
            out[cls_name] = float(prob)
        return out


def kill_probability(field, bombs, explosion_map, my_pos, opp_pos,
                     other_bodies, action_probs, timing=None):
    """P(the opponent dies) if we bomb now, under their predicted actions.

    For each action they might take, move them there and check whether
    an escape still exists. Summing the fatal branches' probabilities
    turns a binary "trapped or not" into a graded shot quality: a 70%
    shot is worth taking when a kill is worth five coins, but the
    worst-case can_kill would refuse it.
    """
    tm = timing or T.TIMING
    w, h = field.shape
    lethal = T.danger_map(
        field, bombs, explosion_map,
        extra_bombs=[(tuple(my_pos), tm.fresh_countdown)], timing=tm)
    blockers = [tuple(p) for p in other_bodies if tuple(p) != tuple(opp_pos)]
    blockers.append(tuple(my_pos))
    free = T.walkable_map(field, bombs=bombs, blocked=blockers)
    total = 0.0
    ox, oy = opp_pos
    for a, prob in action_probs.items():
        if prob <= 0.0:
            continue
        dx, dy = DELTA[a]
        nx, ny = ox + dx, oy + dy
        if not (0 <= nx < w and 0 <= ny < h) or not free[nx * h + ny]:
            nx, ny = ox, oy          # blocked move: they stay put
        if not T.survivable((nx, ny), lethal, free, field.shape, timing=tm).safe:
            total += prob
    return min(1.0, total)


def predicted_threat(field, bombs, others, action_probs_per_opponent,
                     timing=None):
    """Threat map weighted by what opponents are predicted to do.

    The worst-case threat_map marks every tile any opponent could make
    lethal within three steps, which covers most of the board once
    opponents get close and just makes the agent hide. This version only
    counts the branch they're actually likely to take.
    """
    tm = timing or T.TIMING
    w, h = field.shape
    out = np.zeros((w, h), dtype=np.float32)
    for other, probs in zip(others, action_probs_per_opponent):
        _, _, bomb_available, (ox, oy) = other
        if not bomb_available:
            continue
        p_bomb = probs.get("BOMB", 0.0)
        if p_bomb <= 0.0:
            continue
        for (bx, by) in T.blast_coords(field, ox, oy):
            out[bx, by] = max(out[bx, by], p_bomb)
    return out
