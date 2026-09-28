"""
callbacks.py: khaleesi.

Fits a gradient-boosted regression tree on a 47-dim action-conditional
feature vector, Q(s, a) = regressor output.

tactics.py and features.py are duplicated in this folder instead of being
imported from somewhere else, since the tournament only copies the folder
that has callbacks.py in it.

At test time: build the 6x47 feature matrix, one predict call, argmax.
No search, no rule-based fallback.
"""

from __future__ import annotations

import os
import pickle
from collections import deque

import numpy as np

from .features import Context, N_FEATURES, phi_all, survivable_actions
from .tactics import ACTIONS

HERE = os.path.dirname(__file__)
MODEL_FILE = os.path.join(HERE, "model.pkl")

# Exploration during training only.
EPS_START = 0.60
EPS_END = 0.05
EPS_DECAY_ROUNDS = 2500

SAFE_SAMPLING_START = 0.9
SAFE_SAMPLING_END = 0.0
SAFE_SAMPLING_DECAY_ROUNDS = 1500


def setup(self):
    self.rng = np.random.default_rng()
    self.rounds_done = 0
    self.model = _load_model(self)
    self.actions = list(ACTIONS)

    # last 12 own positions, reset each round, for detecting oscillation
    self.coordinate_history = deque([], 12)
    self.current_round = 0

    self.last_context = None
    self.last_phi_all = None
    self.last_action = None

    self.logger.info(
        f"khaleesi: {N_FEATURES} features, model={'yes' if self.model else 'no'}, "
        f"train={getattr(self, 'train', False)}"
    )


def _load_model(self):
    if os.path.isfile(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("n_features") != N_FEATURES:
                self.logger.warning(
                    f"model expects {blob.get('n_features')} features, "
                    f"code produces {N_FEATURES}; ignoring")
                return None
            self.rounds_done = int(blob.get("rounds_done", 0))
            self.logger.info(
                f"loaded model (rounds_done={self.rounds_done}, "
                f"fits={blob.get('fits', 0)})")
            return blob["model"]
        except Exception as exc:
            self.logger.warning(f"could not load model: {exc}")
    if not getattr(self, "train", False):
        self.logger.warning("no model found and not training; acting randomly")
    return None


def save_model(self, model, fits=0):
    tmp = MODEL_FILE + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({
            "model": model,
            "n_features": N_FEATURES,
            "rounds_done": getattr(self, "rounds_done", 0),
            "fits": fits,
        }, fh)
    os.replace(tmp, MODEL_FILE)   # atomic write


def q_values(self, game_state):
    """Q(s, .) for all six actions, plus the context that produced them."""
    if game_state["round"] != getattr(self, "current_round", 0):
        self.coordinate_history = deque([], 12)
        self.current_round = game_state["round"]
    ctx = Context(game_state, coordinate_history=self.coordinate_history)
    phis = phi_all(ctx)
    if self.model is None:
        return np.zeros(len(ACTIONS)), phis, ctx
    return self.model.predict(phis), phis, ctx


def act(self, game_state: dict) -> str:
    q, phis, ctx = q_values(self, game_state)

    self.last_context = ctx
    self.last_phi_all = phis

    if getattr(self, "train", False):
        action = _explore(self, q, ctx)
    else:
        action = ACTIONS[int(np.argmax(q))]

    self.last_action = action
    self.coordinate_history.append(tuple(game_state["self"][3]))
    return action


def _explore(self, q, ctx):
    """Epsilon-greedy, with exploration biased toward survivable moves early on.

    Without this, an exploratory BOMB is usually followed by a random move
    that kills the agent, so Q(BOMB) never gets a chance to look good. This
    bias is annealed to zero well before training ends and isn't used at
    test time.
    """
    rounds = getattr(self, "rounds_done", 0)

    eps = EPS_END + (EPS_START - EPS_END) * max(
        0.0, 1.0 - rounds / EPS_DECAY_ROUNDS)
    p_safe = SAFE_SAMPLING_END + (SAFE_SAMPLING_START - SAFE_SAMPLING_END) * max(
        0.0, 1.0 - rounds / SAFE_SAMPLING_DECAY_ROUNDS)

    if self.rng.random() >= eps and self.model is not None:
        return ACTIONS[int(np.argmax(q))]

    candidates = list(ACTIONS)
    if self.rng.random() < p_safe:
        safe = survivable_actions(ctx)
        if safe:
            candidates = safe
    return candidates[int(self.rng.integers(len(candidates)))]
