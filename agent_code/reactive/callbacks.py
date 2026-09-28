"""
callbacks.py -- Agent A ("reactive").

n-step Q-learning with linear function approximation over action-conditional
features:  Q(s, a) = w . phi(s, a),  one shared weight vector of 29 entries.

At test time the agent does exactly one thing: build the feature matrix, take
the dot product, return argmax.  There is no search, no rule fallback and no
safety mask -- every safety consideration reaches the decision only through a
learned weight.  This is a deliberate design constraint, not an oversight; see
the compliance note in features.py.
"""

from __future__ import annotations

import os

import numpy as np

from .features import Context, N_FEATURES, phi_all, survivable_actions
from .tactics import ACTIONS

MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.npz")

# Exploration schedule (training only; ignored when self.train is False).
TEMP_START = 1.0
TEMP_END = 0.05
TEMP_DECAY_ROUNDS = 2000

SAFE_SAMPLING_START = 0.9      # P(draw exploratory action from survivable set)
SAFE_SAMPLING_END = 0.0
SAFE_SAMPLING_DECAY_ROUNDS = 1500


def setup(self):
    """Called once before the first round.  `self` persists for the session."""
    self.rng = np.random.default_rng()
    self.rounds_done = 0
    self.weights = _load_weights(self)
    self.actions = list(ACTIONS)

    # populated by act() so train.py can reuse the work instead of rebuilding
    # the context for the same state
    self.last_context = None
    self.last_phi_all = None
    self.last_action = None

    self.logger.info(
        f"reactive: {N_FEATURES} features, train={getattr(self, 'train', False)}"
    )


def _load_weights(self):
    if os.path.isfile(MODEL_FILE):
        data = np.load(MODEL_FILE)
        w = data["weights"].astype(np.float64)
        if w.shape == (N_FEATURES,):
            # Restore the schedule position as well as the weights.  Without
            # this, every launch of main.py restarts exploration at maximum
            # temperature and alpha at maximum, so each curriculum stage
            # partly un-learns the one before it.  That is not a fresh start,
            # it is a randomised one, and it silently corrupts any experiment
            # that spans more than a single invocation.
            self.rounds_done = int(data["rounds_done"]) if "rounds_done" in data else 0
            self.logger.info(
                f"loaded weights from {MODEL_FILE} "
                f"(rounds_done={self.rounds_done})")
            return w
        self.logger.warning(
            f"weight shape {w.shape} != ({N_FEATURES},); starting from zero"
        )
    if not getattr(self, "train", False):
        self.logger.warning("no model found and not training -- weights are zero")
    return np.zeros(N_FEATURES, dtype=np.float64)


def save_weights(self):
    np.savez(MODEL_FILE, weights=self.weights,
             rounds_done=getattr(self, "rounds_done", 0))


def q_values(self, game_state):
    """Q(s, .) for all six actions, plus the context that produced them."""
    ctx = Context(game_state)
    phis = phi_all(ctx)
    return phis @ self.weights, phis, ctx


def act(self, game_state: dict) -> str:
    q, phis, ctx = q_values(self, game_state)

    self.last_context = ctx
    self.last_phi_all = phis

    if getattr(self, "train", False):
        action = _explore(self, q, ctx)
    else:
        action = ACTIONS[int(np.argmax(q))]

    self.last_action = action
    self.logger.debug(
        f"step {game_state['step']} q="
        + " ".join(f"{a}:{v:+.2f}" for a, v in zip(ACTIONS, q))
        + f" -> {action}"
    )
    return action


def _explore(self, q, ctx):
    """Boltzmann sampling with an annealed survivability prior.

    Two things are annealed independently:

      temperature      -- softmax sharpness, the usual exploration knob
      safe sampling    -- probability that the exploratory draw is restricted
                          to actions the escape BFS believes are survivable

    The second exists because of a failure mode we hit repeatedly with plain
    epsilon-greedy: an exploratory BOMB is normally followed by *random*
    movement, which normally kills the agent.  Q(BOMB) therefore collapses in
    the first few hundred rounds and never recovers, and the agent converges
    to a coward that never bombs at all.  Restricting early exploration to
    survivable actions decouples "was bombing a good idea" from "did the
    random walk afterwards happen to kill me".

    The prior is annealed to zero well before training ends and is absent
    entirely at test time, so the submitted policy is a pure argmax.
    """
    rounds = getattr(self, "rounds_done", 0)

    temp = TEMP_END + (TEMP_START - TEMP_END) * max(
        0.0, 1.0 - rounds / TEMP_DECAY_ROUNDS
    )
    p_safe = SAFE_SAMPLING_END + (SAFE_SAMPLING_START - SAFE_SAMPLING_END) * max(
        0.0, 1.0 - rounds / SAFE_SAMPLING_DECAY_ROUNDS
    )

    candidates = list(ACTIONS)
    if self.rng.random() < p_safe:
        safe = survivable_actions(ctx)
        if safe:
            candidates = safe

    idx = [ACTIONS.index(a) for a in candidates]
    logits = np.asarray(q, dtype=np.float64)[idx] / max(temp, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return candidates[int(self.rng.choice(len(candidates), p=p))]
