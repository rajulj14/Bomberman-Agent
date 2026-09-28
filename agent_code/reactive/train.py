"""
train.py -- n-step Q-learning for Agent A.

Update rule, semi-gradient n-step Q-learning (Sutton & Barto, ch. 7):

    G_t   = sum_{i=0}^{n-1} gamma^i r_{t+i}  +  gamma^n max_a Q(s_{t+n}, a)
    w    <- w + alpha (G_t - Q(s_t, a_t)) phi(s_t, a_t)

n is set to 4 on purpose.  The consequence of a BOMB lands
BOMB_TIMER + fuse steps after the action that caused it, so one-step TD has to
propagate credit backwards across the whole fuse before bombing means
anything.  With n = 4 the death (or the destroyed crates) is inside the same
return as the action that caused it.

This is off-policy without importance correction, since the behaviour policy
is Boltzmann while the target is greedy.  For n = 4 the bias is small and the
variance saving is large; we note the trade-off in the report rather than
pretending it is exact.
"""

from __future__ import annotations

import csv
import os
from collections import deque

import numpy as np

import events as e

from .callbacks import q_values, save_weights
from .features import Context
from .tactics import ACTIONS, BFS_INF, S

# --------------------------------------------------------------------------
# Hyperparameters.  Every one of these is a row in the experiments table.
# --------------------------------------------------------------------------

N_STEP = 4
GAMMA = 0.95
ALPHA = 0.005
ALPHA_MIN = 0.0005
ALPHA_DECAY_ROUNDS = 3000
SAVE_EVERY = 50

# Reward scale is read from settings at runtime -- the organisers may change
# it up to seven days before the deadline, so nothing below is a literal 5.
R_KILL = float(S["REWARD_KILL"])
R_COIN = float(S["REWARD_COIN"])

# Non-shaping rewards.  These change the objective; they are scaled against
# the 24-point ceiling of a classic episode rather than against intuition.
EVENT_REWARDS = {
    e.COIN_COLLECTED: R_COIN,
    e.KILLED_OPPONENT: R_KILL,
    e.CRATE_DESTROYED: 0.05 * R_COIN,
    e.COIN_FOUND: 0.2 * R_COIN,
    e.KILLED_SELF: -1.5 * R_KILL,
    e.GOT_KILLED: -1.0 * R_KILL,
    e.INVALID_ACTION: -0.1 * R_COIN,
    e.SURVIVED_ROUND: 0.5 * R_COIN,
}

STEP_PENALTY = -0.02 * R_COIN   # attacks passivity directly

# Potential-based shaping weights (state-only -> optimal policy preserved,
# Ng, Harada & Russell 1999, which the brief's own footnote cites).
PHI_COIN = 0.30
PHI_CRATE = 0.10
PHI_DANGER = 0.50

STATS_FILE = os.path.join(os.path.dirname(__file__), "training_stats.csv")


def setup_training(self):
    self.transitions = deque(maxlen=N_STEP)
    # callbacks.setup restores this from the checkpoint -- do not zero it
    self.rounds_done = getattr(self, "rounds_done", 0)
    self.episode = _fresh_episode()
    self.td_errors = []
    _init_stats_file()
    self.logger.info(
        f"training: n={N_STEP} gamma={GAMMA} alpha={ALPHA} "
        f"kill/coin={R_KILL / max(R_COIN, 1e-9):.1f}"
    )


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    old_ctx = _context_for(self, old_game_state)
    old_phis = self.last_phi_all if old_ctx is self.last_context else None
    if old_phis is None:
        from .features import phi_all
        old_phis = phi_all(old_ctx)

    new_ctx = Context(new_game_state)
    reward = _reward(events) + _shaping(old_ctx, new_ctx)

    a = ACTIONS.index(self_action)
    self.transitions.append((old_phis[a].astype(np.float64), reward, new_ctx))
    _record(self.episode, events)

    if len(self.transitions) == N_STEP:
        _update(self, bootstrap_state=new_game_state)


def end_of_round(self, last_game_state, last_action, events):
    if last_action is not None and self.last_phi_all is not None:
        reward = _reward(events) + STEP_PENALTY
        a = ACTIONS.index(last_action)
        self.transitions.append(
            (self.last_phi_all[a].astype(np.float64), reward, None)
        )
    _record(self.episode, events)

    # flush the tail of the episode with no bootstrap -- these returns are
    # exact, which is precisely where the terminal signal lives
    while self.transitions:
        _update(self, bootstrap_state=None)
        self.transitions.popleft()

    self.rounds_done += 1
    _write_stats(self, last_game_state)
    self.episode = _fresh_episode()
    self.td_errors = []

    if self.rounds_done % SAVE_EVERY == 0:
        save_weights(self)
        self.logger.info(f"saved weights after round {self.rounds_done}")


# --------------------------------------------------------------------------
def _update(self, bootstrap_state):
    """One semi-gradient n-step update on the oldest queued transition."""
    g = 0.0
    discount = 1.0
    for _, reward, _ in self.transitions:
        g += discount * reward
        discount *= GAMMA

    if bootstrap_state is not None:
        q_next, _, _ = q_values(self, bootstrap_state)
        g += discount * float(np.max(q_next))

    phi_t = self.transitions[0][0]
    q_t = float(phi_t @ self.weights)
    td = g - q_t

    alpha = max(
        ALPHA_MIN,
        ALPHA * max(0.0, 1.0 - self.rounds_done / ALPHA_DECAY_ROUNDS)
        + ALPHA_MIN,
    )
    self.weights += alpha * td * phi_t
    self.td_errors.append(td)


def _reward(events) -> float:
    total = STEP_PENALTY
    for ev in events:
        total += EVENT_REWARDS.get(ev, 0.0)
    return total


def _potential(ctx: Context) -> float:
    """State-only potential.  Must not depend on the action taken."""
    if ctx is None:
        return 0.0
    phi = 0.0
    d_coin = ctx.dist_to_nearest_coin()
    if d_coin < BFS_INF:
        phi -= PHI_COIN * d_coin
    elif not ctx.coins:
        d_crate = ctx.dist_to_nearest_crate()
        if d_crate < BFS_INF:
            phi -= PHI_CRATE * d_crate
    if ctx.in_danger():
        phi -= PHI_DANGER
    return phi


def _shaping(old_ctx, new_ctx) -> float:
    return GAMMA * _potential(new_ctx) - _potential(old_ctx)


def _context_for(self, game_state):
    """Reuse the context act() already built for this state where possible."""
    ctx = self.last_context
    if ctx is not None and ctx.step == game_state["step"] and ctx.pos == game_state["self"][3]:
        return ctx
    return Context(game_state)


# --------------------------------------------------------------------------
# Results database, populated from round one.
# --------------------------------------------------------------------------

STAT_FIELDS = [
    "round", "score", "steps", "coins", "crates", "kills", "suicide",
    "got_killed", "survived", "invalid", "bombs", "mean_td", "weight_norm",
]


def _fresh_episode():
    return dict.fromkeys(
        ["coins", "crates", "kills", "suicide", "got_killed", "survived",
         "invalid", "bombs"], 0
    )


def _record(ep, events):
    for ev in events:
        if ev == e.COIN_COLLECTED:
            ep["coins"] += 1
        elif ev == e.CRATE_DESTROYED:
            ep["crates"] += 1
        elif ev == e.KILLED_OPPONENT:
            ep["kills"] += 1
        elif ev == e.KILLED_SELF:
            ep["suicide"] = 1
        elif ev == e.GOT_KILLED:
            ep["got_killed"] = 1
        elif ev == e.SURVIVED_ROUND:
            ep["survived"] = 1
        elif ev == e.INVALID_ACTION:
            ep["invalid"] += 1
        elif ev == e.BOMB_DROPPED:
            ep["bombs"] += 1


def _init_stats_file():
    if not os.path.isfile(STATS_FILE):
        with open(STATS_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow(STAT_FIELDS)


def _write_stats(self, last_game_state):
    ep = self.episode
    score = last_game_state["self"][1] if last_game_state else 0
    steps = last_game_state["step"] if last_game_state else 0
    row = [
        self.rounds_done, score, steps, ep["coins"], ep["crates"], ep["kills"],
        ep["suicide"], ep["got_killed"], ep["survived"], ep["invalid"],
        ep["bombs"],
        float(np.mean(self.td_errors)) if self.td_errors else 0.0,
        float(np.linalg.norm(self.weights)),
    ]
    with open(STATS_FILE, "a", newline="") as fh:
        csv.writer(fh).writerow(row)
