"""
train.py: fitted Q-iteration for the GBDT agent.

Batch RL, not online TD. Transitions accumulate in a replay buffer; every
REFIT_EVERY rounds the whole model gets refitted from scratch on targets
computed with the *previous* model:

    y_i = r_i + gamma * max_a' Q_prev(s'_i, a')        (0 for terminal states)
    Q_new <- fit(phi(s_i, a_i) -> y_i)

Refitting from scratch instead of updating incrementally is what makes
this stable: the regression problem is fully specified before the fit
starts, so there's no moving target within a single fit. First fit uses
targets y = r (a one-step reward model), each later round pushes value
one step further.

n-step returns (N_STEP) are used on top of FQI. With one-step targets the
death penalty only moves back one step per refit, and the optimistic max
over next actions assumes escape, so the blame never reached the bomb that
started a four-step fuse (first version scored 0.16). Accumulating N_STEP
rewards puts the death and the bomb in the same return.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import events as e

from .callbacks import q_values, save_model
from .features import Context, N_FEATURES, phi_all
from .tactics import ACTIONS, BFS_INF, S

GAMMA = 0.95
N_STEP = 6                   # swept 3/4/6/8; 6 was best. Must be >= 5 to span the bomb fuse.
BUFFER_SIZE = 250_000        # ~270 MB at float32, fits the 8 GB tournament box
REFIT_EVERY = 100            # rounds
FQI_ITERATIONS = 3           # value-propagation sweeps per refit
MIN_SAMPLES_TO_FIT = 2_000

GBDT_PARAMS = dict(
    max_iter=200,
    learning_rate=0.08,
    max_depth=8,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
)

# reward scale read from settings at runtime, since it can change up to
# a week before the deadline
R_KILL = float(S["REWARD_KILL"])
R_COIN = float(S["REWARD_COIN"])

EVENT_REWARDS = {
    e.COIN_COLLECTED: R_COIN,
    e.KILLED_OPPONENT: 1.5 * R_KILL,
    e.CRATE_DESTROYED: 0.25 * R_COIN,
    e.COIN_FOUND: 0.2 * R_COIN,
    e.KILLED_SELF: -1.5 * R_KILL,
    e.GOT_KILLED: -1.0 * R_KILL,
    e.INVALID_ACTION: -0.1 * R_COIN,
    e.SURVIVED_ROUND: 0.5 * R_COIN,
}
STEP_PENALTY = -0.02 * R_COIN

PHI_COIN = 0.30
PHI_CRATE = 0.10
PHI_DANGER = 0.50

STATS_FILE = os.path.join(os.path.dirname(__file__), "training_stats.csv")
N_ACTIONS = len(ACTIONS)


class ReplayBuffer:
    """Fixed-size ring buffer of (phi(s,a), r, phi(s', .), done).

    phi(s', .) is stored for all six actions since the FQI target needs
    max_a' Q(s', a'), and recomputing at fit time would mean replaying
    episodes. At 47 features that's ~1.1 kB per transition.
    """

    def __init__(self, capacity=BUFFER_SIZE):
        self.capacity = capacity
        self.phi = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.phi_next = np.zeros((capacity, N_ACTIONS, N_FEATURES), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.disc = np.zeros(capacity, dtype=np.float32)   # gamma**n for this entry
        self.pos = 0
        self.full = False

    def add(self, phi_sa, reward, phi_next_all, done, disc=None):
        i = self.pos
        self.disc[i] = GAMMA ** N_STEP if disc is None else disc
        self.phi[i] = phi_sa
        self.reward[i] = reward
        if phi_next_all is None:
            self.phi_next[i] = 0.0
        else:
            self.phi_next[i] = phi_next_all
        self.done[i] = done
        self.pos = (i + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos


def setup_training(self):
    self.buffer = ReplayBuffer()
    self.pending = deque(maxlen=N_STEP)
    self.fits = 0
    self.episode = _fresh_episode()
    self.rounds_done = getattr(self, "rounds_done", 0)
    _init_stats_file()
    self.logger.info(
        f"FQI: gamma={GAMMA} refit_every={REFIT_EVERY} "
        f"iterations={FQI_ITERATIONS} buffer={BUFFER_SIZE}")


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return

    old_ctx = _context_for(self, old_game_state)
    old_phis = (self.last_phi_all if old_ctx is self.last_context
                else phi_all(old_ctx))
    new_ctx = Context(new_game_state)
    new_phis = phi_all(new_ctx)

    reward = _reward(events) + _shaping(old_ctx, new_ctx)
    a = ACTIONS.index(self_action)
    self.pending.append((old_phis[a], reward))
    if len(self.pending) == N_STEP:
        g = sum(GAMMA ** i * r for i, (_, r) in enumerate(self.pending))
        self.buffer.add(self.pending[0][0], g, new_phis, done=False)
    _record(self.episode, events)


def end_of_round(self, last_game_state, last_action, events):
    # the framework passes the final step's events to game_events_occurred
    # and then again to end_of_round, so only the terminal bootstrap
    # (done=True) is recorded here to avoid double-counting that reward
    if last_action is not None and self.last_phi_all is not None:
        reward = _reward(events) + STEP_PENALTY
        a = ACTIONS.index(last_action)
        self.pending.append((self.last_phi_all[a], reward))

    # flush the tail with exact returns: the death penalty sits at the
    # end of this window, so every action within N_STEP of it including
    # the bomb that started the fuse gets blamed directly instead of
    # waiting for it to propagate one refit at a time
    while self.pending:
        g = sum(GAMMA ** i * r for i, (_, r) in enumerate(self.pending))
        self.buffer.add(self.pending[0][0], g, None, done=True)
        self.pending.popleft()

    # send_game_events() skips dead agents, so this is the only place
    # KILLED_SELF / GOT_KILLED ever show up -- dropping this line made
    # deaths invisible to the episode stats
    _record(self.episode, events)

    self.rounds_done += 1
    _write_stats(self, last_game_state)
    self.episode = _fresh_episode()

    if self.rounds_done % REFIT_EVERY == 0 and len(self.buffer) >= MIN_SAMPLES_TO_FIT:
        _refit(self)


def _refit(self):
    """Refit the regressor from scratch on the whole buffer."""
    n = len(self.buffer)
    phi = self.buffer.phi[:n]
    reward = self.buffer.reward[:n]
    phi_next = self.buffer.phi_next[:n]
    done = self.buffer.done[:n]
    disc = self.buffer.disc[:n]

    flat_next = phi_next.reshape(-1, N_FEATURES)
    model = self.model
    t0 = time.time()

    for it in range(FQI_ITERATIONS):
        if model is None:
            targets = reward.astype(np.float64)      # iteration 0: reward model
        else:
            q_next = model.predict(flat_next).reshape(n, N_ACTIONS).max(axis=1)
            q_next[done] = 0.0
            targets = reward + disc * q_next

        model = HistGradientBoostingRegressor(**GBDT_PARAMS)
        model.fit(phi, targets)

    self.model = model
    self.fits += 1
    save_model(self, model, fits=self.fits)
    self.logger.info(
        f"refit #{self.fits} on {n} samples in {time.time() - t0:.1f}s "
        f"(round {self.rounds_done})")


def _reward(events) -> float:
    total = STEP_PENALTY
    for ev in events:
        total += EVENT_REWARDS.get(ev, 0.0)
    return total


def _potential(ctx) -> float:
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
    ctx = self.last_context
    if (ctx is not None and ctx.step == game_state["step"]
            and ctx.pos == game_state["self"][3]):
        return ctx
    return Context(game_state)


STAT_FIELDS = [
    "round", "score", "steps", "coins", "crates", "kills", "suicide",
    "got_killed", "survived", "invalid", "bombs", "buffer", "fits",
]


def _fresh_episode():
    return dict.fromkeys(
        ["coins", "crates", "kills", "suicide", "got_killed", "survived",
         "invalid", "bombs"], 0)


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
    with open(STATS_FILE, "a", newline="") as fh:
        csv.writer(fh).writerow([
            self.rounds_done, score, steps, ep["coins"], ep["crates"],
            ep["kills"], ep["suicide"], ep["got_killed"], ep["survived"],
            ep["invalid"], ep["bombs"], len(self.buffer), self.fits,
        ])
