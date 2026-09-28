import os
import pickle
import numpy as np

import events as e
from .callbacks import ACTIONS, state_to_discrete_id

REWARD_MAP = {
    e.COIN_COLLECTED: 10.0,
    e.KILLED_OPPONENT: 25.0,
    e.CRATE_DESTROYED: 4.0,
    e.COIN_FOUND: 2.0,
    e.SURVIVED_ROUND: 3.0,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -10.0,
    e.INVALID_ACTION: -1.0,
    e.WAITED: -0.2
}

ALPHA = 0.1
GAMMA = 0.95
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05


def setup_training(self):
    self.learning_rate = ALPHA
    self.gamma = GAMMA
    self.epsilon = getattr(self, "epsilon", 0.5)
    self.round_counter = 0


def reward_from_events(events):
    total = 0.0
    for ev in events:
        if ev in REWARD_MAP:
            total += REWARD_MAP[ev]
    return total


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    if old_game_state is None or self_action is None:
        return

    old_s = state_to_discrete_id(old_game_state)
    a_idx = ACTIONS.index(self_action)
    r = reward_from_events(events)

    if new_game_state is not None:
        new_s = state_to_discrete_id(new_game_state)
        target = r + self.gamma * np.max(self.q_table[new_s])
    else:
        target = r

    td_error = target - self.q_table[old_s, a_idx]
    self.q_table[old_s, a_idx] += self.learning_rate * td_error


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    if last_game_state is not None and last_action is not None:
        old_s = state_to_discrete_id(last_game_state)
        a_idx = ACTIONS.index(last_action)
        r = reward_from_events(events)
        td_error = r - self.q_table[old_s, a_idx]
        self.q_table[old_s, a_idx] += self.learning_rate * td_error

    self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
    self.round_counter += 1

    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    with open(model_path, "wb") as f:
        pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon, "round": self.round_counter}, f)
