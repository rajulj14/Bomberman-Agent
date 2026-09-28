from collections import deque
import os
import pickle
import numpy as np

import events as e
from .features import ACTIONS, FEATURE_DIM, extract_action_features, extract_all_features, get_danger_map, has_safe_escape, count_bomb_targets, bfs_distance_and_step, count_enemy_escape_routes

CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
AWAY_FROM_CRATE = "AWAY_FROM_CRATE"
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
AWAY_FROM_ENEMY = "AWAY_FROM_ENEMY"

DROPPED_SAFE_BOMB = "DROPPED_SAFE_BOMB"
DROPPED_ENEMY_BOMB = "DROPPED_ENEMY_BOMB"
DROPPED_TRAP_BOMB = "DROPPED_TRAP_BOMB"
DROPPED_PINCER_BOMB = "DROPPED_PINCER_BOMB"
DROPPED_CLUSTER_BOMB = "DROPPED_CLUSTER_BOMB"
DROPPED_SUICIDE_BOMB = "DROPPED_SUICIDE_BOMB"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
MOVED_IN_LOOP = "MOVED_IN_LOOP"

REWARD_MAP = {
    e.COIN_COLLECTED: 15.0,
    e.KILLED_OPPONENT: 65.0,
    e.CRATE_DESTROYED: 6.0,
    e.COIN_FOUND: 3.0,
    e.SURVIVED_ROUND: 4.0,
    
    e.KILLED_SELF: -30.0,
    e.GOT_KILLED: -12.0,
    e.INVALID_ACTION: -2.0,
    e.WAITED: -1.5,

    CLOSER_TO_COIN: 1.0,
    AWAY_FROM_COIN: -1.0,
    CLOSER_TO_CRATE: 1.0,
    AWAY_FROM_CRATE: -0.8,
    CLOSER_TO_ENEMY: 2.0,
    AWAY_FROM_ENEMY: -1.0,

    DROPPED_SAFE_BOMB: 9.0,
    DROPPED_ENEMY_BOMB: 25.0,
    DROPPED_TRAP_BOMB: 40.0,
    DROPPED_PINCER_BOMB: 50.0,
    DROPPED_CLUSTER_BOMB: 20.0,
    DROPPED_SUICIDE_BOMB: -25.0,
    ESCAPED_DANGER: 4.0,
    ENTERED_DANGER: -5.0,
    MOVED_IN_LOOP: -1.0
}

N_STEPS = 4
GAMMA = 0.95
ALPHA = 0.0008
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.03


def setup_training(self):
    self.n_step_buffer = deque([], maxlen=N_STEPS)
    self.learning_rate = ALPHA
    self.gamma = GAMMA
    self.epsilon = getattr(self, "epsilon", 0.5)
    self.round_counter = 0
    self.update_counter = 0
    self.round_td_errors = []
    self.initial_weights = self.weights.copy()
    self.logger.info(f"Training setup: α={ALPHA}, γ={GAMMA}, n_steps={N_STEPS}")
    self.logger.info(f"Initial weight norm: {np.linalg.norm(self.weights):.4f}")


def reward_from_events(self, events) -> float:
    total = 0.0
    for event in events:
        if event in REWARD_MAP:
            total += REWARD_MAP[event]
    return total


def detect_custom_events(old_game_state, self_action, new_game_state):
    custom_events = []
    if old_game_state is None:
        return custom_events

    old_arena = old_game_state['field']
    _, _, old_bombs_left, old_pos = old_game_state['self']
    old_bombs = old_game_state['bombs']
    old_explosion = old_game_state['explosion_map']
    old_others = [xy for (n, s, b, xy) in old_game_state['others']]
    old_coins = old_game_state['coins']

    old_danger = get_danger_map(old_arena, old_bombs, old_explosion)
    old_my_danger = old_danger[old_pos]

    free_space = (old_arena == 0) & (old_explosion == 0)
    for bx, by in [xy for (xy, t) in old_bombs]:
        free_space[bx, by] = False
    for ox, oy in old_others:
        free_space[ox, oy] = False

    old_crates = [(cx, cy) for cx in range(1, old_arena.shape[0]-1) for cy in range(1, old_arena.shape[1]-1) if old_arena[cx, cy] == 1]
    
    crate_targets = []
    for cx, cy in old_crates:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < old_arena.shape[0] and 0 <= ny < old_arena.shape[1] and free_space[nx, ny]:
                crate_targets.append((nx, ny))

    old_coin_d, _, _ = bfs_distance_and_step(old_arena, old_pos, old_coins, free_space)
    old_crate_d, _, _ = bfs_distance_and_step(old_arena, old_pos, crate_targets, free_space)
    old_enemy_d, _, _ = bfs_distance_and_step(old_arena, old_pos, old_others, free_space)

    if new_game_state is not None:
        new_arena = new_game_state['field']
        _, _, _, new_pos = new_game_state['self']
        new_bombs = new_game_state['bombs']
        new_explosion = new_game_state['explosion_map']
        new_others = [xy for (n, s, b, xy) in new_game_state['others']]
        new_coins = new_game_state['coins']

        new_danger = get_danger_map(new_arena, new_bombs, new_explosion)
        new_my_danger = new_danger[new_pos]

        new_free_space = (new_arena == 0) & (new_explosion == 0)
        for bx, by in [xy for (xy, t) in new_bombs]:
            new_free_space[bx, by] = False
        for ox, oy in new_others:
            new_free_space[ox, oy] = False

        new_crates = [(cx, cy) for cx in range(1, new_arena.shape[0]-1) for cy in range(1, new_arena.shape[1]-1) if new_arena[cx, cy] == 1]
        new_crate_targets = []
        for cx, cy in new_crates:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < new_arena.shape[0] and 0 <= ny < new_arena.shape[1] and new_free_space[nx, ny]:
                    new_crate_targets.append((nx, ny))

        new_coin_d, _, _ = bfs_distance_and_step(new_arena, new_pos, new_coins, new_free_space)
        new_crate_d, _, _ = bfs_distance_and_step(new_arena, new_pos, new_crate_targets, new_free_space)
        new_enemy_d, _, _ = bfs_distance_and_step(new_arena, new_pos, new_others, new_free_space)

        if old_my_danger <= 4 and new_my_danger > 4:
            custom_events.append(ESCAPED_DANGER)
        elif old_my_danger > 4 and new_my_danger <= 4:
            custom_events.append(ENTERED_DANGER)

        if old_coin_d < 99 and new_coin_d < 99:
            if new_coin_d < old_coin_d:
                custom_events.append(CLOSER_TO_COIN)
            elif new_coin_d > old_coin_d:
                custom_events.append(AWAY_FROM_COIN)

        if old_crate_d < 99 and new_crate_d < 99 and (old_coin_d == 99 or old_coin_d > 4):
            if new_crate_d < old_crate_d:
                custom_events.append(CLOSER_TO_CRATE)
            elif new_crate_d > old_crate_d:
                custom_events.append(AWAY_FROM_CRATE)

        if old_enemy_d < 99 and new_enemy_d < 99 and old_enemy_d <= 5:
            if new_enemy_d < old_enemy_d:
                custom_events.append(CLOSER_TO_ENEMY)
            elif new_enemy_d > old_enemy_d:
                custom_events.append(AWAY_FROM_ENEMY)

    if self_action == 'BOMB':
        crates_hit, enemies_hit = count_bomb_targets(old_arena, old_pos, old_others)
        safe = has_safe_escape(old_arena, old_pos, old_bombs, old_explosion)
        if safe and enemies_hit > 0:
            _, trapped = count_enemy_escape_routes(old_arena, old_pos, old_others, old_bombs, old_explosion)
            if trapped >= 1:
                custom_events.append(DROPPED_PINCER_BOMB)
            else:
                custom_events.append(DROPPED_ENEMY_BOMB)
        elif safe and crates_hit >= 2:
            custom_events.append(DROPPED_CLUSTER_BOMB)
        elif safe and crates_hit > 0:
            custom_events.append(DROPPED_SAFE_BOMB)
        elif not safe:
            custom_events.append(DROPPED_SUICIDE_BOMB)

    return custom_events


def update_q_weights(self, state, action, n_step_return, next_state):
    phi_sa = extract_action_features(state, action, getattr(self, "coordinate_history", None))
    current_q = np.dot(self.weights, phi_sa)

    if next_state is not None:
        next_features = extract_all_features(next_state, getattr(self, "coordinate_history", None))
        next_q_values = np.dot(next_features, self.weights)
        target = n_step_return + (self.gamma ** N_STEPS) * np.max(next_q_values)
    else:
        target = n_step_return

    td_error = target - current_q
    td_error = np.clip(td_error, -15.0, 15.0)
    self.weights += self.learning_rate * td_error * phi_sa
    self.update_counter = getattr(self, "update_counter", 0) + 1
    if hasattr(self, "round_td_errors"):
        self.round_td_errors.append(abs(td_error))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    if old_game_state is None:
        return

    custom_events = detect_custom_events(old_game_state, self_action, new_game_state)
    all_events = events + custom_events
    reward = reward_from_events(self, all_events)
    self.n_step_buffer.append((old_game_state, self_action, reward, new_game_state))

    if len(self.n_step_buffer) == N_STEPS:
        s_0, a_0, _, _ = self.n_step_buffer[0]
        n_return = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(N_STEPS))
        s_n = self.n_step_buffer[-1][3]
        update_q_weights(self, s_0, a_0, n_return, s_n)


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    custom_events = detect_custom_events(last_game_state, last_action, None)
    all_events = events + custom_events
    reward = reward_from_events(self, all_events)

    self.n_step_buffer.append((last_game_state, last_action, reward, None))

    while len(self.n_step_buffer) > 0:
        s_0, a_0, _, _ = self.n_step_buffer[0]
        k = len(self.n_step_buffer)
        n_return = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(k))
        update_q_weights(self, s_0, a_0, n_return, None)
        self.n_step_buffer.popleft()

    self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
    self.round_counter += 1

    # Training statistics logging
    weight_norm = np.linalg.norm(self.weights)
    weight_change = np.linalg.norm(self.weights - self.initial_weights) if hasattr(self, "initial_weights") else 0.0
    avg_td = np.mean(self.round_td_errors) if hasattr(self, "round_td_errors") and len(self.round_td_errors) > 0 else 0.0
    updates = getattr(self, "update_counter", 0)

    if self.round_counter % 10 == 0 or self.round_counter <= 3:
        self.logger.info(f"Round {self.round_counter}: updates={updates}, "
                         f"avg_td_err={avg_td:.4f}, weight_norm={weight_norm:.4f}, "
                         f"weight_change={weight_change:.4f}, eps={self.epsilon:.4f}")
        self.logger.info(f"  Weights: {self.weights}")

    # Reset per-round TD errors
    if hasattr(self, "round_td_errors"):
        self.round_td_errors = []

    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    with open(model_path, "wb") as f:
        pickle.dump({"weights": self.weights, "epsilon": self.epsilon, "round": self.round_counter}, f)
