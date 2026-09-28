from collections import deque
import os
import pickle
import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_MAP = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'WAIT': (0, 0)
}
DIR_MAP = {'UP': 1, 'RIGHT': 2, 'DOWN': 3, 'LEFT': 4}


def get_danger_map(arena, bombs, explosion_map):
    danger = np.ones(arena.shape, dtype=int) * 99
    danger[explosion_map > 0] = 0
    for (xb, yb), t in bombs:
        danger[xb, yb] = min(danger[xb, yb], t)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for dist in range(1, 4):
                nx, ny = xb + dx * dist, yb + dy * dist
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    if arena[nx, ny] == -1:
                        break
                    danger[nx, ny] = min(danger[nx, ny], t)
                    if arena[nx, ny] == 1:
                        break
                else:
                    break
    return danger


def bfs_distance_and_step(arena, start, targets, free_space):
    if not targets or len(targets) == 0:
        return 99, None, None

    targets_set = set(targets)
    if start in targets_set:
        return 0, start, start

    queue = deque([start])
    parent = {start: None}
    dist = {start: 0}
    found_target = None

    while queue:
        curr = queue.popleft()
        if curr in targets_set:
            found_target = curr
            break

        cx, cy = curr
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if (neighbor in targets_set or free_space[nx, ny]) and neighbor not in parent:
                    parent[neighbor] = curr
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)

    if found_target is None:
        return 99, None, None

    curr = found_target
    while parent[curr] != start and parent[curr] is not None:
        curr = parent[curr]

    return dist[found_target], curr, found_target


def step_to_dir_idx(start, step_coord):
    if step_coord is None or start is None:
        return 0
    dx, dy = step_coord[0] - start[0], step_coord[1] - start[1]
    if (dx, dy) == (0, -1): return 1
    if (dx, dy) == (1, 0):  return 2
    if (dx, dy) == (0, 1):  return 3
    if (dx, dy) == (-1, 0): return 4
    return 0


def state_to_discrete_id(game_state):
    """
    Maps game_state into a compact integer ID in [0, 12000).
    """
    if game_state is None:
        return 0

    arena = game_state['field']
    _, _, bombs_left, my_pos = game_state['self']
    bombs = game_state['bombs']
    bomb_coords = set(xy for (xy, t) in bombs)
    others = [xy for (n, s, b, xy) in game_state['others']]
    other_coords = set(others)
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    x, y = my_pos
    danger_map = get_danger_map(arena, bombs, explosion_map)
    my_danger = danger_map[x, y]

    # Danger state (0=safe, 1=danger <= 1, 2=danger 2..4)
    if my_danger <= 1:
        d_state = 1
    elif my_danger <= 4:
        d_state = 2
    else:
        d_state = 0

    free_space = (arena == 0) & (explosion_map == 0)
    for bx, by in bomb_coords:
        free_space[bx, by] = False
    for ox, oy in other_coords:
        free_space[ox, oy] = False

    crates = [(cx, cy) for cx in range(1, arena.shape[0]-1) for cy in range(1, arena.shape[1]-1) if arena[cx, cy] == 1]
    crate_targets = []
    for cx, cy in crates:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and free_space[nx, ny]:
                crate_targets.append((nx, ny))

    _, coin_next, _ = bfs_distance_and_step(arena, my_pos, coins, free_space)
    _, crate_next, _ = bfs_distance_and_step(arena, my_pos, crate_targets, free_space)
    _, enemy_next, _ = bfs_distance_and_step(arena, my_pos, others, free_space)

    coin_dir = step_to_dir_idx(my_pos, coin_next)
    crate_dir = step_to_dir_idx(my_pos, crate_next)
    enemy_dir = step_to_dir_idx(my_pos, enemy_next)

    # Valid moves bitmask
    valid_mask = 0
    for i, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
        nx, ny = x + dx, y + dy
        if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
            if arena[nx, ny] == 0 and (nx, ny) not in bomb_coords and (nx, ny) not in other_coords:
                valid_mask |= (1 << i)

    can_bomb = 1 if (bombs_left > 0 and my_pos not in bomb_coords and (len(crates) > 0 or len(others) > 0)) else 0

    # Compact state index
    state_id = (d_state * 2500 + coin_dir * 500 + crate_dir * 100 + enemy_dir * 20 + can_bomb * 10 + (valid_mask % 10)) % 12000
    return state_id


def setup(self):
    self.coordinate_history = deque([], 10)
    model_path = os.path.join(os.path.dirname(__file__), "model.pt")

    if os.path.isfile(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data.get("q_table", np.zeros((12000, len(ACTIONS)), dtype=np.float32))
            self.epsilon = data.get("epsilon", 0.05)
    else:
        self.q_table = np.zeros((12000, len(ACTIONS)), dtype=np.float32)
        self.epsilon = 0.5 if getattr(self, "train", False) else 0.0


def act(self, game_state: dict) -> str:
    if game_state is None:
        return 'WAIT'

    state_id = state_to_discrete_id(game_state)
    q_values = self.q_table[state_id]

    if getattr(self, "train", False):
        if random.random() < getattr(self, "epsilon", 0.1):
            return random.choice(ACTIONS)

    best_idx = int(np.argmax(q_values))
    return ACTIONS[best_idx]


def state_to_features(game_state: dict) -> np.array:
    return np.array([state_to_discrete_id(game_state)], dtype=np.float32)
