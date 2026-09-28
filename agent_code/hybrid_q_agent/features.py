from collections import deque
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_MAP = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'WAIT': (0, 0)
}
FEATURE_DIM = 24


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


def get_bomb_blast(arena, pos):
    xb, yb = pos
    blast = set([(xb, yb)])
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for dist in range(1, 4):
            nx, ny = xb + dx * dist, yb + dy * dist
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if arena[nx, ny] == -1:
                    break
                blast.add((nx, ny))
                if arena[nx, ny] == 1:
                    break
            else:
                break
    return blast


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


def has_safe_escape(arena, my_pos, bombs, explosion_map):
    bomb_blast = get_bomb_blast(arena, my_pos)
    existing_danger = get_danger_map(arena, bombs, explosion_map)

    free = (arena == 0) & (explosion_map == 0)
    for bpos, _ in bombs:
        free[bpos] = False

    queue = deque([(my_pos, 0)])
    visited = {my_pos: 0}

    while queue:
        (cx, cy), d = queue.popleft()
        if (cx, cy) not in bomb_blast and existing_danger[cx, cy] > 4:
            return True

        if d < 4:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    if free[nx, ny] and existing_danger[nx, ny] > d + 1:
                        if (nx, ny) not in visited or visited[(nx, ny)] > d + 1:
                            visited[(nx, ny)] = d + 1
                            queue.append(((nx, ny), d + 1))
    return False


def enemy_can_escape(arena, enemy_pos, bomb_blast, bombs, explosion_map):
    existing_danger = get_danger_map(arena, bombs, explosion_map)
    free = (arena == 0) & (explosion_map == 0)
    for bpos, _ in bombs:
        free[bpos] = False

    queue = deque([(enemy_pos, 0)])
    visited = {enemy_pos: 0}
    safe_count = 0

    while queue:
        (cx, cy), d = queue.popleft()
        if (cx, cy) not in bomb_blast and existing_danger[cx, cy] > 4:
            safe_count += 1
            if safe_count >= 3:
                return safe_count

        if d < 4:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    if free[nx, ny] and existing_danger[nx, ny] > d + 1:
                        if (nx, ny) not in visited or visited[(nx, ny)] > d + 1:
                            visited[(nx, ny)] = d + 1
                            queue.append(((nx, ny), d + 1))
    return safe_count


def count_bomb_targets(arena, pos, others):
    xb, yb = pos
    crate_count = 0
    opponent_count = 0
    other_coords = set(others)

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for dist in range(1, 4):
            nx, ny = xb + dx * dist, yb + dy * dist
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if arena[nx, ny] == -1:
                    break
                if arena[nx, ny] == 1:
                    crate_count += 1
                    break
                if (nx, ny) in other_coords:
                    opponent_count += 1
            else:
                break

    return crate_count, opponent_count


def count_unique_bomb_crates(arena, pos, danger_map):
    xb, yb = pos
    unique_count = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for dist in range(1, 4):
            nx, ny = xb + dx * dist, yb + dy * dist
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if arena[nx, ny] == -1:
                    break
                if arena[nx, ny] == 1:
                    if danger_map[nx, ny] > 4:
                        unique_count += 1
                    break
            else:
                break
    return unique_count


def count_enemy_escape_routes(arena, pos, others, bombs, explosion_map):
    bomb_blast = get_bomb_blast(arena, pos)
    enemies_in_blast = [epos for epos in others if epos in bomb_blast]
    enemies_trapped = 0
    for epos in enemies_in_blast:
        safe_tiles = enemy_can_escape(arena, epos, bomb_blast, bombs, explosion_map)
        if safe_tiles <= 1:
            enemies_trapped += 1
    return len(enemies_in_blast), enemies_trapped


def extract_action_features(game_state, action, coordinate_history=None):
    if game_state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    arena = game_state['field']
    _, score, bombs_left, my_pos = game_state['self']
    bombs = game_state['bombs']
    bomb_coords = set(xy for (xy, t) in bombs)
    others = [xy for (n, s, b, xy) in game_state['others']]
    other_coords = set(others)
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    x, y = my_pos
    danger_map = get_danger_map(arena, bombs, explosion_map)
    my_danger = danger_map[x, y]

    free_space = (arena == 0) & (explosion_map == 0)
    for bx, by in bomb_coords:
        if (bx, by) != my_pos:
            free_space[bx, by] = False
    for ox, oy in other_coords:
        free_space[ox, oy] = False

    # Safe haven escape calculation
    safe_tiles = [(cx, cy) for cx in range(1, arena.shape[0]-1) for cy in range(1, arena.shape[1]-1)
                  if arena[cx, cy] == 0 and danger_map[cx, cy] > 4 and (cx, cy) not in other_coords]
    escape_dist, escape_next, _ = bfs_distance_and_step(arena, my_pos, safe_tiles, free_space)

    # Crate targets calculation
    crates = [(cx, cy) for cx in range(1, arena.shape[0]-1) for cy in range(1, arena.shape[1]-1) if arena[cx, cy] == 1]
    crate_targets = []
    for cx, cy in crates:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] and free_space[nx, ny] and danger_map[nx, ny] > 4:
                crate_targets.append((nx, ny))

    # Independent BFS calculations to all targets
    coin_dist, coin_next, _ = bfs_distance_and_step(arena, my_pos, coins, free_space)
    crate_dist, crate_next, _ = bfs_distance_and_step(arena, my_pos, crate_targets, free_space)
    enemy_dist, enemy_next, _ = bfs_distance_and_step(arena, my_pos, others, free_space)

    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Feature 15: Enemy proximity indicator (state-level feature)
    enemy_is_close = (enemy_dist <= 3 and len(others) > 0)
    features[15] = 1.0 if enemy_is_close else 0.0

    # Feature 13: Constant model bias
    features[13] = 1.0

    # Feature 18: Inverse distance to nearest coin
    features[18] = float(1.0 / (coin_dist + 1.0)) if coin_dist < 99 else 0.0

    # Feature 19: Inverse distance to nearest safe crate target
    features[19] = float(1.0 / (crate_dist + 1.0)) if crate_dist < 99 else 0.0

    # Feature 20: Inverse distance to nearest enemy
    features[20] = float(1.0 / (enemy_dist + 1.0)) if enemy_dist < 99 else 0.0

    # Feature 21: Adjacent enemy melee contact sensor (touching enemy)
    min_enemy_dist = min([abs(ox - x) + abs(oy - y) for ox, oy in others]) if others else 99
    features[21] = 1.0 if min_enemy_dist <= 1 else 0.0

    # Feature 22: Closest opponent has bomb available
    closest_has_bomb = False
    if game_state.get('others'):
        closest_opp = min(game_state['others'], key=lambda o: abs(o[3][0] - x) + abs(o[3][1] - y))
        closest_has_bomb = bool(closest_opp[2])
    features[22] = 1.0 if closest_has_bomb else 0.0

    if action == 'BOMB':
        # Rule 1: is_valid for BOMB means ONLY:
        # bombs_left > 0 AND current tile is not occupied by an existing bomb.
        is_bomb_valid = (bombs_left > 0 and my_pos not in bomb_coords)
        features[0] = 1.0 if is_bomb_valid else -1.0

        # Feature 1 & 2: Danger at current position
        features[1] = 1.0 if my_danger <= 1 else 0.0
        features[2] = 1.0 if 1 < my_danger <= 4 else 0.0

        # Feature 3: Escapes danger (dropping bomb does not escape current danger)
        features[3] = -1.0 if my_danger <= 4 else 0.0

        crates_hit, enemies_hit = count_bomb_targets(arena, my_pos, others)
        can_escape = has_safe_escape(arena, my_pos, bombs, explosion_map) if is_bomb_valid else False

        # Feature 7 & 8: Target counts in blast area
        features[7] = float(crates_hit)
        features[8] = float(enemies_hit)

        # Feature 9: Escape route available after dropping bomb
        features[9] = 1.0 if can_escape else -1.0

        # Feature 12: Tactical blast value (continuous target metric)
        if can_escape:
            features[12] = float(crates_hit * 2.0 + enemies_hit * 8.0)

        # Feature 14: Enemy trapped count
        if enemies_hit > 0 and can_escape:
            _, trapped = count_enemy_escape_routes(arena, my_pos, others, bombs, explosion_map)
            features[14] = float(trapped)

        # Feature 17: Bomb near enemy indicator
        if enemies_hit > 0 and can_escape and enemy_dist <= 2:
            features[17] = float(enemies_hit)

        # Feature 23: Unique fresh crates hit (not already in an active bomb blast)
        if can_escape:
            unique_crates = count_unique_bomb_crates(arena, my_pos, danger_map)
            features[23] = float(unique_crates)

    else:
        # Movement actions (UP, RIGHT, DOWN, LEFT) or WAIT
        dx, dy = ACTION_MAP[action]
        nx, ny = x + dx, y + dy
        in_bounds = (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1])

        # Rule 2 & 3: Movement and WAIT validity
        if action == 'WAIT':
            is_action_valid = True
        else:
            is_action_valid = (in_bounds and arena[nx, ny] == 0
                               and (nx, ny) not in bomb_coords
                               and (nx, ny) not in other_coords)
        features[0] = 1.0 if is_action_valid else -1.0

        if in_bounds and is_action_valid:
            target_danger = danger_map[nx, ny]

            # Feature 1 & 2: Danger level of target tile
            features[1] = 1.0 if target_danger <= 1 else 0.0
            features[2] = 1.0 if 1 < target_danger <= 4 else 0.0

            # Feature 3: Escape progress (Independent)
            if my_danger <= 4:
                if escape_next is not None and (nx, ny) == escape_next:
                    features[3] = 1.5
                elif target_danger > my_danger:
                    features[3] = 0.8
                else:
                    features[3] = -1.0

            # Rule 4, 5, 6, 7: Compute coin, crate, enemy features INDEPENDENTLY!
            # No if/elif hierarchy. Neither danger nor enemy proximity suppresses them.

            # Feature 4: Coin navigation (always computed if coins exist)
            if coin_next is not None:
                features[4] = 1.0 if (nx, ny) == coin_next else -0.5

            # Feature 5: Crate navigation (always computed if crate targets exist)
            if crate_next is not None:
                features[5] = 1.0 if (nx, ny) == crate_next else -0.5

            # Feature 6: Enemy pursuit (always computed if opponents exist)
            if enemy_next is not None:
                features[6] = 1.0 if (nx, ny) == enemy_next else -0.5

            # Feature 10: Dead-end hazard
            neighbors = [(nx + ddx, ny + ddy) for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            walkable_neighbors = sum(1 for gx, gy in neighbors
                                     if 0 <= gx < arena.shape[0] and 0 <= gy < arena.shape[1] and arena[gx, gy] == 0)
            if walkable_neighbors <= 1 and target_danger <= 4:
                features[10] = 1.0

            # Feature 11: Repeated tile / loop penalty
            if coordinate_history and coordinate_history.count((nx, ny)) >= 2:
                features[11] = 1.0

            # Feature 16: Chase nearby enemy
            if enemy_is_close and enemy_next is not None and (nx, ny) == enemy_next:
                features[16] = 1.0
        else:
            # Physically blocked move has high hazard
            features[1] = 1.0

    return features


def extract_all_features(game_state, coordinate_history=None):
    matrix = np.zeros((len(ACTIONS), FEATURE_DIM), dtype=np.float32)
    for i, a in enumerate(ACTIONS):
        matrix[i] = extract_action_features(game_state, a, coordinate_history)
    return matrix
