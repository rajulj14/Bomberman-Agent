from collections import deque
import os
import pickle
import random
import numpy as np

from .features import ACTIONS, FEATURE_DIM, extract_all_features


def setup(self):
    self.coordinate_history = deque([], 12)
    self.current_round = 0
    model_path = os.path.join(os.path.dirname(__file__), "model.pt")

    if os.path.isfile(model_path):
        self.logger.info("Loading learned model from model.pt")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                w = data.get("weights", np.zeros(FEATURE_DIM, dtype=np.float32))
                if len(w) < FEATURE_DIM:
                    new_w = np.zeros(FEATURE_DIM, dtype=np.float32)
                    new_w[:len(w)] = w
                    self.weights = new_w
                else:
                    self.weights = w[:FEATURE_DIM].copy()
                self.epsilon = data.get("epsilon", 0.05)
                train_rounds = data.get("round", 0)
                self.logger.info(f"Loaded weights from round {train_rounds}, "
                                 f"weight norm={np.linalg.norm(self.weights):.4f}, "
                                 f"epsilon={self.epsilon:.4f}")
            else:
                self.logger.warning("model.pt has unexpected format, starting from zeros")
                self.weights = np.zeros(FEATURE_DIM, dtype=np.float32)
                self.epsilon = 0.05
    else:
        self.logger.info("No model.pt found — initializing zero weights for training")
        self.weights = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.epsilon = 0.5 if getattr(self, "train", False) else 0.0

    # Log feature/weight dimensions for verification
    self.logger.info(f"Feature dimension: {FEATURE_DIM}, Weight dimension: {len(self.weights)}")
    self.logger.info(f"Weight vector: {self.weights}")


def act(self, game_state: dict) -> str:
    """
    Purely learned action selection.
    Architecture: game_state → features → learned Q(s,a) = w^T φ(s,a) → argmax → action
    No hard-coded bonuses, penalties, or tactical overrides.
    """
    if game_state is None:
        return 'WAIT'

    # Round tracking for coordinate history
    if game_state["round"] != getattr(self, "current_round", 0):
        self.coordinate_history = deque([], 12)
        self.current_round = game_state["round"]

    # Step 1: Extract feature matrix — all domain knowledge lives here as STATE REPRESENTATION
    #         Shape: (6, FEATURE_DIM) — one row per action
    feature_matrix = extract_all_features(game_state, self.coordinate_history)

    # Step 2: Exploration during TRAINING only (not used in tournament inference)
    if getattr(self, "train", False):
        eps = getattr(self, "epsilon", 0.1)
        if random.random() < eps:
            # Explore: pick a random valid, non-dangerous action
            valid_indices = [i for i in range(len(ACTIONS))
                            if feature_matrix[i, 0] > 0 and feature_matrix[i, 1] == 0]
            if valid_indices:
                chosen_idx = random.choice(valid_indices)
            else:
                chosen_idx = random.randint(0, len(ACTIONS) - 1)
            _, _, _, (x, y) = game_state['self']
            self.coordinate_history.append((x, y))
            return ACTIONS[chosen_idx]

    # Step 3: LEARNED action selection — Q(s,a) = feature_matrix @ weights
    q_values = np.dot(feature_matrix, self.weights)

    # Step 4: Select action with highest learned Q-value
    best_idx = int(np.argmax(q_values))
    chosen_action = ACTIONS[best_idx]

    # Logging for verification
    self.logger.debug(f"Q-values: {dict(zip(ACTIONS, [f'{v:.2f}' for v in q_values]))}")
    self.logger.debug(f"Selected: {chosen_action} (idx={best_idx}, Q={q_values[best_idx]:.2f})")

    # Track position history
    _, _, _, (x, y) = game_state['self']
    self.coordinate_history.append((x, y))
    return chosen_action


def state_to_features(game_state: dict) -> np.array:
    return extract_all_features(game_state)
