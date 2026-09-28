import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import Ridge

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from environment import BombeRLeWorld, WorldArgs
from agent_code.hybrid_q_agent.features import ACTIONS, FEATURE_DIM, extract_action_features


def collect_expert_data(n_rounds=30):
    print(f"[*] Simulating {n_rounds} rounds of expert gameplay from rule_based_agent...")

    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    args = WorldArgs(
        no_gui=True,
        fps=0,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=log_dir,
        save_stats=False,
        match_name="pretrain",
        seed=None,
        silence_errors=True,
        scenario="classic"
    )

    agents = [("rule_based_agent", False) for _ in range(4)]
    world = BombeRLeWorld(args, agents)
    world.user_input = None

    X = []
    y = []

    for r in range(n_rounds):
        world.new_round()
        world.user_input = None
        while world.running:
            for agent in world.active_agents:
                state = world.get_state_for_agent(agent)
                if state is None:
                    continue
                try:
                    expert_action = agent.code.act(agent, state)
                except Exception:
                    expert_action = 'WAIT'
                if expert_action is None:
                    expert_action = 'WAIT'

                for a in ACTIONS:
                    feat = extract_action_features(state, a)
                    X.append(feat)
                    if a == expert_action:
                        y.append(15.0 if a == 'BOMB' else 8.0)
                    else:
                        if feat[0] < 0 or feat[1] > 0:
                            y.append(-10.0)
                        elif a == 'BOMB' and feat[9] < 0:
                            y.append(-15.0)
                        else:
                            y.append(0.0)

            world.do_step()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[+] Collected {len(X)} state-action transition pairs across {n_rounds} rounds.")
    return X, y


def pretrain_model():
    X, y = collect_expert_data(n_rounds=30)
    print("[*] Fitting Linear Q-Weights via Ridge Regression...")
    reg = Ridge(alpha=1.0, fit_intercept=False)
    reg.fit(X, y)
    weights = reg.coef_.astype(np.float32)

    print("\n[+] Learned Feature Weights:")
    feature_names = [
        "is_valid", "immediate_danger", "future_danger", "escapes_danger",
        "delta_coin", "delta_crate", "delta_enemy",
        "bomb_crates", "bomb_enemies", "bomb_escape_safe",
        "dead_end_hazard", "loop_penalty", "tactical_bomb", "bias",
        "enemy_trapped", "enemy_close", "chase_nearby_enemy", "bomb_near_enemy",
        "inv_coin_dist", "inv_crate_dist", "inv_enemy_dist",
        "adjacent_enemy", "enemy_has_bomb", "unique_crates_hit"
    ]
    for name, w in zip(feature_names, weights):
        print(f"  - {name:20s}: {w:+.4f}")

    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    with open(model_path, "wb") as f:
        pickle.dump({"weights": weights, "epsilon": 0.4, "round": 0}, f)
    print(f"\n[+] Successfully saved pre-trained model weights to: {model_path}\n")


if __name__ == "__main__":
    pretrain_model()
