#!/usr/bin/env python3
"""
train_opponent_model.py -- fit P(action | opponent situation) from replays.

Runs games, watches the opponents, infers what they did from consecutive
observations, and fits a classifier.  The result is written next to the agent
as opponent_model.pkl, where features.py picks it up automatically.

    python train_opponent_model.py --rounds 300
    python train_opponent_model.py --rounds 300 --opponents coin_collector_agent

Held-out accuracy is reported against a majority-class baseline; if the model
is not clearly beating that baseline, it is not worth feeding into the agent
and the script says so rather than saving a useless file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent_code.khaleesi.opponent_model import (  # noqa: E402
    N_OPP_FEATURES, OPP_FEATURE_NAMES, OpponentModel, infer_action,
    opponent_features,
)
from evaluate import WorldArgs  # noqa: E402


def collect(opponents, rounds, seed, scenario, quiet=True):
    """Play games and return (features, action_labels) for every opponent step."""
    from environment import BombeRLeWorld

    if quiet:
        logging.disable(logging.INFO)

    agents = [(o, False) for o in opponents]
    world = BombeRLeWorld(WorldArgs(scenario, seed, ROOT / "logs"), agents)

    X, y = [], []
    try:
        for rnd in range(rounds):
            world.new_round()
            prev = None
            while world.running:
                field = np.array(world.arena)
                bombs = [((b.x, b.y), b.timer) for b in world.bombs]
                # the world keeps Explosion objects; rebuild the map the
                # same way get_state_for_agent does
                expl = np.zeros(field.shape, dtype=int)
                for ex in world.explosions:
                    if ex.is_dangerous():
                        for (bx, by) in ex.blast_coords:
                            expl[bx, by] = max(expl[bx, by], ex.timer - 1)
                coins = [(c.x, c.y) for c in world.coins if c.collectable]
                alive = {a.name: a for a in world.active_agents}
                positions = {n: (a.x, a.y) for n, a in alive.items()}

                if prev is not None:
                    for name, (feats, old_pos) in prev.items():
                        new_pos = positions.get(name)
                        act = infer_action(old_pos, new_pos,
                                           prev_bombs, bombs)
                        if act is not None:
                            X.append(feats)
                            y.append(act)

                # snapshot this step's situation for every living agent
                snapshot = {}
                for name, a in alive.items():
                    others = [(x, y) for n, (x, y) in positions.items()
                              if n != name]
                    if not others:
                        continue
                    feats = opponent_features(
                        field, bombs, expl, coins, (a.x, a.y),
                        a.bombs_left, others, others[0])
                    snapshot[name] = (feats, (a.x, a.y))
                prev, prev_bombs = snapshot, bombs

                world.do_step()

            if (rnd + 1) % 25 == 0:
                print(f"  round {rnd + 1}/{rounds}, {len(X)} samples")
    finally:
        try:
            world.end()
        except Exception:
            pass
        if quiet:
            logging.disable(logging.NOTSET)

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opponents", nargs="*",
                   default=["rule_based_agent"] * 4)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario", default="classic")
    args = p.parse_args()

    print(f"collecting from {args.rounds} rounds of "
          f"{', '.join(args.opponents)} ...")
    X, y = collect(args.opponents, args.rounds, args.seed, args.scenario)
    print(f"\n{len(X)} samples, {N_OPP_FEATURES} features")
    counts = Counter(y)
    for a, c in counts.most_common():
        print(f"  {a:6s} {c:7d}  ({100 * c / len(y):5.1f}%)")

    if len(X) < 500:
        print("\ntoo few samples; increase --rounds")
        return

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=0, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=50, learning_rate=0.15,
                                         max_depth=6, early_stopping=False)
    clf.fit(Xtr, ytr)

    acc = clf.score(Xte, yte)
    baseline = counts.most_common(1)[0][1] / len(y)
    print(f"\nheld-out accuracy {acc:.3f}   "
          f"majority-class baseline {baseline:.3f}")

    if acc <= baseline + 0.05:
        print("model barely beats the baseline -- not saving.")
        print("the opponent may be too stochastic, or the features too thin.")
        return

    OpponentModel(clf, sorted(set(y))).save()
    print(f"saved to agent_code/khaleesi/opponent_model.pkl")

    # which situations drive the prediction -- useful for the report
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(clf, Xte[:3000], yte[:3000],
                                   n_repeats=3, random_state=0)
        order = np.argsort(r.importances_mean)[::-1][:8]
        print("\ntop features:")
        for i in order:
            print(f"  {OPP_FEATURE_NAMES[i]:24s} {r.importances_mean[i]:+.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
