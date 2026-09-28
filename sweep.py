#!/usr/bin/env python3
"""
sweep.py -- run several hyperparameter variants in parallel, then rank them.

The agent is single-threaded, so on a multi-core machine the sensible way to
explore hyperparameters is not to make one run faster but to run several at
once.  This script clones the agent directory once per variant, patches the
chosen constants in each copy, and prints the commands to run them
concurrently.

    python sweep.py --param N_STEP --values 3 4 6 8 --base booster
    python sweep.py --param max_depth --values 4 6 8 --base booster --rounds 2500

Each variant starts from the base agent's current model.pkl, so the sweep
measures "what does changing this parameter do from here", not "what does
training from scratch with this parameter give".  That is the question worth
asking when you already have a converged agent.

Afterwards:

    python sweep.py --collect --param N_STEP --values 3 4 6 8

runs the tournament for every variant against a fixed field and prints a
ranking, so the comparison uses the same opponents and the same seeds.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "agent_code"

# which file each tunable lives in
LOCATION = {
    "N_STEP": "train.py",
    "GAMMA": "train.py",
    "FQI_ITERATIONS": "train.py",
    "REFIT_EVERY": "train.py",
    "BUFFER_SIZE": "train.py",
    "max_iter": "train.py",
    "max_depth": "train.py",
    "learning_rate": "train.py",
    "min_samples_leaf": "train.py",
    "l2_regularization": "train.py",
    "EPS_START": "callbacks.py",
    "EPS_DECAY_ROUNDS": "callbacks.py",
}


def variant_name(base, param, value):
    return f"{base}_{param.lower()}{str(value).replace('.', '')}"


def patch(path: Path, param: str, value):
    """Replace one constant, line by line.

    Deliberately not a clever regex: an earlier version silently failed to
    match module-level constants and stripped the trailing comma from dict
    entries, which would have produced four identical agents and a syntax
    error respectively.  Both failures are invisible until you read the
    results and believe them.
    """
    lines = path.read_text().splitlines(keepends=True)
    hit = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # dict entry:  "    max_depth=6,"
        if stripped.startswith(f"{param}="):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{param}={value},\n"
            hit = True
            break
        # module constant:  "N_STEP = 4   # comment"
        if stripped.startswith(f"{param} ="):
            comment = ""
            if "#" in line:
                comment = "  " + line[line.index("#"):].rstrip()
            lines[i] = f"{param} = {value}{comment}\n"
            hit = True
            break
    if not hit:
        raise SystemExit(f"could not find {param} in {path}")
    path.write_text("".join(lines))


def setup(base, param, values, rounds, opponents):
    if param not in LOCATION:
        raise SystemExit(f"unknown parameter {param}; known: "
                         f"{', '.join(sorted(LOCATION))}")
    base_dir = AGENT_DIR / base
    if not base_dir.is_dir():
        raise SystemExit(f"no such agent: {base_dir}")

    cmds = []
    for v in values:
        name = variant_name(base, param, v)
        dst = AGENT_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(base_dir, dst,
                        ignore=shutil.ignore_patterns("logs", "__pycache__",
                                                      "training_stats.csv"))
        patch(dst / LOCATION[param], param, v)
        print(f"  {name}: {param} = {v}")
        cmds.append(
            f"OMP_NUM_THREADS=1 python main.py play --no-gui --agents {name} "
            f"{' '.join(opponents)} --train 1 --scenario classic "
            f"--n-rounds {rounds} > logs/sweep_{name}.out 2>&1 &"
        )

    print("\nRun all variants at once (one core each):\n")
    print("mkdir -p logs")
    for c in cmds:
        print(c)
    print("wait")
    print(f"\nThen:  python sweep.py --collect --param {param} "
          f"--values {' '.join(map(str, values))} --base {base}")


def collect(base, param, values, field, rounds, seed):
    import csv
    from collections import defaultdict

    sys.path.insert(0, str(ROOT))
    from evaluate import bootstrap_ci, run_matches

    rows = []
    for v in values:
        name = variant_name(base, param, v)
        if not (AGENT_DIR / name / "model.pkl").is_file():
            print(f"  {name}: no model, skipped")
            continue
        print(f"  evaluating {name} ...")
        data = run_matches(name, list(field), rounds, seed, "classic")
        subj = [r for r in data if int(r["is_subject"]) == 1]
        mean, lo, hi = bootstrap_ci([float(r["score"]) for r in subj])
        rows.append({
            "value": v, "score": mean, "lo": lo, "hi": hi,
            "kills": sum(float(r["kills"]) for r in subj) / len(subj),
            "coins": sum(float(r["coins"]) for r in subj) / len(subj),
            "suicides": sum(float(r["suicides"]) for r in subj) / len(subj),
        })

    rows.sort(key=lambda r: -r["score"])
    print(f"\n{param} sweep, {rounds} rounds vs {', '.join(field)}")
    print("=" * 70)
    print(f"{param:>16}{'score':>9}{'95% CI':>18}{'coins':>8}{'kills':>8}{'suic':>7}")
    print("=" * 70)
    for r in rows:
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        print(f"{str(r['value']):>16}{r['score']:>9.3f}{ci:>18}"
              f"{r['coins']:>8.2f}{r['kills']:>8.3f}{r['suicides']:>7.2f}")
    print("=" * 70)
    print("Overlapping intervals mean the difference is not significant.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="booster")
    p.add_argument("--param", required=True)
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--rounds", type=int, default=2500,
                   help="training rounds per variant")
    p.add_argument("--opponents", nargs="*",
                   default=["peaceful_agent", "coin_collector_agent",
                            "rule_based_agent"])
    p.add_argument("--collect", action="store_true")
    p.add_argument("--eval-rounds", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.collect:
        collect(args.base, args.param, args.values,
                args.opponents, args.eval_rounds, args.seed)
    else:
        setup(args.base, args.param, args.values, args.rounds, args.opponents)


if __name__ == "__main__":
    main()
