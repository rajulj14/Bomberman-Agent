#!/usr/bin/env python3
"""
evaluate.py -- measure agent strength properly.

Training statistics are not an evaluation.  They are polluted by exploration
noise, they change while you collect them, and they tell you nothing about how
the submitted (greedy) policy actually plays.  This script runs the agent the
way the tournament will: `train = 0`, pure argmax, no exploration.

Two design points that matter for the report:

PAIRED SEEDS (common random numbers)
    Every variant you compare is played on the *same sequence of boards*.
    Crate layouts and coin positions are the dominant source of variance in
    this game -- far larger than the difference between two decent agents.
    Comparing agent A on seed 1 against agent B on seed 2 mostly measures
    which of them got the easier maps.  Fixing the seed removes that entirely,
    so a much smaller sample detects a real difference.

BOOTSTRAP CONFIDENCE INTERVALS
    "Agent A scored 3.4, agent B scored 3.7" is not a result.  Episode scores
    here are heavily skewed (one lucky kill is worth five coins), so the mean
    alone is misleading.  Resampling gives an honest interval.

Usage
-----
    python evaluate.py --agent reactive --n-rounds 200 --label baseline

    python evaluate.py --agent reactive \\
        --opponents peaceful_agent coin_collector_agent \\
        --n-rounds 200 --seed 42 --label vs_weak

    # compare two runs already in the CSV
    python evaluate.py --compare baseline improved

Results append to results/evaluation.csv, one row per agent per round, tagged
with the git commit so a number in the report can always be traced back to the
code that produced it.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "results" / "evaluation.csv"

FIELDS = [
    "label", "commit", "timestamp", "scenario", "seed", "opponents",
    "round", "agent", "is_subject",
    "score", "coins", "kills", "suicides", "crates", "bombs", "invalid",
    "steps", "died", "killed_by_opponent", "survived",
]


# --------------------------------------------------------------------------
def git_commit() -> str:
    """Tag every row with the code that produced it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) or "no-git"
    except Exception:
        return "no-git"


class WorldArgs:
    """Stands in for the argparse namespace BombeRLeWorld expects."""

    def __init__(self, scenario, seed, log_dir):
        self.no_gui = True
        self.scenario = scenario
        self.seed = seed
        self.log_dir = str(log_dir)
        self.match_name = None
        self.save_replay = False
        self.save_stats = False
        self.continue_without_training = True
        self.silence_errors = False
        self.turn_based = False
        self.update_interval = 0.1
        self.make_video = False
        self.skip_frames = True
        self.train = 0
        self.command_name = "play"


def run_matches(subject, opponents, n_rounds, seed, scenario, quiet=True):
    """Play n_rounds and return one dict of metrics per agent per round.

    The world is driven directly rather than through main.py so that we can
    read each agent's per-round statistics, which the JSON export aggregates
    away.
    """
    import logging

    from environment import BombeRLeWorld

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    if quiet:
        logging.disable(logging.INFO)

    agents = [(subject, False)] + [(o, False) for o in opponents]
    world = BombeRLeWorld(WorldArgs(scenario, seed, log_dir), agents)

    rows = []
    try:
        # The subject is the FIRST agent. Identifying it by code_name breaks
        # as soon as an opponent uses the same directory (e.g. self-play, or
        # rule_based_agent evaluated against itself).
        subject_agent = world.agents[0]
        for rnd in range(1, n_rounds + 1):
            world.new_round()
            while world.running:
                world.do_step()

            for agent in world.agents:
                st = agent.statistics
                died = bool(agent.dead)
                suicides = int(st.get("suicides", 0))
                rows.append({
                    "round": rnd,
                    "agent": agent.name,
                    "is_subject": int(agent is subject_agent),
                    "score": int(st.get("score", 0)),
                    "coins": int(st.get("coins", 0)),
                    "kills": int(st.get("kills", 0)),
                    "suicides": suicides,
                    "crates": int(st.get("crates", 0)),
                    "bombs": int(st.get("bombs", 0)),
                    "invalid": int(st.get("invalid", 0)),
                    "steps": int(st.get("steps", 0)),
                    "died": int(died),
                    "killed_by_opponent": int(died and suicides == 0),
                    "survived": int(not died),
                })
    finally:
        try:
            world.end()
        except Exception:
            pass
        if quiet:
            logging.disable(logging.NOTSET)

    return rows


# --------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for the mean.

    Episode scores are skewed and bounded below at zero, so a normal-theory
    interval understates the uncertainty.  Resampling makes no distributional
    assumption.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_difference(a_values, b_values, n_boot=10000, rng=None):
    """Bootstrap the mean paired difference a - b, round by round.

    This is the number that actually answers "is the change an improvement?",
    and because the two runs share seeds the pairing cancels map difficulty.
    """
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return (float("nan"),) * 4
    d = a[:n] - b[:n]
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # fraction of resamples on the wrong side of zero -- a rough two-sided p
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


# --------------------------------------------------------------------------
def append_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def summarise(rows, subject_only=True):
    sel = [r for r in rows if not subject_only or int(r["is_subject"]) == 1]
    if not sel:
        return {}
    out = {}
    for metric in ["score", "coins", "kills", "suicides", "crates",
                   "bombs", "invalid", "steps", "died",
                   "killed_by_opponent", "survived"]:
        vals = [float(r[metric]) for r in sel]
        out[metric] = bootstrap_ci(vals)
    out["_n"] = len(sel)
    return out


def print_summary(label, summary):
    n = summary.get("_n", 0)
    print(f"\n{label}   ({n} rounds)")
    print("-" * 62)
    for metric in ["score", "coins", "kills", "crates", "bombs",
                   "suicides", "killed_by_opponent", "survived", "steps"]:
        if metric not in summary:
            continue
        mean, lo, hi = summary[metric]
        print(f"  {metric:<20} {mean:7.3f}   [{lo:6.3f}, {hi:6.3f}]")


def print_win_rate(rows):
    """How often does the subject outscore every opponent in the same round?"""
    by_round = defaultdict(list)
    for r in rows:
        by_round[r["round"]].append(r)
    wins = ties = losses = 0
    for _, group in by_round.items():
        subj = [g for g in group if int(g["is_subject"]) == 1]
        opps = [g for g in group if int(g["is_subject"]) == 0]
        if not subj or not opps:
            continue
        s = int(subj[0]["score"])
        best = max(int(o["score"]) for o in opps)
        if s > best:
            wins += 1
        elif s == best:
            ties += 1
        else:
            losses += 1
    total = wins + ties + losses
    if total:
        print(f"\n  outright wins {wins}/{total} ({100 * wins / total:.1f}%), "
              f"ties {ties}, losses {losses}")


# --------------------------------------------------------------------------
def cmd_run(args):
    opponents = args.opponents
    if opponents is None:
        opponents = ["rule_based_agent"] * 3

    print(f"running {args.n_rounds} rounds: {args.agent} vs "
          f"{', '.join(opponents) or 'nobody'}")
    print(f"scenario={args.scenario} seed={args.seed} label={args.label}")

    rows = run_matches(args.agent, opponents, args.n_rounds,
                       args.seed, args.scenario)

    meta = dict(
        label=args.label,
        commit=git_commit(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        scenario=args.scenario,
        seed=args.seed,
        opponents="+".join(opponents) if opponents else "none",
    )
    full = [{**meta, **r} for r in rows]
    append_csv(Path(args.out), full)

    print_summary(args.label, summarise(full))
    print_win_rate(full)
    print(f"\nappended {len(full)} rows to {args.out}")


def cmd_compare(args):
    rows = load_csv(Path(args.out))
    if not rows:
        print(f"no data in {args.out}")
        return

    a_label, b_label = args.compare
    a = [r for r in rows if r["label"] == a_label and int(r["is_subject"])]
    b = [r for r in rows if r["label"] == b_label and int(r["is_subject"])]
    if not a or not b:
        have = sorted({r["label"] for r in rows})
        print(f"missing data. labels present: {', '.join(have)}")
        return

    print_summary(a_label, summarise(a))
    print_summary(b_label, summarise(b))

    seeds_a = {r["seed"] for r in a}
    seeds_b = {r["seed"] for r in b}
    paired = seeds_a == seeds_b
    print(f"\npaired seeds: {'yes' if paired else 'NO -- differences below are '
          'confounded by map difficulty'}")

    print(f"\npaired difference ({a_label} - {b_label})")
    print("-" * 62)
    for metric in ["score", "coins", "kills", "suicides",
                   "killed_by_opponent", "survived"]:
        av = [float(r[metric]) for r in sorted(a, key=lambda r: int(r["round"]))]
        bv = [float(r[metric]) for r in sorted(b, key=lambda r: int(r["round"]))]
        mean, lo, hi, p = paired_difference(av, bv)
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"  {metric:<20} {mean:+7.3f}   [{lo:+6.3f}, {hi:+6.3f}]  "
              f"p~{p:.3f}{flag}")
    print("\n  * interval excludes zero")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="reactive", help="agent directory to evaluate")
    p.add_argument("--opponents", nargs="*", default=None,
                   help="opponent agents (default: three rule_based_agent)")
    p.add_argument("--n-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42,
                   help="fixed world seed -- keep identical across variants")
    p.add_argument("--scenario", default="classic")
    p.add_argument("--label", default="run",
                   help="name for this configuration in the results CSV")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="compare two labels already in the CSV instead of running")
    args = p.parse_args()

    if args.compare:
        cmd_compare(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
