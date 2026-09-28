#!/usr/bin/env python3
"""
tournament.py -- rank agents against each other under tournament conditions.

Plays four-agent games, the way the real tournament does, and ranks by mean
score per round with bootstrap confidence intervals.

    python tournament.py --agents rule_based_agent coin_collector_agent \\
        peaceful_agent random_agent --rounds 100

With more than four agents it runs a round-robin over every four-agent subset,
so each agent meets every combination of opponents an equal number of times.
That matters: an agent's score depends heavily on who else is on the board, and
measuring each one against a different opponent mix would rank the schedule
rather than the agents.

All subsets share the same seed sequence (common random numbers), so every
agent sees the same crate layouts.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import bootstrap_ci, run_matches  # noqa: E402


def play_group(group, rounds, seed, scenario):
    """One four-agent game repeated `rounds` times; returns rows for all seats."""
    rows = run_matches(group[0], list(group[1:]), rounds, seed, scenario)
    # run_matches only flags the first agent as subject; re-attach the real
    # agent identity by seat order so every seat's results are usable
    per_round = defaultdict(list)
    for r in rows:
        per_round[r["round"]].append(r)
    out = []
    for rnd, entries in per_round.items():
        for seat, entry in enumerate(entries):
            if seat < len(group):
                out.append({**entry, "code": group[seat]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--rounds", type=int, default=100,
                   help="rounds per four-agent group")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", default="classic")
    args = p.parse_args()

    agents = args.agents
    if len(agents) < 2:
        print("need at least two agents")
        return

    if len(agents) <= 4:
        groups = [tuple(agents)]
    else:
        groups = list(itertools.combinations(agents, 4))

    print(f"{len(agents)} agents, {len(groups)} group(s), "
          f"{args.rounds} rounds each, scenario={args.scenario}\n")

    stats = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)
    games = defaultdict(int)

    for gi, group in enumerate(groups, 1):
        print(f"[{gi}/{len(groups)}] {' vs '.join(group)}")
        rows = play_group(group, args.rounds, args.seed, args.scenario)

        by_round = defaultdict(list)
        for r in rows:
            by_round[r["round"]].append(r)

        for _, entries in by_round.items():
            best = max(int(e["score"]) for e in entries)
            for e in entries:
                code = e["code"]
                games[code] += 1
                for m in ["score", "coins", "kills", "suicides", "crates",
                          "killed_by_opponent", "survived", "steps"]:
                    stats[code][m].append(float(e[m]))
                if int(e["score"]) == best:
                    wins[code] += 1

    # ---- ranking ----
    table = []
    for code in agents:
        if not stats[code]["score"]:
            continue
        mean, lo, hi = bootstrap_ci(stats[code]["score"])
        table.append({
            "agent": code, "score": mean, "lo": lo, "hi": hi,
            "coins": np.mean(stats[code]["coins"]),
            "kills": np.mean(stats[code]["kills"]),
            "suicides": np.mean(stats[code]["suicides"]),
            "killed": np.mean(stats[code]["killed_by_opponent"]),
            "survived": np.mean(stats[code]["survived"]),
            "winrate": 100.0 * wins[code] / max(games[code], 1),
            "n": games[code],
        })
    table.sort(key=lambda r: -r["score"])

    print("\n" + "=" * 92)
    print(f"{'#':<3}{'agent':<24}{'score':>8}{'95% CI':>18}"
          f"{'coins':>8}{'kills':>8}{'suic':>7}{'killed':>8}{'win%':>7}")
    print("=" * 92)
    for i, r in enumerate(table, 1):
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        print(f"{i:<3}{r['agent']:<24}{r['score']:>8.3f}{ci:>18}"
              f"{r['coins']:>8.2f}{r['kills']:>8.3f}{r['suicides']:>7.2f}"
              f"{r['killed']:>8.2f}{r['winrate']:>7.1f}")
    print("=" * 92)
    print(f"{table[0]['n']} rounds per agent. "
          "Overlapping intervals mean the ranking is not significant.")


if __name__ == "__main__":
    main()
