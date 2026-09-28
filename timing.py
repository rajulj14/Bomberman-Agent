#!/usr/bin/env python3
"""
timing.py -- measure per-decision cost across real games.

A single hand-built worst-case board tells you almost nothing: it might be
rarer than one step in a thousand, or it might be typical. What matters is
the distribution over states the agent actually encounters, because the
engine's rule is per step and the overrun is deducted from the next step's
budget -- so a fat tail is worse than a high mean.

    python timing.py --agent khaleesi --rounds 30

Reports mean/p50/p95/p99/max, and scales them by a factor for the tournament
hardware. The tournament runs on an AMD Ryzen 5 2600; a modern laptop core is
roughly 3-4x faster, so --scale 3.5 gives a rough projection. That factor is
a guess, not a measurement, and the report should say so.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import WorldArgs  # noqa: E402

TIMES: list[float] = []


def instrument(agent_name):
    """Wrap the agent's act() so every decision is timed."""
    import importlib
    mod = importlib.import_module(f"agent_code.{agent_name}.callbacks")
    original = mod.act

    def timed(self, game_state):
        t0 = time.perf_counter()
        try:
            return original(self, game_state)
        finally:
            TIMES.append(time.perf_counter() - t0)

    mod.act = timed
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="khaleesi")
    p.add_argument("--opponents", nargs="*",
                   default=["rule_based_agent", "coin_collector_agent",
                            "peaceful_agent"])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", default="classic")
    p.add_argument("--scale", type=float, default=3.5,
                   help="assumed slowdown on tournament hardware")
    p.add_argument("--limit", type=float, default=500.0,
                   help="per-step budget in ms (settings.TIMEOUT)")
    args = p.parse_args()

    instrument(args.agent)

    from environment import BombeRLeWorld
    logging.disable(logging.INFO)
    agents = [(args.agent, False)] + [(o, False) for o in args.opponents]
    world = BombeRLeWorld(WorldArgs(args.scenario, args.seed, ROOT / "logs"),
                          agents)

    print(f"timing {args.agent} over {args.rounds} rounds vs "
          f"{', '.join(args.opponents)} ...")
    try:
        for r in range(args.rounds):
            world.new_round()
            while world.running:
                world.do_step()
            if (r + 1) % 10 == 0:
                print(f"  {r + 1}/{args.rounds} rounds, "
                      f"{len(TIMES)} decisions")
    finally:
        try:
            world.end()
        except Exception:
            pass
        logging.disable(logging.NOTSET)

    if not TIMES:
        print("no decisions recorded -- did the agent name match?")
        return

    t = np.array(TIMES) * 1000.0
    print(f"\n{len(t)} decisions on this machine")
    print("-" * 58)
    rows = [("mean", t.mean()), ("median", np.percentile(t, 50)),
            ("p95", np.percentile(t, 95)), ("p99", np.percentile(t, 99)),
            ("max", t.max())]
    print(f"{'':>10}{'here':>12}{f'x{args.scale} (est.)':>16}")
    for name, v in rows:
        proj = v * args.scale
        flag = "  OVER" if proj > args.limit else ""
        print(f"{name:>10}{v:>10.1f} ms{proj:>13.1f} ms{flag}")
    print("-" * 58)

    over = (t * args.scale > args.limit).mean() * 100
    print(f"projected steps over the {args.limit:.0f} ms budget: {over:.2f}%")
    if over == 0:
        print("comfortable margin.")
    elif over < 0.5:
        print("rare overruns; each forces WAIT and eats into the next step.")
    else:
        print("frequent overruns -- this needs fixing before submission.")


if __name__ == "__main__":
    main()
