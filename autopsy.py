#!/usr/bin/env python3
"""
autopsy.py -- why does the agent die?

Score tells you an agent lost a round; it does not tell you whether the round
was lost at the moment of death or ten steps earlier.  This replays games and,
at the instant of each death, reconstructs what the agent's own world model
said was available.  Deaths sort into three categories that call for entirely
different fixes:

  DOOMED      no survivable action existed at the fatal step.  The mistake was
              upstream -- the agent walked into a position from which nothing
              saved it.  Fix: positioning features, threat map, dead-end
              avoidance.

  AVOIDABLE   a survivable action existed and the agent chose otherwise.  The
              features saw the danger; the learned weights mispriced it.
              Fix: training, reward balance.

  UNSEEN      the danger map did not mark the fatal tile as lethal at all.
              This is a bug in tactics.py, not a policy failure, and it is the
              most valuable of the three to find.

    python autopsy.py --agent khaleesi --rounds 200

Also reports how many steps of warning preceded each death, and runs the same
analysis on a reference agent so the failure profile can be compared against
one that is known to be decent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import WorldArgs  # noqa: E402


def analyse(agent_name, opponents, rounds, seed, scenario, use_tactics=True):
    from environment import BombeRLeWorld

    T = None
    if use_tactics:
        try:
            mod = __import__(f"agent_code.{agent_name}.tactics",
                             fromlist=["tactics"])
            T = mod
        except Exception as exc:
            print(f"  (no tactics module for {agent_name}: {exc})")

    logging.disable(logging.INFO)
    agents = [(agent_name, False)] + [(o, False) for o in opponents]
    world = BombeRLeWorld(WorldArgs(scenario, seed, ROOT / "logs"), agents)

    verdicts = Counter()
    warnings = []
    death_steps = []
    escape_counts = []
    context = Counter()

    try:
        subject = world.agents[0]
        for _ in range(rounds):
            world.new_round()
            subject = [a for a in world.agents if a.code_name == agent_name][0]
            warn = 0            # consecutive steps spent standing in danger
            last = None

            while world.running:
                if not subject.dead and T is not None:
                    field = np.array(world.arena)
                    bombs = [((b.x, b.y), b.timer) for b in world.bombs]
                    expl = np.zeros(field.shape, dtype=int)
                    for ex in world.explosions:
                        if ex.is_dangerous():
                            for (bx, by) in ex.blast_coords:
                                expl[bx, by] = max(expl[bx, by], ex.timer - 1)
                    others = [(a.x, a.y) for a in world.active_agents
                              if a is not subject]

                    lethal = T.danger_map(field, bombs, expl)
                    free = T.walkable_map(field, bombs=bombs, blocked=others)
                    esc = T.survivable((subject.x, subject.y), lethal, free,
                                       field.shape)
                    clean = T._clean_from(lethal)
                    h = field.shape[1]
                    in_danger = clean[subject.x * h + subject.y] > 0

                    last = {
                        "safe": esc.safe,
                        "n_safe": esc.n_safe_tiles,
                        "in_danger": in_danger,
                        "n_bombs": len(bombs),
                        "n_others": len(others),
                        "step": world.step,
                    }
                    warn = warn + 1 if in_danger else 0
                    last["warning"] = warn

                was_alive = not subject.dead
                world.do_step()

                if was_alive and subject.dead:
                    death_steps.append(world.step)
                    if last is None:
                        verdicts["NO DATA"] += 1
                    elif not last["in_danger"]:
                        # the tile the agent stood on was never marked lethal
                        verdicts["UNSEEN"] += 1
                    elif last["safe"]:
                        verdicts["AVOIDABLE"] += 1
                        escape_counts.append(last["n_safe"])
                    else:
                        verdicts["DOOMED"] += 1
                    if last is not None:
                        warnings.append(last["warning"])
                        context[f"{last['n_others']} opponents alive"] += 1
                    break
    finally:
        try:
            world.end()
        except Exception:
            pass
        logging.disable(logging.NOTSET)

    return verdicts, warnings, death_steps, escape_counts, context


def report(name, verdicts, warnings, death_steps, escape_counts, context,
           rounds):
    total = sum(verdicts.values())
    print(f"\n{name}: {total} deaths in {rounds} rounds "
          f"({100 * total / rounds:.0f}% of rounds)")
    print("-" * 60)
    if total == 0:
        print("  no deaths recorded")
        return
    for k in ("DOOMED", "AVOIDABLE", "UNSEEN", "NO DATA"):
        n = verdicts.get(k, 0)
        if n:
            print(f"  {k:<12}{n:>5}  ({100 * n / total:4.1f}%)")
    if warnings:
        w = np.array(warnings)
        print(f"\n  steps in danger before dying: "
              f"mean {w.mean():.1f}, median {np.median(w):.0f}, "
              f"zero-warning {100 * (w == 0).mean():.0f}%")
    if death_steps:
        print(f"  death step: mean {np.mean(death_steps):.0f}, "
              f"median {np.median(death_steps):.0f}")
    if escape_counts:
        print(f"  when avoidable, safe tiles available: "
              f"mean {np.mean(escape_counts):.1f}")
    if context:
        print("  " + ", ".join(f"{k}: {v}" for k, v in context.most_common(4)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--compare", default=None,
                   help="second agent to profile for comparison")
    p.add_argument("--opponents", nargs="*",
                   default=["rule_based_agent", "coin_collector_agent",
                            "peaceful_agent"])
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", default="classic")
    args = p.parse_args()

    print(f"autopsy: {args.agent} over {args.rounds} rounds "
          f"vs {', '.join(args.opponents)}")
    res = analyse(args.agent, args.opponents, args.rounds, args.seed,
                  args.scenario)
    report(args.agent, *res, args.rounds)

    if args.compare:
        print(f"\nautopsy: {args.compare} (reference)")
        res2 = analyse(args.compare, args.opponents, args.rounds, args.seed,
                       args.scenario, use_tactics=False)
        report(args.compare, *res2, args.rounds)

    print("\nDOOMED    -> positioning; the round was lost before the fatal step")
    print("AVOIDABLE -> the policy mispriced a danger its features could see")
    print("UNSEEN    -> the danger map missed it; that is a bug worth finding")


if __name__ == "__main__":
    main()
