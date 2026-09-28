#!/usr/bin/env python3

import sys
import os
import json
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BIN = os.path.join(ROOT_DIR, ".venv/bin/python3") if os.path.isdir(os.path.join(ROOT_DIR, ".venv")) else sys.executable

def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    match_name = f"terminal_score_{n_rounds}rnd"
    stats_file = os.path.join(ROOT_DIR, f"results/{match_name}.json")

    if os.path.isfile(stats_file):
        try: os.remove(stats_file)
        except Exception: pass

    include_khaleesi = any("khaleesi" in a.lower() for a in sys.argv[1:])
    if include_khaleesi:
        agents = ["hybrid_q_agent", "khaleesi", "rule_based_agent", "peaceful_agent"]
    else:
        agents = ["hybrid_q_agent", "rule_based_agent", "coin_collector_agent", "peaceful_agent"]

    cmd = [
        PYTHON_BIN, "main.py", "play", "--no-gui",
        "--agents", *agents,
        "--scenario", "classic",
        "--n-rounds", str(n_rounds),
        "--match-name", match_name,
        "--save-stats"
    ]

    print(f"\n[*] Running {n_rounds} rounds of 4-Player Arena on Classic...")
    print(f"[*] Agents: {' vs '.join(agents)}\n")
    
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    with open(stats_file, "r") as f:
        data = json.load(f)

    by_agent = data.get("by_agent", {})
    rows = []
    for name, stats in by_agent.items():
        score = stats.get("score", 0)
        coins = stats.get("coins", 0)
        kills = stats.get("kills", 0)
        suicides = stats.get("suicides", 0)
        crates = stats.get("crates", 0)
        rows.append((name, score, score / n_rounds, coins, kills, suicides, crates))

    
    rows.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 90)
    print(f"  🏆 FINAL TOURNAMENT SCOREBOARD ({n_rounds} Rounds Classic)")
    print("=" * 90)
    header = f"{'Rank':<5} | {'Agent':<22} | {'Score':>7} | {'Avg/Rnd':>7} | {'Coins':>6} | {'Kills':>6} | {'Suicides':>9} | {'Crates':>7}"
    print(header)
    print("-" * len(header))

    medals = ["🥇 1", "🥈 2", "🥉 3", "   4"]
    for i, (name, sc, avg, co, ki, su, cr) in enumerate(rows):
        rank_label = medals[i] if i < len(medals) else f"   {i+1}"
        agent_label = f"**{name}**" if name == "hybrid_q_agent" else name
        print(f"{rank_label:<5} | {name:<22} | {sc:>7d} | {avg:>7.2f} | {co:>6d} | {ki:>6d} | {su:>9d} | {cr:>7d}")

    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
