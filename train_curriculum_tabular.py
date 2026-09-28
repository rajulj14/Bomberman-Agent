import os
import sys
import subprocess
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON_BIN = sys.executable


def run_stage(title, cmd):
    print(f"\n{'='*70}")
    print(f"[*] Starting Curriculum Stage: {title}")
    print(f"[*] Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    start_t = time.time()
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"[!] Error in stage {title}: return code {res.returncode}")
    else:
        print(f"\n[+] Completed {title} in {elapsed:.2f}s ({elapsed/60:.1f} min)!\n")


def main():
    print("[*] Starting Tabular Q-Agent Curriculum Training...")
    print(f"[*] Working Directory: {ROOT_DIR}\n")

    # Stage 1: Coin Heaven — Pure Navigation (100 rounds)
    run_stage("Stage 1: Coin Heaven — Pure Navigation (100 rounds)", [
        PYTHON_BIN, "main.py", "play", "--no-gui",
        "--agents", "tabular_q_agent",
        "--train", "1",
        "--scenario", "coin-heaven",
        "--n-rounds", "100"
    ])

    # Stage 2: Loot Crate — Crate Blasting & Coins (100 rounds)
    run_stage("Stage 2: Loot Crate — Crate Blasting & Coins (100 rounds)", [
        PYTHON_BIN, "main.py", "play", "--no-gui",
        "--agents", "tabular_q_agent",
        "--train", "1",
        "--scenario", "loot-crate",
        "--n-rounds", "100"
    ])

    # Stage 3: Classic Solo — Full Arena (100 rounds)
    run_stage("Stage 3: Classic Solo — Full Arena (100 rounds)", [
        PYTHON_BIN, "main.py", "play", "--no-gui",
        "--agents", "tabular_q_agent",
        "--train", "1",
        "--scenario", "classic",
        "--n-rounds", "100"
    ])

    # Stage 4: Hunting — vs Peaceful & Coin Collector (100 rounds)
    run_stage("Stage 4: Hunting — vs Peaceful & Coin Collector (100 rounds)", [
        PYTHON_BIN, "main.py", "play", "--no-gui",
        "--agents", "tabular_q_agent", "peaceful_agent", "coin_collector_agent",
        "--train", "1",
        "--scenario", "classic",
        "--n-rounds", "100"
    ])

    # Stage 5: Full Arena — vs Rule-Based Opponents (100 rounds)
    run_stage("Stage 5: Full Arena — vs Rule-Based Opponents (100 rounds)", [
        PYTHON_BIN, "main.py", "play", "--no-gui",
        "--agents", "tabular_q_agent", "rule_based_agent", "coin_collector_agent", "peaceful_agent",
        "--train", "1",
        "--scenario", "classic",
        "--n-rounds", "100"
    ])

    print("\n[+] TABULAR CURRICULUM TRAINING COMPLETE!")
    print("[+] Model saved to agent_code/tabular_q_agent/model.pt\n")


if __name__ == "__main__":
    main()
