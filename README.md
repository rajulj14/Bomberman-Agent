# Reinforcement Learning for Bomberman

Final project for Machine Learning Essentials, Summer Semester 2026, Heidelberg University.
Team Khaleesi.

The code builds on the course framework [ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl).
The framework files are unchanged. This repository adds our agents, their trained models and the scripts
we used for training and evaluation.

## Agents

All agents except the tabular one score each of the six actions with a function of features that describe
what happens if this action is taken, and play the action with the highest value.

| Agent | Method | Features | Role |
|---|---|---|---|
| `khaleesi` | gradient-boosted regression trees, fitted Q-iteration with 6-step returns | 47 | tournament submission |
| `hybrid_q_agent` | linear Q-learning with 4-step returns, weights initialised by ridge regression on `rule_based_agent` games | 24 | second agent |
| `reactive` | linear Q-learning with 4-step returns | 40 | first agent, basis of the `khaleesi` features |
| `tabular_q_agent` | tabular Q-learning over discretised states | | first test, dropped early |

`khaleesi` also uses a small classifier that predicts the next action of an opponent
(`agent_code/khaleesi/opponent_model.py`). It was trained on games of `rule_based_agent`.

## Results

Standard field: `rule_based_agent`, `coin_collector_agent` and `peaceful_agent`. Scores are points per
round with 95 % bootstrap confidence intervals.

`khaleesi` on five seeds, 500 rounds each. Seed 42 was used during development, the other four were not.

| Seed | `khaleesi` | 95 % CI | `rule_based_agent` |
|---|---|---|---|
| 42 | 8.39 | [7.97, 8.80] | 5.10 |
| 7 | 7.94 | [7.54, 8.35] | 5.31 |
| 1 | 8.19 | [7.78, 8.60] | 5.17 |
| 2 | 8.29 | [7.89, 8.71] | 5.14 |
| 3 | 8.17 | [7.75, 8.58] | 5.19 |
| Mean | 8.19 | [7.99, 8.40] | 5.18 |

`hybrid_q_agent` in the standard field, 500 rounds, seed 42: 6.80 [6.38, 7.24], against 5.43 for
`rule_based_agent`.

Both agents in the same games, with `rule_based_agent` and `coin_collector_agent`, 500 rounds, seed 42:
`khaleesi` 6.61 [6.23, 6.99], `hybrid_q_agent` 3.31 [3.04, 3.58].

Decision time on one thread (Apple M4 Max): `khaleesi` 6.0 ms in the median and at most 18.1 ms,
`hybrid_q_agent` 0.78 ms on average. The tournament limit is 0.5 s.

## Repository structure

```
agent_code/
  khaleesi/            submitted agent
    callbacks.py       builds the feature matrix and plays the best action
    features.py        the 47 action-conditional features
    tactics.py         danger map, escape search, threat map, kill test
    opponent_model.py  classifier for opponent actions
    train.py           fitted Q-iteration with a replay buffer
    model.pkl, opponent_model.pkl
  hybrid_q_agent/      linear agent
    callbacks.py, features.py, train.py
    pretrain.py        ridge regression on rule_based_agent games
    model.pt
  reactive/            first linear agent, model.npz
  tabular_q_agent/     tabular Q-learning test
evaluate.py            one agent against three opponents, fixed seed, paired bootstrap comparison
tournament.py          four agents in the same games, ranked with confidence intervals
sweep.py               trains one copy of an agent per hyperparameter value in parallel
timing.py              decision time per step
time_agents.py         decision time of one agent over a few rounds
autopsy.py             sorts every death of an agent into unseen, avoidable or doomed
scoreboard.py          100-round benchmark used during the development of hybrid_q_agent
train_opponent_model.py  collects games of rule_based_agent and trains the opponent model
train_curriculum_tabular.py  runs the five training stages of tabular_q_agent
```

## Setup

Python 3.10 or newer.

```bash
pip install numpy pygame tqdm scikit-learn==1.9.0
```

The trained models are pickled with scikit-learn 1.9.0 and need this version to load.

## Usage

Watch a game:

```bash
python main.py play --agents khaleesi hybrid_q_agent rule_based_agent peaceful_agent
```

Evaluate in the standard field. We set `OMP_NUM_THREADS=1` so that scikit-learn uses one thread, as in the tournament.

```bash
OMP_NUM_THREADS=1 python tournament.py --agents khaleesi rule_based_agent coin_collector_agent peaceful_agent --rounds 500 --seed 42
OMP_NUM_THREADS=1 python evaluate.py --agent khaleesi --opponents rule_based_agent coin_collector_agent peaceful_agent --n-rounds 200 --seed 42 --label test
```

Training writes into the agent folder and overwrites the trained models (`model.pkl`, `opponent_model.pkl`, `model.pt`). Back them up first.

Train `khaleesi` from scratch with the curriculum of the final version. Move `model.pkl` out of
`agent_code/khaleesi/` first, otherwise training continues from the submitted model.

```bash
OMP_NUM_THREADS=1 python main.py play --no-gui --agents khaleesi --train 1 --scenario coin-heaven --n-rounds 2000
OMP_NUM_THREADS=1 python main.py play --no-gui --agents khaleesi --train 1 --scenario loot-crate --n-rounds 3000
OMP_NUM_THREADS=1 python main.py play --no-gui --agents khaleesi --train 1 --scenario classic --n-rounds 4000
OMP_NUM_THREADS=1 python main.py play --no-gui --agents khaleesi peaceful_agent coin_collector_agent rule_based_agent --train 1 --scenario classic --n-rounds 6000
```

Train the opponent model:

```bash
python train_opponent_model.py --rounds 200
```

Train `hybrid_q_agent`: pretraining, then five stages.

```bash
python agent_code/hybrid_q_agent/pretrain.py
python main.py play --no-gui --agents hybrid_q_agent --train 1 --scenario coin-heaven --n-rounds 200
python main.py play --no-gui --agents hybrid_q_agent --train 1 --scenario loot-crate --n-rounds 300
python main.py play --no-gui --agents hybrid_q_agent --train 1 --scenario classic --n-rounds 200
python main.py play --no-gui --agents hybrid_q_agent peaceful_agent coin_collector_agent --train 1 --scenario classic --n-rounds 200
python main.py play --no-gui --agents hybrid_q_agent rule_based_agent coin_collector_agent peaceful_agent --train 1 --scenario classic --n-rounds 300
```

Hyperparameter sweep, for example over the return horizon:

```bash
python sweep.py --base khaleesi --param N_STEP --values 3 4 6 8 --rounds 2500
python sweep.py --base khaleesi --collect --param N_STEP --values 3 4 6 8
```

Analysis:

```bash
OMP_NUM_THREADS=1 python timing.py --agent khaleesi --rounds 30
OMP_NUM_THREADS=1 python time_agents.py khaleesi 5
OMP_NUM_THREADS=1 python autopsy.py --agent khaleesi --rounds 200
python scoreboard.py 100
```
