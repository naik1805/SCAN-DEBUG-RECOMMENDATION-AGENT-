"""Shared DQN training routines and auto-train helpers."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from src.data.paths import COMPILED_DATASET_PATH, DATA_DIR, MODEL_WEIGHTS_PATH
from src.env.debug_env import ScanDebugEnv, REV_ACTION_MAP
from src.models.agent import DQNAgent

from src.config import get_settings

DEFAULT_AUTO_TRAIN_EPISODES = get_settings().auto_train_episodes


def _latest_source_mtime(data_dir: str = DATA_DIR) -> float:
    """Latest modification time across raw scan debug data files."""
    latest = 0.0
    if not os.path.isdir(data_dir):
        return latest
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if name == "compiled_dataset.json":
                continue
            path = os.path.join(root, name)
            try:
                latest = max(latest, os.path.getmtime(path))
            except OSError:
                pass
    return latest


def weights_need_retrain(
    dataset_path: str = COMPILED_DATASET_PATH,
    model_path: str = MODEL_WEIGHTS_PATH,
) -> bool:
    """True when weights are missing or raw source data is newer than saved weights."""
    if not os.path.exists(dataset_path):
        return False
    if not os.path.exists(model_path):
        return True
    source_mtime = _latest_source_mtime()
    weights_mtime = os.path.getmtime(model_path)
    return source_mtime > weights_mtime


def should_auto_train(
    dataset_path: str = COMPILED_DATASET_PATH,
    model_path: str = MODEL_WEIGHTS_PATH,
) -> bool:
    """Startup auto-train when enabled and weights are stale/missing."""
    if not get_settings().auto_train_on_startup:
        return False
    return weights_need_retrain(dataset_path=dataset_path, model_path=model_path)


def run_training(
    agent: DQNAgent,
    episodes: int = DEFAULT_AUTO_TRAIN_EPISODES,
    save_weights: bool = True,
    model_path: str = MODEL_WEIGHTS_PATH,
) -> Dict[str, Any]:
    """Run RL fine-tuning episodes plus supervised pretrain on the compiled dataset."""
    env = ScanDebugEnv()
    total_rewards: list[float] = []
    losses: list[float] = []

    for _ in range(episodes):
        state, _info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, truncated, _info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done or truncated)
            loss = agent.optimize_model()
            if loss is not None:
                losses.append(loss)
            state = next_state
            ep_reward += reward

        total_rewards.append(ep_reward)

    if env.dataset:
        states_dataset = []
        actions_dataset = []
        for case in env.dataset:
            state_vec = env._build_state_for_case(case)
            target_act = REV_ACTION_MAP.get(case["true_action"], 3)
            states_dataset.append(state_vec)
            actions_dataset.append(target_act)
        agent.pretrain_supervised(states_dataset, actions_dataset, epochs=1500)

    if save_weights:
        agent.save(model_path)

    return {
        "status": "Training complete",
        "episodes_trained": episodes,
        "average_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "average_loss": float(np.mean(losses)) if losses else 0.0,
        "final_epsilon": agent.epsilon,
        "weights_saved": save_weights,
        "dataset_cases": len(env.dataset),
    }
