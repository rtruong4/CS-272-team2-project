import gymnasium as gym
import highway_env
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from sb3_contrib import QRDQN


import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import __init__

OUTDIR = "data"
MODEL_PATH = "qrdqn_agent_final"
N_EPISODES = 100

def create_env():
    env = gym.make(
        "highway-construction-v0",
        render_mode="rgb_array",
    )
    return env

def evaluate():
    model = QRDQN.load(MODEL_PATH)
    env = create_env()

    returns = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        ep_ret = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward

            if terminated or truncated:
                returns.append(ep_ret)
                break

    env.close()
    return np.array(returns)

def plot_violin(returns):
    plt.figure(figsize=(7,6))
    sns.violinplot(data=returns)
    plt.title("Custom Env – 1000 Episode Evaluation")
    plt.ylabel("Episode Return")
    plt.savefig("violin_plot.png")
    plt.close()

if __name__ == "__main__":
    results = evaluate()
    np.save("returns.npy", results)
    plot_violin(results)

    print("Evaluation complete! Saved:")
    print("returns.npy, violin_plot.png")
