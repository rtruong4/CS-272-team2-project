import __init__
import gymnasium as gym
import highway_env
import sys
import os
import torch as th 
from typing import Callable 
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn 

MODEL_NAME = "qrdqn_agent_low_gamma_96"
OUTDIR = "data"
FILE_NAME_ZIP = f"{MODEL_NAME}_final.zip"
VEC_NORM_STATS_FILE = f"{OUTDIR}/vec_normalize_stats.pkl"

PHASE1_TIMESTEPS = 100000 
PHASE2_TIMESTEPS = 200000 
PHASE3_TIMESTEPS = 50000 
TOTAL_TIMESTEPS = PHASE1_TIMESTEPS + PHASE2_TIMESTEPS + PHASE3_TIMESTEPS # 350k total

LR_PHASE1 = 5e-4 # Highest LR for initial exploration
LR_PHASE2 = 3e-4 # Medium LR for core learning
LR_PHASE3 = 1e-4 # Lowest LR for stable fine-tuning


ENV_CONFIG = {
    "reward_weights": [0.5, 0, 0.5, 0, -1.0, 0],  
    "screen_width": 1600,
    "screen_height": 600,
    "scaling": 1.2,
    "centering_position": [0.3, 0.5],
    "duration": 120,
}

os.makedirs(OUTDIR, exist_ok=True)

def three_phase_schedule(p1_steps: int, p2_steps: int, lr1: float, lr2: float, lr3: float) -> Callable[[float], float]:
    """Custom learning rate schedule based on total steps completed."""
    def func(progress_remaining: float) -> float:

        progress_completed = 1.0 - progress_remaining
        
        # Calculate steps completed based on total timesteps
        timesteps_completed = progress_completed * TOTAL_TIMESTEPS
        
        boundary_1 = p1_steps
        boundary_2 = p1_steps + p2_steps
        
        if timesteps_completed < boundary_1:
            return lr1
        elif timesteps_completed < boundary_2:
            return lr2
        else:
            return lr3
            
    return func

def create_env(monitor_path=None):
    """Creates the highway-construction environment with custom configuration."""
    env = gym.make(
        "highway-construction-v0",
        render_mode="rgb_array", 
    )
    env.unwrapped.configure(ENV_CONFIG)

    if monitor_path:
        env = Monitor(env, monitor_path) 
    return env

POLICY_KWARGS = dict(
    n_quantiles=50, # Number of quantiles for distribution prediction
)

QRDQN_HYPERPARAMS = dict(
    learning_rate=three_phase_schedule(
        PHASE1_TIMESTEPS, 
        PHASE2_TIMESTEPS, 
        LR_PHASE1, 
        LR_PHASE2, 
        LR_PHASE3
    ), 
    gamma=0.9999,
    
    buffer_size=1_000_000, # Large buffer for off-policy learning
    learning_starts=50000, # Number of steps to fill the buffer before training begins
    train_freq=(4, "step"), # Train every 4 steps
    gradient_steps=1,
    batch_size=512, # Retained large batch size for stable updates
)

if __name__ == "__main__":
    
    train_env = DummyVecEnv([lambda: create_env(monitor_path=f"{OUTDIR}/monitor.csv")])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = QRDQN(
        "MlpPolicy", 
        train_env,
        verbose=1,
        device="auto",
        policy_kwargs=POLICY_KWARGS,
        **QRDQN_HYPERPARAMS
    )
    
    print("Starting QR-DQN training with standard rewards and high Gamma...")
    print(f"GAMMA: {QRDQN_HYPERPARAMS['gamma']}")
    print("-" * 30)
    print(f"Phase 1 (LR={LR_PHASE1}) for {PHASE1_TIMESTEPS} timesteps.")
    print(f"Phase 2 (LR={LR_PHASE2}) for {PHASE2_TIMESTEPS} timesteps.")
    print(f"Phase 3 (LR={LR_PHASE3}) for {PHASE3_TIMESTEPS} timesteps.")
    print(f"Total training duration: {TOTAL_TIMESTEPS} timesteps.")


    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=1,
        )
        print("\nTraining completed successfully without interruption.")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving model and stats")

    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}. Saving model and stats")
    
    model_save_path = f"{OUTDIR}/{FILE_NAME_ZIP}"
    stats_save_path = VEC_NORM_STATS_FILE

    print(f"Saving model to {model_save_path}")
    model.save(model_save_path)
    
    print(f"Saving VecNormalize stats to {stats_save_path}")
    train_env.save(stats_save_path) 

    print(f"\nTraining finished after {model.num_timesteps} timesteps.")
    print(f"Final model saved as {FILE_NAME_ZIP}.")

    train_env.close()