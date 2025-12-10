import __init__
import gymnasium as gym
import highway_env
import sys
import os
from sb3_contrib import QRDQN

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

OUTDIR = "data"
MODEL_NAME = "qrdqn_agent_final"
VEC_NORM_STATS_FILE = f"{OUTDIR}/vec_normalize_stats.pkl" 
monitor_log_path = f"{OUTDIR}/monitor.csv"
modelFile = f"{OUTDIR}/{MODEL_NAME}.zip" 
saveAs = f"{OUTDIR}/{MODEL_NAME}.zip"
ADDITIONAL_TIMESTEPS = 150000
NEW_LEARNING_RATE = 1e-4

def create_env():
    """Creates the highway-construction environment with monitor for logging."""
    env = gym.make(
        "highway-construction-v0",
        render_mode="rgb_array",
    )
    # Use override_existing=False to append data during continued training
    env = Monitor(
        env,
        filename=monitor_log_path,
        allow_early_resets=True,
        override_existing=False, 
    )
    return env

if __name__ == "__main__":
    env = DummyVecEnv([lambda: create_env()]) 

    if os.path.exists(VEC_NORM_STATS_FILE):
        print(f"Loading VecNormalize stats from {VEC_NORM_STATS_FILE}")
        env = VecNormalize.load(VEC_NORM_STATS_FILE, env)
        env.norm_obs = True
        env.norm_reward = True
        env.clip_obs = 10.
    else:
        print(f"CRITICAL ERROR: VecNormalize stats file not found at {VEC_NORM_STATS_FILE}. Exiting.")
        sys.exit(1)

    print(f"Loading existing model from {modelFile}...")
    try:
        model = QRDQN.load(modelFile, env=env, device="auto", custom_objects=None)
        print("Model loaded successfully. training")
        
        #Use chosen learning rate
        model.lr_schedule = lambda remaining_progress: NEW_LEARNING_RATE
        
        if model.policy.optimizer is not None:
             for param_group in model.policy.optimizer.param_groups:
                 param_group['lr'] = NEW_LEARNING_RATE
        
        print(f"Learning rate overridden to: {NEW_LEARNING_RATE}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close() 
        sys.exit(1)

    print(f"\nContinuing training for {ADDITIONAL_TIMESTEPS} more timesteps")
    
    try:
        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS, 
            log_interval=1,
            reset_num_timesteps=False 
        )
        print("\nTraining completed.")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving model and stats.")

    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}. Saving model and stats.")
    
    print(f"Saving model to {saveAs}")
    model.save(saveAs)
    
    print(f"Saving VecNormalize stats to {VEC_NORM_STATS_FILE}")
    env.save(VEC_NORM_STATS_FILE) 

    final_timesteps = model.num_timesteps
    print(f"\nTotal cumulative timesteps trained: {final_timesteps}")

    env.close()