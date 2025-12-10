import __init__
import gymnasium as gym
import time
import highway_env
import numpy as np

env = gym.make(
    "highway-construction-v0",
    render_mode="human", 
)

inner = env.unwrapped
inner.config["screen_width"]  = 1600
inner.config["screen_height"] = 600
inner.config["scaling"]       = 1.2
inner.config["centering_position"] = [0.3, 0.5]
inner.config["duration"] = 120

SIM_FREQ = env.unwrapped.config["simulation_frequency"]
PAUSE_TIME = 1 / SIM_FREQ 

def visualize_agent_performance_on_input(model, env, num_episodes=3):

    
    for episode in range(num_episodes):
        
        if episode > 0:
            input("Press Enter to start the next episode...") 
            
        print(f"\n--- Running Episode {episode + 1}/{num_episodes} ---")
        
        obs, info = env.reset()
        env.render() 


        done = False
        step_count = 0
        total_reward = 0
        
        max_steps = inner.config["duration"] * SIM_FREQ 

        while not done and step_count < max_steps:
            
            action = env.action_space.sample() 
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            env.render() 
            
            time.sleep(PAUSE_TIME)
            
        print(f"Episode finished after {step_count} steps. Total Reward: {total_reward:.2f}")


        time.sleep(1) 



model = None 
visualize_agent_performance_on_input(model, env, num_episodes=20)
env.close()

