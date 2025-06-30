import numpy as np
import time
from stable_baselines3 import PPO as SB3_PPO    
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class SB3TrainingCallback(BaseCallback):
    """Callback per tracciare episodi durante training SB3"""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            # Log ogni 50 episodi
            if len(self.episode_rewards) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"  SB3 Episode {len(self.episode_rewards)}: Avg Reward (last 50) = {avg_reward:.2f}")
        
        return True

def train_sb3_ppo(env_name, total_timesteps):
    """Training Stable-Baselines3 PPO """
    print("🏗️  Training Stable-Baselines3 PPO...")
    
    env = make_vec_env(env_name, n_envs=1)
    
    model = SB3_PPO(
        "MlpPolicy", 
        env,
        learning_rate=3e-4,      
        n_steps=2048,           
        batch_size=64,           
        n_epochs=10,             
        gamma=0.99,              
        gae_lambda=0.95,         
        clip_range=0.2,          
        ent_coef=0.01,           
        vf_coef=1.0,            
        max_grad_norm=0.5,       
    )
    
    callback = SB3TrainingCallback()
    
    # Training
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time
    
    env.close()
    
    print(f"SB3 completed in {training_time:.2f} seconds")
    return model, callback.episode_rewards, training_time