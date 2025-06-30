import numpy as np
import gymnasium as gym
import random

class RandomAgent:
    """Agente baseline che sceglie azioni casuali"""
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def act(self, state):
        return random.randint(0, self.action_dim - 1)
    
def train_random_agent(env_name, total_timesteps):
    """Training del Random Agent (giusto per raccogliere statistiche)"""
    print("Training Random Agent...")
    
    env = gym.make(env_name)
    action_dim = env.action_space.n
    agent = RandomAgent(action_dim)
    
    scores = []
    timesteps = 0
    episode = 0
    
    while timesteps < total_timesteps:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_score = 0
        
        while timesteps < total_timesteps:
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            episode_score += reward
            timesteps += 1
            
            if done:
                scores.append(episode_score)
                episode += 1
                
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                    print(f"  Random Episode {episode}: Avg Score = {avg_score:.2f}")
                
                break
    
    env.close()
    return scores