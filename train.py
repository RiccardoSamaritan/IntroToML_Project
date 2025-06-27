from collections import deque
import numpy as np
import torch
import gymnasium as gym

from ppo import PPOAgent

def training_ppo(env_name='CartPole-v1', total_timesteps=1000000):
    """
    Training PPO seguendo esattamente l'Algorithm 1 del paper
    - Raccoglie horizon=2048 timesteps prima di ogni update
    - Usa minibatch SGD per K=10 epoche
    - Iperparametri identici al paper (Tabella 3)
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Iperparametri ESATTI dal paper (Tabella 3 - MuJoCo)
    agent = PPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=3e-4,           # Adam stepsize (paper)
        gamma=0.99,        # Discount (paper)
        eps_clip=0.2,      # Standard PPO clipping
        k_epochs=10,       # Num. epochs (paper)
        gae_lambda=0.95,   # GAE parameter (paper)
        minibatch_size=64, # Minibatch size (paper)
        horizon=2048,      # Horizon T (paper)
        c1=1.0,           # VF coeff (da Equazione 9)
        c2=0.01           # Entropy coeff (tipico)
    )
    
    scores = deque(maxlen=100)
    episode_scores = []
    timesteps = 0
    episode = 0
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    print("🚀 Training PPO con iperparametri del paper...")
    print(f"📋 Horizon: {agent.horizon}, Epochs: {agent.k_epochs}, Minibatch: {agent.minibatch_size}")
    
    while timesteps < total_timesteps:
        episode_score = 0
        episode_steps = 0
        
        while timesteps < total_timesteps:
            # Seleziona azione
            action, log_prob, value = agent.policy.act(torch.FloatTensor(state).unsqueeze(0))
            
            # Esegui azione nell'ambiente
            next_state, reward, done, _, _ = env.step(action)
            
            # Memorizza esperienza
            agent.remember(state, action, reward, log_prob.item(), value, done)
            
            state = next_state
            episode_score += reward
            episode_steps += 1
            timesteps += 1
            
            # Update ogni horizon timesteps (come nel paper)
            if agent.should_update():
                print(f"🔄 Update a timestep {timesteps} (horizon raggiunto)")
                agent.update()
                
            if done:
                scores.append(episode_score)
                episode_scores.append(episode_score)
                episode += 1
                
                # Reset environment
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                
                # Log progresso
                if episode % 10 == 0:
                    avg_score = np.mean(list(scores)[-10:]) if len(scores) >= 10 else np.mean(scores)
                    print(f"📊 Episodio {episode}, Timesteps: {timesteps}, Score medio (ultimi 10): {avg_score:.2f}")
                
                break
    
    env.close()
    print(f"✅ Training completato! Episodi totali: {episode}, Timesteps: {timesteps}")
    return episode_scores