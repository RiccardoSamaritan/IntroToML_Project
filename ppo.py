import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym

class ActorCritic(nn.Module):
    """
    Rete neurale che combina Actor (policy) e Critic (value function)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Layers condivisi
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Head dell'Actor (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Head del Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Propaga lo stato attraverso la rete per ottenere probabilità delle azioni e valore dello stato
        """
        shared_features = self.shared(state)
    
        action_probs = F.softmax(self.actor(shared_features), dim=-1)
        
        state_value = self.critic(shared_features)
        
        return action_probs, state_value
    
    def act(self, state):
        """
        Seleziona un'azione dato uno stato
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), state_value

class PPOAgent:
    """
    Agente PPO (Proximal Policy Optimization)
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=10, gae_lambda=0.95, 
                 minibatch_size=64, horizon=2048, c1=1.0, c2=0.01):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs  # Paper usa 10 epoche
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size  # Paper: 64 per MuJoCo
        self.horizon = horizon  # Paper: 2048 timesteps
        self.c1 = c1  # Coefficiente value function loss
        self.c2 = c2  # Coefficiente entropy bonus
        
        # Inizializza la rete Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffer per memorizzare le esperienze
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        self.timesteps_collected = 0  # Counter per horizon
        
    def remember(self, state, action, reward, log_prob, value, done):
        """Memorizza un'esperienza nel buffer"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)
        self.timesteps_collected += 1
    
    def should_update(self):
        """Determina se è il momento di aggiornare (ogni horizon timesteps)"""
        return self.timesteps_collected >= self.horizon
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Calcola Generalized Advantage Estimation (GAE)
        Fornisce stime più stabili dell'advantage
        """
        advantages = []
        gae = 0
        
        # Aggiungi il valore del prossimo stato
        values = values + [next_value]
        
        # Calcola GAE all'indietro
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self):
        """Aggiorna la policy usando l'algoritmo PPO"""
        if len(self.memory['states']) == 0:
            return
        
        # Converti in tensori PyTorch
        states = torch.FloatTensor(self.memory['states'])
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.FloatTensor(self.memory['log_probs'])
        rewards = self.memory['rewards']
        values = [v.item() for v in self.memory['values']]
        dones = self.memory['dones']
        
        # Calcola il valore del prossimo stato (0 se episodio terminato)
        with torch.no_grad():
            if dones[-1]:
                next_value = 0
            else:
                _, next_value = self.policy(states[-1:])
                next_value = next_value.item()
        
        # Calcola advantages e returns
        advantages = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.FloatTensor(advantages)
        
        values_tensor = torch.FloatTensor(values[:len(advantages)])
        returns = advantages + values_tensor

        # Normalizza gli advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare data for minibatch training
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        # Aggiornamento PPO per k epoche come nel paper
        for epoch in range(self.k_epochs):
            # Shuffle indices per ogni epoca
            np.random.shuffle(indices)
            
            # Dividi in minibatch come nel paper
            for start_idx in range(0, dataset_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Estrai minibatch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Calcola nuove probabilità e valori
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calcola il rapporto di importance sampling
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calcola l'obiettivo PPO clippato
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Loss del critic (Mean Squared Error)
                critic_loss = F.mse_loss(state_values.squeeze(), batch_returns)
                
                # Loss totale con coefficienti del paper (Equazione 9)
                total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                
                # Aggiornamento gradiente
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping come best practice (non specificato nel paper)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Pulisci il buffer
        self.clear_memory()
        self.timesteps_collected = 0
    
    def clear_memory(self):
        """Pulisce il buffer delle esperienze"""
        for key in self.memory:
            self.memory[key] = []

def train_ppo(env_name='CartPole-v1', total_timesteps=1000000, save_model_flag=True):
    """
    Training PPO:
    - Raccoglie horizon=2048 timesteps prima di ogni update
    - Usa minibatch SGD per K=10 epoche
    - Iperparametri identici al paper "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=10,
        gae_lambda=0.95,
        minibatch_size=64,
        horizon=2048,
        c1=1.0,
        c2=0.01
    )
    
    scores = deque(maxlen=100)
    episode_scores = []
    timesteps = 0
    episode = 0
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    print("🚀 Training PPO...")
    print(f"📋 Horizon: {agent.horizon}, Epochs: {agent.k_epochs}, Minibatch: {agent.minibatch_size}")
    
    best_avg_score = -float('inf')
    
    while timesteps < total_timesteps:
        episode_score = 0
        episode_steps = 0
        
        while timesteps < total_timesteps:
            action, log_prob, value = agent.policy.act(torch.FloatTensor(state).unsqueeze(0))
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, log_prob.item(), value, done)
            
            state = next_state
            episode_score += reward
            episode_steps += 1
            timesteps += 1
            
            if agent.should_update():
                print(f"🔄 Update a timestep {timesteps} (horizon raggiunto)")
                agent.update()
                
            if done:
                scores.append(episode_score)
                episode_scores.append(episode_score)
                episode += 1
                
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                
                # Salva il miglior modello
                if len(scores) >= 100:
                    current_avg = np.mean(list(scores))
                    if current_avg > best_avg_score and save_model_flag:
                        best_avg_score = current_avg
                        print(f"🏆 Nuovo miglior modello! Avg score: {current_avg:.2f}")
                
                if episode % 10 == 0:
                    avg_score = np.mean(list(scores)[-10:]) if len(scores) >= 10 else np.mean(scores)
                    print(f"📊 Episodio {episode}, Timesteps: {timesteps}, Score medio (ultimi 10): {avg_score:.2f}")
                
                break
    
    env.close()
    
    print(f"✅ Training completato! Episodi totali: {episode}, Timesteps: {timesteps}")
    return episode_scores, agent