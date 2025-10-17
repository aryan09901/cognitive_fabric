import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from .policies import PPOPolicy

class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for cognitive agents
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize policy
        self.policy = PPOPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training history
        self.training_history = {
            'losses': [],
            'rewards': [],
            'episode_lengths': []
        }
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor,
              epochs: int = 10) -> float:
        """
        Train the policy using PPO
        """
        batch_size = states.size(0)
        
        # Calculate advantages
        with torch.no_grad():
            current_values = self.policy.get_value(states).squeeze()
            next_values = self.policy.get_value(next_states).squeeze()
            
            # TD target
            targets = rewards + self.gamma * next_values * (~dones).float()
            advantages = targets - current_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action probabilities
        old_action_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze().detach()
        
        total_loss = 0.0
        
        for epoch in range(epochs):
            # Get current action probabilities
            current_action_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
            current_values = self.policy.get_value(states).squeeze()
            
            # Probability ratio
            ratio = current_action_probs / (old_action_probs + 1e-8)
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(current_values, targets)
            
            # Entropy bonus
            action_dist = self.policy(states)
            entropy = -(action_dist * torch.log(action_dist + 1e-8)).sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / epochs
        self.training_history['losses'].append(avg_loss)
        
        return avg_loss
    
    def compute_returns(self, rewards: List[float], gamma: float = None) -> List[float]:
        """Compute discounted returns"""
        if gamma is None:
            gamma = self.gamma
        
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        return returns
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_history['losses']:
            return {}
        
        recent_losses = self.training_history['losses'][-100:]  # Last 100 episodes
        recent_rewards = self.training_history['rewards'][-100:] if self.training_history['rewards'] else []
        
        stats = {
            'current_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0,
            'average_loss': np.mean(recent_losses) if recent_losses else 0,
            'min_loss': np.min(recent_losses) if recent_losses else 0,
            'max_loss': np.max(recent_losses) if recent_losses else 0,
            'training_episodes': len(self.training_history['losses'])
        }
        
        if recent_rewards:
            stats.update({
                'average_reward': np.mean(recent_rewards),
                'max_reward': np.max(recent_rewards),
                'min_reward': np.min(recent_rewards)
            })
        
        return stats

class MultiAgentTrainer:
    """
    Trainer for multi-agent reinforcement learning
    """
    
    def __init__(self, num_agents: int, state_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.trainers = [
            PPOTrainer(state_dim, action_dim) for _ in range(num_agents)
        ]
        self.joint_training_history = []
    
    def train_joint(self,
                   states: List[torch.Tensor],
                   actions: List[torch.Tensor],
                   rewards: List[torch.Tensor],
                   next_states: List[torch.Tensor],
                   dones: List[torch.Tensor]):
        """
        Train multiple agents jointly
        """
        losses = []
        
        for i, trainer in enumerate(self.trainers):
            loss = trainer.train(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
            losses.append(loss)
        
        self.joint_training_history.append({
            'losses': losses,
            'average_loss': np.mean(losses),
            'timestamp': torch.tensor(len(self.joint_training_history))
        })
        
        return losses
    
    def get_agent_trainer(self, agent_index: int) -> PPOTrainer:
        """Get trainer for specific agent"""
        if 0 <= agent_index < self.num_agents:
            return self.trainers[agent_index]
        raise ValueError(f"Agent index {agent_index} out of range")
    
    def save_all_checkpoints(self, base_path: str):
        """Save checkpoints for all agents"""
        for i, trainer in enumerate(self.trainers):
            filepath = f"{base_path}/agent_{i}_checkpoint.pt"
            trainer.save_checkpoint(filepath)
    
    def load_all_checkpoints(self, base_path: str):
        """Load checkpoints for all agents"""
        for i, trainer in enumerate(self.trainers):
            filepath = f"{base_path}/agent_{i}_checkpoint.pt"
            trainer.load_checkpoint(filepath)