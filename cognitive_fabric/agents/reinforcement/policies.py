import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional

class PPOPolicy(nn.Module):
    """
    Proximal Policy Optimization (PPO) implementation for agent decision making
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon: float = 0.2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Policy network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)
    
    def update(
        self, 
        states: torch.Tensor, 
        rewards: torch.Tensor,
        old_probs: Optional[torch.Tensor] = None
    ) -> float:
        """
        Update policy using PPO
        """
        self.train()
        
        # Calculate advantages
        values = self.get_value(states).squeeze()
        advantages = rewards - values.detach()
        
        # Get current probabilities
        current_probs = self(states)
        
        # If old probabilities not provided, use current (for first update)
        if old_probs is None:
            old_probs = current_probs.detach()
        
        # Calculate probability ratio
        ratio = current_probs / old_probs.clamp(min=1e-8)
        
        # PPO loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values, rewards)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()

class MultiAgentPolicy(nn.Module):
    """
    Policy network for multi-agent coordination
    """
    
    def __init__(self, state_dim: int, num_agents: int, action_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Agent-specific heads
        self.agent_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(num_agents)
        ])
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_encoder(state)
        
        # Get actions for all agents
        actions = []
        for head in self.agent_heads:
            actions.append(head(shared_features))
        
        return torch.stack(actions, dim=1)  # [batch_size, num_agents, action_dim]