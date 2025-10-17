#!/usr/bin/env python3
"""
Agent training script for Cognitive Fabric
"""

import asyncio
import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.reinforcement.trainers import PPOTrainer
from agents.reinforcement.environments import MultiAgentEnvironment
from agents.core.cognitive_node import CognitiveNode
from config.base import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentTrainer:
    """
    Trainer for Cognitive Fabric agents
    """
    
    def __init__(self):
        self.env = MultiAgentEnvironment(num_agents=3)
        self.trainer = PPOTrainer(
            state_dim=512,
            action_dim=256,
            learning_rate=config.RL_LEARNING_RATE
        )
        self.agents = []
        
    def setup_agents(self):
        """Setup agents for training"""
        logger.info("Setting up training agents...")
        
        agent_configs = [
            {
                'agent_id': 'train_agent_1',
                'config': {
                    'LLM_MODEL': config.LLM_MODEL,
                    'specialization': 'research'
                }
            },
            {
                'agent_id': 'train_agent_2', 
                'config': {
                    'LLM_MODEL': config.LLM_MODEL,
                    'specialization': 'technical'
                }
            },
            {
                'agent_id': 'train_agent_3',
                'config': {
                    'LLM_MODEL': config.LLM_MODEL,
                    'specialization': 'general'
                }
            }
        ]
        
        for agent_config in agent_configs:
            # Create agent with training configuration
            agent = CognitiveNode(
                agent_config['agent_id'],
                {**config.__dict__, **agent_config['config']}
            )
            self.agents.append(agent)
        
        logger.info(f"Setup {len(self.agents)} agents for training")
    
    async def train_single_agent(self, agent: CognitiveNode, episodes: int = 100):
        """Train a single agent"""
        logger.info(f"Training agent: {agent.agent_id}")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Agent selects action based on state
                action = agent.rl_policy(torch.FloatTensor(state['state_vector']))
                action_idx = torch.multinomial(action, 1).item()
                
                # Execute action in environment
                next_state, reward, done, info = self.env.step({
                    'agent_selection': [action_idx],
                    'retrieval_depth': 2,
                    'verification_level': 1,
                    'collaboration_mode': 0
                })
                
                # Store experience for learning
                agent.training_buffer.append({
                    'state': torch.FloatTensor(state['state_vector']),
                    'action': action_idx,
                    'reward': reward,
                    'next_state': torch.FloatTensor(next_state['state_vector']),
                    'done': done
                })
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done or steps >= 50:
                    break
            
            # Train periodically
            if episode % 10 == 0 and len(agent.training_buffer) >= 32:
                await self._train_agent(agent)
            
            if episode % 20 == 0:
                logger.info(f"Episode {episode}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        logger.info(f"Completed training for {agent.agent_id}")
    
    async def _train_agent(self, agent: CognitiveNode):
        """Train agent on collected experience"""
        if len(agent.training_buffer) < 32:
            return
        
        # Convert buffer to training batches
        states = torch.stack([exp['state'] for exp in agent.training_buffer])
        actions = torch.LongTensor([exp['action'] for exp in agent.training_buffer])
        rewards = torch.FloatTensor([exp['reward'] for exp in agent.training_buffer])
        next_states = torch.stack([exp['next_state'] for exp in agent.training_buffer])
        dones = torch.BoolTensor([exp['done'] for exp in agent.training_buffer])
        
        # Train using PPO
        loss = self.trainer.train(states, actions, rewards, next_states, dones)
        
        # Clear buffer after training
        agent.training_buffer.clear()
        
        logger.debug(f"Training loss: {loss:.4f}")
    
    async def train_multi_agent(self, episodes: int = 500):
        """Train multiple agents collaboratively"""
        logger.info("Starting multi-agent training...")
        
        self.setup_agents()
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_rewards = [0] * len(self.agents)
            steps = 0
            
            while True:
                # Each agent selects action
                actions = []
                for i, agent in enumerate(self.agents):
                    action_probs = agent.rl_policy(torch.FloatTensor(state['state_vector']))
                    action_idx = torch.multinomial(action_probs, 1).item()
                    actions.append(action_idx)
                
                # Execute joint action
                next_state, rewards, done, info = self.env.step({
                    'agent_selection': actions,
                    'retrieval_depth': 2,
                    'verification_level': 1,
                    'collaboration_mode': 1  # Collaborative mode
                })
                
                # Store experiences
                for i, agent in enumerate(self.agents):
                    agent.training_buffer.append({
                        'state': torch.FloatTensor(state['state_vector']),
                        'action': actions[i],
                        'reward': rewards,
                        'next_state': torch.FloatTensor(next_state['state_vector']),
                        'done': done
                    })
                    episode_rewards[i] += rewards
                
                state = next_state
                steps += 1
                
                if done or steps >= 100:
                    break
            
            # Train agents periodically
            if episode % 25 == 0:
                training_tasks = []
                for agent in self.agents:
                    if len(agent.training_buffer) >= 32:
                        training_tasks.append(self._train_agent(agent))
                
                if training_tasks:
                    await asyncio.gather(*training_tasks)
            
            if episode % 50 == 0:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Steps = {steps}")
        
        logger.info("Multi-agent training completed")
    
    def save_models(self, save_path: Path):
        """Save trained models"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        for agent in self.agents:
            model_path = save_path / f"{agent.agent_id}_policy.pt"
            torch.save(agent.rl_policy.state_dict(), model_path)
            logger.info(f"Saved model: {model_path}")
        
        # Save trainer
        trainer_path = save_path / "ppo_trainer.pt"
        torch.save(self.trainer.state_dict(), trainer_path)
        
        logger.info(f"All models saved to: {save_path}")
    
    def load_models(self, load_path: Path):
        """Load trained models"""
        for agent in self.agents:
            model_path = load_path / f"{agent.agent_id}_policy.pt"
            if model_path.exists():
                agent.rl_policy.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded model: {model_path}")
        
        trainer_path = load_path / "ppo_trainer.pt"
        if trainer_path.exists():
            self.trainer.load_state_dict(torch.load(trainer_path))
        
        logger.info("Models loaded successfully")

async def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Cognitive Fabric agents")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi", help="Training mode")
    parser.add_argument("--save-dir", type=str, default="./models/trained", help="Directory to save models")
    parser.add_argument("--load-dir", type=str, help="Directory to load models from")
    
    args = parser.parse_args()
    
    trainer = AgentTrainer()
    
    # Load existing models if specified
    if args.load_dir:
        load_path = Path(args.load_dir)
        if load_path.exists():
            trainer.load_models(load_path)
        else:
            logger.warning(f"Load directory not found: {load_path}")
    
    # Run training
    if args.mode == "multi":
        await trainer.train_multi_agent(episodes=args.episodes)
    else:
        trainer.setup_agents()
        training_tasks = []
        for agent in trainer.agents:
            training_tasks.append(trainer.train_single_agent(agent, args.episodes // len(trainer.agents)))
        await asyncio.gather(*training_tasks)
    
    # Save trained models
    save_path = Path(args.save_dir)
    trainer.save_models(save_path)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    asyncio.run(main())