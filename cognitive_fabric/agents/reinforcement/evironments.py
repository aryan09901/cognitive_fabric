import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch

@dataclass
class AgentState:
    query: str
    context: List[Dict[str, Any]]
    reputation: float
    memory_usage: float
    available_agents: List[str]

class MultiAgentEnvironment(gym.Env):
    """
    Multi-agent reinforcement learning environment for Cognitive Fabric
    """
    
    def __init__(self, num_agents: int = 5, state_dim: int = 512, action_dim: int = 256):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            'state_vector': spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,)),
            'reputation': spaces.Box(low=0, high=1000, shape=(num_agents,)),
            'capabilities': spaces.MultiBinary(num_agents * 10),  # 10 capabilities per agent
            'query_complexity': spaces.Box(low=0, high=1, shape=(1,))
        })
        
        self.action_space = spaces.Dict({
            'agent_selection': spaces.MultiDiscrete([num_agents] * 3),  # Select up to 3 agents
            'retrieval_depth': spaces.Discrete(5),  # How deep to search knowledge
            'verification_level': spaces.Discrete(3),  # Verification strictness
            'collaboration_mode': spaces.Discrete(2)  # Individual vs collaborative
        })
        
        self.agents = []
        self.current_state = None
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment"""
        self.step_count = 0
        self.agents = self._initialize_agents()
        self.current_state = self._get_initial_state()
        
        return self._state_to_observation(self.current_state)
    
    def step(self, action: Dict[str, Any]) -> tuple:
        """Execute one environment step"""
        self.step_count += 1
        
        # Execute actions
        reward = self._execute_actions(action)
        next_state = self._update_state(action)
        done = self.step_count >= self.max_steps
        
        info = {
            'step': self.step_count,
            'agents_used': action.get('agent_selection', []),
            'reward_components': self._get_reward_components()
        }
        
        return self._state_to_observation(next_state), reward, done, info
    
    def _initialize_agents(self) -> List[Dict[str, Any]]:
        """Initialize agents with different capabilities"""
        agents = []
        capabilities_list = [
            ['research', 'analysis', 'verification'],
            ['technical', 'coding', 'debugging'],
            ['general', 'explanation', 'summarization'],
            ['creative', 'brainstorming', 'ideation'],
            ['critical', 'evaluation', 'fact_checking']
        ]
        
        for i in range(self.num_agents):
            agents.append({
                'id': f'agent_{i}',
                'reputation': np.random.uniform(50, 100),
                'capabilities': capabilities_list[i % len(capabilities_list)],
                'specialization': capabilities_list[i % len(capabilities_list)][0],
                'success_rate': np.random.uniform(0.7, 0.95)
            })
        
        return agents
    
    def _get_initial_state(self) -> AgentState:
        """Get initial state for the environment"""
        return AgentState(
            query="What are the ethical implications of artificial general intelligence?",
            context=[],
            reputation=100.0,
            memory_usage=0.1,
            available_agents=[agent['id'] for agent in self.agents]
        )
    
    def _execute_actions(self, action: Dict[str, Any]) -> float:
        """Execute actions and calculate reward"""
        reward = 0.0
        
        # Reward for appropriate agent selection
        selected_agents = action.get('agent_selection', [])
        if self._is_appropriate_selection(selected_agents, self.current_state.query):
            reward += 2.0
        
        # Reward for efficient resource usage
        retrieval_depth = action.get('retrieval_depth', 2)
        if retrieval_depth <= 3:  # Not too deep
            reward += 0.5
        
        # Reward for verification level matching query importance
        verification_level = action.get('verification_level', 1)
        if self._needs_high_verification(self.current_state.query):
            if verification_level == 2:  # Strict verification
                reward += 1.5
        else:
            if verification_level == 0:  # Flexible verification
                reward += 1.0
        
        # Penalty for poor collaboration decisions
        collaboration_mode = action.get('collaboration_mode', 0)
        if len(selected_agents) > 1 and collaboration_mode == 0:
            reward -= 1.0  # Should use collaboration for multiple agents
        
        return reward
    
    def _update_state(self, action: Dict[str, Any]) -> AgentState:
        """Update environment state based on actions"""
        # Simulate state changes
        new_reputation = self.current_state.reputation * 0.99  # Small decay
        new_memory_usage = min(1.0, self.current_state.memory_usage + 0.05)
        
        # Update agent reputations based on performance
        selected_agents = action.get('agent_selection', [])
        for agent_idx in selected_agents:
            if agent_idx < len(self.agents):
                performance = np.random.uniform(0.8, 1.0)
                self.agents[agent_idx]['reputation'] *= performance
        
        return AgentState(
            query=self.current_state.query,  # In real env, this would change
            context=self.current_state.context,
            reputation=new_reputation,
            memory_usage=new_memory_usage,
            available_agents=self.current_state.available_agents
        )
    
    def _state_to_observation(self, state: AgentState) -> Dict[str, np.ndarray]:
        """Convert state to observation for RL agent"""
        # State vector (simplified)
        state_vector = np.random.randn(self.state_dim)
        
        # Reputation vector
        reputation_vector = np.array([agent['reputation'] for agent in self.agents])
        
        # Capabilities matrix (flattened)
        capabilities_vector = np.zeros(self.num_agents * 10)
        for i, agent in enumerate(self.agents):
            for j, capability in enumerate(agent['capabilities']):
                if j < 10:  # Limit to 10 capabilities
                    capabilities_vector[i * 10 + j] = 1
        
        # Query complexity (simplified)
        query_complexity = np.array([self._estimate_complexity(state.query)])
        
        return {
            'state_vector': state_vector,
            'reputation': reputation_vector,
            'capabilities': capabilities_vector,
            'query_complexity': query_complexity
        }
    
    def _is_appropriate_selection(self, selected_agents: List[int], query: str) -> bool:
        """Check if agent selection is appropriate for the query"""
        if not selected_agents:
            return False
        
        # Simple matching based on query keywords and agent capabilities
        query_lower = query.lower()
        
        for agent_idx in selected_agents:
            if agent_idx >= len(self.agents):
                return False
            
            agent = self.agents[agent_idx]
            capabilities = ' '.join(agent['capabilities']).lower()
            
            # Check if agent capabilities match query needs
            if 'technical' in capabilities and any(word in query_lower for word in ['code', 'programming', 'technical']):
                return True
            if 'research' in capabilities and any(word in query_lower for word in ['research', 'study', 'analysis']):
                return True
        
        return len(selected_agents) > 0  # Default to True if agents selected
    
    def _needs_high_verification(self, query: str) -> bool:
        """Check if query needs high verification"""
        high_verification_keywords = [
            'medical', 'health', 'safety', 'legal', 'financial', 
            'investment', 'treatment', 'diagnosis'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in high_verification_keywords)
    
    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)"""
        # Simple heuristic based on query length and keywords
        length_factor = min(len(query.split()) / 50, 1.0)  # Normalize by 50 words
        
        complex_keywords = ['compare', 'analyze', 'evaluate', 'critique', 'synthesize']
        keyword_factor = 0.0
        for keyword in complex_keywords:
            if keyword in query.lower():
                keyword_factor += 0.2
        
        return min(length_factor + keyword_factor, 1.0)
    
    def _get_reward_components(self) -> Dict[str, float]:
        """Get detailed reward components for analysis"""
        return {
            'agent_selection': 2.0,
            'resource_efficiency': 0.5,
            'verification_appropriateness': 1.5,
            'collaboration_decision': -1.0
        }

class SingleAgentEnvironment(gym.Env):
    """
    Single-agent environment for individual agent training
    """
    
    def __init__(self, state_dim: int = 256, action_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.action_space = spaces.Discrete(action_dim)
        
        self.current_state = None
        
    def reset(self):
        """Reset environment"""
        self.current_state = np.random.randn(self.state_dim)
        return self.current_state
    
    def step(self, action):
        """Execute step"""
        # Simple reward based on action
        reward = -0.1  # Small step penalty
        if action < self.action_dim // 2:  # "Good" actions in first half
            reward += 1.0
        
        # Next state (random walk)
        self.current_state = self.current_state + np.random.randn(self.state_dim) * 0.1
        
        done = np.random.random() < 0.05  # 5% chance of episode ending
        
        return self.current_state, reward, done, {}