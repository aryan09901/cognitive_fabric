import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class RewardComponents:
    verification_reward: float
    confidence_reward: float
    efficiency_reward: float
    collaboration_reward: float
    reputation_reward: float
    total_reward: float

class RewardCalculator:
    """
    Advanced reward calculation for reinforcement learning
    """
    
    def __init__(self):
        self.weights = {
            'verification': 0.3,
            'confidence': 0.2,
            'efficiency': 0.15,
            'collaboration': 0.2,
            'reputation': 0.15
        }
        
        self.verification_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'poor': 0.5
        }
    
    def calculate_reward(self, 
                       response_data: Dict[str, Any],
                       context: List[Dict[str, Any]],
                       processing_time: float,
                       agent_reputation: float,
                       collaboration_used: bool = False) -> RewardComponents:
        """
        Calculate comprehensive reward for agent actions
        """
        # Verification-based reward
        verification_reward = self._calculate_verification_reward(
            response_data.get('verification_score', 0),
            response_data.get('sources', [])
        )
        
        # Confidence-based reward
        confidence_reward = self._calculate_confidence_reward(
            response_data.get('confidence', 0),
            len(context)
        )
        
        # Efficiency reward (penalize slow responses)
        efficiency_reward = self._calculate_efficiency_reward(processing_time)
        
        # Collaboration reward
        collaboration_reward = self._calculate_collaboration_reward(
            collaboration_used,
            response_data.get('agents_used', [])
        )
        
        # Reputation-based reward
        reputation_reward = self._calculate_reputation_reward(agent_reputation)
        
        # Total weighted reward
        total_reward = (
            verification_reward * self.weights['verification'] +
            confidence_reward * self.weights['confidence'] +
            efficiency_reward * self.weights['efficiency'] +
            collaboration_reward * self.weights['collaboration'] +
            reputation_reward * self.weights['reputation']
        )
        
        return RewardComponents(
            verification_reward=verification_reward,
            confidence_reward=confidence_reward,
            efficiency_reward=efficiency_reward,
            collaboration_reward=collaboration_reward,
            reputation_reward=reputation_reward,
            total_reward=total_reward
        )
    
    def _calculate_verification_reward(self, verification_score: float, sources: List[Dict]) -> float:
        """Calculate reward based on verification quality"""
        if verification_score >= self.verification_thresholds['excellent']:
            base_reward = 2.0
        elif verification_score >= self.verification_thresholds['good']:
            base_reward = 1.0
        elif verification_score >= self.verification_thresholds['poor']:
            base_reward = 0.0
        else:
            base_reward = -1.0
        
        # Bonus for multiple high-quality sources
        high_quality_sources = sum(1 for source in sources 
                                 if source.get('verification_score', 0) >= 0.7)
        source_bonus = min(high_quality_sources * 0.2, 0.6)
        
        return base_reward + source_bonus
    
    def _calculate_confidence_reward(self, confidence: float, context_size: int) -> float:
        """Calculate reward based on confidence and context utilization"""
        # Base confidence reward
        confidence_reward = max(0.0, confidence - 0.5) * 2.0  # Scale to 0-1 range
        
        # Context utilization bonus
        if context_size > 0:
            context_bonus = min(context_size * 0.1, 0.5)
        else:
            context_bonus = -0.5  # Penalty for no context
        
        return confidence_reward + context_bonus
    
    def _calculate_efficiency_reward(self, processing_time: float) -> float:
        """Calculate reward based on processing efficiency"""
        # Target processing time (seconds)
        target_time = 3.0
        max_penalty_time = 10.0
        
        if processing_time <= target_time:
            return 1.0
        elif processing_time <= max_penalty_time:
            # Linear penalty from 1.0 to -1.0
            penalty = (processing_time - target_time) / (max_penalty_time - target_time) * 2.0
            return 1.0 - penalty
        else:
            return -1.0
    
    def _calculate_collaboration_reward(self, collaboration_used: bool, agents_used: List[str]) -> float:
        """Calculate reward for collaboration decisions"""
        if not collaboration_used:
            # Single agent - neutral reward
            return 0.0
        
        # Collaboration used - reward based on number of agents
        num_agents = len(agents_used)
        if num_agents == 1:
            return -0.5  # Penalty for unnecessary collaboration flag
        elif num_agents == 2:
            return 0.5   # Good for complex tasks
        elif num_agents == 3:
            return 1.0   # Excellent for very complex tasks
        else:
            return 0.2   # Diminishing returns for more agents
    
    def _calculate_reputation_reward(self, reputation: float) -> float:
        """Calculate reward based on agent reputation"""
        # Normalize reputation to 0-1 scale (assuming max reputation 1000)
        normalized_reputation = reputation / 1000.0
        return normalized_reputation
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update reward component weights"""
        # Validate weights sum to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError("Reward weights must sum to 1.0")
        
        self.weights.update(new_weights)
    
    def get_reward_breakdown(self, components: RewardComponents) -> Dict[str, float]:
        """Get detailed breakdown of reward components"""
        return {
            'verification': components.verification_reward * self.weights['verification'],
            'confidence': components.confidence_reward * self.weights['confidence'],
            'efficiency': components.efficiency_reward * self.weights['efficiency'],
            'collaboration': components.collaboration_reward * self.weights['collaboration'],
            'reputation': components.reputation_reward * self.weights['reputation'],
            'total': components.total_reward
        }

class MultiAgentRewardCalculator(RewardCalculator):
    """
    Reward calculator for multi-agent scenarios
    """
    
    def calculate_joint_reward(self, 
                             agent_rewards: List[RewardComponents],
                             collaboration_efficiency: float) -> List[float]:
        """
        Calculate joint rewards for multiple agents considering collaboration
        """
        individual_rewards = [comp.total_reward for comp in agent_rewards]
        
        # Base: average of individual rewards
        average_reward = np.mean(individual_rewards)
        
        # Collaboration efficiency multiplier
        collaboration_multiplier = 0.5 + (collaboration_efficiency * 0.5)
        
        # Calculate joint reward with collaboration bonus
        joint_reward = average_reward * collaboration_multiplier
        
        # Distribute rewards with individual adjustments
        distributed_rewards = []
        for i, individual_reward in enumerate(individual_rewards):
            # Agents who performed better than average get bonus
            performance_ratio = individual_reward / average_reward if average_reward > 0 else 1.0
            adjusted_reward = joint_reward * performance_ratio
            distributed_rewards.append(adjusted_reward)
        
        return distributed_rewards

# Global reward calculator instance
reward_calculator = RewardCalculator()