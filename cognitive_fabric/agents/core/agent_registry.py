import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from .core.cognitive_node import CognitiveNode

logger = logging.getLogger(__name__)

@dataclass
class AgentInfo:
    agent_id: str
    node: CognitiveNode
    capabilities: List[str]
    reputation: float
    status: str  # 'active', 'training', 'inactive'
    last_activity: float

class AgentRegistry:
    """
    Registry for managing all cognitive agents in the system
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {}
        self.capability_index: Dict[str, List[str]] = {}  # capability -> agent_ids
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent_id: str, config: Dict[str, Any], capabilities: List[str]) -> CognitiveNode:
        """Register a new agent in the registry"""
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already registered")
        
        # Create cognitive node
        agent_node = CognitiveNode(agent_id, config)
        
        # Store agent info
        self.agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            node=agent_node,
            capabilities=capabilities,
            reputation=100.0,  # Initial reputation
            status='active',
            last_activity=asyncio.get_event_loop().time()
        )
        
        # Update capability index
        for capability in capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(agent_id)
        
        # Initialize performance metrics
        self.performance_metrics[agent_id] = {
            'total_queries': 0,
            'successful_responses': 0,
            'average_verification_score': 0.0,
            'average_processing_time': 0.0,
            'last_updated': asyncio.get_event_loop().time()
        }
        
        logger.info(f"Registered agent: {agent_id} with capabilities: {capabilities}")
        return agent_node
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the registry"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent_info = self.agents[agent_id]
        
        # Remove from capability index
        for capability in agent_info.capabilities:
            if capability in self.capability_index:
                if agent_id in self.capability_index[capability]:
                    self.capability_index[capability].remove(agent_id)
        
        # Clean up
        del self.agents[agent_id]
        del self.performance_metrics[agent_id]
        
        logger.info(f"Unregistered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[CognitiveNode]:
        """Get an agent by ID"""
        if agent_id in self.agents:
            return self.agents[agent_id].node
        return None
    
    def find_agents_by_capability(self, capability: str, min_reputation: float = 0.0) -> List[str]:
        """Find agents with specific capability and minimum reputation"""
        if capability not in self.capability_index:
            return []
        
        qualified_agents = []
        for agent_id in self.capability_index[capability]:
            if agent_id in self.agents:
                agent_info = self.agents[agent_id]
                if agent_info.reputation >= min_reputation and agent_info.status == 'active':
                    qualified_agents.append(agent_id)
        
        # Sort by reputation (highest first)
        qualified_agents.sort(key=lambda aid: self.agents[aid].reputation, reverse=True)
        return qualified_agents
    
    def update_agent_performance(self, agent_id: str, verification_score: float, processing_time: float, success: bool = True):
        """Update agent performance metrics"""
        if agent_id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[agent_id]
        metrics['total_queries'] += 1
        
        if success:
            metrics['successful_responses'] += 1
        
        # Update running averages
        current_avg_verification = metrics['average_verification_score']
        current_avg_time = metrics['average_processing_time']
        total = metrics['total_queries']
        
        metrics['average_verification_score'] = (
            (current_avg_verification * (total - 1)) + verification_score
        ) / total
        
        metrics['average_processing_time'] = (
            (current_avg_time * (total - 1)) + processing_time
        ) / total
        
        metrics['last_updated'] = asyncio.get_event_loop().time()
        
        # Update reputation based on performance
        self._update_agent_reputation(agent_id, verification_score, success)
    
    def _update_agent_reputation(self, agent_id: str, verification_score: float, success: bool):
        """Update agent reputation based on performance"""
        if agent_id not in self.agents:
            return
        
        agent_info = self.agents[agent_id]
        
        # Base reputation change
        reputation_change = 0
        
        if success:
            # Positive reinforcement for good verification scores
            if verification_score >= 0.8:
                reputation_change += 5
            elif verification_score >= 0.6:
                reputation_change += 2
            else:
                reputation_change -= 1
        else:
            # Penalty for failures
            reputation_change -= 3
        
        # Apply reputation change with bounds
        new_reputation = max(0, min(1000, agent_info.reputation + reputation_change))
        agent_info.reputation = new_reputation
        
        logger.debug(f"Updated reputation for {agent_id}: {reputation_change} (new: {new_reputation})")
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive status for an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent_info = self.agents[agent_id]
        metrics = self.performance_metrics[agent_id]
        
        return {
            'agent_id': agent_id,
            'status': agent_info.status,
            'reputation': agent_info.reputation,
            'capabilities': agent_info.capabilities,
            'last_activity': agent_info.last_activity,
            'performance_metrics': metrics,
            'node_info': agent_info.node.get_status()
        }
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get status for all agents"""
        return [self.get_agent_status(agent_id) for agent_id in self.agents.keys()]
    
    def set_agent_status(self, agent_id: str, status: str):
        """Set agent status (active, training, inactive)"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        valid_statuses = ['active', 'training', 'inactive']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        
        self.agents[agent_id].status = status
        self.agents[agent_id].last_activity = asyncio.get_event_loop().time()
        
        logger.info(f"Set agent {agent_id} status to: {status}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_results = {}
        current_time = asyncio.get_event_loop().time()
        
        for agent_id, agent_info in self.agents.items():
            # Check if agent is responsive
            try:
                node_status = agent_info.node.get_status()
                health_results[agent_id] = {
                    'status': 'healthy',
                    'reputation': agent_info.reputation,
                    'last_activity': agent_info.last_activity,
                    'inactive_duration': current_time - agent_info.last_activity,
                    'node_status': node_status
                }
            except Exception as e:
                health_results[agent_id] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'reputation': agent_info.reputation
                }
        
        return health_results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.status == 'active')
        
        total_queries = sum(metrics['total_queries'] for metrics in self.performance_metrics.values())
        successful_queries = sum(metrics['successful_responses'] for metrics in self.performance_metrics.values())
        
        avg_reputation = sum(agent.reputation for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'total_queries_processed': total_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'average_reputation': avg_reputation,
            'capabilities_available': list(self.capability_index.keys())
        }

# Global agent registry instance
agent_registry = AgentRegistry()