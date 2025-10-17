import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import uuid

from ..core.cognitive_node import CognitiveNode
from ..core.agent_registry import agent_registry
from .communication import CommunicationProtocol, communication_protocol
from .consensus import ConsensusMechanism, consensus_mechanism
from ..reinforcement.rewards import reward_calculator

logger = logging.getLogger(__name__)

@dataclass
class OrchestrationResult:
    response: str
    sources: List[Dict[str, Any]]
    verification_score: float
    confidence: float
    agents_used: List[str]
    agents_participated: int
    processing_time: float
    collaboration_mode: str
    metadata: Dict[str, Any]

class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent collaboration and coordination
    """
    
    def __init__(self, blockchain_address: Optional[str] = None):
        self.agents: Dict[str, CognitiveNode] = {}
        self.blockchain_address = blockchain_address
        self.communication = communication_protocol
        self.consensus = consensus_mechanism
        self.performance_metrics: Dict[str, Any] = {
            'total_queries': 0,
            'collaborative_queries': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0
        }
        
        # Initialize communication
        self._setup_communication_handlers()
    
    def _setup_communication_handlers(self):
        """Setup communication protocol handlers"""
        # Register the orchestrator itself for communication
        self.communication.register_agent("orchestrator")
        
        # Setup message handlers
        self.communication.register_message_handler(
            "orchestrator", 
            communication_protocol.MessageType.COLLABORATION_REQUEST,
            self._handle_collaboration_request
        )
    
    def register_agent(self, agent_id: str, config: Dict[str, Any]):
        """Register a new agent with the orchestrator"""
        try:
            # Create cognitive node
            agent_node = CognitiveNode(agent_id, config)
            self.agents[agent_id] = agent_node
            
            # Register with communication protocol
            self.communication.register_agent(agent_id)
            
            # Register with agent registry
            capabilities = config.get('capabilities', ['general'])
            agent_registry.register_agent(agent_id, config, capabilities)
            
            logger.info(f"Registered agent with orchestrator: {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def route_query(self, 
                         query: str, 
                         specific_agent: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Route query to appropriate agent(s) based on capabilities and load
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.performance_metrics['total_queries'] += 1
            
            # If specific agent requested, use that agent
            if specific_agent and specific_agent in self.agents:
                result = await self._single_agent_processing(
                    specific_agent, query, context
                )
            else:
                # Auto-route to best available agent
                result = await self._auto_route_query(query, context)
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            return OrchestrationResult(
                response=result['response'],
                sources=result.get('sources', []),
                verification_score=result.get('verification_score', 0.0),
                confidence=result.get('confidence', 0.5),
                agents_used=result.get('agents_used', [specific_agent] if specific_agent else ['auto_routed']),
                agents_participated=1,
                processing_time=processing_time,
                collaboration_mode='single_agent',
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_performance_metrics(processing_time, False)
            
            return self._create_error_result(f"Query processing failed: {str(e)}")
    
    async def _single_agent_processing(self, 
                                     agent_id: str, 
                                     query: str, 
                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process query with a single agent"""
        agent = self.agents[agent_id]
        
        try:
            result = await agent.process_query(query, context or {})
            
            # Update agent performance in registry
            agent_registry.update_agent_performance(
                agent_id,
                result.verification_score,
                getattr(result, 'processing_time', 0.0),
                True
            )
            
            return {
                'response': result.response,
                'sources': result.sources,
                'verification_score': result.verification_score,
                'confidence': result.confidence,
                'agents_used': [agent_id],
                'metadata': result.metadata
            }
            
        except Exception as e:
            logger.error(f"Single agent processing failed for {agent_id}: {e}")
            raise
    
    async def _auto_route_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Automatically route query to best available agent"""
        # Analyze query to determine required capabilities
        required_capabilities = self._analyze_query_requirements(query)
        
        # Find suitable agents
        suitable_agents = agent_registry.find_agents_by_capability(
            required_capabilities[0] if required_capabilities else 'general',
            min_reputation=50.0
        )
        
        if not suitable_agents:
            # Fallback to any available agent
            suitable_agents = list(self.agents.keys())
        
        if not suitable_agents:
            raise ValueError("No suitable agents available")
        
        # Select agent with highest reputation
        best_agent = None
        best_reputation = -1
        
        for agent_id in suitable_agents:
            agent_info = agent_registry.get_agent_status(agent_id)
            reputation = agent_info['reputation']
            if reputation > best_reputation:
                best_reputation = reputation
                best_agent = agent_id
        
        if not best_agent:
            best_agent = suitable_agents[0]
        
        return await self._single_agent_processing(best_agent, query, context)
    
    async def collaborative_solving(self, 
                                  query: str, 
                                  context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Solve complex queries using multiple agents collaboratively
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.performance_metrics['total_queries'] += 1
            self.performance_metrics['collaborative_queries'] += 1
            
            # Analyze query complexity
            complexity_analysis = self._analyze_query_complexity(query)
            
            # Select agents for collaboration
            selected_agents = await self._select_collaboration_agents(query, complexity_analysis)
            
            if len(selected_agents) <= 1:
                # Fall back to single agent if not enough suitable agents
                return await self.route_query(query, selected_agents[0] if selected_agents else None, context)
            
            # Execute collaborative solving
            collaboration_result = await self._execute_collaboration(
                query, selected_agents, context
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            return OrchestrationResult(
                response=collaboration_result['response'],
                sources=collaboration_result.get('sources', []),
                verification_score=collaboration_result.get('verification_score', 0.0),
                confidence=collaboration_result.get('confidence', 0.5),
                agents_used=selected_agents,
                agents_participated=len(selected_agents),
                processing_time=processing_time,
                collaboration_mode='multi_agent',
                metadata=collaboration_result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Collaborative solving failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_performance_metrics(processing_time, False)
            
            return self._create_error_result(f"Collaborative solving failed: {str(e)}")
    
    async def _select_collaboration_agents(self, 
                                         query: str, 
                                         complexity_analysis: Dict[str, Any]) -> List[str]:
        """Select agents for collaboration based on query requirements"""
        required_capabilities = complexity_analysis.get('required_capabilities', [])
        
        selected_agents = []
        
        for capability in required_capabilities[:3]:  # Limit to 3 capabilities
            capable_agents = agent_registry.find_agents_by_capability(
                capability, 
                min_reputation=30.0
            )
            
            for agent_id in capable_agents:
                if agent_id not in selected_agents and len(selected_agents) < 4:  # Max 4 agents
                    selected_agents.append(agent_id)
        
        # Ensure we have at least 2 agents for collaboration
        if len(selected_agents) < 2:
            # Add general capability agents
            general_agents = agent_registry.find_agents_by_capability('general', min_reputation=20.0)
            for agent_id in general_agents:
                if agent_id not in selected_agents and len(selected_agents) < 4:
                    selected_agents.append(agent_id)
        
        return selected_agents[:4]  # Return max 4 agents
    
    async def _execute_collaboration(self, 
                                   query: str, 
                                   agents: List[str], 
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute collaborative problem solving with multiple agents"""
        # Send query to all agents in parallel
        tasks = []
        for agent_id in agents:
            task = self._get_agent_response(agent_id, query, context)
            tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        valid_responses = []
        agent_reputations = {}
        
        for i, response in enumerate(responses):
            agent_id = agents[i]
            agent_info = agent_registry.get_agent_status(agent_id)
            agent_reputations[agent_id] = agent_info['reputation']
            
            if not isinstance(response, Exception):
                valid_responses.append({
                    'agent_id': agent_id,
                    'response': response['response'],
                    'confidence': response.get('confidence', 0.5),
                    'sources': response.get('sources', []),
                    'verification_score': response.get('verification_score', 0.0)
                })
                
                # Update agent performance
                agent_registry.update_agent_performance(
                    agent_id,
                    response.get('verification_score', 0.0),
                    response.get('processing_time', 0.0),
                    True
                )
        
        if not valid_responses:
            raise ValueError("No valid responses from collaborating agents")
        
        # Reach consensus on the best response
        consensus_result = await self.consensus.reach_consensus(
            valid_responses, 
            strategy='confidence_based',
            agent_reputations=agent_reputations
        )
        
        # Aggregate sources from all responses
        all_sources = []
        for response in valid_responses:
            all_sources.extend(response.get('sources', []))
        
        # Remove duplicate sources
        unique_sources = []
        seen_source_ids = set()
        
        for source in all_sources:
            source_id = hash(str(source))
            if source_id not in seen_source_ids:
                unique_sources.append(source)
                seen_source_ids.add(source_id)
        
        return {
            'response': consensus_result.final_decision,
            'sources': unique_sources[:10],  # Limit to 10 sources
            'verification_score': consensus_result.confidence,
            'confidence': consensus_result.agreement_level,
            'metadata': {
                'consensus_strategy': consensus_result.winning_strategy,
                'participating_agents': [resp['agent_id'] for resp in valid_responses],
                'consensus_quality': await self.consensus.evaluate_consensus_quality(consensus_result)
            }
        }
    
    async def _get_agent_response(self, 
                                agent_id: str, 
                                query: str, 
                                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get response from a specific agent"""
        try:
            agent = self.agents[agent_id]
            result = await agent.process_query(query, context or {})
            
            return {
                'response': result.response,
                'sources': result.sources,
                'confidence': result.confidence,
                'verification_score': result.verification_score,
                'processing_time': getattr(result, 'processing_time', 0.0)
            }
        except Exception as e:
            logger.error(f"Agent {agent_id} failed to process query: {e}")
            raise
    
    async def _handle_collaboration_request(self, message):
        """Handle collaboration request messages"""
        # This would process incoming collaboration requests from other agents
        # For now, return a simple acknowledgment
        return {
            'status': 'acknowledged',
            'orchestrator_id': 'main_orchestrator',
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def _analyze_query_requirements(self, query: str) -> List[str]:
        """Analyze query to determine required capabilities"""
        query_lower = query.lower()
        capabilities = []
        
        # Detect technical queries
        technical_terms = ['code', 'programming', 'algorithm', 'technical', 'debug', 'api']
        if any(term in query_lower for term in technical_terms):
            capabilities.append('technical')
        
        # Detect research queries
        research_terms = ['research', 'study', 'analysis', 'evidence', 'data']
        if any(term in query_lower for term in research_terms):
            capabilities.append('research')
        
        # Detect creative queries
        creative_terms = ['creative', 'idea', 'brainstorm', 'innovative', 'design']
        if any(term in query_lower for term in creative_terms):
            capabilities.append('creative')
        
        # Default capability
        if not capabilities:
            capabilities.append('general')
        
        return capabilities
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity"""
        word_count = len(query.split())
        sentence_count = query.count('.') + query.count('!') + query.count('?')
        
        # Simple complexity heuristic
        complexity_score = min(1.0, (word_count / 100) + (sentence_count / 10) * 0.1)
        
        required_capabilities = self._analyze_query_requirements(query)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'complexity_score': complexity_score,
            'required_capabilities': required_capabilities,
            'collaboration_recommended': complexity_score > 0.3
        }
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update orchestrator performance metrics"""
        total_queries = self.performance_metrics['total_queries']
        current_avg_time = self.performance_metrics['average_processing_time']
        
        # Update average processing time
        self.performance_metrics['average_processing_time'] = (
            (current_avg_time * (total_queries - 1)) + processing_time
        ) / total_queries
        
        # Update success rate
        if success:
            current_success_rate = self.performance_metrics['success_rate']
            self.performance_metrics['success_rate'] = (
                (current_success_rate * (total_queries - 1)) + 1
            ) / total_queries
    
    def _create_error_result(self, error_message: str) -> OrchestrationResult:
        """Create error result for failed queries"""
        return OrchestrationResult(
            response=error_message,
            sources=[],
            verification_score=0.0,
            confidence=0.0,
            agents_used=[],
            agents_participated=0,
            processing_time=0.0,
            collaboration_mode='error',
            metadata={'error': True, 'message': error_message}
        )
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics"""
        return {
            'total_agents': len(self.agents),
            'performance_metrics': self.performance_metrics,
            'blockchain_address': self.blockchain_address,
            'active_collaborations': 0,  # Would track active collaborations
            'system_health': 'healthy' if len(self.agents) > 0 else 'degraded'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                status = agent.get_status()
                health_results[agent_id] = {
                    'status': 'healthy',
                    'reputation': status.get('reputation', 0),
                    'interaction_count': status.get('interaction_count', 0),
                    'memory_usage': status.get('memory_size', 0)
                }
            except Exception as e:
                health_results[agent_id] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_results

# Global orchestrator instance
orchestrator = MultiAgentOrchestrator()