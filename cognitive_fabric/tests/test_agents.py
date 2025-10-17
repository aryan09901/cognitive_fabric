import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from agents.core.cognitive_node import CognitiveNode, AgentResponse
from agents.reinforcement.policies import PPOPolicy
from agents.multi_agent.orchestrator import MultiAgentOrchestrator

class TestCognitiveNode:
    """Test cases for CognitiveNode"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'LLM_MODEL': 'microsoft/DialoGPT-medium',  # Small model for testing
            'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
            'INITIAL_REPUTATION': 100,
            'MAX_AGENT_MEMORY': 1000,
            'RL_LEARNING_RATE': 0.001
        }
    
    @pytest.fixture
    @pytest.mark.asyncio
    async def cognitive_node(self, sample_config):
        """Create a CognitiveNode instance for testing"""
        with patch('blockchain.core.contracts.blockchain_client') as mock_bc:
            mock_bc.register_agent.return_value = "0x123"
            node = CognitiveNode("test_agent", sample_config)
            await asyncio.sleep(0.1)  # Allow async initialization
            return node
    
    @pytest.mark.asyncio
    async def test_initialization(self, cognitive_node):
        """Test agent initialization"""
        assert cognitive_node.agent_id == "test_agent"
        assert cognitive_node.reputation == 100
        assert cognitive_node.vector_db is not None
        assert cognitive_node.rl_policy is not None
    
    @pytest.mark.asyncio
    async def test_knowledge_retrieval(self, cognitive_node):
        """Test knowledge retrieval functionality"""
        # Mock vector DB response
        with patch.object(cognitive_node.vector_db, 'similarity_search') as mock_search:
            mock_search.return_value = [
                {
                    'content': 'Test knowledge content',
                    'metadata': {'source': 'test'},
                    'similarity_score': 0.9,
                    'verification_score': 0.8
                }
            ]
            
            context = await cognitive_node.retrieve_relevant_knowledge("test query")
            
            assert len(context) == 1
            assert context[0]['content'] == 'Test knowledge content'
            mock_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_response_generation(self, cognitive_node):
        """Test response generation pipeline"""
        # Mock LLM response
        with patch.object(cognitive_node.llm['generator'], '__call__') as mock_generate:
            mock_generate.return_value = [{'generated_text': 'ASSISTANT: This is a test response.'}]
            
            context = [{'content': 'Test context', 'metadata': {}, 'embedding': np.random.rand(384)}]
            action = torch.tensor([0.5])
            
            response = cognitive_node._generate_verified_response(
                "test query", context, action
            )
            
            assert isinstance(response, AgentResponse)
            assert response.response == 'This is a test response.'
            assert response.confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_rl_learning(self, cognitive_node):
        """Test reinforcement learning updates"""
        initial_loss = cognitive_node.rl_policy.update(
            torch.randn(10, 512),
            torch.randn(10)
        )
        
        # Loss should be a float
        assert isinstance(initial_loss, float)
        
        # Test that training buffer works
        cognitive_node.training_buffer = [
            {'state': torch.randn(512), 'reward': 1.0, 'response_quality': 0.8, 'verification_score': 0.9}
        ]
        cognitive_node.interaction_count = 100
        
        await cognitive_node._train_policy()
        
        # Buffer should be cleared after training
        assert len(cognitive_node.training_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, cognitive_node):
        """Test complete query processing pipeline"""
        with patch.object(cognitive_node, 'retrieve_relevant_knowledge') as mock_retrieve, \
             patch.object(cognitive_node, '_generate_verified_response') as mock_generate, \
             patch.object(cognitive_node, '_record_blockchain_interaction') as mock_record:
            
            mock_retrieve.return_value = [{'content': 'test context', 'metadata': {}}]
            mock_generate.return_value = AgentResponse(
                response="Test response",
                sources=[],
                confidence=0.8,
                verification_score=0.9,
                metadata={}
            )
            
            result = await cognitive_node.process_query("test query")
            
            assert isinstance(result, AgentResponse)
            assert result.response == "Test response"
            mock_retrieve.assert_called_once()
            mock_generate.assert_called_once()

class TestMultiAgentOrchestrator:
    """Test cases for MultiAgentOrchestrator"""
    
    @pytest.fixture
    @pytest.mark.asyncio
    async def orchestrator(self):
        """Create MultiAgentOrchestrator instance"""
        orchestrator = MultiAgentOrchestrator("0x123")
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator):
        """Test agent registration"""
        config = {'LLM_MODEL': 'test-model'}
        orchestrator.register_agent("agent1", config)
        
        assert "agent1" in orchestrator.agents
        assert orchestrator.agents["agent1"].agent_id == "agent1"
    
    @pytest.mark.asyncio
    async def test_query_routing(self, orchestrator):
        """Test query routing to appropriate agent"""
        # Register multiple agents
        for i in range(3):
            agent_id = f"agent_{i}"
            mock_agent = AsyncMock()
            mock_agent.agent_id = agent_id
            mock_agent.reputation = 80 + i * 10
            mock_agent.process_query.return_value = AgentResponse(
                response=f"Response from {agent_id}",
                sources=[],
                confidence=0.8,
                verification_score=0.9,
                metadata={}
            )
            orchestrator.agents[agent_id] = mock_agent
        
        result = await orchestrator.route_query("test query")
        
        # Should route to highest reputation agent
        assert "agent_2" in result.get('agents_used', [])
        orchestrator.agents["agent_2"].process_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collaborative_solving(self, orchestrator):
        """Test collaborative problem solving"""
        # Register multiple agents
        for i in range(3):
            agent_id = f"agent_{i}"
            mock_agent = AsyncMock()
            mock_agent.agent_id = agent_id
            mock_agent.process_query.return_value = {
                'response': f"Response from {agent_id}",
                'sources': [],
                'reputation': 80 + i * 10
            }
            orchestrator.agents[agent_id] = mock_agent
        
        result = await orchestrator.collaborative_solving("complex query")
        
        assert 'collaborative_response' in result
        assert result['agents_participated'] == 3
        # All agents should have been called
        for agent in orchestrator.agents.values():
            agent.process_query.assert_called_once()

class TestPPOPolicy:
    """Test cases for PPO policy"""
    
    @pytest.fixture
    def policy(self):
        return PPOPolicy(state_dim=10, action_dim=5)
    
    def test_forward_pass(self, policy):
        """Test policy forward pass"""
        state = torch.randn(1, 10)
        action_probs = policy(state)
        
        assert action_probs.shape == (1, 5)
        # Probabilities should sum to approximately 1
        assert torch.allclose(action_probs.sum(dim=-1), torch.tensor(1.0), atol=1e-6)
    
    def test_value_estimation(self, policy):
        """Test value function estimation"""
        state = torch.randn(1, 10)
        value = policy.get_value(state)
        
        assert value.shape == (1, 1)
    
    def test_policy_update(self, policy):
        """Test policy update with PPO"""
        states = torch.randn(32, 10)
        rewards = torch.randn(32)
        
        loss = policy.update(states, rewards)
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])