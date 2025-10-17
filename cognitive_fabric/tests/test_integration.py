import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.main import app
from fastapi.testclient import TestClient

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "blockchain" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should return some metrics data
    
    @pytest.mark.asyncio
    async def test_query_processing_integration(self, client):
        """Test complete query processing flow"""
        with patch('api.routes.queries.orchestrator') as mock_orchestrator:
            mock_orchestrator.route_query = AsyncMock()
            mock_orchestrator.route_query.return_value = {
                'response': 'Test response from integrated system',
                'sources': [{'content': 'test source', 'metadata': {}}],
                'verification_score': 0.85,
                'confidence': 0.9,
                'agents_used': ['test_agent'],
                'metadata': {'processing_mode': 'single_agent'}
            }
            
            query_data = {
                "query": "Test query for integration testing",
                "collaborative": False,
                "verification_level": "moderate"
            }
            
            response = client.post("/api/v1/queries/process", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data['response'] == 'Test response from integrated system'
            assert data['verification_score'] == 0.85
            assert 'test_agent' in data['agents_used']
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, client):
        """Test batch query processing"""
        with patch('api.routes.queries.orchestrator') as mock_orchestrator:
            mock_orchestrator.route_query = AsyncMock()
            mock_orchestrator.route_query.return_value = {
                'response': 'Batch test response',
                'sources': [],
                'verification_score': 0.8,
                'confidence': 0.85,
                'agents_used': ['batch_agent'],
                'metadata': {}
            }
            
            batch_data = {
                "queries": [
                    {"text": "First batch query"},
                    {"text": "Second batch query"}
                ],
                "verification_level": "moderate"
            }
            
            response = client.post("/api/v1/queries/batch", json=batch_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data['total_processed'] == 2
            assert len(data['successful']) == 2
            assert data['total_failed'] == 0
    
    @pytest.mark.asyncio
    async def test_agent_management_integration(self, client):
        """Test agent registration and management"""
        agent_data = {
            "agent_id": "integration_test_agent",
            "config": {
                "LLM_MODEL": "test-model",
                "capabilities": ["reasoning", "verification"]
            },
            "capabilities": ["reasoning", "verification"],
            "metadata": {"test": True}
        }
        
        response = client.post("/api/v1/agents/register", json=agent_data)
        
        # Should either succeed or give appropriate error
        assert response.status_code in [200, 400, 500]
        
        # Test agent listing
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
    
    def test_knowledge_management_integration(self, client):
        """Test knowledge management endpoints"""
        # Test knowledge addition
        knowledge_data = {
            "content": "Integration test knowledge content",
            "metadata": {
                "source": "integration_test",
                "category": "testing"
            },
            "verification_score": 0.9,
            "source": "test_suite"
        }
        
        response = client.post("/api/v1/knowledge/add", json=knowledge_data)
        assert response.status_code in [200, 400, 500]
        
        # Test knowledge search
        search_data = {
            "query": "integration test",
            "top_k": 5
        }
        
        response = client.post("/api/v1/knowledge/search", json=search_data)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, client):
        """Test error handling across the system"""
        # Test malformed request
        malformed_data = {
            "invalid_field": "should cause error"
        }
        
        response = client.post("/api/v1/queries/process", json=malformed_data)
        assert response.status_code == 422  # Validation error
        
        # Test with non-existent agent
        query_data = {
            "query": "Test query",
            "agent_id": "non_existent_agent",
            "collaborative": False
        }
        
        response = client.post("/api/v1/queries/process", json=query_data)
        # Should handle gracefully, either route to available agent or error
        assert response.status_code in [200, 404, 500]

class TestEndToEnd:
    """End-to-end tests simulating real usage scenarios"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_research_assistant_scenario(self, client):
        """Test research assistant use case"""
        research_queries = [
            "What are the latest advancements in quantum computing?",
            "How do transformer models work in natural language processing?",
            "What are the ethical implications of decentralized AI systems?"
        ]
        
        with patch('api.routes.queries.orchestrator') as mock_orchestrator:
            mock_orchestrator.collaborative_solving = AsyncMock()
            mock_orchestrator.collaborative_solving.return_value = {
                'response': 'Comprehensive research response combining multiple knowledge sources with proper verification and citations.',
                'sources': [
                    {
                        'content': 'Recent quantum computing breakthrough...',
                        'metadata': {'source': 'arxiv', 'year': 2024}
                    },
                    {
                        'content': 'Transformer architecture explanation...',
                        'metadata': {'source': 'research_paper', 'citations': 1250}
                    }
                ],
                'verification_score': 0.88,
                'confidence': 0.92,
                'agents_participated': 3,
                'metadata': {
                    'research_depth': 'comprehensive',
                    'sources_verified': True,
                    'cross_referenced': True
                }
            }
            
            for query in research_queries:
                query_data = {
                    "query": query,
                    "collaborative": True,
                    "verification_level": "strict"
                }
                
                response = client.post("/api/v1/queries/process", json=query_data)
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify research-quality response
                assert len(data['response']) > 50  # Substantial response
                assert data['verification_score'] > 0.7
                assert data['confidence'] > 0.8
                assert len(data['sources']) > 0
    
    @pytest.mark.asyncio
    async def test_verification_focused_scenario(self, client):
        """Test scenarios requiring high verification"""
        critical_queries = [
            "What is the current COVID-19 vaccination efficacy?",
            "What are the verified climate change impacts?",
            "Provide medically accurate information about diabetes treatment."
        ]
        
        with patch('api.routes.queries.orchestrator') as mock_orchestrator:
            mock_orchestrator.route_query = AsyncMock()
            mock_orchestrator.route_query.return_value = {
                'response': 'Verified medical information from peer-reviewed sources...',
                'sources': [
                    {
                        'content': 'Clinical study results...',
                        'metadata': {
                            'source': 'WHO',
                            'verification_score': 0.95,
                            'peer_reviewed': True
                        }
                    }
                ],
                'verification_score': 0.93,
                'confidence': 0.89,
                'agents_used': ['medical_expert_agent'],
                'metadata': {
                    'medically_verified': True,
                    'sources_cited': True,
                    'timestamp': '2024-01-01'
                }
            }
            
            for query in critical_queries:
                query_data = {
                    "query": query,
                    "verification_level": "strict",
                    "context": {"domain": "medical"}
                }
                
                response = client.post("/api/v1/queries/process", json=query_data)
                
                assert response.status_code == 200
                data = response.json()
                
                # High verification requirements
                assert data['verification_score'] >= 0.9
                assert all('source' in src for src in data['sources'])
                assert 'medical_expert_agent' in data['agents_used']

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])