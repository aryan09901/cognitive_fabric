import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_config():
    """Temporary configuration for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            'LLM_MODEL': 'microsoft/DialoGPT-medium',
            'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
            'BLOCKCHAIN_RPC_URL': 'http://localhost:8545',
            'CONTRACT_ADDRESS': '0x' + '0' * 40,
            'PRIVATE_KEY': '0x' + '1' * 64,
            'VECTOR_DB_URL': 'localhost:8000',
            'IPFS_URL': 'localhost:5001',
            'MAX_AGENT_MEMORY': 1000,
            'RL_LEARNING_RATE': 0.001
        }
        json.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def mock_blockchain():
    """Mock blockchain interactions"""
    with patch('blockchain.core.interactions.BlockchainInteractions') as mock:
        mock_instance = Mock()
        mock_instance.register_agent = AsyncMock(return_value="0xmock_tx_hash")
        mock_instance.share_knowledge = AsyncMock(return_value="0xmock_knowledge_tx")
        mock_instance.record_interaction = AsyncMock(return_value="0xmock_interaction_tx")
        mock_instance.get_agent_reputation = AsyncMock(return_value={
            'score': 100,
            'total_interactions': 10,
            'positive_interactions': 8,
            'knowledge_contributions': 5,
            'trust_score': 80
        })
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_vector_db():
    """Mock vector database"""
    with patch('knowledge.core.vector_db.VectorDatabase') as mock:
        mock_instance = Mock()
        mock_instance.add_knowledge = AsyncMock(return_value="mock_doc_id")
        mock_instance.similarity_search = AsyncMock(return_value=[
            type('SearchResult', (), {
                'content': 'Test knowledge content',
                'metadata': {'source': 'test', 'verification_score': 0.8},
                'similarity_score': 0.9,
                'id': 'doc1'
            })()
        ])
        mock_instance.get_collection_stats = AsyncMock(return_value={
            'total_documents': 100,
            'average_verification_score': 0.75,
            'sources_count': 10
        })
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_llm():
    """Mock language model"""
    with patch('nlp.core.language_models.AdvancedLanguageModel') as mock:
        mock_instance = Mock()
        mock_instance.generate_verified_response = AsyncMock(return_value={
            'response': 'This is a test response from the AI agent.',
            'verification_score': 0.85,
            'confidence': 0.9,
            'sources_used': [{'content': 'Test source', 'metadata': {}}],
            'generation_metadata': {'model': 'test-model'}
        })
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing"""
    return {
        'agent_id': 'test_agent',
        'config': {
            'LLM_MODEL': 'microsoft/DialoGPT-medium',
            'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
            'INITIAL_REPUTATION': 100,
            'MAX_AGENT_MEMORY': 1000
        },
        'capabilities': ['research', 'analysis', 'verification']
    }

@pytest.fixture
def sample_knowledge_item():
    """Sample knowledge item for testing"""
    return {
        'content': 'This is a sample knowledge item for testing purposes.',
        'metadata': {
            'source': 'test_source',
            'category': 'testing',
            'verification_score': 0.8
        },
        'verification_score': 0.8,
        'source': 'test'
    }

@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return {
        'query': 'What are the benefits of decentralized AI systems?',
        'agent_id': 'test_agent',
        'collaborative': False,
        'verification_level': 'moderate',
        'context': {'domain': 'ai_research'}
    }

@pytest.fixture
def sample_agent_response():
    """Sample agent response for testing"""
    return {
        'response': 'Decentralized AI systems offer benefits like improved transparency, reduced single points of failure, and enhanced privacy through distributed computation.',
        'sources': [
            {
                'content': 'Research paper on decentralized AI benefits',
                'metadata': {'source': 'academic', 'year': 2023}
            }
        ],
        'verification_score': 0.88,
        'confidence': 0.92,
        'processing_time': 2.1,
        'agents_used': ['test_agent'],
        'metadata': {
            'model': 'test-model',
            'context_utilization': 3
        }
    }

@pytest.fixture
def mock_ipfs_client():
    """Mock IPFS client"""
    with patch('knowledge.storage.ipfs_client.IPFSClient') as mock:
        mock_instance = Mock()
        mock_instance.upload_json = AsyncMock(return_value="QmMockIPFSHash")
        mock_instance.download_json = AsyncMock(return_value={
            'content': 'Test content',
            'metadata': {'source': 'test'},
            'timestamp': 1234567890
        })
        mock_instance.check_connectivity = AsyncMock(return_value=True)
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_communication_protocol():
    """Mock communication protocol"""
    with patch('agents.multi_agent.communication.CommunicationProtocol') as mock:
        mock_instance = Mock()
        mock_instance.send_message = AsyncMock()
        mock_instance.receive_message = AsyncMock(return_value=None)
        mock_instance.register_agent = Mock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_training_data():
    """Sample training data for RL"""
    return {
        'states': [
            {'state_vector': [0.1, 0.2, 0.3], 'reputation': [85, 90, 78]},
            {'state_vector': [0.4, 0.5, 0.6], 'reputation': [88, 92, 80]}
        ],
        'actions': [1, 2],
        'rewards': [1.0, 0.5],
        'next_states': [
            {'state_vector': [0.2, 0.3, 0.4], 'reputation': [86, 91, 79]},
            {'state_vector': [0.5, 0.6, 0.7], 'reputation': [89, 93, 81]}
        ]
    }

@pytest.fixture(autouse=True)
def mock_external_services():
    """Automatically mock external services for all tests"""
    with patch('web3.Web3') as mock_web3, \
         patch('chromadb.HttpClient') as mock_chroma, \
         patch('ipfshttpclient.connect') as mock_ipfs, \
         patch('sentence_transformers.SentenceTransformer') as mock_embedding:
        
        # Mock Web3
        mock_web3_instance = Mock()
        mock_web3_instance.eth.get_transaction_count = Mock(return_value=0)
        mock_web3_instance.eth.gas_price = Mock(return_value=1000000000)
        mock_web3_instance.eth.send_raw_transaction = Mock(return_value=b'mock_tx')
        mock_web3_instance.to_hex = Mock(return_value="0xmock")
        mock_web3.return_value = mock_web3_instance
        
        # Mock ChromaDB
        mock_chroma_instance = Mock()
        mock_chroma_instance.get_or_create_collection = Mock(return_value=Mock())
        mock_chroma.return_value = mock_chroma_instance
        
        # Mock IPFS
        mock_ipfs_instance = Mock()
        mock_ipfs.return_value = mock_ipfs_instance
        
        # Mock Embedding model
        mock_embedding_instance = Mock()
        mock_embedding_instance.encode = Mock(return_value=[[0.1] * 384])
        mock_embedding.return_value = mock_embedding_instance
        
        yield