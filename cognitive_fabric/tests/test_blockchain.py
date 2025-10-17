import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from blockchain.core.interactions import BlockchainInteractions
from blockchain.core.contracts import BlockchainClient

class TestBlockchainInteractions:
    """Test cases for blockchain interactions"""
    
    @pytest.fixture
    def blockchain_interactions(self):
        with patch('web3.Web3') as mock_web3:
            with patch('blockchain.core.interactions.config') as mock_config:
                mock_config.BLOCKCHAIN_RPC_URL = "http://localhost:8545"
                mock_config.PRIVATE_KEY = "test_private_key"
                mock_config.CONTRACT_ADDRESS = "0xTestAddress"
                mock_config.GAS_LIMIT = 500000
                
                interactions = BlockchainInteractions()
                interactions.w3 = mock_web3
                interactions.contracts = {
                    'fabric': Mock(),
                    'token': Mock(),
                    'reputation': Mock()
                }
                return interactions
    
    @pytest.mark.asyncio
    async def test_register_agent(self, blockchain_interactions):
        """Test agent registration on blockchain"""
        with patch.object(blockchain_interactions.w3.eth, 'get_transaction_count', return_value=0):
            with patch.object(blockchain_interactions.w3.eth, 'gas_price', return_value=1000000000):
                with patch.object(blockchain_interactions.w3.eth, 'send_raw_transaction', return_value=b'tx_hash'):
                    with patch.object(blockchain_interactions.w3.eth, 'wait_for_transaction_receipt', return_value=Mock(blockNumber=123)):
                        
                        tx_hash = await blockchain_interactions.register_agent("test_agent", "QmMetadata")
                        assert tx_hash == "0x74785f68617368"
    
    @pytest.mark.asyncio
    async def test_share_knowledge(self, blockchain_interactions):
        """Test knowledge sharing on blockchain"""
        blockchain_interactions.contracts['fabric'].functions.shareKnowledge = Mock()
        blockchain_interactions.contracts['fabric'].functions.shareKnowledge.return_value.build_transaction = Mock(return_value={})
        
        with patch.object(blockchain_interactions, '_distribute_rewards', new_callable=AsyncMock):
            tx_hash = await blockchain_interactions.share_knowledge("QmKnowledge", "QmMetadata")
            assert tx_hash.startswith("mock_tx_knowledge_")
    
    @pytest.mark.asyncio
    async def test_record_interaction(self, blockchain_interactions):
        """Test interaction recording on blockchain"""
        blockchain_interactions.contracts['fabric'].functions.recordInteraction = Mock()
        blockchain_interactions.contracts['fabric'].functions.recordInteraction.return_value.build_transaction = Mock(return_value={})
        
        with patch.object(blockchain_interactions, '_update_reputation', new_callable=AsyncMock):
            with patch.object(blockchain_interactions, '_distribute_rewards', new_callable=AsyncMock):
                tx_hash = await blockchain_interactions.record_interaction(
                    "0xFrom", "0xTo", "QmQuery", "QmResponse", 85
                )
                assert tx_hash.startswith("mock_tx_interaction_")
    
    @pytest.mark.asyncio
    async def test_get_agent_reputation(self, blockchain_interactions):
        """Test reputation retrieval"""
        mock_reputation_data = (100, 10, 8, 5, 1234567890, 3)
        blockchain_interactions.contracts['reputation'].functions.getReputationData.return_value.call = Mock(return_value=mock_reputation_data)
        blockchain_interactions.contracts['reputation'].functions.calculateTrustScore.return_value.call = Mock(return_value=80)
        
        reputation = await blockchain_interactions.get_agent_reputation("0xTestAgent")
        
        assert reputation['score'] == 100
        assert reputation['trust_score'] == 80
        assert reputation['total_interactions'] == 10
    
    @pytest.mark.asyncio
    async def test_get_all_agents(self, blockchain_interactions):
        """Test getting all registered agents"""
        mock_agents = ["0xAgent1", "0xAgent2", "0xAgent3"]
        blockchain_interactions.contracts['fabric'].functions.getAllAgents.return_value.call = Mock(return_value=mock_agents)
        
        agents = await blockchain_interactions.get_all_agents()
        assert len(agents) == 3
        assert "0xAgent1" in agents
    
    @pytest.mark.asyncio
    async def test_get_network_stats(self, blockchain_interactions):
        """Test network statistics retrieval"""
        with patch.object(blockchain_interactions, 'get_all_agents', return_value=["0x1", "0x2"]):
            with patch.object(blockchain_interactions, 'get_agent_reputation', return_value={
                'score': 100, 'total_interactions': 5
            }):
                stats = await blockchain_interactions.get_network_stats()
                
                assert stats['total_agents'] == 2
                assert stats['average_reputation'] == 100

class TestBlockchainClient:
    """Test cases for BlockchainClient"""
    
    @pytest.fixture
    def blockchain_client(self):
        with patch('web3.Web3') as mock_web3:
            with patch('blockchain.core.contracts.config') as mock_config:
                mock_config.GAS_LIMIT = 500000
                
                client = BlockchainClient(
                    rpc_url="http://localhost:8545",
                    private_key="test_key",
                    contract_address="0xTest"
                )
                client.w3 = mock_web3
                client.contract = Mock()
                return client
    
    def test_register_agent(self, blockchain_client):
        """Test agent registration"""
        blockchain_client.contract.functions.registerAgent = Mock()
        blockchain_client.contract.functions.registerAgent.return_value.build_transaction = Mock(return_value={})
        blockchain_client.w3.eth.get_transaction_count = Mock(return_value=0)
        blockchain_client.w3.eth.gas_price = Mock(return_value=1000000000)
        blockchain_client.w3.eth.send_raw_transaction = Mock(return_value=b'tx_hash')
        blockchain_client.w3.to_hex = Mock(return_value="0x123")
        
        tx_hash = blockchain_client.register_agent("QmMetadata")
        assert tx_hash == "0x123"
    
    def test_share_knowledge(self, blockchain_client):
        """Test knowledge sharing"""
        blockchain_client.contract.functions.shareKnowledge = Mock()
        blockchain_client.contract.functions.shareKnowledge.return_value.build_transaction = Mock(return_value={})
        blockchain_client.w3.eth.get_transaction_count = Mock(return_value=1)
        blockchain_client.w3.eth.gas_price = Mock(return_value=1000000000)
        blockchain_client.w3.eth.send_raw_transaction = Mock(return_value=b'tx_hash')
        blockchain_client.w3.to_hex = Mock(return_value="0x456")
        
        tx_hash = blockchain_client.share_knowledge("QmKnowledge", "QmMetadata")
        assert tx_hash == "0x456"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])