import asyncio
import json
from typing import Dict, List, Optional, Any
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import logging

from config.base import config

logger = logging.getLogger(__name__)

class BlockchainInteractions:
    """
    Advanced blockchain interaction manager for Cognitive Fabric
    """
    
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(config.BLOCKCHAIN_RPC_URL))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.account = Account.from_key(config.PRIVATE_KEY)
        self.contracts = self._load_contracts()
        
        logger.info(f"Blockchain interactions initialized for {self.account.address}")
    
    def _load_contracts(self) -> Dict[str, Any]:
        """Load all smart contracts"""
        try:
            # Load contract ABIs
            with open('blockchain/artifacts/deployment.json', 'r') as f:
                deployment = json.load(f)
            
            contracts = {}
            
            # CognitiveFabric contract
            with open('blockchain/artifacts/CognitiveFabric.json', 'r') as f:
                fabric_abi = json.load(f)['abi']
            contracts['fabric'] = self.w3.eth.contract(
                address=Web3.to_checksum_address(deployment['contracts']['CognitiveFabric']),
                abi=fabric_abi
            )
            
            # KnowledgeToken contract
            with open('blockchain/artifacts/KnowledgeToken.json', 'r') as f:
                token_abi = json.load(f)['abi']
            contracts['token'] = self.w3.eth.contract(
                address=Web3.to_checksum_address(deployment['contracts']['KnowledgeToken']),
                abi=token_abi
            )
            
            # ReputationSystem contract
            with open('blockchain/artifacts/ReputationSystem.json', 'r') as f:
                reputation_abi = json.load(f)['abi']
            contracts['reputation'] = self.w3.eth.contract(
                address=Web3.to_checksum_address(deployment['contracts']['ReputationSystem']),
                abi=reputation_abi
            )
            
            return contracts
            
        except Exception as e:
            logger.error(f"Failed to load contracts: {e}")
            # Return mock contracts for development
            return self._create_mock_contracts()
    
    def _create_mock_contracts(self) -> Dict[str, Any]:
        """Create mock contracts for development"""
        logger.warning("Using mock contracts for development")
        
        class MockContract:
            def __init__(self, name):
                self.name = name
            
            def functions(self):
                return self
            
            def call(self):
                return None
            
            def build_transaction(self, *args, **kwargs):
                return {}
        
        return {
            'fabric': MockContract('CognitiveFabric'),
            'token': MockContract('KnowledgeToken'),
            'reputation': MockContract('ReputationSystem')
        }
    
    async def register_agent(self, agent_id: str, metadata_ipfs_hash: str) -> str:
        """Register a new agent on blockchain"""
        try:
            transaction = self.contracts['fabric'].functions.registerAgent(
                metadata_ipfs_hash
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': config.GAS_LIMIT,
                'gasPrice': self.w3.eth.gas_price
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, config.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Agent {agent_id} registered in block {receipt.blockNumber}")
            return self.w3.to_hex(tx_hash)
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return f"mock_tx_{agent_id}"
    
    async def share_knowledge(self, knowledge_hash: str, metadata_hash: str) -> str:
        """Share knowledge to the network"""
        try:
            transaction = self.contracts['fabric'].functions.shareKnowledge(
                knowledge_hash, metadata_hash
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': config.GAS_LIMIT,
                'gasPrice': self.w3.eth.gas_price
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, config.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Distribute rewards
            await self._distribute_rewards(knowledge_shared=True)
            
            logger.info(f"Knowledge shared in block {receipt.blockNumber}")
            return self.w3.to_hex(tx_hash)
            
        except Exception as e:
            logger.error(f"Failed to share knowledge: {e}")
            return f"mock_tx_knowledge_{knowledge_hash[:10]}"
    
    async def record_interaction(
        self, 
        from_agent: str, 
        to_agent: str, 
        query_hash: str, 
        response_hash: str, 
        satisfaction_score: int
    ) -> str:
        """Record agent interaction on blockchain"""
        try:
            transaction = self.contracts['fabric'].functions.recordInteraction(
                Web3.to_checksum_address(to_agent),
                query_hash,
                response_hash,
                satisfaction_score
            ).build_transaction({
                'from': Web3.to_checksum_address(from_agent),
                'nonce': self.w3.eth.get_transaction_count(Web3.to_checksum_address(from_agent)),
                'gas': config.GAS_LIMIT,
                'gasPrice': self.w3.eth.gas_price
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, config.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Update reputation
            await self._update_reputation(to_agent, satisfaction_score, knowledge_shared=False)
            
            # Distribute rewards for high satisfaction
            if satisfaction_score >= 80:
                await self._distribute_rewards(high_satisfaction=True)
            
            logger.info(f"Interaction recorded in block {receipt.blockNumber}")
            return self.w3.to_hex(tx_hash)
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            return f"mock_tx_interaction_{from_agent[:10]}"
    
    async def _distribute_rewards(
        self, 
        knowledge_shared: bool = False,
        verification_done: bool = False, 
        high_satisfaction: bool = False
    ):
        """Distribute token rewards for activities"""
        try:
            transaction = self.contracts['token'].functions.distributeRewards(
                self.account.address,
                knowledge_shared,
                verification_done,
                high_satisfaction
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': config.GAS_LIMIT,
                'gasPrice': self.w3.eth.gas_price
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, config.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info("Rewards distributed successfully")
            
        except Exception as e:
            logger.error(f"Failed to distribute rewards: {e}")
    
    async def _update_reputation(
        self, 
        agent: str, 
        satisfaction_score: int, 
        knowledge_shared: bool
    ):
        """Update agent reputation"""
        try:
            transaction = self.contracts['reputation'].functions.updateReputation(
                Web3.to_checksum_address(agent),
                satisfaction_score,
                knowledge_shared
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': config.GAS_LIMIT,
                'gasPrice': self.w3.eth.gas_price
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, config.PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info(f"Reputation updated for agent {agent}")
            
        except Exception as e:
            logger.error(f"Failed to update reputation: {e}")
    
    async def get_agent_reputation(self, agent_address: str) -> Dict[str, Any]:
        """Get comprehensive reputation data for an agent"""
        try:
            reputation_data = self.contracts['reputation'].functions.getReputationData(
                Web3.to_checksum_address(agent_address)
            ).call()
            
            trust_score = self.contracts['reputation'].functions.calculateTrustScore(
                Web3.to_checksum_address(agent_address)
            ).call()
            
            return {
                'score': reputation_data[0],
                'total_interactions': reputation_data[1],
                'positive_interactions': reputation_data[2],
                'knowledge_contributions': reputation_data[3],
                'last_update': reputation_data[4],
                'streak': reputation_data[5],
                'trust_score': trust_score
            }
            
        except Exception as e:
            logger.error(f"Failed to get reputation: {e}")
            return {
                'score': 100,
                'total_interactions': 0,
                'positive_interactions': 0,
                'knowledge_contributions': 0,
                'last_update': 0,
                'streak': 0,
                'trust_score': 50
            }
    
    async def get_all_agents(self) -> List[str]:
        """Get all registered agent addresses"""
        try:
            return self.contracts['fabric'].functions.getAllAgents().call()
        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            return []
    
    async def get_knowledge_item(self, knowledge_hash: str) -> Dict[str, Any]:
        """Get knowledge item from blockchain"""
        try:
            item = self.contracts['fabric'].functions.getKnowledgeItem(knowledge_hash).call()
            
            return {
                'contributor': item[0],
                'content_hash': item[1],
                'metadata': item[2],
                'timestamp': item[3],
                'verification_score': item[4],
                'usefulness_score': item[5],
                'verified': item[6]
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge item: {e}")
            return {}
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get overall network statistics"""
        try:
            agents = await self.get_all_agents()
            total_reputation = 0
            active_agents = 0
            
            for agent in agents:
                reputation = await self.get_agent_reputation(agent)
                total_reputation += reputation['score']
                if reputation['total_interactions'] > 0:
                    active_agents += 1
            
            return {
                'total_agents': len(agents),
                'active_agents': active_agents,
                'average_reputation': total_reputation / len(agents) if agents else 0,
                'network_health': min(100, total_reputation / 10)  # Simple health metric
            }
            
        except Exception as e:
            logger.error(f"Failed to get network stats: {e}")
            return {
                'total_agents': 0,
                'active_agents': 0,
                'average_reputation': 0,
                'network_health': 0
            }

# Global instance
blockchain_interactions = BlockchainInteractions()