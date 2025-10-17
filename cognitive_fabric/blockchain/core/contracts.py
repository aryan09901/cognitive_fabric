from web3 import Web3
from typing import Dict, Any, List, Optional
import json
import os
from eth_account import Account
from web3.middleware import geth_poa_middleware

from config.base import config

class BlockchainClient:
    """Blockchain client for interacting with Cognitive Fabric contracts"""
    
    def __init__(self, rpc_url: str, private_key: str, contract_address: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add POA middleware for testnets
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.private_key = private_key
        self.account = Account.from_key(private_key)
        self.contract_address = contract_address
        
        # Load contract ABI
        self.contract = self._load_contract()
    
    def _load_contract(self):
        """Load contract ABI and create contract instance"""
        try:
            # In production, you'd load from compiled artifact
            with open('blockchain/artifacts/CognitiveFabric.json', 'r') as f:
                artifact = json.load(f)
                abi = artifact['abi']
        except FileNotFoundError:
            # Fallback ABI for development
            abi = self._get_fallback_abi()
        
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=abi
        )
    
    def _get_fallback_abi(self) -> List[Dict]:
        """Fallback ABI for development"""
        return [
            {
                "inputs": [{"internalType": "string", "name": "_metadata", "type": "string"}],
                "name": "registerAgent",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            # ... other function definitions
        ]
    
    def register_agent(self, metadata_ipfs_hash: str) -> str:
        """Register a new agent on blockchain"""
        transaction = self.contract.functions.registerAgent(metadata_ipfs_hash).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': config.GAS_LIMIT,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        return self.w3.to_hex(tx_hash)
    
    def share_knowledge(self, knowledge_hash: str, metadata_hash: str) -> str:
        """Share knowledge to the network"""
        transaction = self.contract.functions.shareKnowledge(
            knowledge_hash, metadata_hash
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': config.GAS_LIMIT,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        return self.w3.to_hex(tx_hash)
    
    def get_agent_reputation(self, agent_address: str) -> int:
        """Get agent reputation score"""
        return self.contract.functions.getAgentReputation(
            Web3.to_checksum_address(agent_address)
        ).call()
    
    def record_interaction(
        self, 
        to_agent: str, 
        query_hash: str, 
        response_hash: str, 
        satisfaction_score: int
    ) -> str:
        """Record agent interaction on blockchain"""
        transaction = self.contract.functions.recordInteraction(
            Web3.to_checksum_address(to_agent),
            query_hash,
            response_hash,
            satisfaction_score
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': config.GAS_LIMIT,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        return self.w3.to_hex(tx_hash)
    
    def get_all_agents(self) -> List[str]:
        """Get all registered agent addresses"""
        return self.contract.functions.getAllAgents().call()

# Singleton instance
blockchain_client = BlockchainClient(
    rpc_url=config.BLOCKCHAIN_RPC_URL,
    private_key=config.PRIVATE_KEY,
    contract_address=config.CONTRACT_ADDRESS
)