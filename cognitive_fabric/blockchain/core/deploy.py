import asyncio
import json
from web3 import Web3
from typing import Dict, Any
import os

from config.base import config

async def deploy_contracts() -> Dict[str, str]:
    """
    Deploy all smart contracts to blockchain
    Returns dictionary of contract addresses
    """
    print("🚀 Deploying Cognitive Fabric contracts...")
    
    w3 = Web3(Web3.HTTPProvider(config.BLOCKCHAIN_RPC_URL))
    
    # Load contract ABIs
    contracts_dir = os.path.join(os.path.dirname(__file__), '..', 'contracts')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    
    contract_addresses = {}
    
    try:
        # Deploy KnowledgeToken
        print("📦 Deploying KnowledgeToken...")
        with open(os.path.join(artifacts_dir, 'KnowledgeToken.json'), 'r') as f:
            token_artifact = json.load(f)
        
        KnowledgeToken = w3.eth.contract(
            abi=token_artifact['abi'],
            bytecode=token_artifact['bytecode']
        )
        
        token_tx_hash = KnowledgeToken.constructor().transact({
            'from': w3.eth.accounts[0],
            'gas': 2000000
        })
        token_receipt = w3.eth.wait_for_transaction_receipt(token_tx_hash)
        contract_addresses['KnowledgeToken'] = token_receipt.contractAddress
        
        # Deploy ReputationSystem
        print("🏆 Deploying ReputationSystem...")
        with open(os.path.join(artifacts_dir, 'ReputationSystem.json'), 'r') as f:
            reputation_artifact = json.load(f)
        
        ReputationSystem = w3.eth.contract(
            abi=reputation_artifact['abi'],
            bytecode=reputation_artifact['bytecode']
        )
        
        reputation_tx_hash = ReputationSystem.constructor().transact({
            'from': w3.eth.accounts[0],
            'gas': 2000000
        })
        reputation_receipt = w3.eth.wait_for_transaction_receipt(reputation_tx_hash)
        contract_addresses['ReputationSystem'] = reputation_receipt.contractAddress
        
        # Deploy CognitiveFabric
        print("🧠 Deploying CognitiveFabric...")
        with open(os.path.join(artifacts_dir, 'CognitiveFabric.json'), 'r') as f:
            fabric_artifact = json.load(f)
        
        CognitiveFabric = w3.eth.contract(
            abi=fabric_artifact['abi'],
            bytecode=fabric_artifact['bytecode']
        )
        
        fabric_tx_hash = CognitiveFabric.constructor().transact({
            'from': w3.eth.accounts[0],
            'gas': 3000000
        })
        fabric_receipt = w3.eth.wait_for_transaction_receipt(fabric_tx_hash)
        contract_addresses['CognitiveFabric'] = fabric_receipt.contractAddress
        
        print("✅ Contracts deployed successfully!")
        print(f"KnowledgeToken: {contract_addresses['KnowledgeToken']}")
        print(f"ReputationSystem: {contract_addresses['ReputationSystem']}")
        print(f"CognitiveFabric: {contract_addresses['CognitiveFabric']}")
        
        # Save deployment info
        deployment_info = {
            'network': config.BLOCKCHAIN_RPC_URL,
            'contracts': contract_addresses,
            'deployer': w3.eth.accounts[0],
            'timestamp': asyncio.get_event_loop().time()
        }
        
        with open(os.path.join(artifacts_dir, 'deployment.json'), 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        return contract_addresses
        
    except Exception as e:
        print(f"❌ Contract deployment failed: {e}")
        raise