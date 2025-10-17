"""
Blockchain module for Cognitive Fabric
Smart contracts, interactions, and decentralized verification.
"""

from .core.contracts import BlockchainClient, blockchain_client
from .core.interactions import BlockchainInteractions, blockchain_interactions

__all__ = [
    'BlockchainClient',
    'blockchain_client',
    'BlockchainInteractions', 
    'blockchain_interactions'
]