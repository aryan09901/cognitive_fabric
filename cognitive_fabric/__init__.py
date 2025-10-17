"""
AGI-on-Chain: Decentralized Cognitive Fabric
A self-improving, privacy-preserving, multi-agent AI network with blockchain verification.
"""

__version__ = "1.0.0"
__author__ = "Cognitive Fabric Team"
__description__ = "Decentralized AI Network with Blockchain + RAG + RL + NLP"

from config.base import config, get_config

__all__ = [
    'config',
    'get_config',
    '__version__',
    '__author__',
    '__description__'
]