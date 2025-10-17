"""
Knowledge management module
Vector databases, storage, and knowledge processing.
"""

from .core.vector_db import VectorDatabase, vector_db, SearchResult
from .core.knowledge_graph import KnowledgeGraph
from .storage.ipfs_client import IPFSClient, ipfs_client
from .storage.arweave_client import ArweaveClient, arweave_client
from .storage.file_processor import FileProcessor, file_processor

__all__ = [
    'VectorDatabase',
    'vector_db',
    'SearchResult',
    'KnowledgeGraph',
    'IPFSClient',
    'ipfs_client',
    'ArweaveClient',
    'arweave_client',
    'FileProcessor',
    'file_processor'
]