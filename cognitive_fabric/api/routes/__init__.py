"""
API routes module
All endpoint routers for the Cognitive Fabric API.
"""

from .agents import router as agents_router
from .knowledge import router as knowledge_router
from .blockchain import router as blockchain_router
from .queries import router as queries_router

__all__ = [
    'agents_router',
    'knowledge_router',
    'blockchain_router',
    'queries_router'
]