"""
API module
REST API, routes, and web interface for Cognitive Fabric.
"""

from .main import app
from .routes.agents import router as agents_router
from .routes.knowledge import router as knowledge_router
from .routes.blockchain import router as blockchain_router
from .routes.queries import router as queries_router

__all__ = [
    'app',
    'agents_router',
    'knowledge_router',
    'blockchain_router',
    'queries_router'
]