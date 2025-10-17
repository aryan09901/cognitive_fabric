"""
Multi-agent coordination
Orchestration, communication, and collaborative problem solving.
"""

from .orchestrator import MultiAgentOrchestrator
from .communication import CommunicationProtocol, CollaborationManager, communication_protocol
from .consensus import ConsensusMechanism

__all__ = [
    'MultiAgentOrchestrator',
    'CommunicationProtocol',
    'CollaborationManager',
    'communication_protocol',
    'ConsensusMechanism'
]