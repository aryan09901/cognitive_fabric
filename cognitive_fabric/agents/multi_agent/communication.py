import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import uuid
from enum import Enum

class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    COLLABORATION_REQUEST = "collaboration_request"
    KNOWLEDGE_SHARE = "knowledge_share"
    STATUS_UPDATE = "status_update"
    ERROR = "error"

@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1
    requires_response: bool = False
    response_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'priority': self.priority,
            'requires_response': self.requires_response,
            'response_to': self.response_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            message_type=MessageType(data['message_type']),
            content=data['content'],
            timestamp=data['timestamp'],
            priority=data.get('priority', 1),
            requires_response=data.get('requires_response', False),
            response_to=data.get('response_to')
        )

class CommunicationProtocol:
    """
    Communication protocol for multi-agent messaging
    """
    
    def __init__(self):
        self.message_queues = {}  # agent_id -> asyncio.Queue
        self.message_handlers = {}
        self.pending_responses = {}  # message_id -> asyncio.Future
        
    def register_agent(self, agent_id: str):
        """Register an agent for communication"""
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = asyncio.Queue()
            self.message_handlers[agent_id] = {}
    
    async def send_message(self, message: AgentMessage) -> Optional[Any]:
        """Send a message to an agent"""
        if message.receiver_id not in self.message_queues:
            raise ValueError(f"Agent {message.receiver_id} not registered")
        
        # Add to receiver's queue
        await self.message_queues[message.receiver_id].put(message)
        
        # If response required, create future
        if message.requires_response:
            future = asyncio.Future()
            self.pending_responses[message.message_id] = future
            return await future
        
        return None
    
    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive a message for an agent"""
        if agent_id not in self.message_queues:
            raise ValueError(f"Agent {agent_id} not registered")
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queues[agent_id].get(), 
                    timeout=timeout
                )
            else:
                message = await self.message_queues[agent_id].get()
            
            return message
            
        except asyncio.TimeoutError:
            return None
    
    async def send_response(self, original_message: AgentMessage, response_content: Dict[str, Any]):
        """Send a response to a message"""
        response_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=original_message.receiver_id,
            receiver_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            content=response_content,
            timestamp=asyncio.get_event_loop().time(),
            response_to=original_message.message_id
        )
        
        # Send the response
        await self.send_message(response_message)
        
        # Complete the pending future if it exists
        if original_message.message_id in self.pending_responses:
            self.pending_responses[original_message.message_id].set_result(response_content)
            del self.pending_responses[original_message.message_id]
    
    def register_message_handler(self, agent_id: str, message_type: MessageType, handler):
        """Register a handler for specific message types"""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}
        
        self.message_handlers[agent_id][message_type] = handler
    
    async def process_incoming_messages(self, agent_id: str):
        """Process incoming messages for an agent"""
        while True:
            try:
                message = await self.receive_message(agent_id, timeout=1.0)
                if message:
                    await self._handle_message(agent_id, message)
            except Exception as e:
                print(f"Error processing message for {agent_id}: {e}")
    
    async def _handle_message(self, agent_id: str, message: AgentMessage):
        """Handle an incoming message"""
        handlers = self.message_handlers.get(agent_id, {})
        handler = handlers.get(message.message_type)
        
        if handler:
            try:
                response = await handler(message)
                if message.requires_response and response:
                    await self.send_response(message, response)
            except Exception as e:
                print(f"Error in message handler for {agent_id}: {e}")
                
                # Send error response if required
                if message.requires_response:
                    error_response = {
                        'error': str(e),
                        'success': False
                    }
                    await self.send_response(message, error_response)

class CollaborationManager:
    """
    Manager for multi-agent collaboration
    """
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        self.communication = communication_protocol
        self.active_collaborations = {}  # collaboration_id -> collaboration_data
        self.agent_capabilities = {}  # agent_id -> capabilities
    
    def register_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        """Register agent capabilities"""
        self.agent_capabilities[agent_id] = capabilities
    
    async def initiate_collaboration(self, query: str, initiating_agent: str, required_capabilities: List[str]) -> str:
        """Initiate a new collaboration"""
        collaboration_id = str(uuid.uuid4())
        
        # Find suitable agents
        suitable_agents = self._find_agents_with_capabilities(required_capabilities)
        suitable_agents = [agent for agent in suitable_agents if agent != initiating_agent]
        
        if not suitable_agents:
            raise ValueError("No suitable agents found for collaboration")
        
        # Create collaboration
        collaboration = {
            'id': collaboration_id,
            'query': query,
            'initiating_agent': initiating_agent,
            'participants': [initiating_agent] + suitable_agents[:2],  # Limit to 3 agents
            'status': 'active',
            'responses': {},
            'start_time': asyncio.get_event_loop().time()
        }
        
        self.active_collaborations[collaboration_id] = collaboration
        
        # Send collaboration requests
        for agent_id in suitable_agents[:2]:
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=initiating_agent,
                receiver_id=agent_id,
                message_type=MessageType.COLLABORATION_REQUEST,
                content={
                    'collaboration_id': collaboration_id,
                    'query': query,
                    'required_capabilities': required_capabilities
                },
                timestamp=asyncio.get_event_loop().time(),
                requires_response=True
            )
            
            await self.communication.send_message(message)
        
        return collaboration_id
    
    async def add_collaboration_response(self, collaboration_id: str, agent_id: str, response: Dict[str, Any]):
        """Add a response to a collaboration"""
        if collaboration_id not in self.active_collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.active_collaborations[collaboration_id]
        collaboration['responses'][agent_id] = response
        
        # Check if all participants have responded
        if len(collaboration['responses']) == len(collaboration['participants']):
            await self._finalize_collaboration(collaboration_id)
    
    async def _finalize_collaboration(self, collaboration_id: str):
        """Finalize a collaboration and aggregate responses"""
        collaboration = self.active_collaborations[collaboration_id]
        
        # Aggregate responses (simple consensus)
        aggregated_response = self._aggregate_responses(collaboration['responses'])
        
        # Send final result to initiating agent
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id='collaboration_manager',
            receiver_id=collaboration['initiating_agent'],
            message_type=MessageType.RESPONSE,
            content={
                'collaboration_id': collaboration_id,
                'aggregated_response': aggregated_response,
                'individual_responses': collaboration['responses'],
                'participants': collaboration['participants']
            },
            timestamp=asyncio.get_event_loop().time()
        )
        
        await self.communication.send_message(message)
        
        # Mark collaboration as completed
        collaboration['status'] = 'completed'
        collaboration['end_time'] = asyncio.get_event_loop().time()
    
    def _find_agents_with_capabilities(self, required_capabilities: List[str]) -> List[str]:
        """Find agents with the required capabilities"""
        suitable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            if all(cap in capabilities for cap in required_capabilities):
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _aggregate_responses(self, responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple agent responses"""
        if not responses:
            return {}
        
        # Simple aggregation: use the response from the most confident agent
        best_agent = None
        best_confidence = -1
        
        for agent_id, response in responses.items():
            confidence = response.get('confidence', 0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_agent = agent_id
        
        if best_agent:
            return responses[best_agent]
        
        # Fallback: return first response
        return next(iter(responses.values()))

# Global communication instance
communication_protocol = CommunicationProtocol()