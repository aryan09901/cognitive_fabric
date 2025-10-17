from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class VerificationLevel(str, Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to process")
    agent_id: Optional[str] = Field(None, description="Specific agent to use")
    collaborative: bool = Field(False, description="Use collaborative solving")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    verification_level: VerificationLevel = Field(VerificationLevel.MODERATE, description="Verification strictness")

class QueryResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used")
    verification_score: float = Field(..., ge=0.0, le=1.0, description="Verification score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    agents_used: List[str] = Field(..., description="Agents involved")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")

class BatchQueryItem(BaseModel):
    text: str = Field(..., description="Query text")
    agent_id: Optional[str] = Field(None, description="Specific agent")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class BatchQueryRequest(BaseModel):
    queries: List[BatchQueryItem] = Field(..., description="Queries to process")
    verification_level: VerificationLevel = Field(VerificationLevel.MODERATE, description="Verification strictness")

class BatchQueryResponse(BaseModel):
    successful: List[QueryResponse] = Field(..., description="Successful responses")
    failed: List[Dict[str, Any]] = Field(..., description="Failed queries")
    total_processed: int = Field(..., description="Total processed successfully")
    total_failed: int = Field(..., description="Total failed")

class VerificationRequest(BaseModel):
    response: str = Field(..., description="Response to verify")
    context: List[Dict[str, Any]] = Field(..., description="Context for verification")
    verification_level: VerificationLevel = Field(VerificationLevel.STRICT, description="Verification strictness")

class VerificationResponse(BaseModel):
    verification_score: float = Field(..., ge=0.0, le=1.0, description="Overall verification score")
    is_verified: bool = Field(..., description="Whether response is verified")
    details: Dict[str, Any] = Field(..., description="Detailed verification results")
    suggestions: List[str] = Field(..., description="Suggestions for improvement")

class QueryHistoryItem(BaseModel):
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    timestamp: datetime = Field(..., description="When the query was processed")
    verification_score: float = Field(..., description="Verification score")
    agent_id: str = Field(..., description="Agent that processed the query")

class QueryHistoryResponse(BaseModel):
    agent_id: str = Field(..., description="Agent identifier")
    queries: List[QueryHistoryItem] = Field(..., description="Query history")
    total_count: int = Field(..., description="Total queries in history")
    limit: int = Field(..., description="Limit used in query")
    offset: int = Field(..., description="Offset used in query")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Timestamp of check")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment")
    blockchain: Dict[str, Any] = Field(..., description="Blockchain health")
    vector_db: Dict[str, Any] = Field(..., description="Vector DB health")
    models: Dict[str, Any] = Field(..., description="Models health")
    error: Optional[str] = Field(None, description="Error if unhealthy")

class AgentRegistration(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    config: Dict[str, Any] = Field(..., description="Agent configuration")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class KnowledgeItem(BaseModel):
    content: str = Field(..., description="Knowledge content")
    metadata: Dict[str, Any] = Field(..., description="Knowledge metadata")
    verification_score: float = Field(0.0, ge=0.0, le=1.0, description="Initial verification score")
    source: Optional[str] = Field(None, description="Knowledge source")

class BlockchainInteraction(BaseModel):
    from_agent: str = Field(..., description="Source agent")
    to_agent: str = Field(..., description="Target agent")
    query_hash: str = Field(..., description="Query content hash")
    response_hash: str = Field(..., description="Response content hash")
    satisfaction_score: int = Field(..., ge=0, le=100, description="Satisfaction score")