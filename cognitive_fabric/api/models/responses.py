from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"

class BaseResponse(BaseModel):
    """Base response model for all API responses"""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

class ErrorResponse(BaseResponse):
    """Error response model"""
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    suggestion: Optional[str] = Field(None, description="Suggested resolution")

    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid request parameters",
                "error_code": "VALIDATION_ERROR",
                "details": {"field": "query", "issue": "required field missing"},
                "suggestion": "Please provide a valid query parameter",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456"
            }
        }

class SuccessResponse(BaseResponse):
    """Success response model"""
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "data": {"result": "success"},
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456"
            }
        }

class AgentResponse(BaseModel):
    """Agent response model"""
    agent_id: str = Field(..., description="Agent identifier")
    response: str = Field(..., description="Agent's response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    verification_score: float = Field(..., ge=0.0, le=1.0, description="Verification score")
    processing_time: float = Field(..., description="Processing time in seconds")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "agent_id": "research_specialist",
                "response": "This is the agent's response to the query...",
                "confidence": 0.92,
                "verification_score": 0.88,
                "processing_time": 2.45,
                "sources": [
                    {
                        "content": "Source content...",
                        "metadata": {"source": "research_paper", "year": 2023}
                    }
                ],
                "metadata": {
                    "model": "mistralai/Mistral-7B-Instruct-v0.1",
                    "context_utilization": 3
                }
            }
        }

class QueryResponse(SuccessResponse):
    """Query processing response"""
    data: Dict[str, Any] = Field(..., description="Query response data")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Query processed successfully",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "response": "This is the system's response...",
                    "sources": [
                        {
                            "content": "Source content...",
                            "metadata": {"source": "verified_knowledge"}
                        }
                    ],
                    "verification_score": 0.85,
                    "confidence": 0.90,
                    "processing_time": 3.21,
                    "agents_used": ["research_specialist"],
                    "collaboration_mode": "single_agent",
                    "metadata": {
                        "query_complexity": 0.7,
                        "verification_level": "strict"
                    }
                }
            }
        }

class BatchQueryResponse(SuccessResponse):
    """Batch query processing response"""
    data: Dict[str, Any] = Field(..., description="Batch processing results")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Batch processing completed",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "successful": [
                        {
                            "query": "First query",
                            "response": "Response to first query...",
                            "verification_score": 0.82
                        }
                    ],
                    "failed": [
                        {
                            "query": "Second query",
                            "error": "Processing failed",
                            "reason": "Timeout"
                        }
                    ],
                    "total_processed": 1,
                    "total_failed": 1,
                    "average_processing_time": 2.5
                }
            }
        }

class AgentStatusResponse(SuccessResponse):
    """Agent status response"""
    data: Dict[str, Any] = Field(..., description="Agent status information")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Agent status retrieved",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "agent_id": "research_specialist",
                    "status": "active",
                    "reputation": 95.5,
                    "capabilities": ["research", "analysis", "verification"],
                    "performance_metrics": {
                        "total_queries": 150,
                        "success_rate": 0.94,
                        "average_verification_score": 0.87,
                        "average_processing_time": 2.3
                    },
                    "memory_usage": 450,
                    "blockchain_address": "0x742...",
                    "last_activity": "2023-01-01T00:00:00Z"
                }
            }
        }

class KnowledgeResponse(SuccessResponse):
    """Knowledge management response"""
    data: Dict[str, Any] = Field(..., description="Knowledge operation results")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Knowledge added successfully",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "document_id": "doc_123456",
                    "content_hash": "QmKnowledgeHash",
                    "verification_score": 0.8,
                    "source": "user_upload",
                    "metadata": {
                        "category": "research",
                        "timestamp": "2023-01-01T00:00:00Z"
                    }
                }
            }
        }

class BlockchainResponse(SuccessResponse):
    """Blockchain operation response"""
    data: Dict[str, Any] = Field(..., description="Blockchain operation results")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Interaction recorded on blockchain",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "transaction_hash": "0x123456...",
                    "block_number": 1234567,
                    "gas_used": 21000,
                    "status": "confirmed",
                    "interaction_details": {
                        "from_agent": "0xAgent1",
                        "to_agent": "0xAgent2",
                        "satisfaction_score": 85
                    }
                }
            }
        }

class SystemHealthResponse(SuccessResponse):
    """System health check response"""
    data: Dict[str, Any] = Field(..., description="System health information")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "System health check completed",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "overall_status": "healthy",
                    "components": {
                        "blockchain": {
                            "status": "healthy",
                            "network": "polygon",
                            "latest_block": 1234567
                        },
                        "vector_database": {
                            "status": "healthy",
                            "total_documents": 15432,
                            "average_verification": 0.78
                        },
                        "agents": {
                            "status": "healthy",
                            "total_agents": 5,
                            "active_agents": 5
                        },
                        "api": {
                            "status": "healthy",
                            "response_time": 0.15,
                            "throughput": 45.2
                        }
                    },
                    "metrics": {
                        "total_queries": 1247,
                        "success_rate": 0.96,
                        "average_processing_time": 2.8,
                        "system_uptime": 86400
                    }
                }
            }
        }

class VerificationResponse(SuccessResponse):
    """Verification response"""
    data: Dict[str, Any] = Field(..., description="Verification results")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Verification completed",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "verification_score": 0.88,
                    "is_verified": True,
                    "details": {
                        "verified_claims": [
                            {
                                "claim": "Sample claim",
                                "confidence": 0.92,
                                "sources": ["source1", "source2"]
                            }
                        ],
                        "unverified_claims": [],
                        "contradictions": []
                    },
                    "suggestions": [
                        "Provide more specific sources for better verification"
                    ]
                }
            }
        }

class PaginatedResponse(SuccessResponse):
    """Paginated response model"""
    data: Dict[str, Any] = Field(..., description="Paginated data")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Data retrieved successfully",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "items": [
                        {"id": 1, "content": "Item 1"},
                        {"id": 2, "content": "Item 2"}
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 10,
                        "total_items": 100,
                        "total_pages": 10,
                        "has_next": True,
                        "has_prev": False
                    }
                }
            }
        }

class TrainingResponse(SuccessResponse):
    """Training operation response"""
    data: Dict[str, Any] = Field(..., description="Training results")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Training initiated successfully",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "training_id": "train_123456",
                    "status": "started",
                    "agents_training": ["agent1", "agent2"],
                    "estimated_duration": 3600,
                    "progress_url": "/api/v1/training/train_123456/progress"
                }
            }
        }

class CollaborationResponse(SuccessResponse):
    """Collaboration response"""
    data: Dict[str, Any] = Field(..., description="Collaboration results")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Collaboration completed successfully",
                "timestamp": "2023-01-01T00:00:00Z",
                "request_id": "req_123456",
                "data": {
                    "collaboration_id": "collab_123456",
                    "response": "Collaborative response...",
                    "participating_agents": ["agent1", "agent2", "agent3"],
                    "consensus_score": 0.89,
                    "individual_contributions": {
                        "agent1": "Contribution from agent1...",
                        "agent2": "Contribution from agent2...",
                        "agent3": "Contribution from agent3..."
                    },
                    "processing_time": 4.56,
                    "metadata": {
                        "consensus_strategy": "confidence_based",
                        "collaboration_quality": 0.92
                    }
                }
            }
        }

# Common error responses
class ErrorResponses:
    """Common error response templates"""
    
    @staticmethod
    def not_found(resource: str, request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            status=ResponseStatus.ERROR,
            message=f"{resource} not found",
            error_code="NOT_FOUND",
            suggestion=f"Please check the {resource.lower()} identifier",
            request_id=request_id
        )
    
    @staticmethod
    def validation_error(details: Dict[str, Any], request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            status=ResponseStatus.ERROR,
            message="Validation error",
            error_code="VALIDATION_ERROR",
            details=details,
            suggestion="Please check the request parameters",
            request_id=request_id
        )
    
    @staticmethod
    def internal_error(error_message: str, request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            status=ResponseStatus.ERROR,
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"message": error_message},
            suggestion="Please try again later or contact support",
            request_id=request_id
        )
    
    @staticmethod
    def rate_limit_exceeded(request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            status=ResponseStatus.ERROR,
            message="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            suggestion="Please wait before making more requests",
            request_id=request_id
        )
    
    @staticmethod
    def unauthorized(request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            status=ResponseStatus.ERROR,
            message="Unauthorized access",
            error_code="UNAUTHORIZED",
            suggestion="Please provide valid authentication credentials",
            request_id=request_id
        )
    
    @staticmethod
    def service_unavailable(service: str, request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            status=ResponseStatus.ERROR,
            message=f"{service} service unavailable",
            error_code="SERVICE_UNAVAILABLE",
            suggestion=f"Please try again later or check {service.lower()} status",
            request_id=request_id
        )