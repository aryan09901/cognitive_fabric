from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import json
import time
from typing import Dict, Any, Optional
import re

class RequestValidator:
    """
    Request validation middleware
    """
    
    def __init__(self):
        self.max_content_length = 10 * 1024 * 1024  # 10MB
        self.allowed_content_types = [
            'application/json',
            'text/plain'
        ]
    
    async def validate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate incoming request"""
        validation_errors = []
        
        # Check content length
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_content_length:
            validation_errors.append(
                f"Content too large. Maximum size is {self.max_content_length} bytes"
            )
        
        # Check content type
        content_type = request.headers.get('content-type', '').split(';')[0]
        if content_type and content_type not in self.allowed_content_types:
            validation_errors.append(
                f"Unsupported content type: {content_type}. "
                f"Allowed types: {', '.join(self.allowed_content_types)}"
            )
        
        # Validate JSON body if present
        if content_type == 'application/json' and await request.body():
            try:
                body = await request.json()
                await self._validate_json_body(body)
            except json.JSONDecodeError:
                validation_errors.append("Invalid JSON in request body")
            except HTTPException as e:
                validation_errors.append(e.detail)
        
        if validation_errors:
            raise HTTPException(
                status_code=422,
                detail={
                    "errors": validation_errors,
                    "message": "Request validation failed"
                }
            )
        
        return None
    
    async def _validate_json_body(self, body: Dict[str, Any]):
        """Validate JSON request body"""
        if isinstance(body, dict):
            # Check for SQL injection patterns
            self._check_sql_injection(body)
            
            # Validate specific fields if present
            if 'query' in body:
                self._validate_query_field(body['query'])
            
            if 'agent_id' in body:
                self._validate_agent_id(body['agent_id'])
    
    def _check_sql_injection(self, data: Any):
        """Check for potential SQL injection patterns"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION']
        
        def check_value(value):
            if isinstance(value, str):
                upper_value = value.upper()
                for keyword in sql_keywords:
                    # Simple pattern matching (in production, use proper SQL injection detection)
                    if keyword in upper_value and ' ' in value:
                        raise HTTPException(
                            status_code=422,
                            detail="Potential SQL injection detected in request"
                        )
            elif isinstance(value, (list, dict)):
                for item in value.values() if isinstance(value, dict) else value:
                    check_value(item)
        
        check_value(data)
    
    def _validate_query_field(self, query: str):
        """Validate query field"""
        if not isinstance(query, str):
            raise HTTPException(
                status_code=422,
                detail="Query must be a string"
            )
        
        if len(query.strip()) == 0:
            raise HTTPException(
                status_code=422,
                detail="Query cannot be empty"
            )
        
        if len(query) > 1000:
            raise HTTPException(
                status_code=422,
                detail="Query too long. Maximum length is 1000 characters"
            )
    
    def _validate_agent_id(self, agent_id: str):
        """Validate agent ID field"""
        if not isinstance(agent_id, str):
            raise HTTPException(
                status_code=422,
                detail="Agent ID must be a string"
            )
        
        # Basic pattern validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            raise HTTPException(
                status_code=422,
                detail="Agent ID can only contain letters, numbers, underscores, and hyphens"
            )
        
        if len(agent_id) > 50:
            raise HTTPException(
                status_code=422,
                detail="Agent ID too long. Maximum length is 50 characters"
            )

class RateLimiter:
    """
    Rate limiting middleware
    """
    
    def __init__(self, requests_per_hour: int = 1000):
        self.requests_per_hour = requests_per_hour
        self.request_log: Dict[str, list] = {}
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Clean old entries
        if client_id in self.request_log:
            self.request_log[client_id] = [
                timestamp for timestamp in self.request_log[client_id]
                if timestamp > hour_ago
            ]
        else:
            self.request_log[client_id] = []
        
        # Check rate limit
        if len(self.request_log[client_id]) >= self.requests_per_hour:
            return False
        
        # Log current request
        self.request_log[client_id].append(current_time)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        if client_id not in self.request_log:
            return self.requests_per_hour
        
        recent_requests = [
            timestamp for timestamp in self.request_log[client_id]
            if timestamp > hour_ago
        ]
        
        return max(0, self.requests_per_hour - len(recent_requests))

# Global instances
request_validator = RequestValidator()
rate_limiter = RateLimiter()