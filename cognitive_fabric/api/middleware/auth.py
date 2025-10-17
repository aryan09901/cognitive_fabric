from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import secrets
import time

class APIKeyAuth(HTTPBearer):
    """
    API Key authentication middleware
    """
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.valid_api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load valid API keys (in production, use secure storage)"""
        # This is a simplified implementation
        # In production, use environment variables or a secure secrets manager
        return {
            "dev_key_12345": {
                "name": "Development Key",
                "permissions": ["read", "write", "admin"],
                "rate_limit": 1000,  # requests per hour
                "created_at": time.time(),
                "last_used": None
            },
            "test_key_67890": {
                "name": "Test Key", 
                "permissions": ["read", "write"],
                "rate_limit": 100,
                "created_at": time.time(),
                "last_used": None
            }
        }
    
    async def __call__(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate API key for request"""
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if credentials:
            api_key = credentials.credentials
            if self._validate_api_key(api_key):
                # Update last used timestamp
                self.valid_api_keys[api_key]["last_used"] = time.time()
                return self.valid_api_keys[api_key]
            else:
                raise HTTPException(
                    status_code=403, 
                    detail="Invalid or expired API key"
                )
        else:
            raise HTTPException(
                status_code=401, 
                detail="API key required"
            )
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key in self.valid_api_keys
    
    def generate_api_key(self, 
                        name: str, 
                        permissions: list, 
                        rate_limit: int = 100) -> str:
        """Generate a new API key"""
        api_key = f"key_{secrets.token_hex(16)}"
        
        self.valid_api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": time.time(),
            "last_used": None
        }
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.valid_api_keys:
            del self.valid_api_keys[api_key]
            return True
        return False
    
    def check_permission(self, api_key_data: Dict[str, Any], permission: str) -> bool:
        """Check if API key has specific permission"""
        return permission in api_key_data.get("permissions", [])

# Global auth instance
api_key_auth = APIKeyAuth()