from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from typing import Dict, List, Optional, Any
import time
import logging

from config.base import config
from .routes import agents, knowledge, blockchain, queries
from .models.schemas import *
from .middleware.auth import APIKeyAuth
from monitoring.metrics import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
metrics_collector = MetricsCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Cognitive Fabric API...")
    metrics_collector.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cognitive Fabric API...")
    metrics_collector.stop()

# Create FastAPI application
app = FastAPI(
    title="Cognitive Fabric API",
    description="Decentralized AI Network with Blockchain Verification",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication middleware (optional)
# app.add_middleware(APIKeyAuth)

# Include routers
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["Knowledge"])
app.include_router(blockchain.router, prefix="/api/v1/blockchain", tags=["Blockchain"])
app.include_router(queries.router, prefix="/api/v1/queries", tags=["Queries"])

# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"Request completed: {response.status_code} in {process_time:.2f}s")
    
    # Record metrics
    metrics_collector.record_request(
        method=request.method,
        endpoint=str(request.url.path),
        status_code=response.status_code,
        duration=process_time
    )
    
    return response

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": config.VERSION,
        "environment": config.ENVIRONMENT
    }
    
    # Check critical services
    try:
        # Check blockchain connection
        blockchain_status = await _check_blockchain_health()
        health_status["blockchain"] = blockchain_status
        
        # Check vector database
        vector_db_status = await _check_vector_db_health()
        health_status["vector_db"] = vector_db_status
        
        # Check AI models
        model_status = await _check_model_health()
        health_status["models"] = model_status
        
        # Overall status
        all_healthy = all([
            blockchain_status.get("healthy", False),
            vector_db_status.get("healthy", False),
            model_status.get("healthy", False)
        ])
        
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status

async def _check_blockchain_health() -> Dict[str, Any]:
    """Check blockchain connection health"""
    try:
        # This would actually check blockchain connection
        return {"healthy": True, "network": config.BLOCKCHAIN_RPC_URL}
    except Exception as e:
        return {"healthy": False, "error": str(e)}

async def _check_vector_db_health() -> Dict[str, Any]:
    """Check vector database health"""
    try:
        # This would actually check vector DB connection
        return {"healthy": True, "collection": config.VECTOR_DB_COLLECTION}
    except Exception as e:
        return {"healthy": False, "error": str(e)}

async def _check_model_health() -> Dict[str, Any]:
    """Check AI model health"""
    try:
        # This would actually check model loading and inference
        return {"healthy": True, "model": config.LLM_MODEL}
    except Exception as e:
        return {"healthy": False, "error": str(e)}

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return metrics_collector.get_metrics()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Cognitive Fabric API",
        "version": config.VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=config.DEBUG,
        log_level="info"
    )