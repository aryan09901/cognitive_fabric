from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, List, Optional, Any
import asyncio
import uuid

from ...models.schemas import *
from ...agents.multi_agent.orchestrator import MultiAgentOrchestrator
from ...nlp.core.language_models import AdvancedLanguageModel
from ...monitoring.metrics import MetricsCollector

router = APIRouter()

# Initialize components
orchestrator = MultiAgentOrchestrator()
language_model = AdvancedLanguageModel()
metrics = MetricsCollector()

@router.post("/process", response_model=QueryResponse)
async def process_query(
    query: QueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a query through the cognitive fabric
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Process query based on mode
        if query.collaborative:
            result = await orchestrator.collaborative_solving(
                query.query, 
                query.context or {}
            )
        else:
            result = await orchestrator.route_query(
                query.query, 
                query.agent_id,
                query.context or {}
            )
        
        # Calculate processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_query_metrics,
            request_id=request_id,
            query=query.query,
            processing_time=processing_time,
            agent_count=result.get('agents_participated', 1),
            verification_score=result.get('verification_score', 0.0)
        )
        
        return QueryResponse(
            request_id=request_id,
            response=result['response'],
            sources=result.get('sources', []),
            verification_score=result.get('verification_score', 0.0),
            confidence=result.get('confidence', 0.5),
            processing_time=processing_time,
            agents_used=result.get('agents_used', []),
            metadata=result.get('metadata', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/batch", response_model=BatchQueryResponse)
async def process_batch_queries(queries: BatchQueryRequest):
    """
    Process multiple queries in batch
    """
    try:
        tasks = []
        for query in queries.queries:
            task = orchestrator.route_query(
                query.text, 
                query.agent_id,
                query.context or {}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_queries = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_queries.append({
                    'index': i,
                    'error': str(result)
                })
            else:
                successful_results.append(QueryResponse(
                    request_id=str(uuid.uuid4()),
                    response=result['response'],
                    sources=result.get('sources', []),
                    verification_score=result.get('verification_score', 0.0),
                    confidence=result.get('confidence', 0.5),
                    processing_time=0.0,  # Would need individual timing
                    agents_used=result.get('agents_used', []),
                    metadata=result.get('metadata', {})
                ))
        
        return BatchQueryResponse(
            successful=successful_results,
            failed=failed_queries,
            total_processed=len(successful_results),
            total_failed=len(failed_queries)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.post("/verify", response_model=VerificationResponse)
async def verify_response(verification_request: VerificationRequest):
    """
    Verify a response against knowledge base
    """
    try:
        verification_result = await language_model.verification_engine.verify_response(
            verification_request.response,
            verification_request.context
        )
        
        return VerificationResponse(
            verification_score=verification_result.overall_score,
            is_verified=verification_result.overall_score >= 0.7,
            details={
                'verified_claims': verification_result.verified_claims,
                'unverified_claims': verification_result.unverified_claims,
                'contradictions': verification_result.contradictions
            },
            suggestions=verification_result.suggestions if hasattr(verification_result, 'suggestions') else []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@router.get("/history/{agent_id}", response_model=QueryHistoryResponse)
async def get_query_history(
    agent_id: str,
    limit: int = 50,
    offset: int = 0
):
    """
    Get query history for an agent
    """
    try:
        # This would query the agent's memory system
        history = await orchestrator.get_agent_history(agent_id, limit, offset)
        
        return QueryHistoryResponse(
            agent_id=agent_id,
            queries=history,
            total_count=len(history),
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent not found: {str(e)}")