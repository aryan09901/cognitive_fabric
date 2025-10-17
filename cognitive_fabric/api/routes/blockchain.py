from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import asyncio

from ...blockchain.core.interactions import blockchain_interactions
from ...models.schemas import *

router = APIRouter()

@router.get("/agents", response_model=List[Dict[str, Any]])
async def get_blockchain_agents():
    """
    Get all agents registered on blockchain
    """
    try:
        agents = await blockchain_interactions.get_all_agents()
        
        # Get reputation data for each agent
        agents_with_reputation = []
        for agent_address in agents:
            reputation = await blockchain_interactions.get_agent_reputation(agent_address)
            agents_with_reputation.append({
                "address": agent_address,
                "reputation": reputation
            })
        
        return agents_with_reputation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain agents: {str(e)}")

@router.get("/agents/{agent_address}/reputation", response_model=Dict[str, Any])
async def get_agent_reputation(agent_address: str):
    """
    Get reputation data for a specific agent
    """
    try:
        reputation = await blockchain_interactions.get_agent_reputation(agent_address)
        
        return {
            "agent_address": agent_address,
            "reputation_data": reputation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reputation: {str(e)}")

@router.get("/knowledge/{knowledge_hash}", response_model=Dict[str, Any])
async def get_knowledge_item(knowledge_hash: str):
    """
    Get knowledge item from blockchain
    """
    try:
        knowledge_item = await blockchain_interactions.get_knowledge_item(knowledge_hash)
        
        if not knowledge_item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        return {
            "knowledge_hash": knowledge_hash,
            "item": knowledge_item
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge item: {str(e)}")

@router.get("/network/stats", response_model=Dict[str, Any])
async def get_network_stats():
    """
    Get overall network statistics
    """
    try:
        stats = await blockchain_interactions.get_network_stats()
        
        return {
            "success": True,
            "network_stats": stats,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get network stats: {str(e)}")

@router.post("/interactions/record", response_model=Dict[str, Any])
async def record_interaction(interaction: BlockchainInteraction):
    """
    Record an agent interaction on blockchain
    """
    try:
        tx_hash = await blockchain_interactions.record_interaction(
            from_agent=interaction.from_agent,
            to_agent=interaction.to_agent,
            query_hash=interaction.query_hash,
            response_hash=interaction.response_hash,
            satisfaction_score=interaction.satisfaction_score
        )
        
        return {
            "success": True,
            "transaction_hash": tx_hash,
            "message": "Interaction recorded on blockchain"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")

@router.get("/transactions/{tx_hash}", response_model=Dict[str, Any])
async def get_transaction_status(tx_hash: str):
    """
    Get transaction status
    """
    try:
        # This would query the blockchain for transaction status
        # For now, return mock data
        return {
            "transaction_hash": tx_hash,
            "status": "confirmed",
            "block_number": 123456,
            "confirmations": 15
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transaction status: {str(e)}")

@router.get("/rewards/balance", response_model=Dict[str, Any])
async def get_rewards_balance(agent_address: str):
    """
    Get token balance for an agent
    """
    try:
        # This would query the KnowledgeToken contract
        # For now, return mock data
        return {
            "agent_address": agent_address,
            "token_balance": 1500.0,
            "token_symbol": "KNWL",
            "rewards_earned": 500.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rewards balance: {str(e)}")

@router.post("/knowledge/share", response_model=Dict[str, Any])
async def share_knowledge_on_chain(share_request: Dict[str, Any]):
    """
    Share knowledge on blockchain
    """
    try:
        knowledge_hash = share_request.get("knowledge_hash")
        metadata_hash = share_request.get("metadata_hash")
        
        if not knowledge_hash or not metadata_hash:
            raise HTTPException(status_code=400, detail="Knowledge hash and metadata hash are required")
        
        tx_hash = await blockchain_interactions.share_knowledge(knowledge_hash, metadata_hash)
        
        return {
            "success": True,
            "transaction_hash": tx_hash,
            "knowledge_hash": knowledge_hash,
            "message": "Knowledge shared on blockchain"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to share knowledge: {str(e)}")