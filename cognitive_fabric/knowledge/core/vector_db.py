import chromadb
from typing import List, Dict, Any, Optional
import numpy as np
import asyncio
from dataclasses import dataclass
import uuid

from config.base import config

@dataclass
class KnowledgeItem:
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    verification_score: float
    blockchain_hash: Optional[str] = None

class VectorDatabase:
    """
    Vector database manager for semantic knowledge storage and retrieval
    """
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=config.VECTOR_DB_COLLECTION,
            metadata={"description": "Cognitive Fabric Knowledge Base"}
        )
        
        # Cache for frequent queries
        self.query_cache = {}
        
    async def add_knowledge(
        self, 
        content: str, 
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        verification_score: float = 0.0,
        blockchain_hash: Optional[str] = None
    ) -> str:
        """Add knowledge item to vector database"""
        
        item_id = str(uuid.uuid4())
        
        # Create knowledge item
        knowledge_item = KnowledgeItem(
            id=item_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            verification_score=verification_score,
            blockchain_hash=blockchain_hash
        )
        
        # Add to ChromaDB
        self.collection.add(
            documents=[content],
            metadatas=[{
                **metadata,
                'verification_score': verification_score,
                'blockchain_hash': blockchain_hash or '',
                'embedding_shape': embedding.shape
            }],
            embeddings=[embedding.tolist()],
            ids=[item_id]
        )
        
        return item_id
    
    async def similarity_search(
        self, 
        embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search with filters"""
        
        # Build query
        query_embeddings = [embedding.tolist()]
        
        # Apply filters
        where = {}
        if filters:
            if 'min_verification_score' in filters:
                where['verification_score'] = {'$gte': filters['min_verification_score']}
            if 'source' in filters:
                where['source'] = filters['source']
        
        # Perform search
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': 1.0 / (1.0 + distance),  # Convert distance to similarity
                'verification_score': metadata.get('verification_score', 0.0),
                'blockchain_hash': metadata.get('blockchain_hash'),
                'source': metadata.get('source', 'unknown')
            })
        
        return formatted_results
    
    async def update_verification_score(
        self, 
        item_id: str, 
        verification_score: float
    ) -> bool:
        """Update verification score for knowledge item"""
        try:
            # Get current metadata
            results = self.collection.get(ids=[item_id], include=['metadatas'])
            if not results['metadatas']:
                return False
            
            current_metadata = results['metadatas'][0]
            updated_metadata = {
                **current_metadata,
                'verification_score': verification_score
            }
            
            # Update in database
            self.collection.update(
                ids=[item_id],
                metadatas=[updated_metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to update verification score: {e}")
            return False
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        all_items = self.collection.get()
        
        total_items = len(all_items['ids'])
        verification_scores = [
            metadata.get('verification_score', 0) 
            for metadata in all_items['metadatas']
        ]
        
        return {
            'total_items': total_items,
            'average_verification_score': np.mean(verification_scores) if verification_scores else 0,
            'verified_items': sum(1 for score in verification_scores if score >= 0.7),
            'sources': len(set(metadata.get('source', 'unknown') for metadata in all_items['metadatas']))
        }
    
    async def cleanup_low_quality(self, min_verification_score: float = 0.3):
        """Remove low-quality knowledge items"""
        try:
            # Get items below threshold
            low_quality_items = self.collection.get(
                where={'verification_score': {'$lt': min_verification_score}}
            )
            
            if low_quality_items['ids']:
                self.collection.delete(ids=low_quality_items['ids'])
                print(f"Removed {len(low_quality_items['ids'])} low-quality items")
                
        except Exception as e:
            print(f"Cleanup failed: {e}")

class KnowledgeGraph:
    """
    Knowledge graph for semantic relationships between concepts
    """
    
    def __init__(self):
        self.nodes = {}  # concept -> metadata
        self.edges = {}  # (concept1, concept2) -> relationship_type
        
    async def add_concept(self, concept: str, metadata: Dict[str, Any]):
        """Add concept to knowledge graph"""
        self.nodes[concept] = {
            **metadata,
            'frequency': self.nodes.get(concept, {}).get('frequency', 0) + 1
        }
    
    async def add_relationship(self, concept1: str, concept2: str, relationship: str):
        """Add relationship between concepts"""
        edge_key = (concept1, concept2)
        self.edges[edge_key] = relationship
    
    async def find_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Find related concepts in knowledge graph"""
        related = set()
        
        def traverse(current_concept, depth):
            if depth > max_depth:
                return
            
            for (c1, c2), rel in self.edges.items():
                if c1 == current_concept and c2 not in related:
                    related.add(c2)
                    traverse(c2, depth + 1)
                elif c2 == current_concept and c1 not in related:
                    related.add(c1)
                    traverse(c1, depth + 1)
        
        traverse(concept, 0)
        return list(related)