import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import heapq
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0

class EpisodicMemory:
    """
    Episodic memory system for storing agent experiences
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories = {}  # id -> MemoryItem
        self.access_queue = []  # Min-heap for LRU eviction
        self.current_size = 0
        
    def add_memory(self, memory_id: str, content: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add a new memory item"""
        if memory_id in self.memories:
            self.update_memory(memory_id, content, embedding, metadata)
            return
        
        # Evict if necessary
        if self.current_size >= self.max_size:
            self._evict_oldest()
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            timestamp=time.time(),
            last_accessed=time.time()
        )
        
        self.memories[memory_id] = memory
        heapq.heappush(self.access_queue, (memory.last_accessed, memory_id))
        self.current_size += 1
        
        logger.debug(f"Added memory: {memory_id}")
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item"""
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        memory.access_count += 1
        memory.last_accessed = time.time()
        
        # Update access queue
        self.access_queue = [(ts, mid) for ts, mid in self.access_queue if mid != memory_id]
        heapq.heappush(self.access_queue, (memory.last_accessed, memory_id))
        heapq.heapify(self.access_queue)
        
        return memory
    
    def update_memory(self, memory_id: str, content: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Update an existing memory item"""
        if memory_id not in self.memories:
            self.add_memory(memory_id, content, embedding, metadata)
            return
        
        memory = self.memories[memory_id]
        memory.content = content
        memory.embedding = embedding
        memory.metadata.update(metadata)
        memory.last_accessed = time.time()
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """Search for similar memories using cosine similarity"""
        similarities = []
        
        for memory in self.memories.values():
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            similarities.append((memory, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_recent_memories(self, count: int = 10) -> List[MemoryItem]:
        """Get most recent memories"""
        memories = list(self.memories.values())
        memories.sort(key=lambda x: x.timestamp, reverse=True)
        return memories[:count]
    
    def get_frequent_memories(self, count: int = 10) -> List[MemoryItem]:
        """Get most frequently accessed memories"""
        memories = list(self.memories.values())
        memories.sort(key=lambda x: x.access_count, reverse=True)
        return memories[:count]
    
    def _evict_oldest(self):
        """Evict the least recently used memory"""
        if not self.access_queue:
            return
        
        while self.access_queue:
            oldest_time, memory_id = heapq.heappop(self.access_queue)
            if memory_id in self.memories:
                del self.memories[memory_id]
                self.current_size -= 1
                logger.debug(f"Evicted memory: {memory_id}")
                break
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def __len__(self):
        return self.current_size

class SemanticMemory:
    """
    Semantic memory system for structured knowledge storage
    """
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.concept_graph = {}  # concept -> related concepts
        self.fact_triples = []  # (subject, predicate, object) tuples
        
    async def add_knowledge(self, content: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add knowledge to semantic memory"""
        # Add to vector database
        await self.vector_db.add_knowledge(content, embedding, metadata)
        
        # Extract concepts and relationships
        concepts = self._extract_concepts(content)
        relationships = self._extract_relationships(content, concepts)
        
        # Update concept graph
        for concept in concepts:
            if concept not in self.concept_graph:
                self.concept_graph[concept] = set()
            
            # Add relationships
            for related_concept, relation in relationships:
                if related_concept != concept:
                    self.concept_graph[concept].add(related_concept)
                    # Store as fact triple
                    self.fact_triples.append((concept, relation, related_concept))
    
    async def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for knowledge using semantic search"""
        return await self.vector_db.similarity_search(query, top_k)
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Get concepts related to the given concept"""
        visited = set()
        result = []
        
        def traverse(current_concept, depth):
            if depth > max_depth or current_concept in visited:
                return
            
            visited.add(current_concept)
            if current_concept != concept:  # Don't include the starting concept
                result.append(current_concept)
            
            if current_concept in self.concept_graph:
                for related in self.concept_graph[current_concept]:
                    traverse(related, depth + 1)
        
        traverse(concept, 0)
        return result
    
    def get_facts_about(self, concept: str) -> List[Tuple[str, str, str]]:
        """Get all facts about a concept"""
        return [triple for triple in self.fact_triples 
                if triple[0] == concept or triple[2] == concept]
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (simplified)"""
        # In production, use NLP techniques like NER
        words = text.split()
        concepts = [word.lower() for word in words if len(word) > 3 and word.isalpha()]
        return list(set(concepts))[:10]  # Limit to top 10 concepts
    
    def _extract_relationships(self, text: str, concepts: List[str]) -> List[Tuple[str, str]]:
        """Extract relationships between concepts (simplified)"""
        relationships = []
        
        # Simple co-occurrence based relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Check if concepts appear together in meaningful ways
                if self._concepts_related(text, concept1, concept2):
                    relationships.append((concept2, "related_to"))
        
        return relationships
    
    def _concepts_related(self, text: str, concept1: str, concept2: str) -> bool:
        """Check if two concepts are related in the text"""
        text_lower = text.lower()
        concept1_lower = concept1.lower()
        concept2_lower = concept2.lower()
        
        # Simple proximity check
        idx1 = text_lower.find(concept1_lower)
        idx2 = text_lower.find(concept2_lower)
        
        if idx1 == -1 or idx2 == -1:
            return False
        
        # Concepts are related if they appear close to each other
        distance = abs(idx1 - idx2)
        return distance < 100  # Within 100 characters

class WorkingMemory:
    """
    Working memory for temporary storage during reasoning
    """
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.current_focus = None
    
    def add_item(self, item: Any, priority: float = 1.0):
        """Add an item to working memory"""
        self.items.append((item, priority, time.time()))
        
        # Sort by priority and recency
        self.items = deque(
            sorted(self.items, key=lambda x: (x[1], x[2]), reverse=True)[:self.capacity]
        )
    
    def get_items(self, count: int = 5) -> List[Any]:
        """Get top items from working memory"""
        return [item[0] for item in list(self.items)[:count]]
    
    def set_focus(self, item: Any):
        """Set the current focus of attention"""
        self.current_focus = item
    
    def get_focus(self) -> Any:
        """Get the current focus of attention"""
        return self.current_focus
    
    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.current_focus = None
    
    def __len__(self):
        return len(self.items)
    