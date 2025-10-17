import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Advanced embedding model for semantic representations
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self._get_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.model_name = "fallback"
            self.embedding_dim = 384  # Default dimension
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model"""
        if self.model is None:
            return 384
        
        # Test encoding to get dimension
        test_embedding = self.model.encode(["test"])
        return test_embedding.shape[1]
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode list of texts to embeddings"""
        if not texts:
            return np.array([])
        
        if self.model is None:
            # Fallback: random embeddings
            return np.random.rand(len(texts), self.embedding_dim)
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            # Fallback to random embeddings
            return np.random.rand(len(texts), self.embedding_dim)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.encode([text])[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def batch_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate batch cosine similarities"""
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return np.array([])
        
        # Normalize embeddings
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms1 = np.where(norms1 == 0, 1e-8, norms1)
        norms2 = np.where(norms2 == 0, 1e-8, norms2)
        
        embeddings1_normalized = embeddings1 / norms1
        embeddings2_normalized = embeddings2 / norms2
        
        similarities = np.dot(embeddings1_normalized, embeddings2_normalized.T)
        return similarities
    
    def most_similar(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray, top_k: int = 5) -> List[int]:
        """Find most similar embeddings to query"""
        if len(candidate_embeddings) == 0:
            return []
        
        similarities = self.batch_similarity(np.array([query_embedding]), candidate_embeddings)
        similarities = similarities[0]  # Get first row
        
        # Get indices of top_k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices.tolist()
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'status': 'loaded' if self.model is not None else 'fallback'
        }
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate if embedding is reasonable"""
        if embedding is None:
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        # Check if embedding is all zeros (likely error)
        if np.all(embedding == 0):
            return False
        
        # Check magnitude (should not be extremely large or small)
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.1 or magnitude > 100:
            return False
        
        return True

class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputation
    """
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        text_hash = hash(text)
        if text_hash in self.cache:
            self.access_count[text_hash] = self.access_count.get(text_hash, 0) + 1
            return self.cache[text_hash]
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Set embedding in cache"""
        text_hash = hash(text)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        
        self.cache[text_hash] = embedding
        self.access_count[text_hash] = 1
    
    def _evict_least_used(self):
        """Evict least used item from cache"""
        if not self.access_count:
            # Remove random item if no access counts
            key_to_remove = next(iter(self.cache.keys()))
        else:
            # Remove least accessed item
            key_to_remove = min(self.access_count.items(), key=lambda x: x[1])[0]
        
        del self.cache[key_to_remove]
        del self.access_count[key_to_remove]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
    
    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self._calculate_hit_rate(),
            'most_accessed': sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would track hits/misses in production
        return 0.0

# Global embedding model instance
embedding_model = EmbeddingModel()
embedding_cache = EmbeddingCache()