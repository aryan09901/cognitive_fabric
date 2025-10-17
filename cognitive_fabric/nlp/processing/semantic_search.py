import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """
    Advanced semantic search engine for knowledge retrieval
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_cache = {}
            logger.info(f"Loaded semantic search model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.model = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        if not self.model:
            # Return random embedding as fallback
            return np.random.rand(768)
        
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            embedding = self.model.encode([text])[0]
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return np.random.rand(768)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def find_similar_documents(self, 
                              query: str, 
                              documents: List[Dict[str, Any]],
                              top_k: int = 5,
                              similarity_threshold: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find documents semantically similar to query
        """
        if not documents:
            return []
        
        query_embedding = self.encode_text(query)
        document_embeddings = []
        valid_documents = []
        
        # Encode all documents
        for doc in documents:
            content = doc.get('content', '')
            if content:
                doc_embedding = self.encode_text(content)
                document_embeddings.append(doc_embedding)
                valid_documents.append(doc)
        
        if not document_embeddings:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        
        # Pair documents with similarities and filter
        scored_documents = []
        for i, (doc, similarity) in enumerate(zip(valid_documents, similarities)):
            if similarity >= similarity_threshold:
                scored_documents.append((doc, similarity))
        
        # Sort by similarity and return top_k
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents[:top_k]
    
    def cluster_documents(self, 
                         documents: List[Dict[str, Any]], 
                         num_clusters: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Cluster documents by semantic similarity
        """
        from sklearn.cluster import KMeans
        
        if len(documents) < num_clusters:
            # Not enough documents for clustering
            return [documents]
        
        # Encode documents
        embeddings = []
        valid_documents = []
        
        for doc in documents:
            content = doc.get('content', '')
            if content:
                embedding = self.encode_text(content)
                embeddings.append(embedding)
                valid_documents.append(doc)
        
        if not embeddings:
            return [documents]
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group documents by cluster
        clustered_documents = [[] for _ in range(num_clusters)]
        for doc, cluster_id in zip(valid_documents, clusters):
            clustered_documents[cluster_id].append(doc)
        
        # Remove empty clusters
        clustered_documents = [cluster for cluster in clustered_documents if cluster]
        
        return clustered_documents
    
    def find_related_concepts(self, 
                            concept: str, 
                            knowledge_base: List[Dict[str, Any]],
                            max_related: int = 10) -> List[Tuple[str, float]]:
        """
        Find concepts related to the given concept
        """
        concept_embedding = self.encode_text(concept)
        concept_similarities = {}
        
        for item in knowledge_base:
            content = item.get('content', '')
            # Extract potential concepts (simplified)
            words = content.split()
            potential_concepts = [word for word in words if len(word) > 3 and word.isalpha()]
            
            for potential_concept in potential_concepts[:20]:  # Limit per document
                if potential_concept.lower() != concept.lower():
                    concept_embed = self.encode_text(potential_concept)
                    similarity = cosine_similarity([concept_embedding], [concept_embed])[0][0]
                    
                    if potential_concept not in concept_similarities:
                        concept_similarities[potential_concept] = []
                    concept_similarities[potential_concept].append(similarity)
        
        # Average similarities and get top concepts
        avg_similarities = {}
        for concept_name, similarities in concept_similarities.items():
            avg_similarities[concept_name] = np.mean(similarities)
        
        sorted_concepts = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_concepts[:max_related]
    
    def semantic_search_with_filters(self,
                                   query: str,
                                   documents: List[Dict[str, Any]],
                                   filters: Dict[str, Any],
                                   top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Semantic search with additional filters
        """
        # Apply content filters first
        filtered_docs = documents
        
        if 'min_verification_score' in filters:
            min_score = filters['min_verification_score']
            filtered_docs = [doc for doc in filtered_docs 
                           if doc.get('metadata', {}).get('verification_score', 0) >= min_score]
        
        if 'source' in filters:
            source_filter = filters['source']
            filtered_docs = [doc for doc in filtered_docs 
                           if doc.get('metadata', {}).get('source') == source_filter]
        
        if 'max_length' in filters:
            max_len = filters['max_length']
            filtered_docs = [doc for doc in filtered_docs 
                           if len(doc.get('content', '')) <= max_len]
        
        # Perform semantic search on filtered documents
        return self.find_similar_documents(query, filtered_docs, top_k)

class HybridSearchEngine:
    """
    Hybrid search combining semantic and keyword search
    """
    
    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        self.semantic_engine = SemanticSearchEngine()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def hybrid_search(self,
                     query: str,
                     documents: List[Dict[str, Any]],
                     top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform hybrid semantic + keyword search
        """
        # Semantic search
        semantic_results = self.semantic_engine.find_similar_documents(
            query, documents, top_k * 2
        )
        
        # Keyword search
        keyword_results = self._keyword_search(query, documents, top_k * 2)
        
        # Combine results
        combined_scores = {}
        
        # Add semantic scores
        for doc, score in semantic_results:
            doc_id = id(doc)  # Use object ID as key
            combined_scores[doc_id] = {
                'document': doc,
                'semantic_score': score,
                'keyword_score': 0.0
            }
        
        # Add keyword scores
        for doc, score in keyword_results:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = score
            else:
                combined_scores[doc_id] = {
                    'document': doc,
                    'semantic_score': 0.0,
                    'keyword_score': score
                }
        
        # Calculate combined scores
        combined_results = []
        for scores in combined_scores.values():
            combined_score = (
                scores['semantic_score'] * self.semantic_weight +
                scores['keyword_score'] * self.keyword_weight
            )
            combined_results.append((scores['document'], combined_score))
        
        # Sort and return top_k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def _keyword_search(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Simple keyword-based search"""
        query_words = set(query.lower().split())
        scored_documents = []
        
        for doc in documents:
            content = doc.get('content', '').lower()
            content_words = set(content.split())
            
            # Calculate keyword overlap
            overlap = len(query_words.intersection(content_words))
            if len(query_words) > 0:
                score = overlap / len(query_words)
            else:
                score = 0.0
            
            scored_documents.append((doc, score))
        
        # Sort by score and return top_k
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents[:top_k]

# Global instances
semantic_search_engine = SemanticSearchEngine()
hybrid_search_engine = HybridSearchEngine()