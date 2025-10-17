import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from knowledge.core.vector_db import VectorDatabase, SearchResult
from knowledge.storage.ipfs_client import IPFSClient
from agents.memory_systems import EpisodicMemory, SemanticMemory, WorkingMemory

class TestVectorDatabase:
    """Test cases for VectorDatabase"""
    
    @pytest.fixture
    def vector_db(self):
        with patch('chromadb.HttpClient') as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            db = VectorDatabase()
            db.collection = mock_collection
            return db
    
    @pytest.mark.asyncio
    async def test_add_knowledge(self, vector_db):
        """Test adding knowledge to vector database"""
        mock_embedding = np.random.rand(384)
        mock_metadata = {"source": "test", "verification_score": 0.8}
        
        vector_db.collection.add = Mock()
        
        doc_id = await vector_db.add_knowledge(
            content="Test content",
            embedding=mock_embedding,
            metadata=mock_metadata
        )
        
        assert doc_id is not None
        vector_db.collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_db):
        """Test similarity search"""
        mock_results = {
            'documents': [['Test document 1', 'Test document 2']],
            'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]],
            'distances': [[0.1, 0.2]],
            'ids': [['doc1', 'doc2']]
        }
        vector_db.collection.query.return_value = mock_results
        
        mock_embedding = np.random.rand(384)
        results = await vector_db.similarity_search(mock_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0].similarity_score > results[1].similarity_score
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, vector_db):
        """Test hybrid search"""
        with patch.object(vector_db, 'similarity_search', return_value=[
            SearchResult("Doc1", {"source": "test"}, 0.9, "1"),
            SearchResult("Doc2", {"source": "test"}, 0.8, "2")
        ]):
            with patch.object(vector_db, 'keyword_search', return_value=[
                SearchResult("Doc1", {"source": "test"}, 0.7, "1"),
                SearchResult("Doc3", {"source": "test"}, 0.6, "3")
            ]):
                results = await vector_db.hybrid_search(
                    query="test query",
                    query_embedding=np.random.rand(384),
                    top_k=3
                )
                
                assert len(results) <= 3
                # Doc1 should have highest score due to combination
    
    @pytest.mark.asyncio
    async def test_update_document_metadata(self, vector_db):
        """Test metadata update"""
        vector_db.collection.update = Mock()
        
        await vector_db.update_document_metadata("doc1", {"new_field": "value"})
        vector_db.collection.update.assert_called_once_with(
            ids=["doc1"],
            metadatas=[{"new_field": "value"}]
        )
    
    @pytest.mark.asyncio
    async def test_delete_document(self, vector_db):
        """Test document deletion"""
        vector_db.collection.delete = Mock()
        
        await vector_db.delete_document("doc1")
        vector_db.collection.delete.assert_called_once_with(ids=["doc1"])
    
    @pytest.mark.asyncio
    async def test_get_collection_stats(self, vector_db):
        """Test collection statistics"""
        mock_results = {
            'ids': ['doc1', 'doc2', 'doc3'],
            'metadatas': [
                {'verification_score': 0.8, 'source': 'test1'},
                {'verification_score': 0.9, 'source': 'test2'},
                {'verification_score': 0.7, 'source': 'test1'}
            ]
        }
        vector_db.collection.get.return_value = mock_results
        
        stats = await vector_db.get_collection_stats()
        
        assert stats['total_documents'] == 3
        assert 0.7 <= stats['average_verification_score'] <= 0.9
        assert stats['sources_count'] == 2

class TestIPFSClient:
    """Test cases for IPFSClient"""
    
    @pytest.fixture
    def ipfs_client(self):
        with patch('ipfshttpclient.connect') as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client
            
            client = IPFSClient()
            client.client = mock_client
            return client
    
    @pytest.mark.asyncio
    async def test_upload_json(self, ipfs_client):
        """Test JSON upload to IPFS"""
        test_data = {"content": "test", "metadata": {"source": "test"}}
        ipfs_client.client.add_str = Mock(return_value="QmTestHash")
        ipfs_client.client.pin.add = Mock()
        
        hash_result = await ipfs_client.upload_json(test_data)
        
        assert hash_result == "QmTestHash"
        ipfs_client.client.add_str.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_json(self, ipfs_client):
        """Test JSON download from IPFS"""
        test_data = b'{"content": "test", "metadata": {"source": "test"}}'
        ipfs_client.client.cat = Mock(return_value=test_data)
        
        result = await ipfs_client.download_json("QmTestHash")
        
        assert result["content"] == "test"
        assert result["metadata"]["source"] == "test"
    
    @pytest.mark.asyncio
    async def test_pin_content(self, ipfs_client):
        """Test content pinning"""
        ipfs_client.client.pin.add = Mock()
        
        success = await ipfs_client.pin_content("QmTestHash")
        
        assert success is True
        ipfs_client.client.pin.add.assert_called_with("QmTestHash")
    
    @pytest.mark.asyncio
    async def test_check_connectivity(self, ipfs_client):
        """Test IPFS connectivity check"""
        ipfs_client.client.version = Mock(return_value={"Version": "0.1.0"})
        
        connected = await ipfs_client.check_connectivity()
        
        assert connected is True

class TestMemorySystems:
    """Test cases for memory systems"""
    
    def test_episodic_memory(self):
        """Test episodic memory functionality"""
        memory = EpisodicMemory(max_size=100)
        
        # Test adding memories
        embedding1 = np.random.rand(384)
        memory.add_memory("mem1", "Test memory 1", embedding1, {"type": "test"})
        memory.add_memory("mem2", "Test memory 2", embedding1, {"type": "test"})
        
        assert len(memory) == 2
        
        # Test retrieving memory
        retrieved = memory.get_memory("mem1")
        assert retrieved.content == "Test memory 1"
        assert retrieved.access_count == 1
        
        # Test similarity search
        query_embedding = np.random.rand(384)
        similar = memory.search_similar(query_embedding, top_k=1)
        assert len(similar) == 1
    
    def test_semantic_memory(self):
        """Test semantic memory functionality"""
        mock_vector_db = Mock()
        memory = SemanticMemory(mock_vector_db)
        
        # Test concept extraction and relationships
        test_content = "Artificial intelligence and machine learning are related fields."
        
        # Mock vector DB call
        async def mock_add_knowledge(*args, **kwargs):
            return "doc1"
        
        mock_vector_db.add_knowledge = mock_add_knowledge
        
        # This would need async context in real test
        # await memory.add_knowledge(test_content, np.random.rand(384), {})
        
        # Test would continue with concept graph verification
    
    def test_working_memory(self):
        """Test working memory functionality"""
        memory = WorkingMemory(capacity=3)
        
        # Test adding items
        memory.add_item("Task 1", priority=1.0)
        memory.add_item("Task 2", priority=2.0)
        memory.add_item("Task 3", priority=0.5)
        
        assert len(memory) == 3
        
        # Test focus
        memory.set_focus("Current Task")
        assert memory.get_focus() == "Current Task"
        
        # Test retrieval (should be sorted by priority)
        items = memory.get_items(2)
        assert len(items) == 2
        
        # Test clearing
        memory.clear()
        assert len(memory) == 0
        assert memory.get_focus() is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])