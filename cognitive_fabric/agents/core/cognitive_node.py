import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn as nn

from config.base import config
from ..reinforcement.policies import PPOPolicy
from ..memory_systems import EpisodicMemory, SemanticMemory
from knowledge.vector_db import VectorDatabase
from blockchain.core.contracts import blockchain_client

@dataclass
class AgentResponse:
    response: str
    sources: List[Dict]
    confidence: float
    verification_score: float
    metadata: Dict[str, Any]

class CognitiveNode:
    """
    Core Cognitive Agent Node implementing RAG + RL + Blockchain
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.reputation = config.INITIAL_REPUTATION
        
        # Initialize components
        self.llm = self._init_llm()
        self.embedding_model = self._init_embedding_model()
        self.vector_db = VectorDatabase()
        self.rl_policy = PPOPolicy(
            state_dim=512,
            action_dim=256,
            learning_rate=config.RL_LEARNING_RATE
        )
        
        # Memory systems
        self.episodic_memory = EpisodicMemory(max_size=config.MAX_AGENT_MEMORY)
        self.semantic_memory = SemanticMemory(vector_db=self.vector_db)
        
        # Blockchain identity
        self.blockchain_address = None
        self._register_on_blockchain()
        
        # Training state
        self.training_buffer = []
        self.interaction_count = 0
        
    def _init_llm(self):
        """Initialize the language model with optimized settings"""
        model_name = self.config.get('LLM_MODEL', config.LLM_MODEL)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'generator': generator
            }
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            # Fallback to a smaller model
            return self._init_fallback_llm()
    
    def _init_fallback_llm(self):
        """Initialize fallback LLM"""
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'generator': generator
        }
    
    def _init_embedding_model(self):
        """Initialize embedding model for semantic search"""
        from sentence_transformers import SentenceTransformer
        
        model_name = self.config.get('EMBEDDING_MODEL', config.EMBEDDING_MODEL)
        return SentenceTransformer(model_name)
    
    def _register_on_blockchain(self):
        """Register agent on blockchain"""
        try:
            metadata_ipfs_hash = self._upload_agent_metadata()
            tx_hash = blockchain_client.register_agent(metadata_ipfs_hash)
            self.blockchain_address = blockchain_client.account.address
            print(f"Agent {self.agent_id} registered on blockchain: {tx_hash}")
        except Exception as e:
            print(f"Failed to register on blockchain: {e}")
            # Continue without blockchain for development
            self.blockchain_address = "0x" + "0" * 40
    
    def _upload_agent_metadata(self) -> str:
        """Upload agent metadata to IPFS and return hash"""
        # Implementation depends on IPFS client
        return "QmMockMetadataHash"
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Main processing pipeline: RAG + RL + Verification
        """
        # Step 1: Semantic retrieval
        retrieved_context = await self.retrieve_relevant_knowledge(query)
        
        # Step 2: RL-based decision making
        action = self._select_action(query, retrieved_context)
        
        # Step 3: Generate response with verification
        response = self._generate_verified_response(query, retrieved_context, action)
        
        # Step 4: Update memory and learn
        await self._learn_from_interaction(query, response, retrieved_context)
        
        # Step 5: Record on blockchain
        if self.blockchain_address and context and 'to_agent' in context:
            await self._record_blockchain_interaction(
                context['to_agent'], query, response.response
            )
        
        return response
    
    async def retrieve_relevant_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """RAG: Retrieve relevant knowledge using semantic search"""
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search vector database
            results = await self.vector_db.similarity_search(
                embedding=query_embedding,
                top_k=top_k,
                filters={"min_verification_score": 0.7}  # Only verified knowledge
            )
            
            # Enhance with blockchain verification
            verified_results = []
            for result in results:
                if await self._verify_knowledge_on_blockchain(result):
                    verified_results.append(result)
            
            return verified_results
            
        except Exception as e:
            print(f"Knowledge retrieval failed: {e}")
            return []
    
    async def _verify_knowledge_on_blockchain(self, knowledge_item: Dict) -> bool:
        """Verify knowledge item on blockchain"""
        try:
            if not self.blockchain_address:
                return True  # Skip verification in development
            
            # Check if knowledge is registered and verified on blockchain
            # This would involve checking the knowledge hash on chain
            return True  # Simplified for example
            
        except Exception as e:
            print(f"Blockchain verification failed: {e}")
            return False
    
    def _select_action(self, query: str, context: List[Dict]) -> torch.Tensor:
        """RL: Select action based on current state"""
        state = self._encode_state(query, context)
        action_probs = self.rl_policy(state)
        action = torch.multinomial(action_probs, 1)
        return action
    
    def _encode_state(self, query: str, context: List[Dict]) -> torch.Tensor:
        """Encode current state for RL policy"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Encode context
        context_embeddings = []
        for item in context:
            if 'embedding' in item:
                context_embeddings.append(item['embedding'])
        
        if context_embeddings:
            context_embedding = np.mean(context_embeddings, axis=0)
        else:
            context_embedding = np.zeros(384)  # Default embedding size
        
        # Combine features
        state_features = np.concatenate([
            query_embedding,
            context_embedding,
            [self.reputation / 100.0],  # Normalized reputation
            [len(context) / 10.0]  # Normalized context size
        ])
        
        # Pad or truncate to fixed size
        if len(state_features) < 512:
            state_features = np.pad(state_features, (0, 512 - len(state_features)))
        else:
            state_features = state_features[:512]
        
        return torch.FloatTensor(state_features)
    
    def _generate_verified_response(
        self, 
        query: str, 
        context: List[Dict], 
        action: torch.Tensor
    ) -> AgentResponse:
        """Generate response with verification against context"""
        
        # Build context-aware prompt
        prompt = self._build_verification_prompt(query, context, action)
        
        try:
            # Generate response
            response = self.llm['generator'](
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm['tokenizer'].eos_token_id
            )[0]['generated_text']
            
            # Extract the actual response
            response_text = response.split("ASSISTANT:")[-1].strip()
            
            # Calculate verification score
            verification_score = self._calculate_verification_score(response_text, context)
            
            # Build sources list
            sources = [
                {
                    'content': item.get('content', ''),
                    'metadata': item.get('metadata', {}),
                    'verification_score': item.get('verification_score', 0.0)
                }
                for item in context
            ]
            
            return AgentResponse(
                response=response_text,
                sources=sources,
                confidence=float(action.mean()),
                verification_score=verification_score,
                metadata={
                    'context_used': len(context),
                    'generation_method': 'verified_rag',
                    'model': self.config.LLM_MODEL
                }
            )
            
        except Exception as e:
            print(f"Response generation failed: {e}")
            return self._generate_fallback_response(query)
    
    def _build_verification_prompt(self, query: str, context: List[Dict], action: torch.Tensor) -> str:
        """Build verification-focused prompt"""
        
        context_text = ""
        for i, item in enumerate(context):
            context_text += f"Source {i+1}: {item.get('content', '')}\n"
            if 'metadata' in item:
                context_text += f"Metadata: {item['metadata']}\n"
            context_text += "\n"
        
        prompt = f"""You are a verified AI agent in a decentralized cognitive network. 
Your reputation depends on providing accurate, verifiable information.

CONTEXT FROM VERIFIED KNOWLEDGE BASE:
{context_text}

USER QUERY: {query}

INSTRUCTIONS:
1. Answer based ONLY on the verified context above
2. If context doesn't contain relevant information, say "I cannot verify this information"
3. Be precise and cite sources when possible
4. Maintain helpful but truthful tone

ASSISTANT: Based on the verified knowledge, """
        
        return prompt
    
    def _calculate_verification_score(self, response: str, context: List[Dict]) -> float:
        """Calculate how verifiable the response is from context"""
        if not response or not context:
            return 0.0
        
        # Simple semantic similarity check
        response_embedding = self.embedding_model.encode([response])[0]
        
        max_similarity = 0.0
        for item in context:
            if 'embedding' in item:
                similarity = np.dot(response_embedding, item['embedding']) / (
                    np.linalg.norm(response_embedding) * np.linalg.norm(item['embedding'])
                )
                max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
    
    def _generate_fallback_response(self, query: str) -> AgentResponse:
        """Generate fallback response when main generation fails"""
        return AgentResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again later.",
            sources=[],
            confidence=0.1,
            verification_score=0.0,
            metadata={'fallback': True, 'error': 'generation_failed'}
        )
    
    async def _learn_from_interaction(
        self, 
        query: str, 
        response: AgentResponse, 
        context: List[Dict]
    ):
        """RL: Learn from interaction and update policy"""
        
        # Calculate reward
        reward = self._calculate_reward(response, context)
        
        # Get state for learning
        state = self._encode_state(query, context)
        
        # Store in training buffer
        self.training_buffer.append({
            'state': state,
            'reward': reward,
            'response_quality': response.confidence,
            'verification_score': response.verification_score
        })
        
        # Train periodically
        self.interaction_count += 1
        if self.interaction_count % 100 == 0:  # Train every 100 interactions
            await self._train_policy()
    
    def _calculate_reward(self, response: AgentResponse, context: List[Dict]) -> float:
        """Calculate RL reward based on response quality"""
        
        base_reward = 0.0
        
        # Reward for verification
        base_reward += response.verification_score * 2.0
        
        # Reward for confidence (if verified)
        if response.verification_score > 0.7:
            base_reward += response.confidence
        
        # Penalty for no context but high confidence
        if not context and response.confidence > 0.8:
            base_reward -= 1.0
        
        # Reward for using multiple sources
        if len(context) > 1:
            base_reward += 0.5
        
        return float(base_reward)
    
    async def _train_policy(self):
        """Train RL policy on accumulated experience"""
        if len(self.training_buffer) < 32:  # Minimum batch size
            return
        
        # Convert to training batch
        states = torch.stack([item['state'] for item in self.training_buffer])
        rewards = torch.FloatTensor([item['reward'] for item in self.training_buffer])
        
        # Update policy
        loss = self.rl_policy.update(states, rewards)
        
        # Clear buffer
        self.training_buffer = []
        
        print(f"Policy updated with loss: {loss:.4f}")
    
    async def _record_blockchain_interaction(
        self, 
        to_agent: str, 
        query: str, 
        response: str
    ):
        """Record interaction on blockchain"""
        try:
            if not self.blockchain_address:
                return
            
            # In production, you'd upload query/response to IPFS first
            query_hash = f"Qm{hash(query)}"  # Mock IPFS hash
            response_hash = f"Qm{hash(response)}"  # Mock IPFS hash
            
            # Calculate satisfaction score (simplified)
            satisfaction_score = min(int(self.reputation), 100)
            
            tx_hash = blockchain_client.record_interaction(
                to_agent=to_agent,
                query_hash=query_hash,
                response_hash=response_hash,
                satisfaction_score=satisfaction_score
            )
            
            print(f"Interaction recorded on blockchain: {tx_hash}")
            
        except Exception as e:
            print(f"Failed to record interaction on blockchain: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        return {
            'agent_id': self.agent_id,
            'reputation': self.reputation,
            'blockchain_address': self.blockchain_address,
            'interaction_count': self.interaction_count,
            'memory_size': len(self.episodic_memory),
            'model': self.config.LLM_MODEL,
            'training_buffer_size': len(self.training_buffer)
        }