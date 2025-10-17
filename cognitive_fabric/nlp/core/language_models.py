import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Pipeline,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from sentence_transformers import SentenceTransformer
import asyncio
from dataclasses import dataclass
import logging

from config.base import config

logger = logging.getLogger(__name__)

class VerificationStoppingCriteria(StoppingCriteria):
    """Stop generation when verification markers are detected"""
    
    def __init__(self, tokenizer, verification_phrases):
        self.tokenizer = tokenizer
        self.verification_phrases = verification_phrases
        self.verification_ids = [
            tokenizer.encode(phrase, add_special_tokens=False) 
            for phrase in verification_phrases
        ]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        # Check if any verification phrase appears in generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(phrase in generated_text for phrase in self.verification_phrases)

@dataclass
class GenerationConfig:
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True

class AdvancedLanguageModel:
    """
    Advanced language model with verification, fact-checking, and multi-modal capabilities
    """
    
    def __init__(self, model_name: str = None, device: str = "auto"):
        self.model_name = model_name or config.LLM_MODEL
        self.device = device
        self.verification_engine = VerificationEngine()
        
        # Initialize models
        self.model, self.tokenizer, self.generator = self._initialize_models()
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Cache for frequent queries
        self.response_cache = {}
        self.embedding_cache = {}
        
        logger.info(f"Initialized AdvancedLanguageModel with {self.model_name}")
    
    def _initialize_models(self) -> Tuple[Any, Any, Pipeline]:
        """Initialize language models with optimized settings"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="./offload"
            )
            
            # Create text generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map=self.device,
                return_full_text=False
            )
            
            return model, tokenizer, generator
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    async def generate_verified_response(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        generation_config: Optional[GenerationConfig] = None,
        verification_level: str = "strict"
    ) -> Dict[str, Any]:
        """
        Generate response with integrated verification against context
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Build verification-aware prompt
        enhanced_prompt = self._build_verification_prompt(prompt, context, verification_level)
        
        try:
            # Generate with verification constraints
            response = await self._generate_with_verification(
                enhanced_prompt, 
                generation_config,
                verification_level
            )
            
            # Extract and clean response
            clean_response = self._extract_response(response)
            
            # Verify against context
            verification_results = await self.verification_engine.verify_response(
                clean_response, context
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                clean_response, context, verification_results
            )
            
            return {
                'response': clean_response,
                'verification_score': verification_results.overall_score,
                'confidence': confidence_scores.overall_confidence,
                'fact_checking': verification_results.fact_checking_results,
                'sources_used': verification_results.verified_sources,
                'generation_metadata': {
                    'model': self.model_name,
                    'verification_level': verification_level,
                    'context_utilization': len(verification_results.verified_sources),
                    'prompt_tokens': len(self.tokenizer.encode(enhanced_prompt)),
                    'response_tokens': len(self.tokenizer.encode(clean_response))
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return await self._generate_fallback_response(prompt)
    
    def _build_verification_prompt(
        self, 
        prompt: str, 
        context: List[Dict[str, Any]], 
        verification_level: str
    ) -> str:
        """Build verification-focused prompt with context"""
        
        context_section = ""
        for i, item in enumerate(context):
            source_info = f"Source {i+1}"
            if 'source' in item.get('metadata', {}):
                source_info += f" ({item['metadata']['source']})"
            if 'verification_score' in item:
                source_info += f" [Verification: {item['verification_score']:.2f}]"
            
            context_section += f"{source_info}:\n{item.get('content', '')}\n\n"
        
        verification_instructions = self._get_verification_instructions(verification_level)
        
        system_prompt = f"""You are a verified AI agent in a decentralized cognitive network. 
Your responses must be accurate, truthful, and based ONLY on the verified context provided.

VERIFIED CONTEXT:
{context_section}

USER QUERY: {prompt}

VERIFICATION INSTRUCTIONS:
{verification_instructions}

RESPONSE GUIDELINES:
1. Base your answer SOLELY on the verified context above
2. If the context doesn't contain relevant information, explicitly state this
3. Cite specific sources when making claims
4. Maintain a helpful but strictly truthful tone
5. If uncertain, acknowledge the limitations

ASSISTANT: Based on the verified knowledge, """
        
        return system_prompt
    
    def _get_verification_instructions(self, level: str) -> str:
        """Get verification instructions based on level"""
        instructions = {
            "strict": """
            - ONLY use information from the verified context
            - DO NOT add any external knowledge or assumptions
            - Explicitly state when information is not available in context
            - Use precise citations for all claims
            """,
            "moderate": """
            - Primarily use verified context
            - You may supplement with general knowledge if clearly indicated
            - Clearly distinguish between verified and general knowledge
            - Acknowledge uncertainty when present
            """,
            "flexible": """
            - Use verified context as primary source
            - You may use reasoning and general knowledge
            - Indicate when information comes from general knowledge
            - Maintain overall truthfulness
            """
        }
        return instructions.get(level, instructions["moderate"])
    
    async def _generate_with_verification(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        verification_level: str
    ) -> str:
        """Generate response with verification constraints"""
        
        # Add verification stopping criteria for strict mode
        stopping_criteria = None
        if verification_level == "strict":
            verification_phrases = [
                "I cannot verify",
                "The context doesn't contain",
                "No information available"
            ]
            stopping_criteria = StoppingCriteriaList([
                VerificationStoppingCriteria(self.tokenizer, verification_phrases)
            ])
        
        # Generate response
        response = self.generator(
            prompt,
            max_new_tokens=generation_config.max_length,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            repetition_penalty=generation_config.repetition_penalty,
            do_sample=generation_config.do_sample,
            num_beams=generation_config.num_beams,
            early_stopping=generation_config.early_stopping,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return response[0]['generated_text']
    
    def _extract_response(self, generated_text: str) -> str:
        """Extract and clean the response from generated text"""
        # Remove the prompt if it's included
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text.strip()
        
        # Clean up any trailing incomplete sentences
        if response and response[-1] not in ['.', '!', '?']:
            last_sentence_end = max(
                response.rfind('.'),
                response.rfind('!'),
                response.rfind('?')
            )
            if last_sentence_end > len(response) * 0.5:  # Keep if most is complete
                response = response[:last_sentence_end + 1]
        
        return response
    
    async def _generate_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """Generate fallback response when main generation fails"""
        fallback_response = "I apologize, but I'm experiencing technical difficulties accessing verified knowledge. Please try again later or rephrase your question."
        
        return {
            'response': fallback_response,
            'verification_score': 0.0,
            'confidence': 0.1,
            'fact_checking': {},
            'sources_used': [],
            'generation_metadata': {
                'model': 'fallback',
                'verification_level': 'none',
                'context_utilization': 0,
                'fallback_used': True
            }
        }
    
    def _calculate_confidence_scores(
        self,
        response: str,
        context: List[Dict[str, Any]],
        verification_results: Any
    ) -> Any:
        """Calculate various confidence scores for the response"""
        
        # Calculate semantic similarity with context
        response_embedding = self.embedding_model.encode([response])[0]
        context_similarities = []
        
        for item in context:
            if 'embedding' in item:
                similarity = self._cosine_similarity(response_embedding, item['embedding'])
                context_similarities.append(similarity)
        
        avg_similarity = np.mean(context_similarities) if context_similarities else 0.0
        
        # Calculate response coherence (simple proxy)
        coherence_score = min(len(response.split()) / 100, 1.0)  # Longer responses tend to be more coherent
        
        # Combine scores
        overall_confidence = (
            verification_results.overall_score * 0.6 +
            avg_similarity * 0.3 +
            coherence_score * 0.1
        )
        
        return type('ConfidenceScores', (), {
            'overall_confidence': overall_confidence,
            'semantic_similarity': avg_similarity,
            'coherence_score': coherence_score,
            'verification_based_confidence': verification_results.overall_score
        })()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    async def batch_process_queries(
        self, 
        queries: List[str], 
        contexts: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in batch for efficiency"""
        
        tasks = []
        for query, context in zip(queries, contexts):
            task = self.generate_verified_response(query, context)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'vocabulary_size': len(self.tokenizer),
            'device': str(self.model.device),
            'dtype': str(self.model.dtype),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }

class VerificationEngine:
    """
    Advanced verification engine for fact-checking and source validation
    """
    
    def __init__(self):
        self.similarity_threshold = 0.7
        self.contradiction_threshold = 0.3
    
    async def verify_response(
        self, 
        response: str, 
        context: List[Dict[str, Any]]
    ) -> Any:
        """Verify response against context and calculate scores"""
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        verification_results = {
            'verified_claims': [],
            'unverified_claims': [],
            'contradictions': [],
            'source_attributions': []
        }
        
        total_claims = len(claims)
        verified_count = 0
        
        for claim in claims:
            claim_verification = await self._verify_single_claim(claim, context)
            
            if claim_verification['verified']:
                verification_results['verified_claims'].append(claim_verification)
                verified_count += 1
            elif claim_verification['contradiction']:
                verification_results['contradictions'].append(claim_verification)
            else:
                verification_results['unverified_claims'].append(claim_verification)
        
        # Calculate overall verification score
        overall_score = verified_count / total_claims if total_claims > 0 else 0.0
        
        # Penalize for contradictions
        contradiction_penalty = len(verification_results['contradictions']) * 0.2
        overall_score = max(0.0, overall_score - contradiction_penalty)
        
        return type('VerificationResults', (), {
            'overall_score': overall_score,
            'verified_claims': verification_results['verified_claims'],
            'unverified_claims': verification_results['unverified_claims'],
            'contradictions': verification_results['contradictions'],
            'verified_sources': list(set(
                vc['source'] for vc in verification_results['verified_claims'] 
                if vc.get('source')
            )),
            'fact_checking_results': verification_results
        })()
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from response text"""
        # Simple sentence-based claim extraction
        import re
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]  # Minimum length
        
        # Further split on conjunctions
        detailed_claims = []
        for claim in claims:
            sub_claims = re.split(r'[,;]', claim)
            detailed_claims.extend([sc.strip() for sc in sub_claims if len(sc.strip()) > 5])
        
        return detailed_claims
    
    async def _verify_single_claim(
        self, 
        claim: str, 
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify a single claim against context"""
        
        best_similarity = 0.0
        best_match = None
        contradiction_found = False
        
        for source in context:
            source_text = source.get('content', '')
            similarity = self._semantic_similarity(claim, source_text)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = source
            
            # Check for contradiction
            if self._check_contradiction(claim, source_text):
                contradiction_found = True
        
        verified = best_similarity >= self.similarity_threshold
        
        return {
            'claim': claim,
            'verified': verified,
            'contradiction': contradiction_found,
            'similarity_score': best_similarity,
            'source': best_match.get('metadata', {}).get('source', 'unknown') if best_match else None,
            'source_verification_score': best_match.get('verification_score', 0.0) if best_match else 0.0
        }
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # This would use the embedding model in production
        # For now, use a simple word overlap measure
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _check_contradiction(self, claim: str, context_text: str) -> bool:
        """Check if claim contradicts context"""
        # Simple contradiction detection based on negative phrases
        negative_indicators = [
            "not true", "incorrect", "wrong", "false", "misleading", 
            "contradicts", "opposite", "never", "cannot", "will not"
        ]
        
        claim_lower = claim.lower()
        context_lower = context_text.lower()
        
        # Check if context contains negation of claim
        for indicator in negative_indicators:
            if indicator in context_lower and any(
                word in context_lower for word in claim_lower.split()[:5]  # First few words
            ):
                return True
        
        return False