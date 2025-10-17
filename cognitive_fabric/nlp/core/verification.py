import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline
import asyncio

@dataclass
class FactCheckingResult:
    claim: str
    is_supported: bool
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    source_attributions: List[Dict[str, Any]]

class AdvancedVerificationSystem:
    """
    Advanced verification system with multiple verification strategies
    """
    
    def __init__(self):
        self.verification_strategies = [
            SemanticVerification(),
            LogicalConsistencyChecker(),
            SourceCredibilityEvaluator(),
            TemporalConsistencyChecker()
        ]
        self.confidence_threshold = 0.7
    
    async def comprehensive_verify(
        self, 
        response: str, 
        context: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive verification using multiple strategies
        """
        
        verification_tasks = []
        for strategy in self.verification_strategies:
            task = strategy.verify(response, context, query)
            verification_tasks.append(task)
        
        # Execute all verification strategies in parallel
        results = await asyncio.gather(*verification_tasks)
        
        # Aggregate results
        aggregated_result = self._aggregate_verification_results(results)
        
        return aggregated_result
    
    def _aggregate_verification_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple verification strategies"""
        
        total_confidence = 0.0
        supported_claims = 0
        total_claims = 0
        strategy_scores = []
        
        for result in results:
            strategy_scores.append(result.get('confidence', 0.0))
            supported_claims += result.get('supported_claims', 0)
            total_claims += result.get('total_claims', 0)
        
        if total_claims > 0:
            claim_support_ratio = supported_claims / total_claims
        else:
            claim_support_ratio = 0.0
        
        overall_confidence = np.mean(strategy_scores) * 0.7 + claim_support_ratio * 0.3
        
        return {
            'overall_confidence': overall_confidence,
            'claim_support_ratio': claim_support_ratio,
            'strategy_scores': strategy_scores,
            'is_verified': overall_confidence >= self.confidence_threshold,
            'verification_details': results
        }

class SemanticVerification:
    """Semantic verification using embedding similarity"""
    
    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
    
    async def verify(
        self, 
        response: str, 
        context: List[Dict[str, Any]], 
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        
        # Extract claims from response
        claims = self._extract_claims(response)
        total_claims = len(claims)
        supported_claims = 0
        
        for claim in claims:
            if await self._claim_supported(claim, context):
                supported_claims += 1
        
        support_ratio = supported_claims / total_claims if total_claims > 0 else 1.0
        
        return {
            'strategy': 'semantic_verification',
            'supported_claims': supported_claims,
            'total_claims': total_claims,
            'confidence': support_ratio,
            'details': {
                'similarity_threshold': self.similarity_threshold,
                'claims_analyzed': claims
            }
        }
    
    async def _claim_supported(self, claim: str, context: List[Dict[str, Any]]) -> bool:
        """Check if claim is supported by context using semantic similarity"""
        
        claim_embedding = await self._get_embedding(claim)
        max_similarity = 0.0
        
        for item in context:
            content = item.get('content', '')
            content_embedding = await self._get_embedding(content)
            
            similarity = self._cosine_similarity(claim_embedding, content_embedding)
            max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= self.similarity_threshold:
                return True
        
        return False
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding (simplified)"""
        # In production, use actual embedding model
        return np.random.rand(384)  # Mock embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from text"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

class LogicalConsistencyChecker:
    """Check logical consistency within response and with context"""
    
    async def verify(
        self, 
        response: str, 
        context: List[Dict[str, Any]], 
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        
        consistency_issues = await self._check_consistency(response, context)
        issue_count = len(consistency_issues)
        
        # Calculate confidence based on issue count
        confidence = max(0.0, 1.0 - (issue_count * 0.2))
        
        return {
            'strategy': 'logical_consistency',
            'confidence': confidence,
            'consistency_issues': consistency_issues,
            'issue_count': issue_count
        }
    
    async def _check_consistency(
        self, 
        response: str, 
        context: List[Dict[str, Any]]
    ) -> List[str]:
        """Check for logical inconsistencies"""
        
        issues = []
        
        # Check for internal contradictions in response
        issues.extend(await self._check_internal_contradictions(response))
        
        # Check for contradictions with context
        issues.extend(await self._check_context_contradictions(response, context))
        
        return issues
    
    async def _check_internal_contradictions(self, response: str) -> List[str]:
        """Check for contradictions within the response itself"""
        issues = []
        
        # Simple contradiction patterns
        contradiction_patterns = [
            (r"(\w+) is (?!not)(\w+)", r"\1 is not \2"),
            (r"(\w+) are (?!not)(\w+)", r"\1 are not \2"),
            (r"always", r"never"),
            (r"all", r"none")
        ]
        
        # This is simplified - in production, use more sophisticated NLP
        sentences = response.split('.')
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                for pos_pattern, neg_pattern in contradiction_patterns:
                    import re
                    if (re.search(pos_pattern, sent1) and re.search(neg_pattern, sent2) or
                        re.search(pos_pattern, sent2) and re.search(neg_pattern, sent1)):
                        issues.append(f"Contradiction between sentences {i+1} and {j+1}")
        
        return issues
    
    async def _check_context_contradictions(
        self, 
        response: str, 
        context: List[Dict[str, Any]]
    ) -> List[str]:
        """Check for contradictions with context"""
        issues = []
        
        # Simplified implementation
        # In production, use semantic similarity and contradiction detection
        response_lower = response.lower()
        
        for item in context:
            context_text = item.get('content', '').lower()
            
            # Simple keyword-based contradiction check
            if self._has_contradictory_phrases(response_lower, context_text):
                issues.append("Potential contradiction with context")
                break
        
        return issues
    
    def _has_contradictory_phrases(self, text1: str, text2: str) -> bool:
        """Check for contradictory phrases between two texts"""
        contradictory_pairs = [
            ("is true", "is false"),
            ("always", "never"),
            ("all", "none"),
            ("increases", "decreases"),
            ("positive", "negative")
        ]
        
        for pos, neg in contradictory_pairs:
            if (pos in text1 and neg in text2) or (pos in text2 and neg in text1):
                return True
        
        return False

class SourceCredibilityEvaluator:
    """Evaluate credibility of sources in context"""
    
    async def verify(
        self, 
        response: str, 
        context: List[Dict[str, Any]], 
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        
        credibility_scores = []
        for item in context:
            score = await self._evaluate_source_credibility(item)
            credibility_scores.append(score)
        
        avg_credibility = np.mean(credibility_scores) if credibility_scores else 0.0
        
        return {
            'strategy': 'source_credibility',
            'confidence': avg_credibility,
            'source_scores': credibility_scores,
            'average_credibility': avg_credibility
        }
    
    async def _evaluate_source_credibility(self, source: Dict[str, Any]) -> float:
        """Evaluate credibility of a single source"""
        
        base_score = source.get('verification_score', 0.5)
        metadata = source.get('metadata', {})
        
        # Factor in source reputation
        source_reputation = metadata.get('reputation', 0.5)
        
        # Factor in timeliness
        timeliness_score = self._evaluate_timeliness(metadata)
        
        # Combine scores
        credibility = (
            base_score * 0.5 +
            source_reputation * 0.3 +
            timeliness_score * 0.2
        )
        
        return credibility
    
    def _evaluate_timeliness(self, metadata: Dict[str, Any]) -> float:
        """Evaluate timeliness of the source"""
        # Check if source has timestamp
        timestamp = metadata.get('timestamp')
        if not timestamp:
            return 0.5  # Neutral score
        
        # Calculate age-based score (simplified)
        # In production, use actual timestamp parsing
        return 0.8  # Mock score

class TemporalConsistencyChecker:
    """Check temporal consistency of information"""
    
    async def verify(
        self, 
        response: str, 
        context: List[Dict[str, Any]], 
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        
        temporal_issues = await self._check_temporal_consistency(response, context)
        issue_count = len(temporal_issues)
        
        confidence = max(0.0, 1.0 - (issue_count * 0.3))
        
        return {
            'strategy': 'temporal_consistency',
            'confidence': confidence,
            'temporal_issues': temporal_issues,
            'issue_count': issue_count
        }
    
    async def _check_temporal_consistency(
        self, 
        response: str, 
        context: List[Dict[str, Any]]
    ) -> List[str]:
        """Check for temporal inconsistencies"""
        
        issues = []
        
        # Extract temporal references from response
        response_times = self._extract_temporal_references(response)
        
        # Check context for temporal information
        for item in context:
            context_times = self._extract_temporal_references(item.get('content', ''))
            
            # Compare temporal references
            for resp_time in response_times:
                for ctx_time in context_times:
                    if self._are_temporally_inconsistent(resp_time, ctx_time):
                        issues.append(f"Temporal inconsistency: {resp_time} vs {ctx_time}")
        
        return issues
    
    def _extract_temporal_references(self, text: str) -> List[str]:
        """Extract temporal references from text"""
        import re
        
        temporal_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:today|yesterday|tomorrow|now|currently)\b',
            r'\b(?:last|next)\s+(?:week|month|year)\b',
            r'\b\d+\s+(?:days|weeks|months|years)\s+ago\b'
        ]
        
        times = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        return times
    
    def _are_temporally_inconsistent(self, time1: str, time2: str) -> bool:
        """Check if two temporal references are inconsistent"""
        # Simplified implementation
        # In production, use proper temporal reasoning
        time1_lower = time1.lower()
        time2_lower = time2.lower()
        
        # Simple contradiction patterns
        contradictory_pairs = [
            ("today", "yesterday"),
            ("last year", "next year"),
            ("now", "then")
        ]
        
        for pair in contradictory_pairs:
            if (pair[0] in time1_lower and pair[1] in time2_lower) or \
               (pair[1] in time1_lower and pair[0] in time2_lower):
                return True
        
        return False