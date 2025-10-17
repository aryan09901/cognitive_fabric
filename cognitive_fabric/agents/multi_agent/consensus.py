import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class ConsensusResult:
    final_decision: Any
    confidence: float
    agreement_level: float
    participant_votes: Dict[str, Any]
    winning_strategy: str

class ConsensusMechanism:
    """
    Consensus mechanism for multi-agent decision making
    """
    
    def __init__(self):
        self.consensus_strategies = {
            'majority_vote': self._majority_vote,
            'weighted_average': self._weighted_average,
            'confidence_based': self._confidence_based,
            'reputation_weighted': self._reputation_weighted
        }
    
    async def reach_consensus(self, 
                            agent_responses: List[Dict[str, Any]],
                            strategy: str = 'confidence_based',
                            agent_reputations: Optional[Dict[str, float]] = None) -> ConsensusResult:
        """
        Reach consensus among multiple agent responses
        """
        if not agent_responses:
            return ConsensusResult(
                final_decision=None,
                confidence=0.0,
                agreement_level=0.0,
                participant_votes={},
                winning_strategy=strategy
            )
        
        if strategy not in self.consensus_strategies:
            strategy = 'confidence_based'
        
        consensus_func = self.consensus_strategies[strategy]
        result = await consensus_func(agent_responses, agent_reputations)
        
        return result
    
    async def _majority_vote(self, 
                           responses: List[Dict[str, Any]],
                           reputations: Optional[Dict[str, float]] = None) -> ConsensusResult:
        """Majority voting consensus"""
        # Group responses by content
        response_counts = {}
        response_confidences = {}
        
        for response in responses:
            content = response.get('response', '')
            confidence = response.get('confidence', 0.5)
            agent_id = response.get('agent_id', 'unknown')
            
            if content not in response_counts:
                response_counts[content] = 0
                response_confidences[content] = []
            
            response_counts[content] += 1
            response_confidences[content].append(confidence)
        
        if not response_counts:
            return self._create_default_consensus(responses)
        
        # Find majority
        max_count = max(response_counts.values())
        majority_responses = [resp for resp, count in response_counts.items() if count == max_count]
        
        if len(majority_responses) == 1:
            # Clear majority
            final_decision = majority_responses[0]
            avg_confidence = np.mean(response_confidences[final_decision])
            agreement_level = max_count / len(responses)
        else:
            # Tie - use highest average confidence
            best_response = None
            best_confidence = -1
            
            for response in majority_responses:
                avg_conf = np.mean(response_confidences[response])
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_response = response
            
            final_decision = best_response
            avg_confidence = best_confidence
            agreement_level = max_count / len(responses)
        
        participant_votes = {
            resp['agent_id']: {
                'vote': resp.get('response'),
                'confidence': resp.get('confidence', 0.5)
            }
            for resp in responses
        }
        
        return ConsensusResult(
            final_decision=final_decision,
            confidence=avg_confidence,
            agreement_level=agreement_level,
            participant_votes=participant_votes,
            winning_strategy='majority_vote'
        )
    
    async def _weighted_average(self,
                              responses: List[Dict[str, Any]],
                              reputations: Optional[Dict[str, float]] = None) -> ConsensusResult:
        """Weighted average consensus based on confidence"""
        if not responses:
            return self._create_default_consensus(responses)
        
        # Convert responses to numerical scores (simplified)
        response_scores = {}
        
        for response in responses:
            content = response.get('response', '')
            confidence = response.get('confidence', 0.5)
            agent_id = response.get('agent_id', 'unknown')
            
            # Simple content scoring (in production, use embeddings)
            content_hash = hash(content)
            
            if content_hash not in response_scores:
                response_scores[content_hash] = {
                    'content': content,
                    'total_score': 0.0,
                    'total_confidence': 0.0,
                    'count': 0
                }
            
            # Weight by confidence
            weight = confidence
            if reputations and agent_id in reputations:
                weight *= (reputations[agent_id] / 100.0)  # Normalize reputation
            
            response_scores[content_hash]['total_score'] += weight
            response_scores[content_hash]['total_confidence'] += confidence
            response_scores[content_hash]['count'] += 1
        
        # Find response with highest weighted score
        best_score = -1
        best_response = None
        
        for score_data in response_scores.values():
            avg_score = score_data['total_score'] / score_data['count']
            if avg_score > best_score:
                best_score = avg_score
                best_response = score_data['content']
                avg_confidence = score_data['total_confidence'] / score_data['count']
        
        participant_votes = {
            resp['agent_id']: {
                'vote': resp.get('response'),
                'confidence': resp.get('confidence', 0.5),
                'weight': resp.get('confidence', 0.5) * (reputations.get(resp['agent_id'], 50) / 100.0 if reputations else 1.0)
            }
            for resp in responses
        }
        
        return ConsensusResult(
            final_decision=best_response,
            confidence=avg_confidence,
            agreement_level=best_score,
            participant_votes=participant_votes,
            winning_strategy='weighted_average'
        )
    
    async def _confidence_based(self,
                              responses: List[Dict[str, Any]],
                              reputations: Optional[Dict[str, float]] = None) -> ConsensusResult:
        """Confidence-based consensus"""
        if not responses:
            return self._create_default_consensus(responses)
        
        # Select response with highest confidence
        best_confidence = -1
        best_response = None
        best_agent = None
        
        for response in responses:
            confidence = response.get('confidence', 0.5)
            if confidence > best_confidence:
                best_confidence = confidence
                best_response = response.get('response')
                best_agent = response.get('agent_id')
        
        # Calculate agreement level (fraction of agents with similar confidence)
        high_confidence_count = sum(1 for resp in responses 
                                  if resp.get('confidence', 0) >= best_confidence * 0.8)
        agreement_level = high_confidence_count / len(responses)
        
        participant_votes = {
            resp['agent_id']: {
                'vote': resp.get('response'),
                'confidence': resp.get('confidence', 0.5),
                'selected': resp.get('agent_id') == best_agent
            }
            for resp in responses
        }
        
        return ConsensusResult(
            final_decision=best_response,
            confidence=best_confidence,
            agreement_level=agreement_level,
            participant_votes=participant_votes,
            winning_strategy='confidence_based'
        )
    
    async def _reputation_weighted(self,
                                 responses: List[Dict[str, Any]],
                                 reputations: Optional[Dict[str, float]] = None) -> ConsensusResult:
        """Reputation-weighted consensus"""
        if not responses:
            return self._create_default_consensus(responses)
        
        if not reputations:
            # Fall back to confidence-based if no reputations
            return await self._confidence_based(responses, reputations)
        
        response_scores = {}
        
        for response in responses:
            content = response.get('response', '')
            agent_id = response.get('agent_id', 'unknown')
            confidence = response.get('confidence', 0.5)
            
            reputation = reputations.get(agent_id, 50)  # Default to 50 if unknown
            weight = (reputation / 100.0) * confidence
            
            content_hash = hash(content)
            
            if content_hash not in response_scores:
                response_scores[content_hash] = {
                    'content': content,
                    'total_weight': 0.0,
                    'total_confidence': 0.0,
                    'count': 0
                }
            
            response_scores[content_hash]['total_weight'] += weight
            response_scores[content_hash]['total_confidence'] += confidence
            response_scores[content_hash]['count'] += 1
        
        # Find response with highest reputation-weighted score
        best_weight = -1
        best_response = None
        
        for score_data in response_scores.values():
            avg_weight = score_data['total_weight'] / score_data['count']
            if avg_weight > best_weight:
                best_weight = avg_weight
                best_response = score_data['content']
                avg_confidence = score_data['total_confidence'] / score_data['count']
        
        participant_votes = {
            resp['agent_id']: {
                'vote': resp.get('response'),
                'confidence': resp.get('confidence', 0.5),
                'reputation': reputations.get(resp['agent_id'], 50),
                'weight': (reputations.get(resp['agent_id'], 50) / 100.0) * resp.get('confidence', 0.5)
            }
            for resp in responses
        }
        
        return ConsensusResult(
            final_decision=best_response,
            confidence=avg_confidence,
            agreement_level=best_weight,
            participant_votes=participant_votes,
            winning_strategy='reputation_weighted'
        )
    
    def _create_default_consensus(self, responses: List[Dict[str, Any]]) -> ConsensusResult:
        """Create default consensus result when no responses"""
        return ConsensusResult(
            final_decision=None,
            confidence=0.0,
            agreement_level=0.0,
            participant_votes={},
            winning_strategy='default'
        )
    
    async def evaluate_consensus_quality(self, consensus_result: ConsensusResult) -> Dict[str, float]:
        """Evaluate the quality of consensus result"""
        quality_metrics = {
            'confidence_score': consensus_result.confidence,
            'agreement_score': consensus_result.agreement_level,
            'participation_score': len(consensus_result.participant_votes) / 10.0,  # Normalize
            'decision_quality': consensus_result.confidence * consensus_result.agreement_level
        }
        
        # Overall quality score
        quality_metrics['overall_quality'] = (
            quality_metrics['confidence_score'] * 0.4 +
            quality_metrics['agreement_score'] * 0.3 +
            quality_metrics['participation_score'] * 0.2 +
            quality_metrics['decision_quality'] * 0.1
        )
        
        return quality_metrics

# Global consensus instance
consensus_mechanism = ConsensusMechanism()