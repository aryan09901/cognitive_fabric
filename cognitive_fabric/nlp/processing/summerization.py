import torch
from transformers import pipeline
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextSummarizer:
    """
    Advanced text summarization for knowledge compression
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info(f"Loaded summarization model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            self.summarizer = None
    
    def summarize_text(self, 
                      text: str, 
                      max_length: int = 150, 
                      min_length: int = 30,
                      compression_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Summarize text with configurable compression
        """
        if not self.summarizer or not text.strip():
            return {
                'summary': text[:max_length] + '...' if len(text) > max_length else text,
                'compression_ratio': 1.0,
                'success': False if not self.summarizer else True
            }
        
        try:
            # Calculate dynamic length based on compression ratio
            word_count = len(text.split())
            target_length = max(min_length, min(max_length, int(word_count * compression_ratio)))
            
            # Generate summary
            summary_result = self.summarizer(
                text,
                max_length=target_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            original_length = len(text)
            compressed_length = len(summary)
            actual_ratio = compressed_length / original_length if original_length > 0 else 1.0
            
            return {
                'summary': summary,
                'compression_ratio': actual_ratio,
                'original_length': original_length,
                'compressed_length': compressed_length,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: extract first few sentences
            sentences = text.split('. ')
            fallback_summary = '. '.join(sentences[:3]) + '.'
            
            return {
                'summary': fallback_summary,
                'compression_ratio': len(fallback_summary) / len(text),
                'success': False,
                'error': str(e)
            }
    
    def summarize_batch(self, 
                       texts: List[str], 
                       max_length: int = 150,
                       min_length: int = 30) -> List[Dict[str, Any]]:
        """
        Summarize multiple texts in batch
        """
        results = []
        for text in texts:
            result = self.summarize_text(text, max_length, min_length)
            results.append(result)
        
        return results
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from text (simplified implementation)
        """
        # Simple sentence scoring based on position and keywords
        sentences = text.split('. ')
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            score = 0
            
            # Score based on position (first sentences are often important)
            position_score = max(0, 1.0 - (i * 0.1))
            score += position_score
            
            # Score based on keywords
            important_keywords = ['important', 'key', 'summary', 'conclusion', 'therefore', 'however']
            keyword_count = sum(1 for keyword in important_keywords if keyword in sentence.lower())
            score += keyword_count * 0.2
            
            # Score based on length (medium-length sentences are often more informative)
            word_count = len(sentence.split())
            if 5 <= word_count <= 25:
                score += 0.3
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        key_points = [sentence for sentence, score in scored_sentences[:num_points]]
        
        return key_points

class KnowledgeCompressor:
    """
    Compress knowledge for efficient storage and retrieval
    """
    
    def __init__(self):
        self.summarizer = TextSummarizer()
    
    def compress_knowledge(self, 
                          knowledge_items: List[Dict[str, Any]],
                          target_compression: float = 0.5) -> List[Dict[str, Any]]:
        """
        Compress multiple knowledge items
        """
        compressed_items = []
        
        for item in knowledge_items:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            
            if len(content) > 1000:  # Only compress long content
                summary_result = self.summarizer.summarize_text(
                    content, 
                    compression_ratio=target_compression
                )
                
                compressed_item = {
                    'content': summary_result['summary'],
                    'metadata': {
                        **metadata,
                        'compressed': True,
                        'original_length': summary_result['original_length'],
                        'compressed_length': summary_result['compressed_length'],
                        'compression_ratio': summary_result['compression_ratio']
                    },
                    'key_points': self.summarizer.extract_key_points(content)
                }
                
                compressed_items.append(compressed_item)
            else:
                # Keep short content as is
                compressed_items.append({
                    'content': content,
                    'metadata': {**metadata, 'compressed': False},
                    'key_points': self.summarizer.extract_key_points(content)
                })
        
        return compressed_items
    
    def calculate_compression_savings(self, original_items: List[Dict], compressed_items: List[Dict]) -> Dict[str, float]:
        """Calculate storage savings from compression"""
        original_size = sum(len(item.get('content', '')) for item in original_items)
        compressed_size = sum(len(item.get('content', '')) for item in compressed_items)
        
        if original_size == 0:
            return {'savings_ratio': 0.0, 'total_savings': 0}
        
        savings_ratio = (original_size - compressed_size) / original_size
        total_savings = original_size - compressed_size
        
        return {
            'savings_ratio': savings_ratio,
            'total_savings': total_savings,
            'original_size': original_size,
            'compressed_size': compressed_size
        }

# Global instances
text_summarizer = TextSummarizer()
knowledge_compressor = KnowledgeCompressor()