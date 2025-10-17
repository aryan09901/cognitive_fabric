import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class TokenizationResult:
    tokens: List[str]
    token_ids: List[int]
    attention_mask: List[int]
    special_tokens_mask: List[int]
    token_count: int
    char_count: int

class AdvancedTokenizer:
    """
    Advanced tokenization utilities for text processing
    """
    
    def __init__(self):
        self.special_tokens = {
            '[VERIFICATION]': 1000,
            '[SOURCE]': 1001,
            '[CITATION]': 1002,
            '[KNOWLEDGE]': 1003,
            '[QUERY]': 1004,
            '[RESPONSE]': 1005
        }
        
        # Common stop words for basic processing
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can'
        }
    
    def tokenize_text(self, text: str, method: str = 'word') -> TokenizationResult:
        """Tokenize text using specified method"""
        if method == 'word':
            return self._word_tokenize(text)
        elif method == 'sentence':
            return self._sentence_tokenize(text)
        elif method == 'subword':
            return self._subword_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
    
    def _word_tokenize(self, text: str) -> TokenizationResult:
        """Word-level tokenization"""
        # Simple word tokenization (improve with NLTK/spaCy in production)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Convert to token IDs (simple hash-based mapping)
        token_ids = [self._word_to_id(word) for word in words]
        
        return TokenizationResult(
            tokens=words,
            token_ids=token_ids,
            attention_mask=[1] * len(words),
            special_tokens_mask=[0] * len(words),
            token_count=len(words),
            char_count=len(text)
        )
    
    def _sentence_tokenize(self, text: str) -> TokenizationResult:
        """Sentence-level tokenization"""
        # Simple sentence splitting (improve with NLTK in production)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Use sentence hashes as IDs
        token_ids = [hash(s) % 10000 for s in sentences]
        
        return TokenizationResult(
            tokens=sentences,
            token_ids=token_ids,
            attention_mask=[1] * len(sentences),
            special_tokens_mask=[0] * len(sentences),
            token_count=len(sentences),
            char_count=len(text)
        )
    
    def _subword_tokenize(self, text: str) -> TokenizationResult:
        """Subword tokenization (simplified BPE-like)"""
        # Simple subword tokenization (in production, use actual BPE/tokenizer)
        words = re.findall(r'\b\w+\b', text.lower())
        subwords = []
        
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Split into character triples
                for i in range(0, len(word) - 2):
                    subwords.append(word[i:i+3])
        
        token_ids = [self._word_to_id(sw) for sw in subwords]
        
        return TokenizationResult(
            tokens=subwords,
            token_ids=token_ids,
            attention_mask=[1] * len(subwords),
            special_tokens_mask=[0] * len(subwords),
            token_count=len(subwords),
            char_count=len(text)
        )
    
    def _word_to_id(self, word: str) -> int:
        """Convert word to numeric ID"""
        return hash(word) % 10000
    
    def calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics"""
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sentence lengths
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        
        # Vocabulary
        vocabulary = set(words)
        
        # Readability metrics (simplified)
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'vocabulary_size': len(vocabulary),
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'most_common_words': sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10],
            'readability_score': self._calculate_readability(avg_sentence_length, avg_word_length)
        }
    
    def _calculate_readability(self, avg_sentence_length: float, avg_word_length: float) -> float:
        """Calculate simplified readability score"""
        # Simplified Flesch-like score
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0, min(100, score))
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract key phrases from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        content_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Calculate word scores (TF-like)
        word_scores = {}
        for word in content_words:
            word_scores[word] = word_scores.get(word, 0) + 1
        
        # Normalize scores
        max_score = max(word_scores.values()) if word_scores else 1
        scored_phrases = [(word, score/max_score) for word, score in word_scores.items()]
        
        # Sort by score
        scored_phrases.sort(key=lambda x: x[1], reverse=True)
        return scored_phrases[:top_k]
    
    def detect_language(self, text: str) -> str:
        """Detect language of text (simplified)"""
        # Simple language detection based on common words
        common_english = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i'}
        common_spanish = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser', 'se', 'no'}
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        english_matches = len(words.intersection(common_english))
        spanish_matches = len(words.intersection(common_spanish))
        
        if english_matches > spanish_matches:
            return 'en'
        elif spanish_matches > english_matches:
            return 'es'
        else:
            return 'unknown'
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Chunk text into overlapping segments"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks

# Global tokenizer instance
advanced_tokenizer = AdvancedTokenizer()