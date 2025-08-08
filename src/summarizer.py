import nltk
import spacy
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
import logging
from typing import List, Dict, Any
import re

class DocumentSummarizer:
    """Handles document summarization using multiple approaches."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.transformer_summarizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load transformer model for abstractive summarization
            self.transformer_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
            
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', '', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.split()) > 3]
        
        return '\n'.join(cleaned_lines).strip()
    
    def extractive_summarization(self, text: str, sentence_count: int = 5) -> str:
        """Generate extractive summary using TextRank."""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, sentence_count)
            
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            self.logger.error(f"Error in extractive summarization: {e}")
            return ""
    
    
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into manageable chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > max_length:
                chunks.append(' '.join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def generate_summary(self, text: str, method: str = "hybrid") -> Dict[str, Any]:
        """
        Generate summary using specified method.
        
        Args:
            text: Input text to summarize
            method: Summarization method ('extractive', 'abstractive', 'hybrid')
            
        Returns:
            Dictionary containing summary and metadata
        """
        preprocessed_text = self.preprocess_text(text)
        
        if method == "extractive":
            summary = self.extractive_summarization(preprocessed_text)
        elif method == "abstractive":
            summary = self.abstractive_summarization(preprocessed_text)
        elif method == "hybrid":
            # First extractive, then abstractive
            extractive_summary = self.extractive_summarization(preprocessed_text, 10)
            summary = self.abstractive_summarization(extractive_summary)
        else:
            raise ValueError(f"Unknown summarization method: {method}")
        
        return {
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()),
            'compression_ratio': len(summary.split()) / len(text.split()),
            'method': method
        }