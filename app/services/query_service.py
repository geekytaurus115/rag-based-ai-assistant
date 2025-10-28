"""
Query Service
Handles query processing, similarity search, and RAG workflow
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from config import Config
from app.utils import format_timestamp, get_youtube_url
from .embedding_service import EmbeddingService
from .llm_service import LLMService


class QueryService:
    """Service for processing user queries using RAG"""
    
    def __init__(self, embeddings_df, video_mapping):
        self.embeddings_df = embeddings_df
        self.video_mapping = video_mapping
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
    
    def process_query(self, user_query, top_k=None):
        """
        Process user query and return relevant video chunks with LLM response
        
        Args:
            user_query: User's question
            top_k: Number of top results to retrieve (default from config)
            
        Returns:
            Dictionary with response and metadata
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        if self.embeddings_df is None:
            return {
                "success": False,
                "error": "Embeddings not loaded. Please check server logs."
            }
        
        # Generate embedding for user query
        question_embedding = self.embedding_service.create_embedding([user_query])
        
        if question_embedding is None:
            return {
                "success": False,
                "error": "Failed to generate embedding. Make sure Ollama is running."
            }
        
        question_embedding = question_embedding[0]
        
        # Calculate cosine similarity with all embeddings
        similarities = cosine_similarity(
            np.vstack(self.embeddings_df['embedding']), 
            [question_embedding]
        ).flatten()
        
        # Get top K most similar chunks
        top_indices = similarities.argsort()[::-1][:top_k]
        relevant_chunks = self.embeddings_df.loc[top_indices]
        
        # Format chunks for display
        chunks_display = self._format_chunks_for_display(relevant_chunks, similarities)
        
        # Build prompt and get LLM response
        prompt = self.llm_service.build_rag_prompt(relevant_chunks, user_query)
        llm_response = self.llm_service.generate_response(prompt)
        
        return {
            "success": True,
            "query": user_query,
            "response": llm_response,
            "chunks": chunks_display,
            "total_chunks": len(self.embeddings_df),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _format_chunks_for_display(self, relevant_chunks, similarities):
        """
        Format chunks for frontend display
        
        Args:
            relevant_chunks: DataFrame of relevant chunks
            similarities: Array of similarity scores
            
        Returns:
            List of formatted chunk dictionaries
        """
        chunks_display = []
        for _, row in relevant_chunks.iterrows():
            start_seconds = int(row['start'])
            youtube_url = get_youtube_url(
                self.video_mapping,
                row['serial no'],
                start_seconds
            )
            
            chunks_display.append({
                "video_num": row['serial no'],
                "title": row['title'],
                "start": format_timestamp(row['start']),
                "end": format_timestamp(row['end']),
                "start_seconds": start_seconds,
                "text": row['text'],
                "similarity": float(similarities[_]),
                "youtube_url": youtube_url,
                "has_youtube": youtube_url is not None
            })
        
        return chunks_display

