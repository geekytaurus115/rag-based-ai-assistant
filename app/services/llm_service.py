"""
LLM Service
Handles Large Language Model inference using Ollama
"""

import requests
from config import Config


class LLMService:
    """Service for LLM inference via Ollama"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_LLM_MODEL
        self.timeout = Config.LLM_TIMEOUT
    
    def generate_response(self, prompt):
        """
        Send prompt to LLM and get response
        
        Args:
            prompt: The prompt string
            
        Returns:
            Generated response text
        """
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            if r.status_code == 200:
                response = r.json()
                return response['response']
            else:
                return f"Error: LLM API returned status {r.status_code}"
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure it's running."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def build_rag_prompt(self, relevant_chunks, user_query):
        """
        Build a RAG prompt for the LLM
        
        Args:
            relevant_chunks: DataFrame of relevant video chunks
            user_query: User's question
            
        Returns:
            Formatted prompt string
        """
        prompt = f'''Here are video subtitle chunks containing video title, video number, start time, end time, and the text:

{relevant_chunks[["title", "serial no", "start", "end", "text"]].to_json(orient="records")}
------------------------------
User Question: "{user_query}"

Instructions:
- Answer the user's question based on the video chunks provided
- Mention specific video numbers and timestamps in readable format (e.g., "Video 3, 4:50 mins")
- Be conversational and helpful
- If the question is unrelated to the videos, politely say you can only answer questions about the video content
- Format timestamps as MM:SS (e.g., 4:50, not 290 seconds)
- Provide specific guidance on which video and timestamp to watch

Example format:
Video 1, 4:50 mins - Discussed topic about...
Video 2, 6:25 mins - Explained how to...
'''
        return prompt

