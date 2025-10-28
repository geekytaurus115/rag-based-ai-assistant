"""
Embedding Service
Handles embedding generation using Ollama BGE-M3 model
"""

import requests
from config import Config


class EmbeddingService:
    """Service for generating embeddings via Ollama"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_EMBEDDING_MODEL
        self.timeout = Config.EMBEDDING_TIMEOUT
    
    def create_embedding(self, prompt_list):
        """
        Generate embeddings using Ollama BGE-M3 model
        
        Args:
            prompt_list: List of text strings to embed
            
        Returns:
            List of embedding vectors or None on error
        """
        try:
            r = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": prompt_list
                },
                timeout=self.timeout
            )
            
            if r.status_code == 200:
                embedding = r.json()['embeddings']
                return embedding
            else:
                print(f"❌ Ollama API error: {r.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Make sure it's running (ollama serve)")
            return None
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    def check_connection(self):
        """
        Check if Ollama service is available
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            r = requests.get(
                f"{self.base_url}/api/tags",
                timeout=Config.HEALTH_CHECK_TIMEOUT
            )
            return r.status_code == 200
        except:
            return False

