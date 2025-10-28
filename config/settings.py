"""
Configuration settings for RAG-Based AI Assistant
Centralized configuration management
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Application configuration"""
    
    # Flask Configuration
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # File Paths
    DATA_DIR = BASE_DIR / 'data'
    VIDEOS_DIR = BASE_DIR / 'videos'
    AUDIOS_DIR = BASE_DIR / 'audios'
    JSONS_DIR = BASE_DIR / 'jsons'
    
    # Data Files
    EMBEDDINGS_FILE = DATA_DIR / 'embeddings_df.joblib'
    VIDEO_MAPPING_FILE = DATA_DIR / 'video_mapping.json'
    
    # Ollama Configuration
    OLLAMA_BASE_URL = 'http://localhost:11434'
    OLLAMA_EMBEDDING_MODEL = 'bge-m3'
    OLLAMA_LLM_MODEL = 'llama3.2'
    
    # API Timeouts (seconds)
    EMBEDDING_TIMEOUT = 30
    LLM_TIMEOUT = 120
    HEALTH_CHECK_TIMEOUT = 2
    
    # RAG Configuration
    TOP_K_RESULTS = 5
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VIDEOS_DIR.mkdir(exist_ok=True)
        cls.AUDIOS_DIR.mkdir(exist_ok=True)
        cls.JSONS_DIR.mkdir(exist_ok=True)

