"""Flask Application Factory - Minimal Production Version"""

import json
import joblib
from flask import Flask

from config import Config
from app.services import QueryService
from app.routes import main_bp, api_bp
from app.routes.api import set_query_service

# Application state
_app_state = {'embeddings_df': None, 'video_mapping': {}}

def get_app_state():
    """Get application state for routes"""
    return _app_state


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.ensure_directories()
    
    # Load data
    _load_data()
    
    # Initialize services
    if _app_state['embeddings_df'] is not None:
        query_service = QueryService(_app_state['embeddings_df'], _app_state['video_mapping'])
        set_query_service(query_service)
    
    # Register routes
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    
    return app


def _load_data():
    """Load embeddings and video mapping"""
    # Load video mapping
    if Config.VIDEO_MAPPING_FILE.exists():
        with open(Config.VIDEO_MAPPING_FILE, 'r', encoding='utf-8') as f:
            _app_state['video_mapping'] = json.load(f)
    
    # Load embeddings
    if Config.EMBEDDINGS_FILE.exists():
        _app_state['embeddings_df'] = joblib.load(Config.EMBEDDINGS_FILE)
        print(f"✅ Loaded {len(_app_state['embeddings_df'])} chunks")
    else:
        print("⚠️  Run: python scripts/preprocess_json.py")

