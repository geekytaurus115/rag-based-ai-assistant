"""
API Routes
Handles all API endpoints for the application
"""

from flask import Blueprint, request, jsonify
from app.services import EmbeddingService

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global query service (will be set by app factory)
query_service = None


def set_query_service(service):
    """Set the query service instance"""
    global query_service
    query_service = service


@api_bp.route('/query', methods=['POST'])
def query():
    """
    API endpoint to process user queries
    
    Expected JSON: {"query": "user question"}
    Returns: JSON with response and relevant chunks
    """
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400
        
        # Process the query
        result = query_service.process_query(user_query)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    from app import get_app_state
    
    app_state = get_app_state()
    
    # Check if embeddings are loaded
    embeddings_loaded = app_state['embeddings_df'] is not None
    
    # Check Ollama connection
    embedding_service = EmbeddingService()
    ollama_online = embedding_service.check_connection()
    ollama_status = "online" if ollama_online else "offline"
    
    return jsonify({
        "status": "healthy" if embeddings_loaded and ollama_online else "degraded",
        "embeddings_loaded": embeddings_loaded,
        "total_chunks": len(app_state['embeddings_df']) if embeddings_loaded else 0,
        "ollama_status": ollama_status
    })


@api_bp.route('/stats', methods=['GET'])
def stats():
    """Get statistics about the loaded data"""
    from app import get_app_state
    
    app_state = get_app_state()
    embeddings_df = app_state['embeddings_df']
    
    if embeddings_df is None:
        return jsonify({"error": "Embeddings not loaded"}), 500
    
    unique_videos = embeddings_df['serial no'].nunique()
    total_chunks = len(embeddings_df)
    
    # Get video titles
    video_info = embeddings_df.groupby('serial no').agg({
        'title': 'first',
        'text': 'count'
    }).reset_index()
    video_info.columns = ['video_num', 'title', 'chunk_count']
    
    return jsonify({
        "total_videos": unique_videos,
        "total_chunks": total_chunks,
        "videos": video_info.to_dict(orient='records')
    })

