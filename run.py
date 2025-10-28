"""
Main entry point for the RAG-Based AI Assistant application
Run this file to start the Flask web server
"""

from app import create_app
from config import Config

if __name__ == '__main__':
    app = create_app()
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )

