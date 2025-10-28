"""
Main Routes
Handles main web interface routes
"""

import os
from flask import Blueprint, render_template, send_from_directory
from config import Config

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@main_bp.route('/videos/<path:filename>')
def serve_video(filename):
    """
    Serve video files from the videos directory
    
    Args:
        filename: Name of the video file
        
    Returns:
        Video file
    """
    videos_dir = str(Config.VIDEOS_DIR)
    return send_from_directory(videos_dir, filename)

