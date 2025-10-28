"""
Helper utility functions
Reusable utility functions for formatting and URL generation
"""


def format_timestamp(seconds):
    """
    Convert seconds to MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "4:30")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def get_youtube_url(video_mapping, video_num, start_seconds):
    """
    Get YouTube URL with timestamp for a video
    
    Args:
        video_mapping: Dictionary mapping video numbers to YouTube IDs
        video_num: Serial number of the video
        start_seconds: Start time in seconds
        
    Returns:
        YouTube URL with timestamp or None if not found
    """
    video_key = str(video_num)
    if video_key in video_mapping:
        video_id = video_mapping[video_key]['video_id']
        # YouTube timestamp format: ?v=VIDEO_ID&t=SECONDSs
        return f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
    
    return None

