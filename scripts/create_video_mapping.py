"""
Create a mapping between video numbers and YouTube URLs
This script reads video_urls.txt and creates a JSON mapping file
"""

import json
import re
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    """
    Extract YouTube video ID from various URL formats
    
    Args:
        url: YouTube URL (youtu.be or youtube.com)
        
    Returns:
        Video ID string or None
    """
    url = url.strip().split('?')[0]  # Remove query parameters for extraction
    
    # Handle youtu.be short links
    if 'youtu.be' in url:
        return url.split('/')[-1]
    
    # Handle youtube.com links
    if 'youtube.com' in url:
        parsed = urlparse(url)
        if 'v=' in url:
            query = parse_qs(parsed.query)
            return query.get('v', [None])[0]
    
    return None

def create_mapping():
    """Create video number to YouTube URL mapping"""
    
    import os
    
    # Get paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_file = os.path.join(script_dir, 'video_urls.txt')
    output_file = os.path.join(project_root, 'data', 'video_mapping.json')
    
    mapping = {}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        for idx, url in enumerate(urls, start=1):
            video_id = extract_video_id(url)
            if video_id:
                # Store clean YouTube URL
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
                mapping[str(idx)] = {
                    "url": clean_url,
                    "video_id": video_id
                }
                print(f"Video {idx}: {video_id}")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save mapping to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"\n✅ Created mapping for {len(mapping)} videos")
        print(f"📁 Saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found!")
        print("Make sure the file exists in the scripts directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎬 Creating YouTube video mapping...\n")
    create_mapping()

