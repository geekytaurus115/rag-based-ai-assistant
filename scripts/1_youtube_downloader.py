import yt_dlp
import os
import re
from urllib.parse import urlparse, parse_qs

def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filenames."""
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, '')
    return name.strip()

def clean_youtube_url(url: str) -> str:
    """
    Cleans YouTube URL by removing tracking parameters like ?si=...
    Handles both youtu.be and youtube.com formats.
    """
    url = url.strip()
    parsed = urlparse(url)
    
    # Handle youtu.be short links
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/").split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # Handle youtube.com links
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return f"https://www.youtube.com/watch?v={query['v'][0]}"
    
    return url

def check_ffmpeg():
    """Check if ffmpeg is available."""
    import shutil
    return shutil.which('ffmpeg') is not None

def download_youtube_video(url: str, save_path: str):
    """Download a single YouTube video using yt-dlp."""
    try:
        cleaned_url = clean_youtube_url(url)
        
        # Check if ffmpeg is available
        has_ffmpeg = check_ffmpeg()
        
        # yt-dlp options
        ydl_opts = {
            # If ffmpeg is available, get best quality and merge
            # Otherwise, get best pre-merged format
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' if has_ffmpeg else 'best[ext=mp4]/best',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'cookiesfrombrowser': None,  # Can use 'chrome', 'firefox', etc. if needed
        }
        
        # Only add merge option if ffmpeg is available
        if has_ffmpeg:
            ydl_opts['merge_output_format'] = 'mp4'
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to check if already downloaded
            info = ydl.extract_info(cleaned_url, download=False)
            title = sanitize_filename(info['title'])
            filename = f"{title}.mp4"
            filepath = os.path.join(save_path, filename)
            
            if os.path.exists(filepath):
                print(f"⚠️ Skipping '{title}' (already downloaded).")
                return
            
            # Download the video
            print(f"\n🎬 Downloading: {title}")
            print(f"📺 Channel: {info.get('uploader', 'Unknown')}")
            duration = info.get('duration', 0)
            print(f"⏱ Duration: {duration // 60} min {duration % 60} sec")
            
            if not has_ffmpeg:
                print("⚠️ Note: ffmpeg not found. Downloading best available pre-merged format.")
            
            ydl.download([cleaned_url])
            print(f"✅ Done! Saved as: {filename}")

    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

def download_from_file(file_path: str, save_path: str = '.'):
    """Read URLs from file and download videos sequentially."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("⚠️ No URLs found in file.")
        return

    print(f"🔗 Found {len(urls)} video links in '{file_path}'")
    os.makedirs(save_path, exist_ok=True)

    for url in urls:
        download_youtube_video(url, save_path)

    print("\n🎉 All downloads completed!")

if __name__ == "__main__":
    file_path = "video_urls.txt"  # your text file with video URLs
    save_path = "videos"
    os.makedirs(save_path, exist_ok=True)
    download_from_file(file_path, save_path)
