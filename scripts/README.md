# Data Processing Scripts

This directory contains scripts for preparing video data for the RAG-based AI assistant.

## Scripts Overview

### 1. `youtube_downloader.py`
Downloads videos from YouTube URLs listed in `video_urls.txt`.
- **Input**: `video_urls.txt` (list of YouTube URLs)
- **Output**: Videos saved to `../videos/` directory
- **Dependencies**: yt-dlp or pytube

### 2. `extract_audio.py` / `video_to_mp3.py`
Extracts audio from video files.
- **Input**: Video files from `../videos/`
- **Output**: Audio files to `../audios/` directory
- **Dependencies**: ffmpeg

### 3. `speech_to_text.py` / `mp3_to_json.py`
Converts audio to text using Whisper ASR.
- **Input**: Audio files from `../audios/`
- **Output**: JSON transcripts to `../jsons/` directory
- **Dependencies**: OpenAI Whisper

### 4. `preprocess_json.py`
Creates embeddings from JSON transcripts.
- **Input**: JSON files from `../jsons/`
- **Output**: `../data/embeddings_df.joblib`
- **Dependencies**: Ollama with bge-m3 model
- **Usage**: `python preprocess_json.py`

### 5. `create_video_mapping.py`
Creates a mapping between video numbers and YouTube URLs.
- **Input**: `video_urls.txt`
- **Output**: `../data/video_mapping.json`
- **Usage**: `python create_video_mapping.py`

### 6. `process_incoming.py`
End-to-end processing for new videos.
- Handles the complete pipeline for new content

## Pipeline Order

To process videos from scratch, run scripts in this order:

1. Add YouTube URLs to `video_urls.txt`
2. Run `youtube_downloader.py` to download videos
3. Run `extract_audio.py` or `video_to_mp3.py` to extract audio
4. Run `speech_to_text.py` or `mp3_to_json.py` to transcribe
5. Run `create_video_mapping.py` to create URL mappings
6. Run `preprocess_json.py` to generate embeddings

## Requirements

- **Ollama**: Must be running with `bge-m3` model installed
- **Whisper**: OpenAI Whisper for speech-to-text
- **ffmpeg**: For audio/video processing

## Notes

All scripts are designed to work from the scripts directory and automatically handle paths relative to the project root.

