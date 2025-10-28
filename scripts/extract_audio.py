# Extract 15 seconds from an audio file using ffmpeg

import subprocess
import os
from pathlib import Path

def extract_audio_segment(input_file, output_file, start_time=0, duration=15):
    """
    Extract a segment from an audio file using ffmpeg.
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to save the extracted segment
        start_time (float): Start time in seconds (default: 0)
        duration (float): Duration in seconds (default: 15)
    """
    try:
        if not os.path.exists(input_file):
            print(f"[ERROR] File not found - {input_file}")
            return
        
        print(f"Extracting {duration} seconds starting from {start_time}s...")
        print(f"Input: {input_file}")
        
        # Use ffmpeg to extract the audio segment
        # -ss: start time in seconds
        # -t: duration in seconds
        # -i: input file
        # -acodec copy: copy audio codec without re-encoding (faster)
        # -avoid_negative_ts make_zero: handle negative timestamps
        
        command = [
            "ffmpeg",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(input_file),
            "-acodec", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_file),
            "-y"  # Overwrite output file if it exists
        ]
        
        # Run ffmpeg (suppress output for cleaner console)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024
            print(f"[OK] Successfully saved to: {output_file}")
            print(f"  File size: {file_size:.2f} KB")
        else:
            print(f"[ERROR] {result.stderr}")
            
    except FileNotFoundError:
        print(f"[ERROR] ffmpeg not found. Please install ffmpeg.")
    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    # Configuration
    input_dir = Path("audios")
    output_dir = Path("audios") / "segments"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Example: Extract 15 seconds from the first audio file
    audio_files = list(input_dir.glob("2_*.mp3"))
    
    if not audio_files:
        print("No audio files found in the 'audios' directory")
        return
    
    print(f"Found {len(audio_files)} audio file(s)\n")
    
    # Process the first file as an example
    input_file = audio_files[0]
    output_file = output_dir / f"{input_file.stem}_15s.mp3"
    
    print(f"Processing: {input_file.name}\n")
    
    # Extract first 15 seconds
    extract_audio_segment(
        input_file=str(input_file),
        output_file=str(output_file),
        start_time=0,      # Start from the beginning
        duration=15        # Extract 15 seconds
    )
    
    print("\n" + "="*50)
    print("Usage examples:")
    print("  # Extract first 15 seconds")
    print("  extract_audio_segment('audios/file.mp3', 'output.mp3', start_time=0, duration=15)")
    print()
    print("  # Extract 15 seconds starting from 1 minute")
    print("  extract_audio_segment('audios/file.mp3', 'output.mp3', start_time=60, duration=15)")
    print()
    print("  # Extract custom duration")
    print("  extract_audio_segment('audios/file.mp3', 'output.mp3', start_time=30, duration=10)")

if __name__ == "__main__":
    main()

