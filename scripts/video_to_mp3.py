# Convert videos to mp3

import os
import subprocess

files = os.listdir("videos")
c = 0

for file in files:
    print(file)
    c += 1
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{c}_{file}.mp3"])
    