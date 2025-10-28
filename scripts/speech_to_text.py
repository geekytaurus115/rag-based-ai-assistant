import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/segments/2_I Copied a YouTube Channel with AI 3D Animation Tools.mp4_15s.mp3", 
                          language="en",
                          task="translate",
                           word_timestamps=False )


chunks = []
# for segment in result["segments"]:
#     chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

#print(chunks)

# with open("output.json", "w") as f:
#     json.dump(chunks,f)

for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)