import whisper
import json
import os

model = whisper.load_model("large-v2")

mp3_audios = os.listdir("audios")


for audio in mp3_audios:
    print(audio)
    number = audio.split("_")[0]
    title = audio.split("_")[1][:8]
    print(number, " ", title)
    result = model.transcribe(audio = f"audios/{audio}", 
                            language="hi",
                            task="translate",
                            word_timestamps=False )

    chunks = []
    for segment in result["segments"]:
        chunks.append({"serial no": number, "title": title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
    
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    #print(chunks)

    with open(f"jsons/{number}_{title}.json", "w") as f:
        json.dump(chunks_with_metadata, f)