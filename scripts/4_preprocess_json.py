import requests
import os
import json
import pandas as pd
import joblib

# def create_embedding(prompt_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": prompt_list
#     })

#     embedding = r.json()['embeddings']
#     return embedding


# Get paths relative to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jsons_dir = os.path.join(project_root, "jsons")
output_file = os.path.join(project_root, "data", "embeddings_df.joblib")

jsons = os.listdir(jsons_dir) # List all the jsons
my_dict = []
chunk_id = 0

for jsn in jsons:
    with open(os.path.join(jsons_dir, jsn)) as f:
        content = json.load(f)
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)
        # print(chunk)

df = pd.DataFrame.from_records(my_dict)
# print(df)
"""
Save this DataFrame
"""
# Ensure data directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)
joblib.dump(df, output_file)
print(f"✅ Saved embeddings to: {output_file}")

