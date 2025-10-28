import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests


def create_embedding(prompt_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": prompt_list
    })

    embedding = r.json()['embeddings']
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    return(response)


df = joblib.load('embeddings_df.joblib')

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# print(df['embedding'].values)
# print(df['embedding'].shape)
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)


# Find similarities of question_embedding with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)

top_results = 7
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)

new_df = df.loc[max_indx]
# print(new_df)
# print(new_df[["title", "serial no", "text"]])


prompt = f''' Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "serial no", "start", "end", "text"]].to_json(orient="records")}
------------------------------
"{incoming_query}"

User asked this question related to the video chunks, you have to answer in a human way(don't mention the above format, it's just for you) where and how much content is taught (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the show. Don't ask me cross question, only you have answer give this else "Ask regarding show only", and don't add extra text and also mention me video time in time formate (like 4:50, not like 1215 seconds which in unreadable)
More specifiacally you can answer:
for example:
    Video1, 4:50 mins - Talked about ....
    Video2, 6:25 mins - Talked about ...
'''

# with open("prompt.txt", "w") as f:
#     f.write(prompt)

resp = inference(prompt)['response']
print(resp)

# with open("resp.txt", "w") as f:
#     f.write(resp)

# for index, item in new_df.iterrows():
#     print(index, item['title'], item['serial no'], item['text'], item['start'], item['end'])