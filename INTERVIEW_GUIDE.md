# RAG-Based AI Assistant - Interview Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Complete Project Flow](#complete-project-flow)
3. [Detailed Step-by-Step Breakdown](#detailed-step-by-step-breakdown)
4. [Vector Database & Embedding Storage](#vector-database--embedding-storage)
5. [Technology Stack Deep Dive](#technology-stack-deep-dive)
6. [Common Interview Questions & Answers](#common-interview-questions--answers)
7. [Architecture Decisions & Rationale](#architecture-decisions--rationale)

---

## Project Overview

**What is this project?**
A Retrieval-Augmented Generation (RAG) system that:
- Downloads YouTube videos
- Transcribes them with timestamps
- Creates semantic searchable embeddings
- Answers questions about video content with precise timestamps

**Key Capabilities:**
- Multilingual video transcription (Hindi → English translation)
- Semantic search across video transcripts
- AI-powered Q&A with video references
- Web-based user interface

---

## Complete Project Flow

### High-Level Pipeline

```
YouTube URLs → Video Download → Audio Extraction → Transcription → 
Embedding Generation → Storage → Query Processing → LLM Response
```

### Detailed Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

Step 1: Video Download
  Input:  video_urls.txt (list of YouTube URLs)
  Tool:   yt-dlp
  Output: videos/*.mp4 files
  Location: scripts/youtube_downloader.py

Step 2: Audio Extraction
  Input:  videos/*.mp4
  Tool:   FFmpeg (via subprocess)
  Output: audios/*.mp3 files
  Location: scripts/video_to_mp3.py

Step 3: Speech-to-Text Transcription
  Input:  audios/*.mp3
  Model:  OpenAI Whisper (large-v2)
  Output: jsons/*.json (transcripts with timestamps)
  Location: scripts/mp3_to_json.py

Step 4: Embedding Generation
  Input:  jsons/*.json (text chunks)
  Model:  BGE-M3 via Ollama API
  Output: data/embeddings_df.joblib (DataFrame with embeddings)
  Location: scripts/preprocess_json.py

Step 5: Video Mapping Creation
  Input:  jsons/*.json
  Output: data/video_mapping.json (video ID to YouTube URL mapping)
  Location: scripts/create_video_mapping.py

┌─────────────────────────────────────────────────────────────────┐
│                      RUNTIME QUERY PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

Step 6: Application Startup
  Input:  data/embeddings_df.joblib, data/video_mapping.json
  Tool:   Flask application factory
  Output: Loaded embeddings in memory (pandas DataFrame)
  Location: app/__init__.py

Step 7: User Query Processing
  Input:  User question (text string)
  Process:
    a) Generate query embedding (BGE-M3)
    b) Calculate cosine similarity with all stored embeddings
    c) Retrieve top K most similar chunks
    d) Build RAG prompt with context
    e) Generate LLM response (Llama 3.2)
  Output: JSON response with answer + relevant chunks + timestamps
  Location: app/services/query_service.py
```

---

## Detailed Step-by-Step Breakdown

### **STEP 1: Video Download**

**Script:** `scripts/youtube_downloader.py`

**Input:**
- File: `scripts/video_urls.txt`
- Format: One YouTube URL per line
- Example:
  ```
  https://www.youtube.com/watch?v=VIDEO_ID_1
  https://youtu.be/VIDEO_ID_2
  ```

**Tool/Library Used:**
- **yt-dlp**: Python library for downloading YouTube videos
- **Why yt-dlp?** It's the most reliable and actively maintained YouTube downloader

**Process:**
1. Reads URLs from `video_urls.txt`
2. Uses `yt-dlp` to download each video
3. Saves videos with original titles

**Output:**
- Location: `videos/` directory
- Format: `VideoTitle.mp4`
- Example: `Aman को पहले नहीं था honey खाने का शौक ｜ Male Entrepreneurs.mp4`

**Key Points for Interview:**
- yt-dlp handles various YouTube URL formats
- Downloads best quality video by default
- Can be configured for specific quality/resolution

---

### **STEP 2: Audio Extraction**

**Script:** `scripts/video_to_mp3.py`

**Input:**
- Directory: `videos/` containing `.mp4` files

**Tool/Library Used:**
- **FFmpeg**: Command-line tool for audio/video processing
- **subprocess**: Python module to execute FFmpeg commands
- **Why FFmpeg?** Industry standard, supports all formats, high quality conversion

**Process:**
1. Lists all `.mp4` files in `videos/` directory
2. For each video:
   - Extracts audio track using FFmpeg
   - Converts to MP3 format
   - Adds numbered prefix (1_, 2_, etc.) for ordering
3. Saves audio files

**Output:**
- Location: `audios/` directory
- Format: `{number}_{VideoTitle}.mp4.mp3`
- Example: `1_Aman को .mp4.mp3`

**FFmpeg Command (conceptual):**
```bash
ffmpeg -i input.mp4 -vn -acodec libmp3lame output.mp3
```

**Key Points for Interview:**
- FFmpeg must be installed separately (not a Python package)
- Audio extraction is necessary because Whisper works on audio, not video
- MP3 format chosen for compatibility and file size

---

### **STEP 3: Speech-to-Text Transcription**

**Script:** `scripts/mp3_to_json.py`

**Input:**
- Directory: `audios/` containing `.mp3` files

**Model/Library Used:**
- **OpenAI Whisper**: `large-v2` model
- **PyTorch**: Deep learning framework (for GPU acceleration)
- **Why Whisper?** 
  - State-of-the-art accuracy
  - Multilingual support
  - Automatic language detection
  - Word-level timestamps capability

**Process:**
1. Loads Whisper `large-v2` model (downloads on first use)
2. For each audio file:
   - Transcribes audio to text
   - **Language**: Hindi (`language="hi"`)
   - **Task**: Translation (`task="translate"`) - translates Hindi to English
   - Generates segments with start/end timestamps
3. Structures output as JSON

**Whisper Configuration:**
```python
model = whisper.load_model("large-v2")
result = model.transcribe(
    audio=f"audios/{audio}",
    language="hi",          # Source language: Hindi
    task="translate",       # Translate to English
    word_timestamps=False   # Segment-level timestamps
)
```

**Output:**
- Location: `jsons/` directory
- Format: `{number}_{title}.json`
- Structure:
  ```json
  {
    "chunks": [
      {
        "serial no": "1",
        "title": "Aman को ",
        "start": 0.0,
        "end": 4.5,
        "text": "Translated English text here..."
      },
      ...
    ],
    "text": "Full transcript..."
  }
  ```

**Key Points for Interview:**
- **GPU Acceleration**: Whisper automatically uses CUDA if available (5-6x faster)
- **Model Size**: `large-v2` is ~1.5GB, provides highest accuracy
- **Translation**: `task="translate"` converts non-English audio to English text
- **Timestamps**: Each chunk has precise start/end times for video navigation
- **Why large-v2?** Best accuracy-to-speed ratio for production use

**Performance:**
- GPU (CUDA): ~10-15 minutes for 1-hour video
- CPU: ~60-90 minutes for 1-hour video

---

### **STEP 4: Embedding Generation**

**Script:** `scripts/preprocess_json.py`

**Input:**
- Directory: `jsons/` containing transcript JSON files

**Model/Library Used:**
- **BGE-M3**: Embedding model via Ollama API
- **Ollama**: Local LLM server
- **pandas**: Data manipulation
- **joblib**: Model serialization
- **requests**: HTTP client for Ollama API

**Why BGE-M3?**
- Multilingual support (works with English translations)
- High-quality embeddings (1024 dimensions)
- Balanced performance and accuracy
- Available via Ollama (easy local deployment)

**Process:**
1. Reads all JSON files from `jsons/` directory
2. For each JSON file:
   - Extracts all text chunks
   - Sends batch of text chunks to Ollama embedding API
   - Receives embedding vectors (one per chunk)
3. Combines metadata with embeddings:
   - `chunk_id`: Unique identifier
   - `serial no`: Video number
   - `title`: Video title
   - `start`, `end`: Timestamps
   - `text`: Original text
   - `embedding`: Vector representation (1024 dimensions for BGE-M3)
4. Creates pandas DataFrame
5. Serializes DataFrame to joblib format

**Ollama API Call:**
```python
r = requests.post("http://localhost:11434/api/embed", json={
    "model": "bge-m3",
    "input": ["text chunk 1", "text chunk 2", ...]
})
embeddings = r.json()['embeddings']  # List of vectors
```

**Output:**
- Location: `data/embeddings_df.joblib`
- Format: Serialized pandas DataFrame
- Structure:
  ```
  DataFrame columns:
  - chunk_id: int (unique identifier)
  - serial no: str (video number)
  - title: str (video title)
  - start: float (start time in seconds)
  - end: float (end time in seconds)
  - text: str (transcript text)
  - embedding: list (1024-dimensional vector)
  ```

**Key Points for Interview:**
- **Batch Processing**: Sends all chunks from one video at once (efficient)
- **Embedding Dimension**: BGE-M3 produces 1024-dimensional vectors
- **Storage Format**: joblib is faster than pickle for large DataFrames
- **Why joblib?** Optimized for NumPy arrays (embeddings are NumPy arrays)

---

### **STEP 5: Video Mapping Creation**

**Script:** `scripts/create_video_mapping.py`

**Input:**
- Directory: `jsons/` containing transcript JSON files
- Original: `scripts/video_urls.txt` (YouTube URLs)

**Tool/Library Used:**
- Standard Python libraries (json, os)

**Process:**
1. Maps video serial numbers to YouTube URLs
2. Creates lookup dictionary

**Output:**
- Location: `data/video_mapping.json`
- Format:
  ```json
  {
    "1": "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "2": "https://www.youtube.com/watch?v=VIDEO_ID_2",
    ...
  }
  ```

**Purpose:**
- Enables generating clickable YouTube links with timestamps
- Links users directly to relevant video segments

---

### **STEP 6: Application Startup**

**File:** `app/__init__.py`

**Input:**
- `data/embeddings_df.joblib`: Pre-computed embeddings
- `data/video_mapping.json`: Video URL mapping

**Tool/Library Used:**
- **Flask**: Web framework
- **joblib**: Load serialized DataFrame
- **json**: Load video mapping

**Process:**
1. Flask application factory pattern
2. Loads embeddings DataFrame into memory
3. Loads video mapping dictionary
4. Initializes services:
   - `EmbeddingService`: For generating query embeddings
   - `LLMService`: For generating responses
   - `QueryService`: Orchestrates RAG workflow
5. Registers routes (main, API)

**Output:**
- Running Flask application
- Embeddings loaded in memory (pandas DataFrame)
- Services initialized and ready

**Key Points for Interview:**
- **In-Memory Storage**: All embeddings loaded at startup (fast queries)
- **Application Factory**: Allows testing and multiple app instances
- **Lazy Loading**: Only loads if files exist (graceful degradation)

---

### **STEP 7: User Query Processing (RAG Workflow)**

**File:** `app/services/query_service.py`

**Input:**
- User question: Text string (e.g., "How can AI help with YouTube content?")

**Models/Libraries Used:**
- **BGE-M3** (via Ollama): Query embedding generation
- **scikit-learn**: Cosine similarity calculation
- **NumPy**: Vector operations
- **Llama 3.2** (via Ollama): Response generation

**Detailed Process:**

#### **7a. Query Embedding Generation**
- **Service**: `EmbeddingService.create_embedding()`
- **Input**: User query string
- **Model**: BGE-M3 via Ollama API
- **Output**: 1024-dimensional embedding vector
- **API Call**:
  ```python
  POST http://localhost:11434/api/embed
  {
    "model": "bge-m3",
    "input": ["user question"]
  }
  ```

#### **7b. Similarity Search**
- **Library**: `scikit-learn.metrics.pairwise.cosine_similarity`
- **Input**: 
  - Query embedding (1 vector)
  - All stored embeddings (N vectors from DataFrame)
- **Process**:
  1. Stacks all embeddings into matrix: `np.vstack(df['embedding'])`
  2. Calculates cosine similarity between query and all chunks
  3. Returns similarity scores (0 to 1, where 1 = most similar)
- **Output**: Array of similarity scores (one per chunk)

#### **7c. Top-K Retrieval**
- **Input**: Similarity scores array
- **Process**:
  1. Sorts indices by similarity (descending)
  2. Takes top K indices (default: 5, configurable)
  3. Retrieves corresponding chunks from DataFrame
- **Output**: DataFrame with top K most relevant chunks

#### **7d. Format Chunks for Display**
- **Service**: `QueryService._format_chunks_for_display()`
- **Input**: Top K chunks DataFrame
- **Process**:
  1. Formats timestamps (seconds → MM:SS format)
  2. Generates YouTube URLs with timestamp parameters
  3. Calculates similarity percentages
- **Output**: List of formatted chunk dictionaries

#### **7e. RAG Prompt Construction**
- **Service**: `LLMService.build_rag_prompt()`
- **Input**: 
  - Relevant chunks (top K)
  - User query
- **Process**: Builds structured prompt with:
  - Context: JSON of relevant chunks with metadata
  - User question
  - Instructions for LLM
- **Output**: Formatted prompt string

**Prompt Template:**
```
Here are video subtitle chunks containing video title, video number, start time, end time, and the text:

[{chunk1}, {chunk2}, ...]
------------------------------
User Question: "{user_query}"

Instructions:
- Answer the user's question based on the video chunks provided
- Mention specific video numbers and timestamps in readable format
- Be conversational and helpful
- Format timestamps as MM:SS
```

#### **7f. LLM Response Generation**
- **Service**: `LLMService.generate_response()`
- **Model**: Llama 3.2 via Ollama API
- **Input**: RAG prompt
- **API Call**:
  ```python
  POST http://localhost:11434/api/generate
  {
    "model": "llama3.2",
    "prompt": "{rag_prompt}",
    "stream": False
  }
  ```
- **Output**: Natural language response text

#### **7g. Response Assembly**
- Combines:
  - LLM-generated answer
  - Formatted chunks with metadata
  - Similarity scores
  - YouTube URLs
  - Timestamps

**Final Output:**
```json
{
  "success": true,
  "query": "user question",
  "response": "LLM-generated answer with video references...",
  "chunks": [
    {
      "video_num": "3",
      "title": "Video Title",
      "start": "4:50",
      "end": "6:25",
      "start_seconds": 290,
      "text": "Relevant transcript text...",
      "similarity": 0.87,
      "youtube_url": "https://youtube.com/watch?v=...&t=290s",
      "has_youtube": true
    },
    ...
  ],
  "total_chunks": 1250,
  "timestamp": "2024-01-15 14:30:00"
}
```

**Key Points for Interview:**
- **RAG Pattern**: Retrieval (similarity search) + Augmentation (context injection) + Generation (LLM)
- **Cosine Similarity**: Measures angle between vectors (semantic similarity)
- **Top-K Selection**: Balances context size vs. relevance
- **Prompt Engineering**: Structured prompt guides LLM to use context effectively
- **Why Llama 3.2?** Fast inference, good instruction following, local deployment

---

## Vector Database & Embedding Storage

### **Database Usage: NO Database Used**

**Critical Answer for Interview:**
- **NO traditional database** (PostgreSQL, MySQL, MongoDB, etc.) is used
- **NO vector database** (Pinecone, Weaviate, Chroma, etc.) is used
- **Storage Method**: In-memory pandas DataFrame + file-based persistence (joblib)

**What We Use:**
- **File Storage**: `data/embeddings_df.joblib` (serialized pandas DataFrame)
- **Runtime Storage**: In-memory pandas DataFrame (loaded at startup)
- **Search Method**: Brute-force cosine similarity using scikit-learn

### **Current Storage Architecture**

1. **Persistent Storage (Disk)**
   - **File**: `data/embeddings_df.joblib`
   - **Format**: Serialized pandas DataFrame
   - **Size**: Depends on number of chunks × embedding dimension
   - **Example**: 1000 chunks × 1024 dims × 4 bytes = ~4MB

2. **Runtime Storage (Memory)**
   - **Location**: Application memory (RAM)
   - **Format**: pandas DataFrame
   - **Loaded**: At application startup via `joblib.load()`
   - **Lifetime**: Until application shutdown

3. **Embedding Structure**
   ```python
   DataFrame:
   - chunk_id: int
   - serial no: str
   - title: str
   - start: float
   - end: float
   - text: str
   - embedding: list[float]  # 1024 dimensions for BGE-M3
   ```

4. **Similarity Search Method**
   - **Library**: `scikit-learn.metrics.pairwise.cosine_similarity`
   - **Process**: Brute-force comparison with all embeddings
   - **Algorithm**: Cosine similarity on NumPy arrays
   - **Code**:
     ```python
     from sklearn.metrics.pairwise import cosine_similarity
     import numpy as np
     
     # Load embeddings
     df = joblib.load('data/embeddings_df.joblib')
     
     # Query embedding
     query_emb = embedding_service.create_embedding([user_query])[0]
     
     # Calculate similarity (brute-force)
     similarities = cosine_similarity(
         np.vstack(df['embedding']), 
         [query_emb]
     ).flatten()
     
     # Get top K
     top_indices = similarities.argsort()[::-1][:top_k]
     ```

### **Why NOT Use a Database?**

**Current Approach (In-Memory DataFrame):**
- ✅ **Simple**: No additional infrastructure
- ✅ **Fast for Small Datasets**: Direct NumPy operations (<100ms for <10K chunks)
- ✅ **No Dependencies**: Works out of the box
- ✅ **Easy Development**: No database setup required
- ✅ **Zero Cost**: No database hosting costs
- ❌ **Scalability Limit**: ~10K-100K chunks (depends on RAM)
- ❌ **No Persistence**: Lost on restart (reloaded from file)
- ❌ **No Advanced Features**: No filtering, metadata search, etc.
- ❌ **Memory Intensive**: All embeddings must fit in RAM

**Vector Database Approach (Not Currently Used):**
- ✅ **Scalable**: Millions of vectors
- ✅ **Advanced Search**: Filtering, hybrid search, etc.
- ✅ **Persistent**: Database-backed
- ✅ **Optimized Indexes**: HNSW, IVF, etc. for fast search (<10ms)
- ✅ **Metadata Filtering**: Query by video title, date, etc.
- ✅ **Real-time Updates**: Add new chunks without restart
- ❌ **Complexity**: Additional infrastructure
- ❌ **Overhead**: Network calls, serialization
- ❌ **Cost**: Hosting/maintenance costs
- ❌ **Overkill**: For small-medium datasets

### **When Would We Need a Vector Database?**

**Consider migrating to vector database if:**
- Dataset grows beyond 100K chunks
- Need sub-10ms search latency
- Require filtering by metadata (e.g., "videos from 2024")
- Need hybrid search (semantic + keyword)
- Multiple applications need shared access
- Want real-time updates without restart
- Need to scale horizontally

---

## How to Use Vector Database (Migration Guide)

### **Option 1: Chroma (Lightweight, Embedded)**

**Why Chroma:**
- Easiest to integrate (Python-native)
- No separate server needed
- Good for small-medium scale
- Free and open-source

**Installation:**
```bash
pip install chromadb
```

**Migration Code:**

**Step 1: Create Chroma Collection (Preprocessing)**
```python
# scripts/preprocess_json_with_chroma.py
import chromadb
from chromadb.config import Settings
import json
import os
import requests

# Initialize Chroma client (persistent mode)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="video_chunks",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

def create_embedding(prompt_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": prompt_list
    })
    return r.json()['embeddings']

# Load JSON transcripts
jsons_dir = "jsons"
chunk_id = 0

for jsn in os.listdir(jsons_dir):
    with open(os.path.join(jsons_dir, jsn)) as f:
        content = json.load(f)
    
    # Generate embeddings
    texts = [c['text'] for c in content['chunks']]
    embeddings = create_embedding(texts)
    
    # Prepare data for Chroma
    ids = []
    embeddings_list = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(content['chunks']):
        ids.append(f"chunk_{chunk_id}")
        embeddings_list.append(embeddings[i])
        documents.append(chunk['text'])
        metadatas.append({
            "chunk_id": chunk_id,
            "serial_no": chunk['serial no'],
            "title": chunk['title'],
            "start": chunk['start'],
            "end": chunk['end']
        })
        chunk_id += 1
    
    # Add to Chroma collection
    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )
    
print(f"✅ Added {chunk_id} chunks to Chroma")
```

**Step 2: Update Query Service**
```python
# app/services/query_service_chroma.py
import chromadb
from chromadb.config import Settings

class QueryServiceChroma:
    def __init__(self, video_mapping):
        self.video_mapping = video_mapping
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection("video_chunks")
        self.embedding_service = EmbeddingService()
    
    def process_query(self, user_query, top_k=5):
        # Generate query embedding
        query_embedding = self.embedding_service.create_embedding([user_query])[0]
        
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        chunks = []
        for i, metadata in enumerate(results['metadatas'][0]):
            chunks.append({
                "video_num": metadata['serial_no'],
                "title": metadata['title'],
                "start": metadata['start'],
                "end": metadata['end'],
                "text": results['documents'][0][i],
                "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
            })
        
        # Build RAG prompt and generate response
        # ... (same as before)
        
        return {"chunks": chunks, "response": llm_response}
```

**Benefits:**
- Persistent storage (survives restarts)
- Fast similarity search with HNSW index
- Metadata filtering support
- Easy to integrate

---

### **Option 2: Pinecone (Managed, Cloud-Based)**

**Why Pinecone:**
- Fully managed (no infrastructure)
- Highly scalable (millions of vectors)
- Fast search (<10ms)
- Good for production

**Installation:**
```bash
pip install pinecone-client
```

**Migration Code:**

**Step 1: Create Pinecone Index**
```python
# scripts/preprocess_json_with_pinecone.py
import pinecone
import json
import os
import requests

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="us-east-1")
index = pinecone.Index("video-chunks")

def create_embedding(prompt_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": prompt_list
    })
    return r.json()['embeddings']

# Process and upload chunks
chunk_id = 0
batch_size = 100
vectors_batch = []

for jsn in os.listdir("jsons"):
    with open(os.path.join("jsons", jsn)) as f:
        content = json.load(f)
    
    texts = [c['text'] for c in content['chunks']]
    embeddings = create_embedding(texts)
    
    for i, chunk in enumerate(content['chunks']):
        vector_id = f"chunk_{chunk_id}"
        vector_data = {
            "id": vector_id,
            "values": embeddings[i],
            "metadata": {
                "chunk_id": chunk_id,
                "serial_no": chunk['serial no'],
                "title": chunk['title'],
                "start": chunk['start'],
                "end": chunk['end'],
                "text": chunk['text']
            }
        }
        vectors_batch.append(vector_data)
        chunk_id += 1
        
        # Upload in batches
        if len(vectors_batch) >= batch_size:
            index.upsert(vectors=vectors_batch)
            vectors_batch = []
            print(f"Uploaded {chunk_id} chunks...")

# Upload remaining
if vectors_batch:
    index.upsert(vectors=vectors_batch)

print(f"✅ Uploaded {chunk_id} chunks to Pinecone")
```

**Step 2: Update Query Service**
```python
# app/services/query_service_pinecone.py
import pinecone

class QueryServicePinecone:
    def __init__(self, video_mapping):
        self.video_mapping = video_mapping
        pinecone.init(api_key="YOUR_API_KEY", environment="us-east-1")
        self.index = pinecone.Index("video-chunks")
        self.embedding_service = EmbeddingService()
    
    def process_query(self, user_query, top_k=5):
        # Generate query embedding
        query_embedding = self.embedding_service.create_embedding([user_query])[0]
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        chunks = []
        for match in results['matches']:
            metadata = match['metadata']
            chunks.append({
                "video_num": metadata['serial_no'],
                "title": metadata['title'],
                "start": metadata['start'],
                "end": metadata['end'],
                "text": metadata['text'],
                "similarity": match['score']
            })
        
        # Build RAG prompt and generate response
        # ... (same as before)
        
        return {"chunks": chunks, "response": llm_response}
```

**Benefits:**
- Fully managed (no server maintenance)
- Highly scalable
- Fast search performance
- Metadata filtering: `index.query(..., filter={"serial_no": "1"})`

---

### **Option 3: Qdrant (High-Performance, Self-Hosted)**

**Why Qdrant:**
- High performance (Rust-based)
- Self-hosted or cloud
- Advanced filtering
- Good for production

**Installation:**
```bash
# Using Docker
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

# Python client
pip install qdrant-client
```

**Migration Code:**

**Step 1: Create Qdrant Collection**
```python
# scripts/preprocess_json_with_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
import os
import requests

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Create collection
collection_name = "video_chunks"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

def create_embedding(prompt_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": prompt_list
    })
    return r.json()['embeddings']

# Process and upload chunks
chunk_id = 0
points = []

for jsn in os.listdir("jsons"):
    with open(os.path.join("jsons", jsn)) as f:
        content = json.load(f)
    
    texts = [c['text'] for c in content['chunks']]
    embeddings = create_embedding(texts)
    
    for i, chunk in enumerate(content['chunks']):
        point = PointStruct(
            id=chunk_id,
            vector=embeddings[i],
            payload={
                "chunk_id": chunk_id,
                "serial_no": chunk['serial no'],
                "title": chunk['title'],
                "start": chunk['start'],
                "end": chunk['end'],
                "text": chunk['text']
            }
        )
        points.append(point)
        chunk_id += 1
        
        # Upload in batches
        if len(points) >= 100:
            client.upsert(collection_name=collection_name, points=points)
            points = []
            print(f"Uploaded {chunk_id} chunks...")

# Upload remaining
if points:
    client.upsert(collection_name=collection_name, points=points)

print(f"✅ Uploaded {chunk_id} chunks to Qdrant")
```

**Step 2: Update Query Service**
```python
# app/services/query_service_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

class QueryServiceQdrant:
    def __init__(self, video_mapping):
        self.video_mapping = video_mapping
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "video_chunks"
        self.embedding_service = EmbeddingService()
    
    def process_query(self, user_query, top_k=5, video_filter=None):
        # Generate query embedding
        query_embedding = self.embedding_service.create_embedding([user_query])[0]
        
        # Optional: Filter by video number
        query_filter = None
        if video_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key="serial_no", match=MatchValue(value=video_filter))
                ]
            )
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        chunks = []
        for result in results:
            payload = result.payload
            chunks.append({
                "video_num": payload['serial_no'],
                "title": payload['title'],
                "start": payload['start'],
                "end": payload['end'],
                "text": payload['text'],
                "similarity": result.score
            })
        
        # Build RAG prompt and generate response
        # ... (same as before)
        
        return {"chunks": chunks, "response": llm_response}
```

**Benefits:**
- High performance (Rust-based)
- Advanced filtering capabilities
- Self-hosted or cloud options
- Good for production scale

---

### **Comparison: Current vs. Vector Database**

| Feature | Current (In-Memory) | Chroma | Pinecone | Qdrant |
|---------|-------------------|--------|----------|--------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Medium | ⭐⭐ Complex |
| **Scalability** | ~100K chunks | ~1M chunks | Millions | Millions |
| **Search Speed** | ~50-100ms | ~10-20ms | ~5-10ms | ~5-10ms |
| **Metadata Filtering** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cost** | Free | Free | Paid | Free (self-hosted) |
| **Infrastructure** | None | Local file | Cloud | Server/Docker |
| **Real-time Updates** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

---

### **Migration Strategy**

**Step-by-Step Migration:**

1. **Keep Current System Running**
   - Don't break existing functionality
   - Run both systems in parallel initially

2. **Create New Vector Database Service**
   - Implement new query service with vector DB
   - Use feature flag to switch between systems

3. **Migrate Data**
   - Run preprocessing script to populate vector DB
   - Verify data integrity

4. **Test Performance**
   - Compare search results
   - Measure latency
   - Validate accuracy

5. **Switch Over**
   - Update configuration
   - Deploy new service
   - Monitor for issues

6. **Deprecate Old System**
   - Remove in-memory code after validation
   - Clean up old files

**Code Example (Feature Flag):**
```python
# config/settings.py
USE_VECTOR_DB = False  # Feature flag

# app/services/query_service.py
if Config.USE_VECTOR_DB:
    from .query_service_chroma import QueryServiceChroma
    query_service = QueryServiceChroma(video_mapping)
else:
    from .query_service_memory import QueryServiceMemory
    query_service = QueryServiceMemory(embeddings_df, video_mapping)
```

---

### **Interview Answer: "How Would You Add a Vector Database?"**

**Answer:**
"I would migrate to a vector database in these steps:

1. **Choose Database**: For our scale, I'd start with Chroma (easiest) or Qdrant (best performance). For production at scale, Pinecone is excellent.

2. **Create Migration Script**: Modify `preprocess_json.py` to:
   - Generate embeddings (same as now)
   - Store in vector database instead of joblib
   - Include metadata (video number, timestamps, text)

3. **Update Query Service**: Replace in-memory similarity search with vector DB query:
   - Generate query embedding (same)
   - Call vector DB search API
   - Get top K results with metadata

4. **Add Filtering**: Leverage vector DB's metadata filtering (e.g., filter by video number, date range)

5. **Performance Testing**: Compare latency and accuracy before full migration

6. **Gradual Rollout**: Use feature flags to test both systems, then switch over.

The main benefits would be:
- Scalability to millions of chunks
- Sub-10ms search latency
- Metadata filtering capabilities
- Real-time updates without restart

For our current dataset size, the in-memory approach works well, but I'd migrate when we exceed 100K chunks or need advanced features."

---

## Technology Stack Deep Dive

### **Core Technologies**

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.12 | Main programming language |
| **Web Framework** | Flask | 3.0+ | REST API and web UI |
| **ML Framework** | PyTorch | 2.0+ | Deep learning (for Whisper) |
| **Data Processing** | pandas | 2.0+ | DataFrame operations |
| **Numerical Computing** | NumPy | 1.24+ | Vector operations |
| **ML Utilities** | scikit-learn | 1.3+ | Cosine similarity |
| **Serialization** | joblib | 1.3+ | Model/data persistence |

### **AI/ML Models**

| Model | Provider | Purpose | Dimensions | Location |
|-------|----------|---------|------------|----------|
| **Whisper large-v2** | OpenAI | Speech-to-text | N/A | Local (downloaded) |
| **BGE-M3** | Ollama | Text embeddings | 1024 | Ollama server |
| **Llama 3.2** | Ollama | Text generation | N/A | Ollama server |

### **External Tools**

| Tool | Purpose | Installation |
|------|---------|--------------|
| **FFmpeg** | Audio/video processing | System-level install |
| **Ollama** | Local LLM server | Separate installation |
| **yt-dlp** | YouTube downloader | Python package |

### **Data Flow Architecture**

```
┌─────────────┐
│  YouTube    │
│    URLs     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     yt-dlp
│   Videos    │ ◄──────────
│   (.mp4)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     FFmpeg
│   Audios    │ ◄──────────
│   (.mp3)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     Whisper
│ Transcripts │ ◄──────────
│   (.json)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     BGE-M3
│ Embeddings  │ ◄──────────
│  (.joblib)  │     (Ollama)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   In-Memory │
│  DataFrame  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Query     │
│  Processing │
└─────────────┘
```

---

## Common Interview Questions & Answers

### **Q1: Explain the RAG architecture in your project.**

**Answer:**
"Our RAG system follows the standard Retrieval-Augmented Generation pattern:

1. **Retrieval Phase**: When a user asks a question, we:
   - Generate an embedding for the query using BGE-M3
   - Calculate cosine similarity with all stored video transcript embeddings
   - Retrieve the top 5 most semantically similar chunks

2. **Augmentation Phase**: We inject the retrieved context into a structured prompt that includes:
   - The relevant video chunks with metadata (title, timestamps, text)
   - The user's original question
   - Instructions for the LLM on how to format the response

3. **Generation Phase**: The augmented prompt is sent to Llama 3.2, which generates a natural language answer that:
   - References specific videos and timestamps
   - Synthesizes information from multiple chunks
   - Provides actionable guidance

This approach combines the knowledge retrieval capabilities of embeddings with the language understanding of LLMs."

---

### **Q2: Why did you choose BGE-M3 for embeddings?**

**Answer:**
"We chose BGE-M3 for several reasons:

1. **Multilingual Support**: Our videos are in Hindi, but we translate to English. BGE-M3 handles both languages well and works excellently with translated text.

2. **Quality**: BGE-M3 produces 1024-dimensional embeddings that capture semantic meaning effectively, as shown by its strong performance on MTEB benchmarks.

3. **Accessibility**: Available via Ollama, which allows local deployment without API costs or rate limits.

4. **Balance**: Good trade-off between embedding quality and inference speed compared to alternatives like OpenAI's text-embedding-ada-002 or sentence-transformers models.

5. **Open Source**: No API dependencies, full control over the embedding process."

---

### **Q3: Which database do you use in this project?**

**Answer:**
"We do NOT use any database in this project - neither a traditional database nor a vector database.

**What We Use Instead:**
- **File-based storage**: Embeddings are stored in `data/embeddings_df.joblib` (serialized pandas DataFrame)
- **In-memory processing**: At startup, we load the DataFrame into RAM using `joblib.load()`
- **Similarity search**: We use scikit-learn's `cosine_similarity` for brute-force comparison with all embeddings in memory

**Why No Database:**
- Our dataset size is manageable (~1000-5000 chunks)
- In-memory search is fast enough (<100ms) for our use case
- Simpler architecture: no database setup, no network calls, no infrastructure overhead
- Zero cost: no database hosting or maintenance
- Sufficient for single-user, moderate dataset size

**Storage Flow:**
1. Preprocessing: Generate embeddings → Save to `embeddings_df.joblib` (disk)
2. Application startup: Load from disk → Store in pandas DataFrame (RAM)
3. Query time: Search in-memory DataFrame using cosine similarity

**When We'd Use a Database:**
- If dataset exceeds 100K chunks
- Need sub-10ms search latency
- Require metadata filtering (e.g., filter by video title)
- Need real-time updates without restart
- Multiple applications need shared access

If we needed a database, we'd choose:
- **Chroma**: Easiest integration, good for small-medium scale
- **Pinecone**: Managed, highly scalable, best for production
- **Qdrant**: High performance, self-hosted option"

---

### **Q3b: How do you handle similarity search? Do you use a vector database?**

**Answer:**
"We use an in-memory approach rather than a vector database:

**Current Implementation:**
- Embeddings are stored in a pandas DataFrame loaded into RAM at startup
- We use scikit-learn's `cosine_similarity` for brute-force comparison
- All embeddings are compared against the query embedding using NumPy operations

**Code Flow:**
```python
# Load embeddings into memory
df = joblib.load('data/embeddings_df.joblib')

# Generate query embedding
query_emb = embedding_service.create_embedding([user_query])[0]

# Brute-force cosine similarity
similarities = cosine_similarity(
    np.vstack(df['embedding']), 
    [query_emb]
).flatten()

# Get top K
top_indices = similarities.argsort()[::-1][:top_k]
```

**Why This Approach:**
- For our dataset size (~1000-5000 chunks), in-memory search is fast enough (<100ms)
- Simpler architecture: no database setup, no network calls
- Direct NumPy operations are very efficient for this scale
- Sufficient for our use case: single-user, moderate dataset size

**When We'd Migrate to Vector Database:**
- If we scale beyond 100K chunks
- If we need sub-10ms latency
- If we need advanced features like filtering or hybrid search
- If multiple applications need shared access
- If we need real-time updates without restart

For production at scale, we'd consider:
- **Chroma**: Easiest to integrate, Python-native
- **Pinecone**: Fully managed, highly scalable
- **Qdrant**: High performance, self-hosted
- **Weaviate**: Open-source, feature-rich"

---

### **Q4: Explain the transcription pipeline. Why Whisper?**

**Answer:**
"Our transcription pipeline:

1. **Audio Extraction**: FFmpeg extracts audio from videos (MP4 → MP3)
2. **Transcription**: Whisper large-v2 transcribes audio to text
3. **Translation**: Since videos are in Hindi, we use `task="translate"` to convert to English
4. **Segmentation**: Whisper automatically segments audio into chunks with timestamps
5. **Storage**: Transcripts saved as JSON with metadata

**Why Whisper:**
- **State-of-the-art accuracy**: Best open-source speech-to-text model
- **Multilingual**: Handles Hindi and many other languages
- **Timestamp generation**: Automatic word/segment-level timestamps
- **Translation capability**: Can translate while transcribing
- **GPU acceleration**: 5-6x faster with CUDA support
- **Open source**: No API costs, runs locally

**Model Choice (large-v2):**
- Highest accuracy for production use
- Good balance between quality and speed
- ~1.5GB model size (manageable)"

---

### **Q5: How do you ensure the LLM uses the retrieved context correctly?**

**Answer:**
"We use prompt engineering to guide the LLM:

1. **Structured Context**: We format retrieved chunks as JSON with clear metadata (video number, title, timestamps, text)

2. **Explicit Instructions**: The prompt includes:
   - Clear directive to answer based on provided chunks
   - Format requirements (e.g., "Video 3, 4:50 mins")
   - Boundary conditions (only answer about video content)

3. **Example Format**: We include an example of the desired output format

4. **Context Ordering**: We present chunks in order of relevance (highest similarity first)

5. **Metadata Inclusion**: By including video numbers and timestamps in the context, the LLM naturally references them

**Prompt Structure:**
```
Context: [JSON of relevant chunks]
Question: "{user_query}"
Instructions: [Clear guidelines]
Example: [Sample response format]
```

This structured approach ensures the LLM follows the RAG pattern correctly."

---

### **Q6: What are the limitations of your current architecture?**

**Answer:**
"Several limitations we're aware of:

1. **Scalability**: In-memory storage limits us to ~10K-100K chunks depending on RAM. For larger datasets, we'd need a vector database.

2. **No Incremental Updates**: Adding new videos requires:
   - Re-running the entire preprocessing pipeline
   - Regenerating all embeddings
   - Restarting the application

3. **Single-User**: Current architecture assumes single-user access. For multi-user, we'd need:
   - Session management
   - Rate limiting
   - Caching layer

4. **No Filtering**: Can't filter by metadata (e.g., "only videos from 2024") without post-processing.

5. **Synchronous Processing**: All operations are blocking. For better UX, we'd want:
   - Async embedding generation
   - Streaming LLM responses
   - Background preprocessing

6. **No Caching**: Every query regenerates embeddings and LLM responses, even for similar queries.

**Future Improvements:**
- Vector database for scalability
- Redis for caching
- Async processing with Celery
- Incremental embedding updates"

---

### **Q7: How do you measure the quality of your RAG system?**

**Answer:**
"Currently, we use qualitative evaluation:

1. **Similarity Scores**: Cosine similarity scores indicate relevance (0-1 scale)
2. **Manual Review**: Test queries and verify:
   - Retrieved chunks are relevant
   - LLM responses are accurate
   - Timestamps are correct

**For Production, We'd Add:**
- **Retrieval Metrics**: Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
- **Generation Metrics**: BLEU, ROUGE, BERTScore for answer quality
- **End-to-End Metrics**: Human evaluation scores, user feedback
- **A/B Testing**: Compare different models/configurations

**Challenges:**
- No ground truth labels for retrieval
- Subjective evaluation of answer quality
- Need domain experts for validation"

---

### **Q8: Why did you choose Ollama over OpenAI API or other services?**

**Answer:**
"Ollama provides several advantages:

1. **Cost**: No API costs - runs entirely locally
2. **Privacy**: Data never leaves our infrastructure
3. **Control**: Full control over models and versions
4. **Latency**: No network calls (except local API)
5. **Flexibility**: Easy to swap models (bge-m3, llama3.2, etc.)
6. **Offline**: Works without internet connection

**Trade-offs:**
- Requires local GPU/CPU resources
- Model management overhead
- Less convenient than managed APIs

**When We'd Use Managed APIs:**
- For prototyping or demos
- If we need cutting-edge models (GPT-4, Claude)
- If infrastructure management is a concern
- For serverless deployments"

---

### **Q9: Explain the cosine similarity calculation in detail.**

**Answer:**
"Cosine similarity measures the angle between two vectors:

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A · B = dot product of vectors
- ||A|| = magnitude (L2 norm) of vector A
- ||B|| = magnitude (L2 norm) of vector B

**In Our Code:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Query embedding: 1 × 1024 vector
query_emb = [0.1, 0.2, ..., 0.9]  # 1024 dimensions

# All embeddings: N × 1024 matrix
all_embs = np.vstack(df['embedding'])  # N chunks × 1024 dims

# Calculate similarity: N × 1 array
similarities = cosine_similarity(all_embs, [query_emb]).flatten()
```

**Why Cosine Similarity:**
- **Normalization**: Measures direction, not magnitude (good for embeddings)
- **Range**: Returns values between -1 and 1 (typically 0 to 1 for embeddings)
- **Semantic Meaning**: Captures semantic similarity well for text embeddings
- **Efficient**: O(n) computation, optimized in scikit-learn

**Interpretation:**
- 1.0 = identical meaning
- 0.8-0.9 = very similar
- 0.5-0.7 = somewhat related
- <0.3 = unrelated"

---

### **Q10: How would you scale this system to handle 1 million video chunks?**

**Answer:**
"To scale to 1 million chunks, we'd need several changes:

1. **Vector Database Migration**
   - Replace in-memory DataFrame with Pinecone, Weaviate, or Qdrant
   - Use optimized indexes (HNSW, IVF) for sub-10ms search
   - Enable filtering and hybrid search

2. **Infrastructure**
   - Horizontal scaling: Multiple Flask workers behind load balancer
   - Caching layer: Redis for query results and embeddings
   - Database: PostgreSQL for metadata, pgvector for embeddings

3. **Processing Pipeline**
   - Async processing: Celery for background tasks
   - Batch processing: Process videos in parallel
   - Incremental updates: Add new videos without full rebuild

4. **Optimization**
   - Embedding caching: Cache query embeddings
   - Response caching: Cache LLM responses for similar queries
   - Model optimization: Quantization, distillation for faster inference

5. **Monitoring**
   - Query latency tracking
   - Error monitoring
   - Resource usage (CPU, GPU, memory)

**Architecture:**
```
Load Balancer → Flask Workers → Vector DB → Ollama Cluster
                ↓
              Redis Cache
```

This would handle millions of chunks with <100ms query latency."

---

### **Q11: How would you integrate a vector database into this project?**

**Answer:**
"I would integrate a vector database in these steps:

**1. Choose the Database:**
- For easiest integration: **Chroma** (Python-native, no server needed)
- For production scale: **Pinecone** (managed, highly scalable)
- For self-hosted: **Qdrant** (high performance, Rust-based)

**2. Modify Preprocessing Script:**
Instead of saving to `embeddings_df.joblib`, I'd:
- Generate embeddings (same as now using BGE-M3)
- Store embeddings in vector database with metadata
- Include all chunk metadata (video number, title, timestamps, text)

**Example with Chroma:**
```python
# Create collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="video_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Add chunks with embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    documents=[chunk['text'] for chunk in chunks],
    metadatas=[{
        "serial_no": chunk['serial no'],
        "title": chunk['title'],
        "start": chunk['start'],
        "end": chunk['end']
    } for chunk in chunks]
)
```

**3. Update Query Service:**
Replace in-memory similarity search with vector DB query:
```python
# Generate query embedding (same)
query_emb = embedding_service.create_embedding([user_query])[0]

# Search in vector database (instead of cosine_similarity)
results = collection.query(
    query_embeddings=[query_emb],
    n_results=top_k,
    include=['documents', 'metadatas', 'distances']
)

# Format results (same structure as before)
chunks = format_results(results)
```

**4. Add Metadata Filtering:**
Leverage vector DB's filtering capabilities:
```python
# Filter by video number
results = collection.query(
    query_embeddings=[query_emb],
    n_results=top_k,
    where={"serial_no": "3"}  # Only search in video 3
)
```

**5. Gradual Migration:**
- Use feature flag to switch between systems
- Run both in parallel for testing
- Compare results and performance
- Switch over after validation

**Benefits After Migration:**
- Scalability to millions of chunks
- Sub-10ms search latency (vs. ~50-100ms now)
- Metadata filtering capabilities
- Real-time updates without restart
- Persistent storage (survives restarts)

**Code Structure:**
```python
# config/settings.py
USE_VECTOR_DB = True  # Feature flag

# app/services/query_service.py
if Config.USE_VECTOR_DB:
    from .query_service_chroma import QueryServiceChroma
    query_service = QueryServiceChroma(video_mapping)
else:
    from .query_service_memory import QueryServiceMemory
    query_service = QueryServiceMemory(embeddings_df, video_mapping)
```

This migration would maintain backward compatibility while adding scalability."

---

## Architecture Decisions & Rationale

### **Decision 1: In-Memory vs. Vector Database**

**Choice**: In-memory pandas DataFrame

**Rationale:**
- Dataset size is manageable (<10K chunks)
- Simpler architecture for MVP
- Fast enough for single-user use case
- Easy to prototype and iterate

**Trade-off**: Limited scalability, but acceptable for current needs

---

### **Decision 2: Ollama vs. Managed APIs**

**Choice**: Ollama (local deployment)

**Rationale:**
- Cost-effective (no API costs)
- Privacy (data stays local)
- Full control over models
- Works offline

**Trade-off**: Requires local infrastructure, but provides independence

---

### **Decision 3: Whisper large-v2 vs. Smaller Models**

**Choice**: large-v2

**Rationale:**
- Highest accuracy for production
- Multilingual support needed
- Translation quality is critical
- GPU acceleration makes it feasible

**Trade-off**: Slower than smaller models, but accuracy is priority

---

### **Decision 4: Flask vs. FastAPI/Django**

**Choice**: Flask

**Rationale:**
- Lightweight and simple
- Sufficient for REST API + web UI
- Easy to understand and maintain
- Good for MVP/prototype

**Trade-off**: Less features than Django, but simpler for our needs

---

### **Decision 5: joblib vs. pickle/JSON**

**Choice**: joblib

**Rationale:**
- Optimized for NumPy arrays (embeddings)
- Faster serialization/deserialization
- Handles large DataFrames efficiently
- Standard in ML workflows

**Trade-off**: Less human-readable than JSON, but much faster

---

## Key Takeaways for Interview

1. **RAG Pattern**: Understand retrieval → augmentation → generation flow
2. **Embeddings**: Know why BGE-M3, how they're generated, where stored
3. **Similarity Search**: Understand cosine similarity, why in-memory works for us
4. **Models**: Whisper for transcription, BGE-M3 for embeddings, Llama 3.2 for generation
5. **Architecture**: Know the trade-offs of our choices (in-memory, Ollama, etc.)
6. **Scalability**: Understand limitations and how to scale (vector DB, caching, etc.)
7. **Data Flow**: Be able to trace a query from input to output
8. **Technical Details**: Embedding dimensions, similarity calculation, prompt structure

---

## Quick Reference: Input/Output Summary

| Step | Input | Tool/Model | Output | Location |
|------|-------|------------|--------|----------|
| 1. Download | YouTube URLs | yt-dlp | MP4 videos | `videos/` |
| 2. Extract Audio | MP4 videos | FFmpeg | MP3 audio | `audios/` |
| 3. Transcribe | MP3 audio | Whisper large-v2 | JSON transcripts | `jsons/` |
| 4. Embed | JSON transcripts | BGE-M3 (Ollama) | Embeddings DataFrame | `data/embeddings_df.joblib` |
| 5. Map Videos | JSON + URLs | Python script | Video mapping | `data/video_mapping.json` |
| 6. Startup | Embeddings file | Flask + joblib | In-memory DataFrame | RAM |
| 7. Query | User question | BGE-M3 + Llama 3.2 | JSON response | HTTP response |

---

**End of Interview Guide**

