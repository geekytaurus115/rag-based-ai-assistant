# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                    (Web Browser)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Request
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                         │
│                   (Flask Routes)                             │
│  ┌──────────────┐          ┌──────────────┐                │
│  │  main.py     │          │   api.py     │                │
│  │              │          │              │                │
│  │ GET /        │          │ POST /api/   │                │
│  │ GET /videos/ │          │      query   │                │
│  └──────┬───────┘          │ GET  /api/   │                │
│         │                  │      health  │                │
│         │                  │ GET  /api/   │                │
│         │                  │      stats   │                │
│         │                  └──────┬───────┘                │
└─────────┼─────────────────────────┼─────────────────────────┘
          │                         │
          │                         ▼
┌─────────┼─────────────────────────────────────────────────┐
│         │            BUSINESS LOGIC LAYER                  │
│         │               (Services)                         │
│         │                                                  │
│         │      ┌──────────────────────────┐               │
│         │      │   QueryService           │               │
│         │      │  - process_query()       │               │
│         └──────┤  - format_chunks()       │               │
│                └────┬──────────────┬──────┘               │
│                     │              │                       │
│         ┌───────────▼─────┐   ┌───▼──────────────┐       │
│         │ EmbeddingService │   │   LLMService     │       │
│         │                  │   │                  │       │
│         │ - create_       │   │ - generate_      │       │
│         │   embedding()   │   │   response()     │       │
│         │ - check_        │   │ - build_rag_     │       │
│         │   connection()  │   │   prompt()       │       │
│         └─────────┬────────┘   └────────┬─────────┘      │
└───────────────────┼─────────────────────┼─────────────────┘
                    │                     │
                    │                     │
┌───────────────────┼─────────────────────┼─────────────────┐
│                   │   UTILITY LAYER     │                  │
│                   │                     │                  │
│           ┌───────▼─────────────────────▼───────┐         │
│           │         helpers.py                   │         │
│           │  - format_timestamp()                │         │
│           │  - get_youtube_url()                 │         │
│           └──────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                    │                     │
                    │                     │
┌───────────────────┼─────────────────────┼─────────────────┐
│                   │  EXTERNAL SERVICES  │                  │
│                   │                     │                  │
│         ┌─────────▼────────┐   ┌────────▼─────────┐      │
│         │  Ollama API      │   │  Ollama API       │      │
│         │  (bge-m3)        │   │  (llama3.2)       │      │
│         │  Embeddings      │   │  LLM Inference    │      │
│         └──────────────────┘   └───────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                    │                     │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                     DATA LAYER                               │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  embeddings_df   │  │  video_mapping   │               │
│  │  .joblib         │  │  .json           │               │
│  │                  │  │                  │               │
│  │ - Vector         │  │ - Video IDs      │               │
│  │   embeddings     │  │ - YouTube URLs   │               │
│  └──────────────────┘  └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## Request Flow Diagram

### User Query Processing

```
User submits query via Web UI
        │
        ▼
┌───────────────────────────────────┐
│ POST /api/query                   │
│ (app/routes/api.py)               │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ query_service.process_query()    │
│ (app/services/query_service.py)  │
└───────────┬───────────────────────┘
            │
            ├──► Step 1: Generate query embedding
            │    │
            │    ▼
            │    ┌────────────────────────────────┐
            │    │ embedding_service              │
            │    │   .create_embedding()          │
            │    │ → Ollama API (bge-m3)         │
            │    └────────────────────────────────┘
            │
            ├──► Step 2: Calculate similarity
            │    │
            │    ▼
            │    ┌────────────────────────────────┐
            │    │ cosine_similarity()            │
            │    │ (scikit-learn)                 │
            │    └────────────────────────────────┘
            │
            ├──► Step 3: Get top K chunks
            │    │
            │    ▼
            │    ┌────────────────────────────────┐
            │    │ embeddings_df.loc[top_indices] │
            │    └────────────────────────────────┘
            │
            ├──► Step 4: Format for display
            │    │
            │    ▼
            │    ┌────────────────────────────────┐
            │    │ format_chunks_for_display()    │
            │    │ → format_timestamp()           │
            │    │ → get_youtube_url()            │
            │    └────────────────────────────────┘
            │
            └──► Step 5: Generate LLM response
                 │
                 ▼
                 ┌────────────────────────────────┐
                 │ llm_service                    │
                 │   .build_rag_prompt()          │
                 │   .generate_response()         │
                 │ → Ollama API (llama3.2)       │
                 └────────────────────────────────┘
                 │
                 ▼
            ┌────────────────────────────────┐
            │ Return JSON response           │
            │ {                              │
            │   response: "...",             │
            │   chunks: [...],               │
            │   timestamp: "..."             │
            │ }                              │
            └────────────────────────────────┘
                 │
                 ▼
            Display in Web UI
```

---

## Module Dependencies

```
run.py
  │
  └─► app.create_app()
       │
       ├─► config.Config
       │    └─► settings.py
       │
       ├─► app.routes.main_bp
       │    └─► main.py
       │
       ├─► app.routes.api_bp
       │    │
       │    └─► api.py
       │         │
       │         └─► query_service
       │              │
       │              ├─► EmbeddingService
       │              │    └─► Ollama API
       │              │
       │              ├─► LLMService
       │              │    └─► Ollama API
       │              │
       │              └─► helpers
       │                   ├─► format_timestamp()
       │                   └─► get_youtube_url()
       │
       └─► Data Loading
            ├─► embeddings_df.joblib
            └─► video_mapping.json
```

---

## Service Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EmbeddingService                       │
├─────────────────────────────────────────────────────────┤
│  Responsibilities:                                       │
│  - Generate embeddings via Ollama                        │
│  - Manage connection to embedding model                  │
│  - Health check for Ollama service                       │
├─────────────────────────────────────────────────────────┤
│  Methods:                                                │
│  + create_embedding(prompt_list) → embeddings           │
│  + check_connection() → bool                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     LLMService                           │
├─────────────────────────────────────────────────────────┤
│  Responsibilities:                                       │
│  - Generate text responses via Ollama                    │
│  - Build RAG prompts with context                        │
│  - Format LLM queries                                    │
├─────────────────────────────────────────────────────────┤
│  Methods:                                                │
│  + generate_response(prompt) → text                      │
│  + build_rag_prompt(chunks, query) → prompt             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QueryService                          │
├─────────────────────────────────────────────────────────┤
│  Responsibilities:                                       │
│  - Orchestrate RAG workflow                              │
│  - Process user queries end-to-end                       │
│  - Format results for display                            │
├─────────────────────────────────────────────────────────┤
│  Dependencies:                                           │
│  - EmbeddingService                                      │
│  - LLMService                                            │
│  - helpers utilities                                     │
├─────────────────────────────────────────────────────────┤
│  Methods:                                                │
│  + process_query(query, top_k) → result                 │
│  - _format_chunks_for_display(chunks) → formatted       │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Preprocessing Pipeline (Scripts)

```
video_urls.txt
  │
  ▼
youtube_downloader.py
  │
  ▼
videos/*.mp4
  │
  ▼
video_to_mp3.py (FFmpeg)
  │
  ▼
audios/*.mp3
  │
  ▼
mp3_to_json.py (Whisper)
  │
  ▼
jsons/*.json (Transcripts)
  │
  ├──► create_video_mapping.py
  │     │
  │     ▼
  │    data/video_mapping.json
  │
  └──► preprocess_json.py (Ollama bge-m3)
        │
        ▼
       data/embeddings_df.joblib
```

### Runtime Data Flow

```
Application Startup
  │
  ├─► Load embeddings_df.joblib
  │    └─► Store in memory
  │
  └─► Load video_mapping.json
       └─► Store in memory
            │
            ▼
       QueryService initialized
            │
            ▼
       Ready to process queries
```

---

## Configuration Management

```
config/settings.py
  │
  ├─► Flask Config
  │    ├─ DEBUG
  │    ├─ HOST
  │    └─ PORT
  │
  ├─► File Paths
  │    ├─ DATA_DIR
  │    ├─ VIDEOS_DIR
  │    ├─ EMBEDDINGS_FILE
  │    └─ VIDEO_MAPPING_FILE
  │
  ├─► Ollama Config
  │    ├─ OLLAMA_BASE_URL
  │    ├─ OLLAMA_EMBEDDING_MODEL
  │    └─ OLLAMA_LLM_MODEL
  │
  └─► RAG Config
       └─ TOP_K_RESULTS
```

---

## Deployment Architecture

### Development

```
Developer Machine
  │
  ├─► Virtual Environment (torch_env)
  │    └─► Python 3.12 + Dependencies
  │
  ├─► Flask Development Server
  │    └─► python run.py (port 5000)
  │
  └─► Ollama Service
       └─► localhost:11434
```

### Production (Recommended)

```
Production Server
  │
  ├─► Virtual Environment
  │    └─► Production dependencies
  │
  ├─► WSGI Server (Gunicorn/Waitress)
  │    └─► Multiple workers
  │
  ├─► Reverse Proxy (Nginx)
  │    └─► SSL termination
  │
  └─► Ollama Service
       └─► GPU acceleration
```

---

## Technology Stack

```
┌─────────────────────────────────────────┐
│          Frontend Layer                  │
│  - HTML5 (Jinja2 templates)             │
│  - CSS3 (Modern dark theme)             │
│  - Vanilla JavaScript (Fetch API)       │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│          Backend Layer                   │
│  - Flask 3.0+ (Web framework)           │
│  - Python 3.12                          │
│  - Blueprint architecture               │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│         Processing Layer                 │
│  - pandas (Data manipulation)           │
│  - numpy (Numerical operations)         │
│  - scikit-learn (Similarity)            │
│  - joblib (Serialization)               │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│           AI Layer                       │
│  - Ollama (Local LLM server)            │
│  - BGE-M3 (Embeddings)                  │
│  - Llama 3.2 (Text generation)          │
│  - Whisper (Speech-to-text)             │
└─────────────────────────────────────────┘
```

---

## Scalability Considerations

### Current Architecture
- Single-threaded Flask
- In-memory data storage
- Synchronous request handling

### Future Enhancements
1. **Caching Layer**
   - Redis for embedding cache
   - Response caching

2. **Database Layer**
   - PostgreSQL + pgvector
   - Persistent storage

3. **Async Processing**
   - Celery for background tasks
   - Message queue (RabbitMQ)

4. **Horizontal Scaling**
   - Multiple Flask workers
   - Load balancer
   - Shared cache

---

This architecture provides a solid foundation for a production RAG application with clear separation of concerns and room for growth.

