# Project Structure Documentation

## 📂 Directory Structure

```
rag_based_ai/
│
├── 📄 run.py                    # Main entry point - Start the application here
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore               # Git ignore patterns
├── 📄 README.md                # Complete project documentation
│
├── 📁 app/                     # Main Flask application package
│   ├── __init__.py             # App factory & initialization logic
│   │
│   ├── 📁 routes/              # Blueprint routes (URL handlers)
│   │   ├── __init__.py         # Routes package exports
│   │   ├── main.py             # Main web routes (/, /videos/<path>)
│   │   └── api.py              # API endpoints (/api/*)
│   │
│   ├── 📁 services/            # Business logic layer
│   │   ├── __init__.py         # Services package exports
│   │   ├── embedding_service.py    # Ollama embedding operations
│   │   ├── llm_service.py          # Ollama LLM operations
│   │   └── query_service.py        # RAG query processing
│   │
│   ├── 📁 utils/               # Utility functions
│   │   ├── __init__.py         # Utils package exports
│   │   └── helpers.py          # Helper functions (timestamps, URLs)
│   │
│   ├── 📁 static/              # Static assets (served by Flask)
│   │   └── style.css           # Application CSS styles
│   │
│   └── 📁 templates/           # Jinja2 HTML templates
│       └── index.html          # Main web interface
│
├── 📁 config/                  # Configuration package
│   ├── __init__.py             # Config package exports
│   └── settings.py             # Application settings & constants
│
├── 📁 scripts/                 # Data preprocessing scripts
│   ├── README.md               # Scripts documentation
│   ├── youtube_downloader.py   # Download YouTube videos
│   ├── extract_audio.py        # Audio extraction utility
│   ├── video_to_mp3.py         # Convert videos to MP3
│   ├── mp3_to_json.py          # Transcribe audio with Whisper
│   ├── speech_to_text.py       # Alternative speech-to-text
│   ├── preprocess_json.py      # Generate embeddings
│   ├── create_video_mapping.py # Create video URL mappings
│   ├── process_incoming.py     # CLI query interface
│   └── video_urls.txt          # Input: List of YouTube URLs
│
├── 📁 data/                    # Data storage (generated files)
│   ├── embeddings_df.joblib    # Serialized embeddings DataFrame
│   ├── video_mapping.json      # Video number to YouTube URL mapping
│   └── output.json             # Processed output data
│
├── 📁 videos/                  # Downloaded YouTube videos
├── 📁 audios/                  # Extracted audio files (MP3)
├── 📁 jsons/                   # Transcription JSON files
└── 📁 torch_env/               # Python virtual environment
```

---

## 🏗️ Architecture Overview

### Separation of Concerns

The project follows a **modular architecture** with clear separation:

1. **Presentation Layer** (`app/routes/`)
   - Handles HTTP requests/responses
   - Routes user requests to appropriate services
   - Returns formatted responses

2. **Business Logic Layer** (`app/services/`)
   - Core RAG functionality
   - Embedding generation
   - LLM inference
   - Query processing

3. **Utility Layer** (`app/utils/`)
   - Reusable helper functions
   - Format conversions
   - URL generation

4. **Configuration Layer** (`config/`)
   - Centralized settings
   - Environment variables
   - Path management

5. **Data Processing Layer** (`scripts/`)
   - Independent preprocessing scripts
   - Data pipeline stages
   - One-time setup tasks

---

## 🔄 Application Flow

### Startup Flow
```
run.py
  └─> create_app() in app/__init__.py
       ├─> Load configuration from config/settings.py
       ├─> Load embeddings from data/embeddings_df.joblib
       ├─> Load video mapping from data/video_mapping.json
       ├─> Initialize QueryService
       ├─> Register blueprints (main_bp, api_bp)
       └─> Start Flask server
```

### Request Flow (Query Processing)
```
User Query (Web UI)
  └─> POST /api/query (app/routes/api.py)
       └─> query_service.process_query()
            ├─> embedding_service.create_embedding()
            ├─> Calculate cosine similarity
            ├─> Get top K relevant chunks
            ├─> llm_service.build_rag_prompt()
            ├─> llm_service.generate_response()
            └─> Return formatted response with chunks
```

---

## 📦 Package Responsibilities

### `app/` - Main Application
- **Purpose**: Flask application with all web interface logic
- **Key Files**:
  - `__init__.py`: App factory, initialization, data loading
  - Entry point for the Flask application

### `app/routes/` - Route Handlers
- **Purpose**: HTTP endpoint definitions (URL → Function mapping)
- **Files**:
  - `main.py`: Web page routes (`/`, `/videos/<path>`)
  - `api.py`: REST API routes (`/api/query`, `/api/health`, `/api/stats`)

### `app/services/` - Business Logic
- **Purpose**: Core functionality separated from HTTP layer
- **Files**:
  - `embedding_service.py`: Generate embeddings via Ollama
  - `llm_service.py`: LLM inference and prompt building
  - `query_service.py`: RAG pipeline orchestration

### `app/utils/` - Helpers
- **Purpose**: Reusable utility functions
- **Files**:
  - `helpers.py`: Timestamp formatting, YouTube URL generation

### `config/` - Configuration
- **Purpose**: Centralized settings management
- **Files**:
  - `settings.py`: All configurable parameters (paths, models, timeouts)

### `scripts/` - Data Pipeline
- **Purpose**: Independent data preprocessing scripts
- **Usage**: Run manually to prepare data
- **Note**: Scripts can be run from scripts directory, paths are handled automatically

### `data/` - Processed Data
- **Purpose**: Storage for generated data files
- **Files**:
  - `embeddings_df.joblib`: Pre-computed embeddings
  - `video_mapping.json`: Video ID to YouTube URL mapping

---

## 🚀 Running the Application

### Development Server
```bash
# Activate virtual environment
torch_env\Scripts\activate  # Windows
# source torch_env/bin/activate  # Linux/macOS

# Start the server
python run.py

# Access at: http://localhost:5000
```

### Production Deployment
```bash
# Use a production WSGI server
pip install gunicorn

# Run with Gunicorn (Linux/macOS)
gunicorn -w 4 -b 0.0.0.0:5000 'app:create_app()'

# Or use waitress (Windows compatible)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:create_app
```

---

## 🔧 Configuration

All configuration is centralized in `config/settings.py`:

```python
# Change these values to customize behavior
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_EMBEDDING_MODEL = 'bge-m3'
OLLAMA_LLM_MODEL = 'llama3.2'
TOP_K_RESULTS = 7
```

**No need to modify code** - just update `config/settings.py`!

---

## 🧪 Testing Individual Components

### Test Embedding Service
```python
from app.services import EmbeddingService

service = EmbeddingService()
embeddings = service.create_embedding(["test text"])
print(embeddings)
```

### Test LLM Service
```python
from app.services import LLMService

service = LLMService()
response = service.generate_response("What is AI?")
print(response)
```

### Test Query Service
```python
import joblib
from app.services import QueryService

embeddings_df = joblib.load('data/embeddings_df.joblib')
video_mapping = {}  # Load from JSON
service = QueryService(embeddings_df, video_mapping)
result = service.process_query("How to use AI?")
print(result)
```

---

## 🔍 Code Organization Benefits

✅ **Modularity**: Each component has a single responsibility  
✅ **Testability**: Easy to test individual services  
✅ **Maintainability**: Clear structure makes updates simple  
✅ **Scalability**: Easy to add new routes or services  
✅ **Reusability**: Services can be used independently  
✅ **Configuration**: Centralized settings management  
✅ **Best Practices**: Follows Flask application factory pattern  

---

## 📝 Adding New Features

### Add a New API Endpoint
1. Define route in `app/routes/api.py`
2. Add business logic to appropriate service
3. Update configuration if needed

### Add a New Service
1. Create new file in `app/services/`
2. Implement service class
3. Export from `app/services/__init__.py`
4. Use in routes or other services

### Add New Configuration
1. Add parameter to `config/settings.py`
2. Use via `from config import Config`
3. Access as `Config.PARAMETER_NAME`

---

## 🐛 Debugging

### Check Application State
```python
from app import get_app_state

state = get_app_state()
print(f"Embeddings loaded: {state['embeddings_df'] is not None}")
print(f"Videos mapped: {len(state['video_mapping'])}")
```

### Enable Flask Debug Mode
- Already enabled in `config/settings.py`: `DEBUG = True`
- Shows detailed error pages
- Auto-reloads on code changes

### Check Logs
- Flask logs appear in terminal where you ran `python run.py`
- Look for startup messages about embeddings and Ollama connection

---

## 📚 Related Documentation

- **README.md**: Complete usage guide and setup instructions
- **scripts/README.md**: Data preprocessing pipeline documentation
- **config/settings.py**: All configurable parameters with comments

---

**This structure follows industry best practices for Flask applications and provides a solid foundation for growth and maintenance.**

