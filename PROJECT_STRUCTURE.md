### 2. **Directory Structure**
   ```
   rag_based_ai/
   ├── run.py              # 🆕 Clean entry point
   ├── app/                # 🆕 Flask application package
   │   ├── routes/         # 🆕 URL handlers (blueprints)
   │   ├── services/       # 🆕 Business logic
   │   ├── utils/          # 🆕 Helper functions
   │   ├── static/         # ✅ Moved from root
   │   └── templates/      # ✅ Moved from root
   ├── config/             # 🆕 Configuration management
   ├── scripts/            # ✅ Organized preprocessing scripts
   └── data/               # ✅ Centralized data storage
   ```

### 4. **Centralized Configuration**
   - All settings in `config/settings.py`
   - Easy to modify without code changes
   - Environment-aware setup ready

### 5. **Organized Scripts**
   - Moved all preprocessing scripts to `scripts/`
   - Updated paths to work from any directory
   - Added comprehensive scripts documentation

---

## 📂 File Organization

### Before (Monolithic)
```
app.py 
├── Flask app setup
├── Data loading functions
├── Embedding service
├── LLM service
├── Query processing
├── All route handlers
└── Main execution
```

### After (Modular)
```
run.py                               # Entry point
app/__init__.py                      # App factory
app/routes/main.py                   # Web routes
app/routes/api.py                    # API routes
app/services/embedding_service.py    # Embeddings
app/services/llm_service.py          # LLM inference
app/services/query_service.py        # RAG logic
app/utils/helpers.py                 # Utilities
config/settings.py                   # Configuration
```

---

## 🚀 How to Use

### Start the Application
```bash
# 1. Activate virtual environment
torch_env\Scripts\activate

# 2. Run the application
python run.py

# 3. Open browser
# http://localhost:5000
```

### Modify Configuration
Edit `config/settings.py`:
```python
# Change models
OLLAMA_EMBEDDING_MODEL = 'bge-m3'
OLLAMA_LLM_MODEL = 'llama3.2'

# Change settings
TOP_K_RESULTS = 7
PORT = 5000
```

### Run Scripts
```bash
python scripts/preprocess_json.py
python scripts/create_video_mapping.py
```

## 📝 Important Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry point to start the app |
| `app/__init__.py` | Flask app factory and initialization |
| `config/settings.py` | All configuration in one place |
| `requirements.txt` | Python dependencies |

---

## 🔧 Configuration Options

All in `config/settings.py`:

```python
# Flask
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Ollama
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_EMBEDDING_MODEL = 'bge-m3'
OLLAMA_LLM_MODEL = 'llama3.2'

# RAG
TOP_K_RESULTS = 5

# Paths (auto-managed)
DATA_DIR = BASE_DIR / 'data'
VIDEOS_DIR = BASE_DIR / 'videos'
# ... and more
```

---

## ✨ Next Steps

1. **Test the Application**
   ```bash
   python run.py
   # Visit http://localhost:5000
   ```

3. **Customize Configuration**
   - Edit `config/settings.py` as needed
   - No code changes required!

4. **Add New Features**
   - New routes → `app/routes/`
   - New services → `app/services/`
   - New utilities → `app/utils/`

5. **Deploy to Production**
   - Use Gunicorn or Waitress
   - Environment variables for secrets
   - Production-ready structure!

---

## 🎓 Learning Resources

- **Flask Application Factory**: [Flask Docs](https://flask.palletsprojects.com/en/latest/patterns/appfactories/)
- **Blueprints**: [Flask Blueprints](https://flask.palletsprojects.com/en/latest/blueprints/)
- **Project Structure**: [Flask Best Practices](https://flask.palletsprojects.com/en/latest/tutorial/layout/)

---

## 🐛 Troubleshooting

### Application won't start
```bash
# Check you're in project root
pwd  # Should show: .../rag_based_ai

# Activate virtual environment
torch_env\Scripts\activate

# Check Ollama is running
curl http://localhost:11434/api/tags
```

### Import errors
```bash
# Ensure virtual environment is activated
# Should see (torch_env) in prompt
```

### Scripts not working
```bash
# Scripts handle paths automatically
python scripts/script_name.py  # From project root
# OR
cd scripts && python script_name.py
```

**Happy coding! 🚀**

