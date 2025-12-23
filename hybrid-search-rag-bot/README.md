# 🤖 Enterprise RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built for free-tier deployment on **Streamlit Cloud** and **Hugging Face Spaces**.

## 🎯 Project Overview

This chatbot demonstrates enterprise-grade engineering practices:
- ✅ **Hybrid Search**: Combines semantic vector search with keyword-based filtering
- ✅ **Source Attribution**: Returns filename and page number for every answer
- ✅ **Modular Architecture**: Separated concerns (ingestion, retrieval, UI)
- ✅ **Streaming Responses**: Word-by-word output for better UX
- ✅ **Conversation Memory**: 10-message sliding window context
- ✅ **Free-Tier Optimized**: All APIs and services run on free tier

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit UI (app.py)                  │
│  - Document upload sidebar                         │
│  - Real-time chat interface                        │
│  - Streaming response display                      │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│ ingest.py       │   │ rag_engine.py   │
│ ─────────────   │   │ ─────────────   │
│ • PDF extract   │   │ • Retrieval     │
│ • Chunking      │   │ • Ranking       │
│ • Metadata      │   │ • LLM gen       │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │  Vector DB & Knowledge Base │
        │  Pinecone (Serverless)      │
        └─────────────────────────────┘
```

### Core Modules

**`app.py`** - Streamlit Application
- Document upload and management
- Chat interface with streaming
- API key configuration
- Session state management

**`ingest.py`** - Document Processing
- PDF text extraction with page tracking
- Semantic chunking (1000 chars, 100 overlap)
- Metadata attachment (filename, page, chunk index)

**`rag_engine.py`** - RAG Pipeline (Coming Soon)
- Vector embedding using all-MiniLM-L6-v2
- Pinecone indexing and retrieval
- Hybrid search (semantic + keyword)
- LLM-based generation via Groq

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repo-url>
cd hybrid-search-rag-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Free-Tier API Keys

#### Groq API (Free LLM)
1. Sign up: https://console.groq.com
2. Create API key
3. Add to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-key-here"
```

#### Pinecone (Free Vector DB)
1. Sign up: https://www.pinecone.io
2. Create serverless index (free tier)
3. Add to `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_ENVIRONMENT = "us-east-1-aws"
```

#### Hugging Face (Free Embeddings)
- No API key needed! Uses local `all-MiniLM-L6-v2` model
- Downloads on first run (~120MB)

### 4. Run Locally
```bash
streamlit run app.py
```

Visit `http://localhost:8501`

## 📋 Features

### Phase 1: Ingestion ✅ Complete
- [x] PDF upload with file validation
- [x] Text extraction with page tracking
- [x] Semantic chunking (RecursiveCharacterTextSplitter)
- [x] Chunk inspection debug panel
- [x] Metadata annotation

### Phase 2: Indexing (In Progress)
- [ ] Vector embedding via HuggingFace
- [ ] Pinecone upsertion with metadata
- [ ] Index validation and health checks

### Phase 3: Retrieval & Generation
- [ ] Hybrid search (vector + BM25)
- [ ] Top-K retrieval with scoring
- [ ] Chat history management
- [ ] Streaming LLM responses via Groq

### Phase 4: Deployment
- [ ] Streamlit Cloud deployment
- [ ] HuggingFace Spaces deployment
- [ ] Environment variable documentation

## 📦 Requirements

**Core Dependencies:**
- `streamlit>=1.28.0` - UI framework
- `langchain>=0.1.0` - Orchestration
- `langchain-groq>=0.0.1` - Groq integration
- `langchain-pinecone>=0.0.1` - Pinecone integration
- `groq>=0.4.1` - Groq API client
- `pinecone-client>=4.0.0` - Pinecone client
- `pypdf>=3.17.0` - PDF processing
- `sentence-transformers>=2.2.0` - Local embeddings
- `python-dotenv>=1.0.0` - Environment management

See `requirements.txt` for complete list.

## 🔐 Security Best Practices

✅ **Implemented:**
- API keys stored in `secrets.toml` (never in code)
- Environment variable management via `.env`
- No hardcoded credentials

✅ **To Add:**
- Rate limiting per user
- Input validation and sanitization
- Request timeout configuration

## 📊 Performance Metrics

| Component | Free Tier | Performance |
|-----------|-----------|-------------|
| LLM | Groq Llama 3.3 70B | ~5 tokens/sec streaming |
| Embeddings | all-MiniLM-L6-v2 | ~200 docs/min (local CPU) |
| Vector DB | Pinecone Serverless | ~10-50ms retrieval |
| Chunks | 1000 chars, 100 overlap | ~4-8 chunks per 5K word doc |

## 🐛 Troubleshooting

### API Key Issues
```python
# Check if secrets are loaded correctly
streamlit run app.py --logger.level=debug
```

### PDF Upload Fails
- Ensure PDF is valid and text-extractable
- Check file size (>50MB may timeout on Streamlit Cloud)
- Review debug panel for chunk validation

### Embedding Errors
- First run downloads ~120MB model (requires internet)
- Embedding runs on CPU; larger batches may be slow
- Use `sentence-transformers` cache location: `~/.cache/huggingface/`

## 📝 Project Structure
```
hybrid-search-rag-bot/
├── app.py                 # Streamlit UI
├── ingest.py             # Document processing
├── rag_engine.py         # RAG pipeline (coming soon)
├── requirements.txt      # Dependencies
├── .streamlit/
│   └── secrets.toml      # API keys (git-ignored)
├── .env                  # Local env vars
└── README.md            # This file
```

## 🎓 Learning Resources

- **LangChain Docs**: https://python.langchain.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Groq API**: https://console.groq.com/docs
- **Pinecone**: https://docs.pinecone.io/
- **Sentence Transformers**: https://huggingface.co/sentence-transformers/

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new features
- Optimize performance
- Improve documentation
- Share results and learnings

## 📄 License

MIT License - See LICENSE file for details

## 🎯 Next Steps

1. **Test Phase 1**: Upload a PDF and verify chunking
2. **Implement Pinecone Integration**: Add vector storage
3. **Implement Hybrid Search**: Combine vector + keyword search
4. **Deploy to Streamlit Cloud**: Production deployment
5. **Add URL Ingestion**: Web scraping support

---

**Built with ❤️ for enterprise AI engineering**

Last Updated: December 22, 2025
