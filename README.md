# PDF Q&A with RAG

Upload PDFs and ask questions. The app retrieves the most relevant passages and has an LLM answer using only those, with file and page citations.

## How it works

```
PDF -> extract text (PyMuPDF) -> chunk (800 chars, 150 overlap)
    -> embed (all-MiniLM-L6-v2) -> FAISS index
Question -> embed -> top-k similar chunks -> LLM (answers from context only) -> answer + sources
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option A: Claude API
```bash
export ANTHROPIC_API_KEY="your-key"     # Windows: set ANTHROPIC_API_KEY=your-key
streamlit run app.py
```

### Option B: free local model with Ollama
```bash
ollama pull llama3.2
export LLM_PROVIDER=ollama
streamlit run app.py
```

## Docker

```bash
docker build -t pdf-rag .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY="your-key" pdf-rag
```

Open http://localhost:8501

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `ollama` |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Claude model name |
| `OLLAMA_MODEL` | `llama3.2` | Local model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

## Ideas for next steps

- Re-rank retrieved chunks with a cross-encoder
- Evaluate retrieval on a set of 20 hand-written questions (hit rate / MRR)
- Persist the index to disk so documents survive restarts
- OCR support for scanned PDFs
