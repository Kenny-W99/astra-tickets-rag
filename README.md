# AstraTickets — Enterprise RAG Demo

A small but **production-minded** RAG (Retrieval-Augmented Generation) project inspired by the DreambigCareer *AstraTickets* project series.

This repo is designed to be:
- **Runnable** in minutes (local FAISS, FastAPI, Streamlit)
- **Safe** (no secrets committed; `.env` is gitignored)
- **Interview-ready** (clear architecture, evaluation script, CI)

## Architecture

1. **Ingest** sample tickets (`data/sample/tickets.jsonl`)
2. **Chunk** text (configurable)
3. **Embed** with OpenAI embeddings
4. **Index** locally with FAISS (cosine similarity)
5. **Serve** RAG as an API (FastAPI)
6. **Demo UI** (Streamlit)

## Quickstart

### 0) Prereqs
- Python 3.11+
- An OpenAI API key

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env to set OPENAI_API_KEY
```

### 2) Build index

```bash
make ingest
```

### 3) Run API

```bash
make api
# http://localhost:8000/docs
```

### 4) Run UI

```bash
make ui
# http://localhost:8501
```

## Docker (optional)

```bash
cp .env.example .env
# set OPENAI_API_KEY

docker compose up --build
```

## Evaluation

```bash
make eval
```

This runs a tiny sanity-check suite:
- Retrieval "must-hit" checks
- A simple LLM-based faithfulness verdict (demo only)

## Repository hygiene
- `.env` is ignored
- `data/index/` is ignored (regenerate by running `make ingest`)

## Resume bullets (draft)
- Built an end-to-end **RAG support assistant** with OpenAI embeddings + **FAISS** vector search, exposing retrieval + generation via **FastAPI** and an interactive **Streamlit** UI.
- Implemented a configurable ingestion pipeline (chunking, overlap, top-k retrieval) and added lightweight **retrieval caching** to reduce latency/cost.
- Added **evaluation harness** (retrieval hit-rate + faithfulness checks) and CI (ruff + pytest) to enforce code quality.

---

**Note:** This repo uses sample data for demonstration.
