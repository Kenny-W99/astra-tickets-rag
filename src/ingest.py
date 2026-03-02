from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .config import settings
from .llm import embed_texts
from .store import VectorStore
from .utils_text import chunk_text


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_documents(tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for t in tickets:
        tid = str(t.get("id"))
        title = (t.get("title") or "").strip()
        body = (t.get("body") or "").strip()
        tags = t.get("tags") or []
        created_at = t.get("created_at")

        full = f"Ticket ID: {tid}\nTitle: {title}\nTags: {', '.join(tags)}\nCreated: {created_at}\n\nBody:\n{body}\n"
        docs.append({
            "doc_id": tid,
            "text": full,
            "source": "sample:tickets",
            "title": title,
            "tags": tags,
            "created_at": created_at,
        })
    return docs


def main() -> None:
    tickets = load_jsonl(settings.data_path)
    docs = build_documents(tickets)

    chunks = []
    meta = []
    for d in docs:
        parts = chunk_text(d["text"], settings.chunk_size, settings.chunk_overlap)
        for j, p in enumerate(parts):
            chunks.append(p)
            meta.append({
                "doc_id": d["doc_id"],
                "chunk_id": f"{d['doc_id']}::chunk{j}",
                "text": p,
                "source": d["source"],
                "title": d["title"],
                "tags": d["tags"],
                "created_at": d["created_at"],
            })

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    embeds = embed_texts(chunks)
    dim = len(embeds[0])

    import faiss

    index = faiss.IndexFlatIP(dim)
    vecs = np.array(embeds, dtype="float32")
    # normalize for cosine similarity
    faiss.normalize_L2(vecs)
    index.add(vecs)

    store = VectorStore(index=index, meta=meta, dim=dim)
    store.save(settings.index_dir)
    print(f"Saved index to {settings.index_dir}")


if __name__ == "__main__":
    main()
