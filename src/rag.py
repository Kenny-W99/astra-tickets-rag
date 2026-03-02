from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np

from .config import settings
from .llm import chat, embed_texts
from .store import VectorStore


SYSTEM_PROMPT = """You are AstraTickets Support AI.
You answer using the provided context. If the answer is not in context, say you don't know.
Be concise, accurate, and cite ticket ids when relevant.
"""


@dataclass
class CacheEntry:
    value: list[dict]
    ts: float


# Simple in-memory retrieval cache (good enough for demo)
_RETRIEVAL_CACHE: dict[str, CacheEntry] = {}
_CACHE_TTL_S = 60.0


def load_store() -> VectorStore:
    return VectorStore.load(settings.index_dir)


def _cache_get(key: str) -> list[dict] | None:
    ent = _RETRIEVAL_CACHE.get(key)
    if not ent:
        return None
    if (time.time() - ent.ts) > _CACHE_TTL_S:
        _RETRIEVAL_CACHE.pop(key, None)
        return None
    return ent.value


def _cache_set(key: str, value: list[dict]) -> None:
    _RETRIEVAL_CACHE[key] = CacheEntry(value=value, ts=time.time())


def retrieve(store: VectorStore, query: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.top_k

    cache_key = f"q={query}::k={k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    qemb = embed_texts([query])[0]
    qvec = np.array(qemb, dtype="float32")
    faiss.normalize_L2(qvec.reshape(1, -1))

    out = store.search(qvec, k)
    _cache_set(cache_key, out)
    return out


def answer(query: str) -> dict:
    store = load_store()
    ctx = retrieve(store, query)

    context_block = "\n\n".join(
        [
            f"[Source {i+1}] (Ticket {c['doc_id']}, score={c.get('score', 0):.3f})\n{c['text']}"
            for i, c in enumerate(ctx)
        ]
    )

    user = f"""Question: {query}

Context:
{context_block}

Instructions:
- Use only the context.
- If missing, say: 'I don't know based on the provided context.'
- Provide a short answer and list relevant ticket ids.
"""

    completion = chat(SYSTEM_PROMPT, user)
    return {"query": query, "answer": completion, "contexts": ctx}
