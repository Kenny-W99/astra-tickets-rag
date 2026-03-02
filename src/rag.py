from __future__ import annotations

import numpy as np
import faiss

from .config import settings
from .llm import chat, embed_texts
from .store import VectorStore


SYSTEM_PROMPT = """You are AstraTickets Support AI.
You answer using the provided context. If the answer is not in context, say you don't know.
Be concise, accurate, and cite ticket ids when relevant.
"""


def load_store() -> VectorStore:
    return VectorStore.load(settings.index_dir)


def retrieve(store: VectorStore, query: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.top_k
    qemb = embed_texts([query])[0]
    qvec = np.array(qemb, dtype="float32")
    faiss.normalize_L2(qvec.reshape(1, -1))
    return store.search(qvec, k)


def answer(query: str) -> dict:
    store = load_store()
    ctx = retrieve(store, query)
    context_block = "\n\n".join(
        [f"[Source {i+1}] (Ticket {c['doc_id']})\n{c['text']}" for i, c in enumerate(ctx)]
    )

    user = f"""Question: {query}

Context:
{context_block}

Instructions:
- Use only the context.
- If missing, say: 'I don't know based on the provided context.'
- Provide a short answer and list any relevant ticket ids.
"""

    completion = chat(SYSTEM_PROMPT, user)
    return {"query": query, "answer": completion, "contexts": ctx}
