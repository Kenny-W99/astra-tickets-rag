from __future__ import annotations

import json

from .config import settings
from .rag import retrieve, load_store
from .llm import chat

EVAL_QUESTIONS = [
    {
        "q": "Customer booked a flight but it was cancelled. They want a refund. What should we tell them?",
        "must_hit": "TCK-001",
    },
    {
        "q": "A user can't reset their password because the reset email never arrives. What are common causes?",
        "must_hit": "TCK-002",
    },
]

SYSTEM = """You are a strict evaluator. Given contexts and an answer, judge if the answer is supported by contexts.
Return JSON with keys: supported (true/false), notes (string).
"""


def main() -> None:
    store = load_store()

    results = []
    for item in EVAL_QUESTIONS:
        ctx = retrieve(store, item["q"], top_k=settings.top_k)
        hit = any(c.get("doc_id") == item["must_hit"] for c in ctx)

        context_block = "\n\n".join([f"(Ticket {c['doc_id']})\n{c['text']}" for c in ctx])
        answer = chat(
            "You are a helpful support agent.",
            f"Question: {item['q']}\n\nContext:\n{context_block}\n\nAnswer briefly.",
        )

        verdict_raw = chat(
            SYSTEM,
            f"Question: {item['q']}\n\nAnswer: {answer}\n\nContexts:\n{context_block}\n\nReturn JSON.",
        )
        try:
            verdict = json.loads(verdict_raw)
        except Exception:
            verdict = {"supported": None, "notes": verdict_raw}

        results.append({
            "q": item["q"],
            "must_hit": item["must_hit"],
            "retrieval_hit": hit,
            "answer": answer,
            "verdict": verdict,
        })

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
