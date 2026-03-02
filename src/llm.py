from __future__ import annotations

from openai import OpenAI

from .config import settings


def client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    c = client()
    resp = c.embeddings.create(
        model=settings.openai_embed_model,
        input=texts,
    )
    # Keep ordering
    return [d.embedding for d in resp.data]


def chat(system: str, user: str) -> str:
    c = client()
    resp = c.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
