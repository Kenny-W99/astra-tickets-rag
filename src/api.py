from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .rag import answer

app = FastAPI(title="AstraTickets RAG API", version="0.1.0")


class QueryIn(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(q: QueryIn):
    return answer(q.query)
