from __future__ import annotations

import json
import os
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class VectorStore:
    index: faiss.Index
    meta: list[dict]
    dim: int

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "meta": self.meta}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(index_dir: str) -> "VectorStore":
        index_path = os.path.join(index_dir, "faiss.index")
        meta_path = os.path.join(index_dir, "meta.json")
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return VectorStore(index=index, meta=payload["meta"], dim=int(payload["dim"]))

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idx = self.index.search(query_vec.astype("float32"), top_k)
        out = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i == -1:
                continue
            m = dict(self.meta[i])
            m["score"] = float(s)
            out.append(m)
        return out
