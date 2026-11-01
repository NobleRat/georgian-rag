"""Hybrid retrieval: BM25 keyword search + cosine similarity, fused with Reciprocal Rank Fusion.

The keyword index is rebuilt in memory from SQLite at startup (milliseconds, no
API calls). Vectors are read from the embeddings table for the configured space only.
"""
from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import db
from .embeddings import EMBEDDED_LAYERS, from_blob
from .importer import LAYER_FORMULA, LAYER_NATIVE, LAYER_OCR, LAYER_VISUAL
from .textproc import tokenize

MAIN_LAYERS = (LAYER_NATIVE, LAYER_VISUAL, LAYER_FORMULA)
RRF_K = 60


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in dict.fromkeys(query_tokens):
            for i, tf in self.postings.get(term, ()):
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
                out[i] += self.idf[term] * tf * (self.k1 + 1) / denom
        return out


class SearchIndex:
    def __init__(self, db_path: Path, space: str):
        self.db_path, self.space = db_path, space
        with db.connect(db_path) as con:
            self.version = db.get_meta(con, "index_version", "0")
            rows = [db.row_to_chunk(r) for r in con.execute("SELECT * FROM chunks ORDER BY document_id, chunk_id")]
            vec_rows = con.execute(
                f"SELECT c.chunk_id, e.vector FROM chunks c JOIN embeddings e ON e.embed_hash=c.embed_hash "
                f"AND e.space=? WHERE c.layer IN ({','.join('?' * len(EMBEDDED_LAYERS))})",
                (space, *EMBEDDED_LAYERS)).fetchall()
        self.main = [c for c in rows if c["layer"] in MAIN_LAYERS]
        self.ocr = [c for c in rows if c["layer"] == LAYER_OCR]
        self.by_id = {c["chunk_id"]: c for c in rows}
        # Formula supplements are attached when their parent chunk or a direct neighbour is
        # retrieved (an equation often sits in a tiny paragraph right after its explanation).
        self.formulas_by_parent: dict[str, list[dict]] = defaultdict(list)
        native_by_doc: dict[str, list[str]] = defaultdict(list)
        for c in self.main:
            if c["layer"] == LAYER_NATIVE:
                native_by_doc[c["document_id"]].append(c["chunk_id"])
        neighbours: dict[str, list[str]] = {}
        for ids in native_by_doc.values():
            for i, cid in enumerate(ids):
                neighbours[cid] = ids[max(0, i - 1): i + 2]
        for c in self.main:
            parent = c["extra"].get("parent_chunk_id") if c["layer"] == LAYER_FORMULA else None
            for cid in neighbours.get(parent, [parent] if parent else []):
                self.formulas_by_parent[cid].append(c)
        self.bm25_main = BM25([tokenize(c["embed_text"]) for c in self.main])
        self.bm25_ocr = BM25([tokenize(c["embed_text"]) for c in self.ocr])
        vecs = {r["chunk_id"]: from_blob(r["vector"]) for r in vec_rows}
        self.dims = len(next(iter(vecs.values()))) if vecs else 0
        self.has_vec = np.array([c["chunk_id"] in vecs for c in self.main], dtype=bool)
        self.matrix = np.zeros((len(self.main), self.dims), dtype=np.float32)
        for i, c in enumerate(self.main):
            if c["chunk_id"] in vecs:
                self.matrix[i] = vecs[c["chunk_id"]]

    @property
    def vector_count(self) -> int:
        return int(self.has_vec.sum())

    def _doc_mask(self, items: list[dict], document_ids) -> np.ndarray:
        if not document_ids:
            return np.ones(len(items), dtype=bool)
        allowed = set(document_ids)
        return np.array([c["document_id"] in allowed for c in items], dtype=bool)

    def search(self, query: str, query_vec: np.ndarray | None = None, top_k: int = 8,
               document_ids: list[str] | None = None, include_ocr: bool = False, candidates: int = 40,
               vector_weight: float = 1.0, vector_candidates: int | None = None,
               vector_slots: int | None = None) -> dict:
        mask = self._doc_mask(self.main, document_ids)
        q_tokens = tokenize(query, query=True)
        bm = self.bm25_main.scores(q_tokens)
        bm_order = [i for i in np.argsort(-bm) if mask[i] and bm[i] > 0][:candidates]
        bm_rank = {int(i): r for r, i in enumerate(bm_order)}

        vec_rank: dict[int, int] = {}
        cos = None
        use_vec = query_vec is not None and self.vector_count > 0 and len(query_vec) == self.dims
        if use_vec:
            cos = self.matrix @ np.asarray(query_vec, dtype=np.float32)
            valid = mask & self.has_vec
            order = [i for i in np.argsort(-cos) if valid[i]][:vector_candidates or candidates]
            vec_rank = {int(i): r for r, i in enumerate(order)}

        fused: dict[int, float] = defaultdict(float)
        for i, r in bm_rank.items():
            fused[i] += 1.0 / (RRF_K + r + 1)
        for i, r in vec_rank.items():
            fused[i] += vector_weight / (RRF_K + r + 1)
        if vector_slots is None or not vec_rank:
            ranked = sorted(fused, key=lambda i: -fused[i])[:top_k]
        else:
            # Keyword-first: measured on Georgian eval questions, text-embedding-3-large alone found only
            # 10/22 expected sources while BM25 found 22/22, so vectors only add extra candidates.
            kw_slots = max(top_k - vector_slots, 0)
            ranked = bm_order[:kw_slots]
            for i in sorted(vec_rank, key=vec_rank.get):
                if len(ranked) >= top_k:
                    break
                if i not in ranked:
                    ranked.append(i)
            ranked = [int(i) for i in ranked]

        hits: list[dict] = []
        seen: set[str] = set()

        def add(chunk: dict, **scores):
            if chunk["chunk_id"] in seen:
                return
            seen.add(chunk["chunk_id"])
            hits.append({"chunk": chunk, **scores})

        for i in ranked:
            c = self.main[i]
            add(c, rrf=round(fused[i], 5), bm25=round(float(bm[i]), 3),
                cosine=round(float(cos[i]), 4) if cos is not None and self.has_vec[i] else None,
                keyword_rank=bm_rank.get(i, None), vector_rank=vec_rank.get(i, None), via="search")
            for f in self.formulas_by_parent.get(c["chunk_id"], []):
                add(f, rrf=None, bm25=None, cosine=None, keyword_rank=None, vector_rank=None,
                    via="formula_supplement")

        if include_ocr and self.ocr:
            omask = self._doc_mask(self.ocr, document_ids)
            osc = self.bm25_ocr.scores(q_tokens)
            for i in [i for i in np.argsort(-osc) if omask[i] and osc[i] > 0][:3]:
                add(self.ocr[i], rrf=None, bm25=round(float(osc[i]), 3), cosine=None, keyword_rank=None,
                    vector_rank=None, via="ocr_candidate_page")

        return {
            "mode": "hybrid" if use_vec else "keyword",
            "query_tokens": q_tokens,
            "vector_coverage": f"{self.vector_count}/{len(self.main)}",
            "hits": hits,
        }


class IndexHolder:
    """Keeps one SearchIndex in memory and reloads it when data or the embedding space changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index: SearchIndex | None = None

    def get(self, db_path: Path, space: str) -> SearchIndex:
        with db.connect(db_path) as con:
            version = db.get_meta(con, "index_version", "0")
        with self._lock:
            idx = self._index
            if idx is None or idx.version != version or idx.space != space or idx.db_path != db_path:
                idx = SearchIndex(db_path, space)
                self._index = idx
            return idx
