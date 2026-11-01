"""Compares keyword/vector fusion settings on the eval questions using CACHED query vectors.

Run after `python -m eval.run_retrieval --hybrid` (which caches the query embeddings); makes no new API calls
for already-cached questions.
"""
import sys

from app import embeddings, rag
from app.config import get_settings
from app.openai_client import get_client

from .common import load_questions


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    index = rag.index_holder().get(s.database_path, s.embedding_space)
    client = get_client(s)
    qs = [q for q in load_questions()["questions"] if q["expected_chunks"]]
    vecs = {q["id"]: embeddings.embed_query(s, s.database_path, client, q["question"]) for q in qs}
    # (vector_weight, vector_candidates, vector_slots); slots=None means plain weighted RRF
    configs = [(0.0, None, None), (0.5, 10, None), (1.0, None, None), (1.0, None, 2), (1.0, None, 3),
               (1.0, None, 4)]
    for k in (8, 10):
        for w, depth, slots in configs:
            hits, rr, misses = 0, 0.0, []
            for q in qs:
                res = index.search(q["question"], vecs[q["id"]] if w else None, top_k=k,
                                   vector_weight=w, vector_candidates=depth, vector_slots=slots)
                ids = [h["chunk"]["chunk_id"] for h in res["hits"]]
                pos = [ids.index(e) + 1 for e in q["expected_chunks"] if e in ids]
                ok = len(pos) == len(q["expected_chunks"]) if q.get("require_all") else bool(pos)
                hits += ok
                rr += 1 / min(pos) if pos else 0
                if not ok:
                    misses.append(q["id"])
            print(f"k={k:<3} w={w:<4} depth={str(depth):<5} vec_slots={str(slots):<5} found {hits}/{len(qs)}  MRR {rr / len(qs):.3f}  miss {misses}")


if __name__ == "__main__":
    main()
