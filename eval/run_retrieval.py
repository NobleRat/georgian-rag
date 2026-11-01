"""Retrieval evaluation on the 25 questions.

    python -m eval.run_retrieval            # keyword only: offline, no API calls
    python -m eval.run_retrieval --hybrid   # REAL API: embeds the questions (needs key + document embeddings)
"""
from __future__ import annotations

import argparse
import sys

from app import embeddings, rag
from app.config import get_settings
from app.openai_client import get_client

from .common import load_questions, write_result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("-k", type=int, default=None)
    args = p.parse_args()
    s = get_settings()
    k = args.k or s.top_k
    index = rag.index_holder().get(s.database_path, s.embedding_space)
    client = None
    if args.hybrid:
        if index.vector_count == 0:
            sys.exit("embeddings ჯერ არ არის შექმნილი — ჯერ გაუშვით: python -m app.cli embed")
        client = get_client(s)
    rows, found_all, rr = [], 0, []
    qs = [q for q in load_questions()["questions"] if q["expected_chunks"]]
    for q in qs:
        qvec = embeddings.embed_query(s, s.database_path, client, q["question"]) if client else None
        res = index.search(q["question"], qvec, top_k=k, vector_slots=s.vector_slots)
        ids = [h["chunk"]["chunk_id"] for h in res["hits"]]
        exp = q["expected_chunks"]
        positions = {e: (ids.index(e) + 1 if e in ids else None) for e in exp}
        ok = all(positions.values()) if q.get("require_all") else any(positions.values())
        best = min([p for p in positions.values() if p], default=None)
        rr.append(1 / best if best else 0)
        found_all += ok
        rows.append({"id": q["id"], "category": q["category"], "mode": res["mode"], "ok": ok,
                     "positions": positions, "top_ids": ids})
        print(f"{q['id']:<4} {'OK ' if ok else 'MISS'} {positions}  top: {ids[:5]}")
    n = len(qs)
    summary = {"mode": "hybrid" if client else "keyword", "k": k, "questions": n, "hit_rate": found_all / n,
               "mrr": sum(rr) / n, "space": s.embedding_space if client else None}
    print(f"\nრეჟიმი: {summary['mode']} · top-{k} · მოიძებნა {found_all}/{n} · MRR {summary['mrr']:.3f}")
    md = [f"# მოძიების შემოწმება ({summary['mode']})\n",
          f"- რეალური API: {'კი (კითხვების embeddings)' if client else 'არა — მხოლოდ ადგილობრივი საკვანძო სიტყვებით ძიება'}",
          f"- top-{k}, სწორად მოძიებული: **{found_all}/{n}**, MRR: {summary['mrr']:.3f}\n",
          "| ID | კატეგორია | შედეგი | მოსალოდნელი წყაროს პოზიცია |", "|---|---|---|---|"]
    md += [f"| {r['id']} | {r['category']} | {'✅' if r['ok'] else '❌'} | {r['positions']} |" for r in rows]
    j, m = write_result(f"retrieval-{summary['mode']}", {"summary": summary, "rows": rows}, "\n".join(md) + "\n")
    print("ანგარიში:", m)


if __name__ == "__main__":
    main()
