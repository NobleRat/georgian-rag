"""End-to-end answer evaluation with the REAL OpenAI API (costs money; prints a cost total).

    python -m eval.run_answers                 # all 25 questions + 2 follow-up conversations
    python -m eval.run_answers --ids G1,V1     # selected questions only

Automatic checks only. The Markdown report has an empty column for human review:
automatic, visual and human checks are kept separate on purpose.
"""
from __future__ import annotations

import argparse
import sys

from app import db, rag
from app.config import get_settings
from app.openai_client import get_client

from .common import facts_found, load_questions, write_result


def run_turns(s, turns: list[str], image_mode: str = "auto") -> list[dict]:
    conv = None
    out = []
    for question in turns:
        rec = {"question": question, "answer": "", "sources": [], "images": [], "error": None, "done": None}
        for ev in rag.answer_stream(s, s.database_path, question, conv, image_mode=image_mode):
            t = ev["type"]
            if t == "conversation":
                conv = ev["conversation_id"]
            elif t == "sources":
                rec["sources"] = ev["sources"]
                rec["images"] = [i["path"] for i in ev["images"]]
                rec["search_query"] = ev["search_query"]
                rec["search_mode"] = ev["search_mode"]
            elif t == "delta":
                rec["answer"] += ev["text"]
            elif t == "done":
                rec["done"] = ev
            elif t == "error":
                rec["error"] = ev["message"]
        out.append(rec)
    return out


def evaluate(q: dict, rec: dict, expected_chunks, expected_facts) -> dict:
    offered = [x["chunk_id"] for x in rec["sources"] if x["sent_to_model"]]
    cit = (rec["done"] or {}).get("citations") or rag.check_citations(rec["answer"], offered)
    checks = {}
    if q.get("expect_insufficient"):
        checks["insufficient_acknowledged"] = rag.INSUFFICIENT in rec["answer"]
    else:
        if expected_chunks:
            hit = [c for c in expected_chunks if c in offered]
            checks["retrieved"] = bool(hit) if not q.get("require_all") else len(hit) == len(expected_chunks)
            cited_exp = [c for c in expected_chunks if c in cit["valid"]]
            checks["expected_source_cited"] = bool(cited_exp) if not q.get("require_all") else \
                len(cited_exp) == len(expected_chunks)
        ff = facts_found(rec["answer"], expected_facts)
        checks["facts_present"] = all(ff.values())
        checks["no_false_insufficient"] = rag.INSUFFICIENT not in rec["answer"]
        if q.get("expected_images"):
            checks["page_image_sent"] = all(i in rec["images"] for i in q["expected_images"])
    checks["citations_all_valid"] = not cit["invalid"]
    checks["no_error"] = rec["error"] is None
    return {"checks": checks, "passed": all(checks.values()), "citations": cit}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--ids")
    p.add_argument("--no-conversations", action="store_true")
    args = p.parse_args()
    s = get_settings()
    get_client(s)  # fails early with a Georgian message if the key is missing
    data = load_questions()
    wanted = set(args.ids.split(",")) if args.ids else None
    with db.connect(s.database_path) as con:
        usage_before = con.execute("SELECT COALESCE(MAX(id), 0) FROM api_usage").fetchone()[0]

    results = []
    for q in data["questions"]:
        if wanted and q["id"] not in wanted:
            continue
        rec = run_turns(s, [q["question"]])[0]
        ev = evaluate(q, rec, q["expected_chunks"], q["expected_facts"])
        results.append({"id": q["id"], "category": q["category"], "question": q["question"],
                        "reference_answer": q["reference_answer"], **rec, **ev})
        print(f"{q['id']:<4} {'PASS' if ev['passed'] else 'FAIL'} {ev['checks']}")

    convs = []
    if not args.no_conversations and not wanted:
        for c in data["conversations"]:
            recs = run_turns(s, c["turns"])
            ev = evaluate({}, recs[-1], c["final_expected_chunks"], c["final_expected_facts"])
            convs.append({"id": c["id"], "turns": recs, **ev})
            print(f"{c['id']:<4} {'PASS' if ev['passed'] else 'FAIL'} {ev['checks']} "
                  f"(საძიებო ფორმულირება: {recs[-1].get('search_query')})")

    with db.connect(s.database_path) as con:
        cost = con.execute("SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM api_usage WHERE id>?",
                           (usage_before,)).fetchone()
    n_pass = sum(r["passed"] for r in results) + sum(c["passed"] for c in convs)
    n_all = len(results) + len(convs)
    summary = {"model": s.chat_model, "embedding_space": s.embedding_space, "passed": n_pass, "total": n_all,
               "api_calls": cost[1], "cost_usd": round(cost[0], 5)}
    print(f"\nავტომატური შემოწმება: {n_pass}/{n_all} · API გამოძახებები: {cost[1]} · ხარჯი ≈ ${cost[0]:.4f}")

    md = ["# პასუხების შემოწმება რეალური OpenAI API-ით\n",
          f"- მოდელი: `{s.chat_model}`, embeddings: `{s.embedding_space}`",
          f"- ავტომატური შემოწმება გაიარა: **{n_pass}/{n_all}**; ხარჯი ≈ ${cost[0]:.4f} ({cost[1]} გამოძახება)",
          "- „ადამიანური შეფასება“ სვეტი შესავსებია ადამიანის მიერ; ავტომატური შედეგი მას არ ანაცვლებს.\n",
          "| ID | კატეგორია | ავტომატური | ჩავარდნილი შემოწმებები | ადამიანური შეფასება |", "|---|---|---|---|---|"]
    for r in results + convs:
        failed = [k for k, v in r["checks"].items() if not v]
        md.append(f"| {r['id']} | {r.get('category', 'საუბარი')} | {'✅' if r['passed'] else '❌'} | "
                  f"{', '.join(failed) or '—'} | |")
    md.append("\n## პასუხები\n")
    for r in results:
        md += [f"### {r['id']}. {r['question']}", f"**საცნობარო პასუხი:** {r['reference_answer']}\n",
               f"**აპლიკაციის პასუხი:**\n\n{r['answer'] or '(ცარიელი)'}\n",
               f"სურათები: {', '.join(r['images']) or '—'} · მითითებები: {r['citations']['valid']}"
               f"{' · ვერ დადასტურდა: ' + str(r['citations']['invalid']) if r['citations']['invalid'] else ''}"
               f"{' · შეცდომა: ' + r['error'] if r['error'] else ''}\n"]
    for c in convs:
        md.append(f"### საუბარი {c['id']}")
        for t in c["turns"]:
            md += [f"- **კითხვა:** {t['question']} (ძიება: „{t.get('search_query')}“)", f"\n{t['answer']}\n"]
    j, m = write_result("answers-real-api", {"summary": summary, "questions": results, "conversations": convs},
                        "\n".join(md) + "\n")
    print("ანგარიში:", m)


if __name__ == "__main__":
    main()
