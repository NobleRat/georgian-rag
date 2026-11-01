"""Command line tools.

    .venv\\Scripts\\python -m app.cli import [--library PATH --library-id ID --id-prefix P] [--documents 01,07]
    .venv\\Scripts\\python -m app.cli estimate [--documents 01,07]
    .venv\\Scripts\\python -m app.cli embed [--documents 01,07] [--yes]
    .venv\\Scripts\\python -m app.cli status
    .venv\\Scripts\\python -m app.cli export-embeddings   # prebuilt/embeddings/<space>.npz (commit it)
    .venv\\Scripts\\python -m app.cli check-openai
    .venv\\Scripts\\python -m app.cli search "კითხვა" [--documents 01]
    .venv\\Scripts\\python -m app.cli ask "კითხვა"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import embeddings, rag
from .config import get_settings
from .importer import import_library
from .openai_client import describe_error, get_client


def _docs(value: str | None):
    return [d.strip() for d in value.split(",") if d.strip()] if value else None


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_import(args):
    s = get_settings()
    root = Path(args.library) if args.library else s.data_library
    report = import_library(s.database_path, root, library_id=args.library_id, id_prefix=args.id_prefix,
                            document_ids=_docs(args.documents))
    _print(report)


def cmd_estimate(args):
    s = get_settings()
    _print(embeddings.estimate(s, s.database_path, _docs(args.documents)))


def cmd_embed(args):
    s = get_settings()
    est = embeddings.estimate(s, s.database_path, _docs(args.documents))
    cost = est["estimated_cost_usd"]
    print(f"დასამუშავებელი: {est['items']} ჩანაწერი, ~{est['tokens']:,} ტოკენი, {est['requests']} მოთხოვნა, "
          f"მიახლოებითი ხარჯი: {'$%.4f' % cost if cost is not None else 'უცნობია'} ({est['model']})")
    if not est["items"]:
        print("ყველა ჩანაწერს უკვე აქვს ვექტორი ამ სივრცეში — API-ს გამოძახება საჭირო არ არის.")
        return
    if not args.yes and input("გავაგრძელო? (y/n): ").strip().lower() not in ("y", "yes", "კი", "დიახ"):
        print("გაუქმდა.")
        return
    job = embeddings.EmbeddingJob()
    state = job.run(s, s.database_path, _docs(args.documents))
    _print(state)
    if state["status"] != "done":
        sys.exit(1)


def cmd_export(_args):
    s = get_settings()
    _print(embeddings.export_prebuilt(s, s.database_path))


def cmd_status(_args):
    from .server import status
    _print(status())


def cmd_check(_args):
    s = get_settings()
    try:
        client = get_client(s)
    except Exception as exc:
        print(describe_error(exc, s)[1])
        sys.exit(1)
    ok = True
    for model in (s.chat_model, s.embedding_model):
        try:
            client.models.retrieve(model)
            print(f"OK: {model}")
        except Exception as exc:
            ok = False
            print(f"შეცდომა: {model}: {describe_error(exc, s, model)[1]}")
    try:
        ids = sorted(m.id for m in client.models.list() if m.id.startswith(("gpt-", "text-embedding")))
        print("ანგარიშისთვის ხელმისაწვდომი GPT/embedding მოდელები:")
        for i in ids:
            print("  ", i)
    except Exception as exc:
        print("მოდელების სია ვერ მივიღე:", describe_error(exc, s)[1])
    sys.exit(0 if ok else 1)


def cmd_search(args):
    s = get_settings()
    index = rag.index_holder().get(s.database_path, s.embedding_space)
    qvec = None
    if s.has_api_key and index.vector_count:
        qvec = embeddings.embed_query(s, s.database_path, get_client(s), args.query)
    res = index.search(args.query, qvec, top_k=args.k, document_ids=_docs(args.documents),
                       include_ocr=args.ocr, vector_slots=s.vector_slots)
    print("mode:", res["mode"], "| tokens:", res["query_tokens"])
    for h in res["hits"]:
        c = h["chunk"]
        print(f"{c['chunk_id']:<24} {c['layer']:<14} rrf={h['rrf']} bm25={h['bm25']} cos={h['cosine']} "
              f"| {rag.location(c)['label']}\n    {c['text'][:150].replace(chr(10), ' ')}")


def cmd_ask(args):
    s = get_settings()
    for ev in rag.answer_stream(s, s.database_path, args.question, image_mode=args.images):
        t = ev["type"]
        if t == "delta":
            print(ev["text"], end="", flush=True)
        elif t == "sources":
            print(f"[ძიება: {ev['search_mode']}; წყაროები: {', '.join(x['chunk_id'] for x in ev['sources'])}; "
                  f"სურათები: {len(ev['images'])}]")
        elif t in ("error", "done", "status"):
            print(f"\n[{t}] " + json.dumps({k: v for k, v in ev.items() if k != "type"}, ensure_ascii=False))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import")
    a.add_argument("--library")
    a.add_argument("--library-id", default="main")
    a.add_argument("--id-prefix", default="")
    a.add_argument("--documents")
    a.set_defaults(fn=cmd_import)
    a = sub.add_parser("estimate")
    a.add_argument("--documents")
    a.set_defaults(fn=cmd_estimate)
    a = sub.add_parser("embed")
    a.add_argument("--documents")
    a.add_argument("--yes", action="store_true")
    a.set_defaults(fn=cmd_embed)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("export-embeddings").set_defaults(fn=cmd_export)
    sub.add_parser("check-openai").set_defaults(fn=cmd_check)
    a = sub.add_parser("search")
    a.add_argument("query")
    a.add_argument("--documents")
    a.add_argument("-k", type=int, default=10)
    a.add_argument("--ocr", action="store_true")
    a.set_defaults(fn=cmd_search)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--images", default="auto", choices=["auto", "always", "never"])
    a.set_defaults(fn=cmd_ask)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
