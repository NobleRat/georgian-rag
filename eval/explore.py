"""Helper for preparing evaluation questions: print chunks or grep the main corpus.

    python -m eval.explore show 07-c0058 23-c0057
    python -m eval.explore grep "გამყინვარ" [--layer native]
    python -m eval.explore heads
"""
import json
import re
import sys

from app import db
from app.config import get_settings


def rows(layer=None):
    with db.connect(get_settings().database_path) as con:
        sql = "SELECT * FROM chunks" + (" WHERE layer=?" if layer else "") + " ORDER BY chunk_id"
        return [db.row_to_chunk(r) for r in con.execute(sql, (layer,) if layer else ())]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "show":
        by = {r["chunk_id"]: r for r in rows()}
        for cid in args:
            r = by[cid]
            print(f"=== {cid} [{r['layer']}] {r['title']} | {json.dumps(r['locators'][:2], ensure_ascii=False)}")
            print(r["text"])
    elif cmd == "grep":
        layer = "native"
        if "--layer" in args:
            layer = args[args.index("--layer") + 1]
        pat = re.compile(args[0], re.I)
        for r in rows(layer):
            for m in pat.finditer(r["text"]):
                s = max(0, m.start() - 90)
                print(f"{r['chunk_id']}: …{r['text'][s:m.end() + 90].replace(chr(10), ' ')}…")
    elif cmd == "heads":
        seen = set()
        for r in rows("native"):
            if r["document_id"] not in seen:
                seen.add(r["document_id"])
                print(f"{r['document_id']} {r['title'][:60]} :: {r['text'][:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
