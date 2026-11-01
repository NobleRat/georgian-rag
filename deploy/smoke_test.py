"""Smoke test for a running deployment (local copy or the real host).

    python deploy/smoke_test.py http://127.0.0.1:8765
    python deploy/smoke_test.py https://<service>.onrender.com --password <APP_PASSWORD> [--no-api]

Without --no-api it asks two real questions (one text, one visual) — about $0.01 in OpenAI usage.
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

TEXT_Q = "რამდენი მილიონი წლის წინ იყო კარბონული გლობალური გამყინვარება?"
VISUAL_Q = "რა არის ნაჩვენები გუდაუთის კერნის პალინოლოგიურ დიაგრამაზე?"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'} {name}{' — ' + detail if detail else ''}", flush=True)


def ask(c: httpx.Client, question: str) -> dict:
    out = {"answer": "", "sources": [], "images": [], "done": None, "error": None, "deltas": 0}
    with c.stream("POST", "/api/chat", json={"question": question}, headers={"X-Rag-App": "1"}, timeout=180) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "sources":
                out["sources"], out["images"], out["mode"] = ev["sources"], ev["images"], ev["search_mode"]
            elif ev["type"] == "delta":
                out["answer"] += ev["text"]
                out["deltas"] += 1
            elif ev["type"] == "done":
                out["done"] = ev
            elif ev["type"] == "error":
                out["error"] = ev["message"]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--user", default="rag")
    p.add_argument("--password")
    p.add_argument("--no-api", action="store_true")
    a = p.parse_args()
    auth = (a.user, a.password) if a.password else None
    c = httpx.Client(base_url=a.url.rstrip("/"), auth=auth, timeout=60)

    check("healthz", c.get("/healthz").status_code == 200)
    if a.password:
        check("password required without login", httpx.get(a.url.rstrip("/") + "/api/status").status_code == 401)
    st = c.get("/api/status").json()
    cov = st["embedding_coverage"]
    check("documents imported", st["documents"] == 24, str(st["documents"]))
    check("550 main chunks", st["chunks_by_layer"]["native"] == 550, json.dumps(st["chunks_by_layer"]))
    check("embeddings complete without re-indexing", cov["done"] == cov["total"] > 0, f"{cov['done']}/{cov['total']}")
    api_embeds = [u for u in st["usage"] if u["kind"] == "embedding"]
    # Informational: on a freshly deployed host this must be 0; locally it shows the original indexing run.
    print(f"INFO document-embedding API calls logged on this server: {sum(u['calls'] for u in api_embeds)}"
          f" (on a fresh host expected 0; startup restored {st['startup'].get('prebuilt', {}).get('loaded')} "
          f"vectors from prebuilt/)")
    check("library path is relative", not any(":" in l["root_path"] or l["root_path"].startswith("/")
                                              for l in st["libraries"]), str([l["root_path"] for l in st["libraries"]]))
    check("OpenAI key configured on server", st["settings"]["api_key_configured"])
    check("UI page served", c.get("/").status_code == 200 and "დოკუმენტების ასისტენტი" in c.get("/").text)
    docs = c.get("/api/documents").json()
    check("document list", len(docs) == 24)
    ch = c.get("/api/chunks/07-c0058").json()
    check("open source excerpt (Word table)", "<table>" in ch["markdown_clean"] and "<sup>" in ch["markdown_clean"],
          ch["location"]["label"])
    img = c.get("/files/main/documents/06/pages/007.webp")
    check("page image served", img.status_code == 200 and img.headers["content-type"] == "image/webp"
          and len(img.content) > 10_000, f"{len(img.content)} bytes")
    orig = c.get("/files/main/" + docs[0]["source_file"])
    check("original file served", orig.status_code == 200 and len(orig.content) > 1000, docs[0]["source_file"])
    check("path traversal blocked", c.get("/files/main/../../app/config.py").status_code == 404)

    if not a.no_api:
        t = ask(c, TEXT_Q)
        valid = (t["done"] or {}).get("citations", {}).get("valid", [])
        check("text question: streamed answer", t["deltas"] > 1 and not t["error"], f"{t['deltas']} parts")
        check("text question: correct fact + verified citation", "280" in t["answer"] and "07-c0060" in valid,
              t["answer"][:120].replace("\n", " "))
        v = ask(c, VISUAL_Q)
        vdone = v["done"] or {}
        check("visual question: page image sent to model",
              "documents/06/pages/007.webp" in [i["path"] for i in v["images"]] and vdone.get("images_sent", 0) >= 1,
              f"images_sent={vdone.get('images_sent')}")
        check("visual question: verified citation, no invalid ids",
              "06-p007-visual-note" in vdone.get("citations", {}).get("valid", [])
              and not vdone.get("citations", {}).get("invalid"), v["answer"][:120].replace("\n", " "))
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} OK" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
