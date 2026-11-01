"""FastAPI server: JSON API + streaming chat (SSE) + static Georgian UI.

Locally it binds to 127.0.0.1. On a host, set APP_PASSWORD so the whole site requires a login.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import secrets
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, embeddings, rag
from .config import STATIC_DIR, get_settings, project_relative, resolve_project_path
from .importer import ALL_LAYERS, ImportErrorKa, import_library
from .openai_client import MISSING_KEY_MESSAGE, describe_error, get_client
from .pricing import PRICES_CHECKED_ON

log = logging.getLogger("rag")
ALLOWED_FILE_TYPES = {".webp": "image/webp", ".png": "image/png", ".jpeg": "image/jpeg", ".jpg": "image/jpeg",
                      ".pdf": "application/pdf", ".doc": "application/msword",
                      ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
CSRF_HEADER = "x-rag-app"

job = embeddings.EmbeddingJob()
_import_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(_app):
    ensure_ready()
    yield


app = FastAPI(title="ქართული RAG", docs_url=None, redoc_url=None, lifespan=_lifespan)
startup_report: dict = {}
OPEN_PATHS = {"/healthz"}


@app.middleware("http")
async def _password_gate(request: Request, call_next):
    """HTTP Basic login for the whole site when APP_PASSWORD is set (required for public hosting)."""
    settings = get_settings()
    if not settings.app_password or request.url.path in OPEN_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
        except (binascii.Error, UnicodeDecodeError):
            user, pwd = "", ""
        if secrets.compare_digest(user.encode(), settings.app_username.encode()) and                 secrets.compare_digest(pwd.encode(), settings.app_password.encode()):
            return await call_next(request)
    return PlainTextResponse("საჭიროა ავტორიზაცია.", status_code=401,
                             headers={"WWW-Authenticate": 'Basic realm="RAG", charset="UTF-8"'})


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _db() -> Path:
    return get_settings().database_path


def _require_app_header(request: Request):
    # Blocks cross-site POSTs from other web pages (a custom header forces a CORS preflight we never allow).
    if request.headers.get(CSRF_HEADER) != "1":
        raise HTTPException(status_code=403, detail="მოთხოვნა უარყოფილია.")


def ensure_ready() -> dict:
    """Builds the local database from files in the project, without API calls.

    Runs on every start: the import is idempotent (unchanged rows are skipped), and missing
    vectors are restored from the committed snapshot in prebuilt/. This makes a host with an
    empty or temporary disk start in the same state as the local machine.
    """
    settings = get_settings()
    db.init_db(settings.database_path)
    report: dict = {"data_library": project_relative(settings.data_library), "import": None, "prebuilt": None,
                    "warnings": []}
    if (settings.data_library / "manifest.json").exists():
        with _import_lock:
            imp = import_library(settings.database_path, settings.data_library)
        report["import"] = {"documents": imp["documents"], "layers": imp["layers"],
                            "sources_verified": imp["sources_verified"], "warnings": imp["warnings"]}
    else:
        report["warnings"].append(f"მონაცემთა საქაღალდე ვერ მოიძებნა: {report['data_library']}")
    report["prebuilt"] = embeddings.load_prebuilt(settings, settings.database_path)
    cov = embeddings.coverage(settings.database_path, settings.embedding_space)
    report["embedding_coverage"] = f"{cov['done']}/{cov['total']}"
    if cov["pending"]:
        report["warnings"].append(f"{cov['pending']} ჩანაწერს ვექტორი არ აქვს; სემანტიკური ძიება ნაწილობრივია.")
    startup_report.clear()
    startup_report.update(report)
    log.info("Startup: %s", json.dumps(report, ensure_ascii=False))
    return report


# ---------------------------------------------------------------- status

@app.get("/api/status")
def status():
    settings = get_settings()
    path = settings.database_path
    with db.connect(path) as con:
        layers = {r["layer"]: r["n"] for r in con.execute("SELECT layer, COUNT(*) n FROM chunks GROUP BY layer")}
        n_docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        last = con.execute("SELECT report FROM import_runs ORDER BY id DESC LIMIT 1").fetchone()
        usage = [dict(r) for r in con.execute(
            "SELECT kind, model, COUNT(*) calls, SUM(input_tokens) input_tokens, SUM(cached_tokens) cached_tokens, "
            "SUM(output_tokens) output_tokens, SUM(images) images, SUM(cost_usd) cost_usd, SUM(1-ok) errors "
            "FROM api_usage GROUP BY kind, model ORDER BY kind")]
        libs = [dict(r) for r in con.execute("SELECT library_id, root_path, id_prefix, imported_at FROM libraries")]
    return {
        "settings": settings.public_dict(),
        "documents": n_docs,
        "chunks_by_layer": {layer: layers.get(layer, 0) for layer in ALL_LAYERS},
        "embedding_coverage": embeddings.coverage(path, settings.embedding_space),
        "index_job": job.snapshot(),
        "last_import": json.loads(last["report"]) if last else None,
        "libraries": libs,
        "usage": usage,
        "usage_total_usd": round(sum((u["cost_usd"] or 0) for u in usage), 6),
        "prices_checked_on": PRICES_CHECKED_ON,
        "startup": startup_report,
    }


@app.post("/api/openai/check")
def openai_check(request: Request):
    _require_app_header(request)
    settings = get_settings()
    result = {"api_key_configured": settings.has_api_key, "checks": []}
    try:
        client = get_client(settings)
    except Exception as exc:
        code, msg = describe_error(exc, settings)
        result["checks"].append({"what": "API გასაღები", "ok": False, "code": code, "message": msg})
        return result
    for what, model in (("პასუხების მოდელი", settings.chat_model), ("embeddings მოდელი", settings.embedding_model)):
        try:
            client.models.retrieve(model)
            result["checks"].append({"what": what, "model": model, "ok": True, "message": "ხელმისაწვდომია"})
        except Exception as exc:
            code, msg = describe_error(exc, settings, model)
            result["checks"].append({"what": what, "model": model, "ok": False, "code": code, "message": msg})
    return result


# ---------------------------------------------------------------- documents

@app.get("/api/documents")
def documents():
    with db.connect(_db()) as con:
        counts = {}
        for r in con.execute("SELECT document_id, layer, COUNT(*) n FROM chunks GROUP BY document_id, layer"):
            counts.setdefault(r["document_id"], {})[r["layer"]] = r["n"]
        docs = []
        for r in con.execute("SELECT * FROM documents ORDER BY document_id"):
            d = dict(r)
            d["page_images"] = json.loads(d["page_images"] or "[]")
            d["review_flags"] = json.loads(d["review_flags"] or "{}")
            d["meta"] = json.loads(d["meta"] or "{}")
            d["chunk_counts"] = counts.get(d["document_id"], {})
            docs.append(d)
    return docs


@app.get("/api/documents/{document_id}/chunks")
def document_chunks(document_id: str, layer: str = "native"):
    with db.connect(_db()) as con:
        rows = [db.row_to_chunk(r) for r in con.execute(
            "SELECT * FROM chunks WHERE document_id=? AND layer=? ORDER BY chunk_id", (document_id, layer))]
    return [{"chunk_id": c["chunk_id"], "location": rag.location(c), "snippet": c["text"][:200],
             "page_images": c["page_images"]} for c in rows]


@app.get("/api/chunks/{chunk_id}")
def chunk(chunk_id: str):
    with db.connect(_db()) as con:
        row = con.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if not row:
            raise HTTPException(404, "ჩანაწერი ვერ მოიძებნა")
        c = db.row_to_chunk(row)
        doc = con.execute("SELECT title, kind, library_id, source_file FROM documents WHERE document_id=?",
                          (c["document_id"],)).fetchone()
    c.pop("embed_text", None)
    c["location"] = rag.location(c)
    c["markdown_clean"] = rag.strip_anchors(c["markdown"])
    c["document"] = dict(doc) if doc else None
    return c


@app.get("/files/{library_id}/{rel_path:path}")
def files(library_id: str, rel_path: str):
    with db.connect(_db()) as con:
        lib = con.execute("SELECT root_path FROM libraries WHERE library_id=?", (library_id,)).fetchone()
    if not lib:
        raise HTTPException(404)
    root = resolve_project_path(lib["root_path"]).resolve()
    target = (root / rel_path).resolve()
    if root not in target.parents or target.suffix.lower() not in ALLOWED_FILE_TYPES or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, media_type=ALLOWED_FILE_TYPES[target.suffix.lower()])


# ---------------------------------------------------------------- import / indexing

class ImportRequest(BaseModel):
    documents: list[str] | None = None


@app.post("/api/import")
def run_import(req: ImportRequest, request: Request):
    _require_app_header(request)
    settings = get_settings()
    if not _import_lock.acquire(blocking=False):
        raise HTTPException(409, "იმპორტი უკვე მიმდინარეობს.")
    try:
        with db.connect(settings.database_path) as con:
            libs = [dict(r) for r in con.execute("SELECT * FROM libraries")]
        if not libs:
            libs = [{"library_id": "main", "root_path": str(settings.data_library), "id_prefix": ""}]
        reports = [import_library(settings.database_path, resolve_project_path(l["root_path"]), library_id=l["library_id"],
                                  id_prefix=l["id_prefix"], document_ids=req.documents) for l in libs]
        return {"reports": reports}
    except ImportErrorKa as exc:
        raise HTTPException(400, str(exc))
    finally:
        _import_lock.release()


@app.get("/api/index/estimate")
def index_estimate(documents: str | None = None):
    settings = get_settings()
    docs = [d for d in (documents or "").split(",") if d] or None
    return embeddings.estimate(settings, settings.database_path, docs)


class IndexStart(BaseModel):
    documents: list[str] | None = None
    confirm: bool = False


@app.post("/api/index/start")
def index_start(req: IndexStart, request: Request):
    _require_app_header(request)
    if not req.confirm:
        raise HTTPException(400, "ინდექსირება საჭიროებს დადასტურებას (confirm=true).")
    settings = get_settings()
    if not settings.has_api_key:
        raise HTTPException(400, MISSING_KEY_MESSAGE.split(" ქვემოთ")[0])
    if not job.start(settings, settings.database_path, req.documents):
        raise HTTPException(409, "ინდექსირება უკვე მიმდინარეობს.")
    return {"started": True}


@app.post("/api/index/stop")
def index_stop(request: Request):
    _require_app_header(request)
    job.stop()
    return {"stopping": True}


@app.get("/api/index/status")
def index_status():
    settings = get_settings()
    return {"job": job.snapshot(), "coverage": embeddings.coverage(settings.database_path, settings.embedding_space)}


# ---------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    document_ids: list[str] | None = None
    image_mode: str = "auto"
    include_ocr: bool = False


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    _require_app_header(request)
    settings = get_settings()
    image_mode = req.image_mode if req.image_mode in ("auto", "always", "never") else "auto"

    def events():
        try:
            for ev in rag.answer_stream(settings, settings.database_path, req.question, req.conversation_id,
                                        req.document_ids or None, image_mode, req.include_ocr):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # never leak a stack trace or key to the browser
            log.exception("chat failed")
            code, msg = describe_error(exc, settings)
            yield f"data: {json.dumps({'type': 'error', 'code': code, 'message': msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str):
    return rag.get_messages(_db(), conversation_id)


# ---------------------------------------------------------------- static UI

@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ImportErrorKa)
def _import_error(_, exc: ImportErrorKa):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="ქართული RAG სერვერი")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    url = f"http://{settings.host}:{settings.port}"
    print(f"\n  აპლიკაცია: {url}\n  გასაჩერებლად დააჭირეთ Ctrl+C\n", flush=True)
    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
