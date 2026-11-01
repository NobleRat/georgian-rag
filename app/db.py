"""SQLite storage: documents, chunks, embeddings, usage and conversations."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS libraries(
    library_id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    id_prefix TEXT NOT NULL DEFAULT '',
    manifest_sha256 TEXT,
    imported_at TEXT
);
CREATE TABLE IF NOT EXISTS documents(
    document_id TEXT PRIMARY KEY,
    library_id TEXT NOT NULL,
    source_document_id TEXT NOT NULL,
    title TEXT,
    original_filename TEXT,
    kind TEXT,
    source_file TEXT,
    source_sha256 TEXT,
    source_verified INTEGER,
    page_count INTEGER,
    page_images TEXT,
    native_units INTEGER,
    review_flags TEXT,
    meta TEXT,
    imported_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks(
    chunk_id TEXT PRIMARY KEY,
    source_chunk_id TEXT NOT NULL,
    library_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    title TEXT,
    text TEXT,
    markdown TEXT,
    embed_text TEXT,
    embed_hash TEXT,
    content_hash TEXT,
    unit_ids TEXT,
    locators TEXT,
    page_images TEXT,
    page_images_derived INTEGER DEFAULT 0,
    source_file TEXT,
    source_sha256 TEXT,
    review_flags TEXT,
    human_verified INTEGER,
    status_note TEXT,
    extra TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_layer ON chunks(layer);
CREATE TABLE IF NOT EXISTS embeddings(
    embed_hash TEXT NOT NULL,
    space TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT,
    PRIMARY KEY(embed_hash, space)
);
CREATE TABLE IF NOT EXISTS query_embeddings(
    query_hash TEXT NOT NULL,
    space TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT,
    PRIMARY KEY(query_hash, space)
);
CREATE TABLE IF NOT EXISTS import_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id TEXT,
    started_at TEXT,
    finished_at TEXT,
    scope TEXT,
    report TEXT
);
CREATE TABLE IF NOT EXISTS api_usage(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    kind TEXT,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    images INTEGER DEFAULT 0,
    cost_usd REAL,
    ok INTEGER,
    error TEXT
);
CREATE TABLE IF NOT EXISTS conversations(
    id TEXT PRIMARY KEY,
    created_at TEXT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    search_query TEXT,
    sources TEXT,
    citations TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as con:
        con.executescript(SCHEMA)


@contextmanager
def connect(path: Path):
    con = sqlite3.connect(str(path), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def get_meta(con, key: str, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(con, key: str, value) -> None:
    con.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def bump_index_version(con) -> None:
    """Signals the in-memory search index that it must reload."""
    current = int(get_meta(con, "index_version", "0"))
    set_meta(con, "index_version", current + 1)


def row_to_chunk(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("unit_ids", "locators", "page_images", "review_flags", "extra"):
        if d.get(key):
            d[key] = json.loads(d[key])
        else:
            d[key] = [] if key != "extra" else {}
    d["human_verified"] = bool(d.get("human_verified"))
    d["page_images_derived"] = bool(d.get("page_images_derived"))
    return d


def log_usage(path: Path, *, kind: str, model: str, input_tokens: int = 0, cached_tokens: int = 0,
              output_tokens: int = 0, reasoning_tokens: int = 0, images: int = 0,
              cost_usd: float | None = None, ok: bool = True, error: str | None = None) -> None:
    with connect(path) as con:
        con.execute(
            "INSERT INTO api_usage(ts, kind, model, input_tokens, cached_tokens, output_tokens, reasoning_tokens,"
            " images, cost_usd, ok, error) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (now_iso(), kind, model, input_tokens, cached_tokens, output_tokens, reasoning_tokens, images,
             cost_usd, 1 if ok else 0, error),
        )
