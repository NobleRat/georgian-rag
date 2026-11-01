"""Embedding creation, caching and cost estimation.

Vectors are stored per (embed_hash, space) where space = "<model>:<dimensions>".
Changing OPENAI_EMBEDDING_MODEL or OPENAI_EMBEDDING_DIMENSIONS therefore starts
a new, separate space; old vectors stay on disk but are never compared with new ones.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import openai

from . import db
from .config import Settings
from .openai_client import describe_error, get_client, sanitize
from .pricing import embedding_cost
from .textproc import normalize, sha256_text

EMBEDDED_LAYERS = ("native", "visual_note", "formula")
MAX_TOKENS_PER_REQUEST = 250_000  # API limit is 300K summed tokens per request
TRANSIENT_ERRORS = (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError)

_encoder = None


def count_tokens(text: str) -> int:
    """Tokens as counted by the embedding tokenizer (cl100k_base)."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # offline and no cached encoding: rough upper estimate for Georgian
            _encoder = False
    if _encoder is False:
        return int(len(text) * 1.9) + 1
    return len(_encoder.encode(text))


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize_vec(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def pending_items(db_path: Path, space: str, document_ids: list[str] | None = None) -> list[tuple[str, str]]:
    """(embed_hash, embed_text) pairs that still need a vector in this space."""
    sql = (f"SELECT DISTINCT c.embed_hash, c.embed_text FROM chunks c WHERE c.layer IN "
           f"({','.join('?' * len(EMBEDDED_LAYERS))}) AND NOT EXISTS (SELECT 1 FROM embeddings e "
           f"WHERE e.embed_hash=c.embed_hash AND e.space=?)")
    params: list = [*EMBEDDED_LAYERS, space]
    if document_ids:
        sql += f" AND c.document_id IN ({','.join('?' * len(document_ids))})"
        params += list(document_ids)
    sql += " ORDER BY c.document_id, c.chunk_id"
    with db.connect(db_path) as con:
        return [(r["embed_hash"], r["embed_text"]) for r in con.execute(sql, params)]


def coverage(db_path: Path, space: str) -> dict:
    with db.connect(db_path) as con:
        total = con.execute(
            f"SELECT COUNT(DISTINCT embed_hash) FROM chunks WHERE layer IN ({','.join('?' * len(EMBEDDED_LAYERS))})",
            EMBEDDED_LAYERS).fetchone()[0]
        done = con.execute(
            f"SELECT COUNT(DISTINCT c.embed_hash) FROM chunks c JOIN embeddings e ON e.embed_hash=c.embed_hash "
            f"AND e.space=? WHERE c.layer IN ({','.join('?' * len(EMBEDDED_LAYERS))})",
            (space, *EMBEDDED_LAYERS)).fetchone()[0]
        other = [dict(r) for r in con.execute(
            "SELECT space, COUNT(*) AS n FROM embeddings WHERE space<>? GROUP BY space", (space,))]
    return {"space": space, "total": total, "done": done, "pending": total - done, "other_spaces": other}


def estimate(settings: Settings, db_path: Path, document_ids: list[str] | None = None) -> dict:
    items = pending_items(db_path, settings.embedding_space, document_ids)
    tokens = sum(count_tokens(t) for _, t in items)
    cost = embedding_cost(settings.embedding_model, tokens)
    batches, cur_n, cur_t = 0, 0, 0
    for _, t in items:
        n = count_tokens(t)
        if cur_n and (cur_n >= settings.embed_batch_size or cur_t + n > MAX_TOKENS_PER_REQUEST):
            batches, cur_n, cur_t = batches + 1, 0, 0
        cur_n, cur_t = cur_n + 1, cur_t + n
    batches += 1 if cur_n else 0
    return {
        "model": settings.embedding_model,
        "space": settings.embedding_space,
        "items": len(items),
        "tokens": tokens,
        "requests": batches,
        "estimated_cost_usd": cost,
        "price_known": cost is not None,
    }


def _embed_batch(client, settings: Settings, texts: list[str]):
    kwargs = {"model": settings.embedding_model, "input": texts}
    if settings.embedding_dimensions:
        kwargs["dimensions"] = settings.embedding_dimensions
    return client.embeddings.create(**kwargs)


def _space_dims(con, space: str) -> int | None:
    row = con.execute("SELECT dims FROM embeddings WHERE space=? LIMIT 1", (space,)).fetchone()
    return row["dims"] if row else None


class EmbeddingJob:
    """Background embedding job with progress reporting (one at a time)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state: dict = {"status": "idle"}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **kw):
        with self._lock:
            self.state.update(kw)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings: Settings, db_path: Path, document_ids: list[str] | None = None,
              client_factory=get_client, retry_sleeps=(10, 30)) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self.state = {"status": "starting", "documents": document_ids or "all"}
        self._thread = threading.Thread(
            target=self.run, args=(settings, db_path, document_ids, client_factory, retry_sleeps), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def run(self, settings: Settings, db_path: Path, document_ids=None, client_factory=get_client,
            retry_sleeps=(10, 30)) -> dict:
        space = settings.embedding_space
        items = pending_items(db_path, space, document_ids)
        token_counts = [count_tokens(t) for _, t in items]
        self._update(status="running", space=space, model=settings.embedding_model, total_items=len(items),
                     done_items=0, total_tokens=sum(token_counts), done_tokens=0, api_tokens=0, cost_usd=0.0,
                     requests=0, retries=0, error=None, error_code=None, started_at=db.now_iso(), finished_at=None)
        if not items:
            self._update(status="done", finished_at=db.now_iso())
            return self.snapshot()
        try:
            client = client_factory(settings)
        except Exception as exc:  # missing key
            code, msg = describe_error(exc, settings, settings.embedding_model)
            self._update(status="error", error=msg, error_code=code, finished_at=db.now_iso())
            return self.snapshot()

        batches: list[list[int]] = []
        cur, cur_t = [], 0
        for i, n in enumerate(token_counts):
            if cur and (len(cur) >= settings.embed_batch_size or cur_t + n > MAX_TOKENS_PER_REQUEST):
                batches.append(cur)
                cur, cur_t = [], 0
            cur.append(i)
            cur_t += n
        if cur:
            batches.append(cur)

        for batch in batches:
            if self._stop.is_set():
                self._update(status="stopped", finished_at=db.now_iso())
                return self.snapshot()
            texts = [items[i][1] for i in batch]
            attempt = 0
            while True:
                try:
                    resp = _embed_batch(client, settings, texts)
                    break
                except TRANSIENT_ERRORS as exc:
                    code, msg = describe_error(exc, settings, settings.embedding_model)
                    if code == "quota" or attempt >= len(retry_sleeps):
                        db.log_usage(db_path, kind="embedding", model=settings.embedding_model, ok=False,
                                     error=sanitize(str(exc))[:500])
                        self._update(status="error", error=msg, error_code=code, finished_at=db.now_iso())
                        return self.snapshot()
                    self._update(retries=self.state.get("retries", 0) + 1,
                                 last_retry_reason=msg)
                    time.sleep(retry_sleeps[attempt])
                    attempt += 1
                except Exception as exc:
                    code, msg = describe_error(exc, settings, settings.embedding_model)
                    db.log_usage(db_path, kind="embedding", model=settings.embedding_model, ok=False,
                                 error=sanitize(str(exc))[:500])
                    self._update(status="error", error=msg, error_code=code, finished_at=db.now_iso())
                    return self.snapshot()
            vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            if len(vectors) != len(texts):
                self._update(status="error", error="OpenAI-მ დაბრუნა მოსალოდნელზე ნაკლები ვექტორი.",
                             error_code="bad_response", finished_at=db.now_iso())
                return self.snapshot()
            api_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None) or sum(
                token_counts[i] for i in batch)
            cost = embedding_cost(settings.embedding_model, api_tokens)
            normed = [_normalize_vec(vec) for vec in vectors]
            with db.connect(db_path) as con:
                expected = _space_dims(con, space)
            dims = {len(v) for v in normed} | ({expected} if expected else set())
            if len(dims) != 1:
                self._update(status="error", error_code="dims_mismatch", finished_at=db.now_iso(),
                             error=f"ვექტორების განზომილებები არ ემთხვევა ერთმანეთს ({sorted(dims)}); "
                                   "ჩაწერა შეჩერდა, რომ სხვადასხვა სივრცე არ აირიოს.")
                return self.snapshot()
            with db.connect(db_path) as con:
                now = db.now_iso()
                for i, v in zip(batch, normed):
                    con.execute(
                        "INSERT OR REPLACE INTO embeddings(embed_hash, space, dims, vector, created_at) VALUES(?,?,?,?,?)",
                        (items[i][0], space, len(v), to_blob(v), now))
                db.bump_index_version(con)
            db.log_usage(db_path, kind="embedding", model=settings.embedding_model, input_tokens=api_tokens,
                         cost_usd=cost)
            s = self.snapshot()
            self._update(done_items=s["done_items"] + len(batch),
                         done_tokens=s["done_tokens"] + sum(token_counts[i] for i in batch),
                         api_tokens=s["api_tokens"] + api_tokens, cost_usd=s["cost_usd"] + (cost or 0),
                         requests=s["requests"] + 1)
        try:
            snap = export_prebuilt(settings, db_path)
        except OSError as exc:  # read-only disk on a host: the vectors are still in the database
            snap = {"exported": 0, "error": str(exc)}
        self._update(status="done", finished_at=db.now_iso(), snapshot=snap)
        return self.snapshot()


# ---------------------------------------------------------------- prebuilt snapshot (committed to the repo)

def prebuilt_path(settings: Settings) -> Path:
    safe = settings.embedding_space.replace(":", "__").replace("/", "_")
    return settings.prebuilt_dir / "embeddings" / f"{safe}.npz"


def export_prebuilt(settings: Settings, db_path: Path) -> dict:
    """Writes the vectors of the current space for the current chunks to prebuilt/embeddings/<space>.npz.

    The file is meant to be committed, so a fresh deployment (empty, temporary disk) can
    restore all document embeddings without calling the API.
    """
    space = settings.embedding_space
    with db.connect(db_path) as con:
        rows = con.execute(
            f"SELECT DISTINCT e.embed_hash, e.vector FROM embeddings e JOIN chunks c ON c.embed_hash=e.embed_hash "
            f"WHERE e.space=? AND c.layer IN ({','.join('?' * len(EMBEDDED_LAYERS))}) ORDER BY e.embed_hash",
            (space, *EMBEDDED_LAYERS)).fetchall()
    if not rows:
        return {"exported": 0, "path": None}
    hashes = np.array([r["embed_hash"] for r in rows], dtype="<U64")
    vectors = np.stack([from_blob(r["vector"]) for r in rows]).astype(np.float32)
    path = prebuilt_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, hashes=hashes, vectors=vectors, space=np.array(space), model=np.array(settings.embedding_model))
    tmp.replace(path)
    return {"exported": len(rows), "dims": int(vectors.shape[1]), "path": str(path), "bytes": path.stat().st_size}


def load_prebuilt(settings: Settings, db_path: Path) -> dict:
    """Restores missing vectors of the current space from the committed snapshot (no API calls)."""
    path = prebuilt_path(settings)
    if not path.exists():
        return {"loaded": 0, "reason": "snapshot not found", "path": path.name}
    with np.load(path, allow_pickle=False) as data:
        if str(data["space"]) != settings.embedding_space:
            return {"loaded": 0, "reason": "snapshot belongs to another embedding space", "path": path.name}
        hashes, vectors = data["hashes"], data["vectors"]
    with db.connect(db_path) as con:
        needed = {r["embed_hash"] for r in con.execute(
            f"SELECT DISTINCT c.embed_hash FROM chunks c WHERE c.layer IN ({','.join('?' * len(EMBEDDED_LAYERS))}) "
            f"AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.embed_hash=c.embed_hash AND e.space=?)",
            (*EMBEDDED_LAYERS, settings.embedding_space))}
        expected = _space_dims(con, settings.embedding_space)
        if expected is not None and expected != vectors.shape[1]:
            return {"loaded": 0, "reason": f"dimension mismatch ({vectors.shape[1]} != {expected})",
                    "path": path.name}
        now = db.now_iso()
        loaded = 0
        for h, v in zip(hashes, vectors):
            h = str(h)
            if h in needed:
                con.execute("INSERT OR IGNORE INTO embeddings(embed_hash, space, dims, vector, created_at) "
                            "VALUES(?,?,?,?,?)", (h, settings.embedding_space, len(v), to_blob(v), now))
                loaded += 1
        if loaded:
            db.bump_index_version(con)
    return {"loaded": loaded, "in_snapshot": int(len(hashes)), "still_missing": len(needed) - loaded,
            "path": path.name}


def embed_query(settings: Settings, db_path: Path, client, text: str) -> np.ndarray:
    """Embeds a search query with the same model/dimensions as the documents (cached on disk)."""
    space = settings.embedding_space
    qhash = sha256_text(normalize(text).strip())
    with db.connect(db_path) as con:
        row = con.execute("SELECT vector FROM query_embeddings WHERE query_hash=? AND space=?",
                          (qhash, space)).fetchone()
        if row:
            return from_blob(row["vector"])
    resp = _embed_batch(client, settings, [text])
    v = _normalize_vec(resp.data[0].embedding)
    tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None) or count_tokens(text)
    db.log_usage(db_path, kind="query_embedding", model=settings.embedding_model, input_tokens=tokens,
                 cost_usd=embedding_cost(settings.embedding_model, tokens))
    with db.connect(db_path) as con:
        con.execute("INSERT OR REPLACE INTO query_embeddings(query_hash, space, dims, vector, created_at) "
                    "VALUES(?,?,?,?,?)", (qhash, space, len(v), to_blob(v), db.now_iso()))
    return v
