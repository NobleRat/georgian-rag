"""Imports a prepared library (manifest.json + index/*.jsonl + documents/ID/...).

Rules:
* `index/native-chunks.jsonl` is the only main text source. content.md and
  document.json are NOT indexed (they contain the same text); document.json is
  only read to derive page images for Word chunks that lack them.
* visual-navigation and checked-formulas go into their own layers with their
  verification status preserved.
* OCR goes into the `ocr_unverified` layer, which is never searched by default.
* Re-importing is idempotent: records are keyed by stable chunk_id and compared
  by content hash; unchanged records are not touched, removed ones are deleted.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import db
from .config import project_relative
from .textproc import clean_title, sha256_text

LAYER_NATIVE = "native"
LAYER_VISUAL = "visual_note"
LAYER_FORMULA = "formula"
LAYER_OCR = "ocr_unverified"
ALL_LAYERS = (LAYER_NATIVE, LAYER_VISUAL, LAYER_FORMULA, LAYER_OCR)

STATUS_NOTES = {
    LAYER_NATIVE: "დოკუმენტის მშობლიური ტექსტი (ტექნიკური ნორმალიზაციით); ადამიანის მიერ სიტყვა-სიტყვით არ არის გადამოწმებული.",
    LAYER_VISUAL: "ასისტენტის მოკლე ვიზუალური სანავიგაციო აღწერა. ეს არ არის დიაგრამის სრული ტრანსკრიფცია და ადამიანის მიერ არ არის გადამოწმებული; ფაქტებისთვის საჭიროა გვერდის გამოსახულება.",
    LAYER_FORMULA: "ფორმულა ასისტენტმა შეადარა Word-ის საჩვენებელ გამოსახულებას; ადამიანის მიერ არ არის გადამოწმებული.",
    LAYER_OCR: "გადაუმოწმებელი OCR: შეიცავს ამოცნობის შეცდომებს; გამოიყენება მხოლოდ კანდიდატი გვერდის მოსაძებნად.",
}

_ANCHOR_ID_RE = re.compile(r'<a id="([^"]+)"></a>')
_UNIT_NUM_RE = re.compile(r"-w(\d+)$")


class ImportErrorKa(Exception):
    """Import error with a Georgian, user-facing message."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(record: dict, *extra: str) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "|" + "|".join(extra)
    return sha256_text(payload)


def _doc_prefix_path(doc_id: str, rel: str) -> str:
    """'pages/004.webp' -> 'documents/ID/pages/004.webp'."""
    return rel if rel.startswith("documents/") else f"documents/{doc_id}/{rel}"


class _UnitPages:
    """Derives page images for Word chunks whose own record has none."""

    def __init__(self, root: Path, doc_id: str):
        self.units: dict[str, dict] = {}
        self.ordered: list[tuple[int, str]] = []
        self.footnote_ref: dict[str, str] = {}
        path = root / "documents" / doc_id / "document.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for u in data.get("units", []):
            self.units[u["unit_id"]] = u
            loc = u.get("locator") or {}
            m = _UNIT_NUM_RE.search(u["unit_id"])
            if m and loc.get("page_image") and loc.get("part") == "word/document.xml":
                self.ordered.append((int(m.group(1)), loc["page_image"]))
            for note in u.get("footnote_references") or []:
                if loc.get("page_image"):
                    self.footnote_ref[str(note)] = loc["page_image"]
        self.ordered.sort()

    def derive(self, unit_ids: list[str]) -> list[str]:
        pages: list[str] = []
        for uid in unit_ids:
            u = self.units.get(uid)
            if not u:
                continue
            loc = u.get("locator") or {}
            page = None
            if loc.get("page_image"):
                page = loc["page_image"]
            elif loc.get("note_id") is not None:
                page = self.footnote_ref.get(str(loc["note_id"]))
            else:
                m = _UNIT_NUM_RE.search(uid)
                if m and self.ordered:
                    n = int(m.group(1))
                    before = [p for num, p in self.ordered if num < n]
                    after = [p for num, p in self.ordered if num > n]
                    for cand in (before[-1:] + after[:1]):
                        if cand not in pages:
                            pages.append(cand)
            if page and page not in pages:
                pages.append(page)
        return pages


def _native_record(raw: dict, doc: dict, prefix: str, unit_pages: _UnitPages) -> dict:
    title = doc["title"]
    page_images = list(raw.get("page_images") or [])
    derived = False
    if not page_images:
        ids = list(raw.get("unit_ids") or []) + _ANCHOR_ID_RE.findall(raw.get("markdown") or "")
        page_images = [_doc_prefix_path(raw["document_id"], p) for p in unit_pages.derive(ids)]
        derived = bool(page_images)
    embed_text = f"{title}\n\n{raw['text']}"
    return {
        "chunk_id": prefix + raw["chunk_id"],
        "source_chunk_id": raw["chunk_id"],
        "document_id": prefix + raw["document_id"],
        "layer": LAYER_NATIVE,
        "title": title,
        "text": raw["text"],
        "markdown": raw.get("markdown") or raw["text"],
        "embed_text": embed_text,
        "unit_ids": raw.get("unit_ids") or [],
        "locators": raw.get("locators") or [],
        "page_images": page_images,
        "page_images_derived": derived,
        "source_file": raw.get("source_file"),
        "source_sha256": raw.get("source_sha256"),
        "review_flags": raw.get("review_flags") or [],
        "human_verified": bool(raw.get("human_verified")),
        "status_note": STATUS_NOTES[LAYER_NATIVE],
        "extra": {"extraction": raw.get("extraction"), "ocr_included": raw.get("ocr_included"),
                  "language": raw.get("language")},
        "content_hash": _canonical_hash(raw, title, json.dumps(page_images)),
    }


def _visual_record(raw: dict, doc: dict, prefix: str) -> dict:
    page_image = raw["page_image"]
    return {
        "chunk_id": prefix + raw["chunk_id"],
        "source_chunk_id": raw["chunk_id"],
        "document_id": prefix + raw["document_id"],
        "layer": LAYER_VISUAL,
        "title": doc["title"],
        "text": raw["text"],
        "markdown": raw["text"],
        "embed_text": f"{doc['title']}\n\n{raw['text']}",
        "unit_ids": [],
        "locators": [{"pdf_page": raw.get("pdf_page"), "page_image": page_image.split("/", 2)[-1]}],
        "page_images": [page_image],
        "page_images_derived": False,
        "source_file": doc["source_file"],
        "source_sha256": doc["source_sha256"],
        "review_flags": ["visual_navigation_only"],
        "human_verified": False,
        "status_note": STATUS_NOTES[LAYER_VISUAL],
        "extra": {k: raw.get(k) for k in ("extraction", "full_transcription", "use", "verification")},
        "content_hash": _canonical_hash(raw, doc["title"]),
    }


def _formula_record(raw: dict, doc: dict, prefix: str, parents: dict[str, str]) -> dict:
    page_image = raw["page_image"]
    md = f"{raw['text']}\n\nLaTeX: `{raw['latex']}`\n\nშენიშვნა აღნიშვნებზე: {raw.get('source_notation_note', '')}"
    return {
        "chunk_id": prefix + raw["chunk_id"],
        "source_chunk_id": raw["chunk_id"],
        "document_id": prefix + raw["document_id"],
        "layer": LAYER_FORMULA,
        "title": doc["title"],
        "text": raw["text"],
        "markdown": md,
        "embed_text": f"{doc['title']}\n\n{raw['text']}\n{raw['latex']}",
        "unit_ids": [raw["source_unit_id"]],
        "locators": [{"part": "word/document.xml", "paragraph_id": raw["source_unit_id"],
                      "reference_render_page": raw.get("reference_render_page"),
                      "page_image": page_image.split("/", 2)[-1]}],
        "page_images": [page_image],
        "page_images_derived": False,
        "source_file": doc["source_file"],
        "source_sha256": doc["source_sha256"],
        "review_flags": ["assistant_visual_transcription"],
        "human_verified": False,
        "status_note": STATUS_NOTES[LAYER_FORMULA],
        "extra": {
            "latex": raw.get("latex"),
            "verification": raw.get("verification"),
            "extraction": raw.get("extraction"),
            "source_notation_note": raw.get("source_notation_note"),
            "parent_chunk_id": parents.get(raw["source_unit_id"]),
        },
        "content_hash": _canonical_hash(raw, doc["title"], parents.get(raw["source_unit_id"], "")),
    }


def _ocr_record(raw: dict, doc: dict, prefix: str) -> dict:
    src = raw.get("source") or {}
    page_image = src.get("page_image")
    loc = {"page_image": page_image.split("/", 2)[-1] if page_image else None}
    if src.get("derived_word_layout"):
        loc["reference_render_page"] = src.get("page")
    else:
        loc["pdf_page"] = src.get("page")
    return {
        "chunk_id": prefix + raw["chunk_id"],
        "source_chunk_id": raw["chunk_id"],
        "document_id": prefix + raw["document_id"],
        "layer": LAYER_OCR,
        "title": doc["title"],
        "text": raw["text"],
        "markdown": raw["text"],
        "embed_text": f"{doc['title']}\n\n{raw['text']}",
        "unit_ids": [],
        "locators": [loc],
        "page_images": [page_image] if page_image else [],
        "page_images_derived": False,
        "source_file": doc["source_file"],
        "source_sha256": doc["source_sha256"],
        "review_flags": ["unverified_ocr"],
        "human_verified": False,
        "status_note": STATUS_NOTES[LAYER_OCR],
        "extra": {"engine_confidence": raw.get("engine_confidence"), "status": raw.get("status"),
                  "must_verify_against_page_before_answer": raw.get("must_verify_against_page_before_answer")},
        "content_hash": _canonical_hash(raw, doc["title"]),
    }


def _doc_record(m: dict, root: Path, prefix: str, verify_sources: bool) -> tuple[dict, list[str]]:
    warnings = []
    doc_id = m["document_id"]
    original = root / m["original"]
    verified = None
    if verify_sources:
        if not original.exists():
            warnings.append(f"დოკუმენტი {doc_id}: ორიგინალი ფაილი ვერ მოიძებნა ({m['original']}).")
            verified = False
        else:
            verified = _file_sha256(original) == m["original_sha256"]
            if not verified:
                warnings.append(f"დოკუმენტი {doc_id}: ორიგინალის SHA-256 არ ემთხვევა manifest-ს.")
    pages_dir = root / "documents" / doc_id / "pages"
    page_images = sorted(f"documents/{doc_id}/pages/{p.name}" for p in pages_dir.glob("*.webp")) if pages_dir.exists() else []
    ext = original.suffix.lower()
    kind = "word" if ext in (".doc", ".docx") else "pdf"
    record = {
        "document_id": prefix + doc_id,
        "source_document_id": doc_id,
        "title": clean_title(m["original_filename"]),
        "original_filename": m["original_filename"],
        "kind": kind,
        "source_file": m["original"],
        "source_sha256": m["original_sha256"],
        "source_verified": verified,
        "page_count": len(page_images),
        "page_images": page_images,
        "native_units": m.get("native_units"),
        "review_flags": m.get("review_flags") or {},
        "meta": {k: m.get(k) for k in ("reference_render_pages", "reference_render_note", "source_locator",
                                        "ocr_pages", "primary_language", "markdown", "structured")},
    }
    if kind == "word" and (root / "documents" / doc_id / "reference-render.pdf").exists():
        record["meta"]["reference_render_pdf"] = f"documents/{doc_id}/reference-render.pdf"
    return record, warnings


def import_library(db_path: Path, root: Path, *, library_id: str = "main", id_prefix: str = "",
                   document_ids: list[str] | None = None, verify_sources: bool = True) -> dict:
    """Imports or re-imports a library. Returns a report dict (counts per layer)."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    native_path = root / "index" / "native-chunks.jsonl"
    if not manifest_path.exists() or not native_path.exists():
        raise ImportErrorKa(
            f"ბიბლიოთეკა ვერ მოიძებნა ან არასრულია: {root}. საჭიროა manifest.json და index/native-chunks.jsonl.")
    db.init_db(db_path)
    _unit_cache.clear()
    started = db.now_iso()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = set(document_ids) if document_ids else None
    warnings: list[str] = []

    docs: dict[str, dict] = {}
    for m in manifest.get("documents", []):
        if scope and m["document_id"] not in scope:
            continue
        rec, w = _doc_record(m, root, id_prefix, verify_sources)
        docs[m["document_id"]] = rec
        warnings.extend(w)
    if scope and (missing := scope - set(docs)):
        raise ImportErrorKa(f"manifest-ში ვერ მოიძებნა დოკუმენტ(ებ)ი: {', '.join(sorted(missing))}")

    records: list[dict] = []
    parents: dict[str, str] = {}
    native_raw = _read_jsonl(native_path)
    for raw in native_raw:
        doc = docs.get(raw["document_id"])
        if doc is None:
            if not scope:
                warnings.append(f"ჩანაწერი {raw['chunk_id']} ეკუთვნის manifest-ში არარსებულ დოკუმენტს.")
            continue
        records.append(_native_record(raw, doc, id_prefix, _unit_pages(root, raw["document_id"])))
        for uid in raw.get("unit_ids") or []:
            parents.setdefault(uid, id_prefix + raw["chunk_id"])
    for raw in _read_jsonl(root / "index" / "visual-navigation.jsonl"):
        if raw["document_id"] in docs:
            records.append(_visual_record(raw, docs[raw["document_id"]], id_prefix))
    for raw in _read_jsonl(root / "index" / "checked-formulas.jsonl"):
        if raw["document_id"] in docs:
            records.append(_formula_record(raw, docs[raw["document_id"]], id_prefix, parents))
    for raw in _read_jsonl(root / "index" / "ocr-unverified.jsonl"):
        if raw["document_id"] in docs:
            records.append(_ocr_record(raw, docs[raw["document_id"]], id_prefix))

    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ImportErrorKa(f"წყაროში ერთი და იგივე chunk_id მეორდება: {', '.join(dupes[:10])}")

    report = {layer: {"added": 0, "changed": 0, "unchanged": 0, "removed": 0, "total": 0} for layer in ALL_LAYERS}
    now = db.now_iso()
    with db.connect(db_path) as con:
        # Guard against document_id collisions with another library.
        for doc in docs.values():
            row = con.execute("SELECT library_id, source_sha256 FROM documents WHERE document_id=?",
                              (doc["document_id"],)).fetchone()
            if row and row["library_id"] != library_id:
                raise ImportErrorKa(
                    f"დოკუმენტის ID {doc['document_id']} უკვე გამოიყენება ბიბლიოთეკაში „{row['library_id']}“. "
                    "ახალი პაკეტი შემოიტანეთ განსხვავებული პრეფიქსით (მაგ. --id-prefix B-).")
        con.execute(
            "INSERT INTO libraries(library_id, root_path, id_prefix, manifest_sha256, imported_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(library_id) DO UPDATE SET root_path=excluded.root_path, id_prefix=excluded.id_prefix, "
            "manifest_sha256=excluded.manifest_sha256, imported_at=excluded.imported_at",
            (library_id, project_relative(root), id_prefix, _file_sha256(manifest_path), now),
        )
        for doc in docs.values():
            con.execute(
                "INSERT INTO documents(document_id, library_id, source_document_id, title, original_filename, kind,"
                " source_file, source_sha256, source_verified, page_count, page_images, native_units, review_flags,"
                " meta, imported_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET "
                "title=excluded.title, original_filename=excluded.original_filename, kind=excluded.kind,"
                " source_file=excluded.source_file, source_sha256=excluded.source_sha256,"
                " source_verified=excluded.source_verified, page_count=excluded.page_count,"
                " page_images=excluded.page_images, native_units=excluded.native_units,"
                " review_flags=excluded.review_flags, meta=excluded.meta, imported_at=excluded.imported_at",
                (doc["document_id"], library_id, doc["source_document_id"], doc["title"], doc["original_filename"],
                 doc["kind"], doc["source_file"], doc["source_sha256"],
                 None if doc["source_verified"] is None else int(doc["source_verified"]), doc["page_count"],
                 json.dumps(doc["page_images"]), doc["native_units"],
                 json.dumps(doc["review_flags"], ensure_ascii=False), json.dumps(doc["meta"], ensure_ascii=False), now),
            )
        existing = {
            row["chunk_id"]: row["content_hash"]
            for row in con.execute(
                f"SELECT chunk_id, content_hash FROM chunks WHERE library_id=? AND document_id IN "
                f"({','.join('?' * len(docs))})", (library_id, *[d["document_id"] for d in docs.values()]))
        } if docs else {}
        seen = set()
        for r in records:
            seen.add(r["chunk_id"])
            stat = report[r["layer"]]
            stat["total"] += 1
            old = existing.get(r["chunk_id"])
            if old == r["content_hash"]:
                stat["unchanged"] += 1
                continue
            stat["changed" if old else "added"] += 1
            con.execute(
                "INSERT INTO chunks(chunk_id, source_chunk_id, library_id, document_id, layer, title, text, markdown,"
                " embed_text, embed_hash, content_hash, unit_ids, locators, page_images, page_images_derived,"
                " source_file, source_sha256, review_flags, human_verified, status_note, extra, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(chunk_id) DO UPDATE SET"
                " source_chunk_id=excluded.source_chunk_id, library_id=excluded.library_id,"
                " document_id=excluded.document_id, layer=excluded.layer, title=excluded.title, text=excluded.text,"
                " markdown=excluded.markdown, embed_text=excluded.embed_text, embed_hash=excluded.embed_hash,"
                " content_hash=excluded.content_hash, unit_ids=excluded.unit_ids, locators=excluded.locators,"
                " page_images=excluded.page_images, page_images_derived=excluded.page_images_derived,"
                " source_file=excluded.source_file, source_sha256=excluded.source_sha256,"
                " review_flags=excluded.review_flags, human_verified=excluded.human_verified,"
                " status_note=excluded.status_note, extra=excluded.extra, updated_at=excluded.updated_at",
                (r["chunk_id"], r["source_chunk_id"], library_id, r["document_id"], r["layer"], r["title"], r["text"],
                 r["markdown"], r["embed_text"], sha256_text(r["embed_text"]), r["content_hash"],
                 json.dumps(r["unit_ids"], ensure_ascii=False), json.dumps(r["locators"], ensure_ascii=False),
                 json.dumps(r["page_images"]), int(r["page_images_derived"]), r["source_file"], r["source_sha256"],
                 json.dumps(r["review_flags"]), int(r["human_verified"]), r["status_note"],
                 json.dumps(r["extra"], ensure_ascii=False), now),
            )
        removed = [cid for cid in existing if cid not in seen]
        for cid in removed:
            layer = con.execute("SELECT layer FROM chunks WHERE chunk_id=?", (cid,)).fetchone()["layer"]
            report[layer]["removed"] += 1
            con.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
        removed_docs = []
        if not scope:
            keep = [d["document_id"] for d in docs.values()]
            for row in con.execute("SELECT document_id FROM documents WHERE library_id=?", (library_id,)).fetchall():
                if row["document_id"] not in keep:
                    removed_docs.append(row["document_id"])
            for did in removed_docs:
                n = con.execute("DELETE FROM chunks WHERE document_id=?", (did,)).rowcount
                report[LAYER_NATIVE]["removed"] += n
                con.execute("DELETE FROM documents WHERE document_id=?", (did,))
        db.bump_index_version(con)
        result = {
            "library_id": library_id,
            "root": project_relative(root),
            "documents": len(docs),
            "documents_removed": removed_docs,
            "scope": sorted(scope) if scope else "all",
            "manifest_document_count": manifest.get("document_count"),
            "manifest_native_chunk_count": manifest.get("native_chunk_count"),
            "native_chunks_in_file": len(native_raw),
            "layers": report,
            "word_chunks_with_derived_pages": sum(1 for r in records if r["page_images_derived"]),
            "chunks_without_page_image": sum(1 for r in records if r["layer"] == LAYER_NATIVE and not r["page_images"]),
            "sources_verified": sum(1 for d in docs.values() if d["source_verified"]),
            "warnings": warnings,
        }
        con.execute("INSERT INTO import_runs(library_id, started_at, finished_at, scope, report) VALUES(?,?,?,?,?)",
                    (library_id, started, db.now_iso(), json.dumps(result["scope"]),
                     json.dumps(result, ensure_ascii=False)))
    return result


_unit_cache: dict[tuple[str, str], _UnitPages] = {}


def _unit_pages(root: Path, doc_id: str) -> _UnitPages:
    key = (str(root), doc_id)
    if key not in _unit_cache:
        _unit_cache[key] = _UnitPages(root, doc_id)
    return _unit_cache[key]
