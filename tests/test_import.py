"""Import: sample -> full, idempotency, change detection, layer separation, ID collisions. Offline."""
import json
import sqlite3

import pytest

from app import embeddings
from app.importer import ImportErrorKa, import_library
from tests.conftest import LIBRARY

pytestmark = pytest.mark.needs_library


def counts(db_path):
    con = sqlite3.connect(db_path)
    try:
        return dict(con.execute("SELECT layer, COUNT(*) FROM chunks GROUP BY layer").fetchall()), \
            con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        con.close()


def test_sample_then_full_import_without_duplicates(settings):
    s = settings()
    sample = import_library(s.database_path, LIBRARY, document_ids=["01", "06", "07", "17", "23"])
    assert sample["documents"] == 5
    assert sample["layers"]["native"]["added"] == 6 + 20 + 60 + 21 + 108
    assert sample["sources_verified"] == 5

    full = import_library(s.database_path, LIBRARY)
    assert full["documents"] == 24
    assert full["native_chunks_in_file"] == 550
    assert full["layers"]["native"]["total"] == 550
    assert full["layers"]["native"]["unchanged"] == 215  # the sample documents were not re-written
    assert full["layers"]["visual_note"]["total"] == 7
    assert full["layers"]["formula"]["total"] == 4
    assert full["layers"]["ocr_unverified"]["total"] == 234

    again = import_library(s.database_path, LIBRARY)
    for layer, stat in again["layers"].items():
        assert stat["added"] == stat["changed"] == stat["removed"] == 0, layer
    layers, n_docs = counts(s.database_path)
    assert n_docs == 24
    assert layers == {"native": 550, "visual_note": 7, "formula": 4, "ocr_unverified": 234}


def test_source_fields_are_preserved(settings):
    s = settings()
    import_library(s.database_path, LIBRARY, document_ids=["07"])
    src = {}
    for line in (LIBRARY / "index" / "native-chunks.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["document_id"] == "07":
            src[r["chunk_id"]] = r
    con = sqlite3.connect(s.database_path)
    con.row_factory = sqlite3.Row
    rows = {r["chunk_id"]: r for r in con.execute("SELECT * FROM chunks WHERE layer='native'")}
    con.close()
    assert set(rows) == set(src)
    for cid, r in src.items():
        row = rows[cid]
        assert row["document_id"] == r["document_id"]
        assert json.loads(row["unit_ids"]) == r["unit_ids"]
        assert json.loads(row["locators"]) == r["locators"]
        assert row["source_file"] == r["source_file"]
        assert row["source_sha256"] == r["source_sha256"]
        assert row["markdown"] == r["markdown"]  # tables, <sup>/<sub>, footnote links unchanged
        if r["page_images"]:
            assert json.loads(row["page_images"]) == r["page_images"]
    table = rows["07-c0058"]
    assert "<table>" in table["markdown"] and "<sup>-33</sup>" in table["markdown"]


def test_ocr_is_a_separate_layer(settings):
    s = settings()
    import_library(s.database_path, LIBRARY, document_ids=["06"])
    con = sqlite3.connect(s.database_path)
    rows = con.execute("SELECT chunk_id, layer, review_flags FROM chunks").fetchall()
    con.close()
    ocr = [r for r in rows if r[1] == "ocr_unverified"]
    assert ocr and all(r[0].endswith("-ocr") for r in ocr)
    assert all("unverified_ocr" in r[2] for r in ocr)
    assert not any(r[0].endswith("-ocr") for r in rows if r[1] != "ocr_unverified")
    assert "ocr_unverified" not in embeddings.EMBEDDED_LAYERS


def test_change_detection_and_removal(settings, mini_library):
    s = settings()
    first = import_library(s.database_path, mini_library)
    assert first["documents"] == 3
    path = mini_library / "index" / "native-chunks.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["text"] += " (შესწორება)"
    lines[0] = json.dumps(rec, ensure_ascii=False)
    removed_id = json.loads(lines[-1])["chunk_id"]
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    second = import_library(s.database_path, mini_library)
    native = second["layers"]["native"]
    assert native["changed"] == 1 and native["removed"] == 1 and native["added"] == 0
    con = sqlite3.connect(s.database_path)
    assert con.execute("SELECT COUNT(*) FROM chunks WHERE chunk_id=?", (removed_id,)).fetchone()[0] == 0
    con.close()


def test_document_id_collision_requires_prefix(settings, mini_library):
    s = settings()
    import_library(s.database_path, mini_library, library_id="main")
    with pytest.raises(ImportErrorKa):
        import_library(s.database_path, mini_library, library_id="second")
    rep = import_library(s.database_path, mini_library, library_id="second", id_prefix="B-")
    assert rep["documents"] == 3
    con = sqlite3.connect(s.database_path)
    ids = [r[0] for r in con.execute("SELECT chunk_id FROM chunks WHERE library_id='second' LIMIT 3")]
    con.close()
    assert all(i.startswith("B-") for i in ids)


def test_missing_library_gives_georgian_error(settings, tmp_path):
    with pytest.raises(ImportErrorKa, match="ბიბლიოთეკა ვერ მოიძებნა"):
        import_library(settings().database_path, tmp_path / "nothing")
