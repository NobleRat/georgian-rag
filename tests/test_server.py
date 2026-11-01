"""HTTP API: status, file serving safety, CSRF header, key never exposed. Offline."""
import json

import pytest
from fastapi.testclient import TestClient

from app import config
from app.config import STATIC_DIR
from tests.conftest import LIBRARY

pytestmark = pytest.mark.needs_library
SECRET = "sk-test-VERYSECRET0123456789abcdef"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "srv.sqlite"))
    monkeypatch.setenv("DATA_LIBRARY", str(LIBRARY))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(config, "_cache", None)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "none.env")
    from app.server import app
    with TestClient(app) as c:
        yield c
    monkeypatch.setattr(config, "_cache", None)


def test_first_start_imports_library_and_status_hides_key(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.text
    data = r.json()
    assert data["documents"] == 24 and data["chunks_by_layer"]["native"] == 550
    assert data["settings"]["api_key_configured"] is True
    assert SECRET not in body and "VERYSECRET" not in body


def test_documents_and_chunk(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 24 and {d["kind"] for d in docs} == {"pdf", "word"}
    c = client.get("/api/chunks/07-c0058").json()
    assert "<table>" in c["markdown_clean"] and 'colspan="1"' not in c["markdown_clean"]


def test_file_serving_is_confined(client):
    assert client.get("/files/main/documents/06/pages/007.webp").status_code == 200
    assert client.get("/files/main/../../.env").status_code == 404
    assert client.get("/files/main/manifest.json").status_code == 404  # type not allowed
    assert client.get("/files/other/documents/06/pages/007.webp").status_code == 404


def test_post_requires_app_header(client):
    assert client.post("/api/chat", json={"question": "x"}).status_code == 403
    assert client.post("/api/index/start", json={"confirm": True}).status_code == 403


def test_index_start_requires_confirmation(client):
    r = client.post("/api/index/start", json={}, headers={"X-Rag-App": "1"})
    assert r.status_code == 400


def test_static_files_do_not_contain_secrets():
    for f in STATIC_DIR.iterdir():
        text = f.read_text(encoding="utf-8")
        assert "sk-" not in text and "OPENAI_API_KEY=" not in text.replace("OPENAI_API_KEY=თქვენი", "")


def test_error_messages_mask_keys():
    from app.openai_client import sanitize
    assert SECRET not in sanitize(f"Incorrect API key provided: {SECRET}")


def test_password_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "pw.sqlite"))
    monkeypatch.setenv("DATA_LIBRARY", str(LIBRARY))
    monkeypatch.setenv("APP_PASSWORD", "საიდუმლო-123")
    monkeypatch.setattr(config, "_cache", None)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "none.env")
    from app.server import app
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200  # health check stays open for the host
        assert c.get("/").status_code == 401
        assert c.get("/api/status").status_code == 401
        assert c.get("/files/main/documents/06/pages/007.webp").status_code == 401
        assert c.get("/api/status", auth=("rag", "wrong")).status_code == 401
        ok = c.get("/api/status", auth=("rag", "საიდუმლო-123"))
        assert ok.status_code == 200 and "საიდუმლო" not in ok.text
        assert ok.json()["settings"]["password_protected"] is True
    monkeypatch.setattr(config, "_cache", None)


def test_startup_restores_embeddings_from_snapshot_without_api(client):
    data = client.get("/api/status").json()
    start = data["startup"]
    assert start["data_library"] == "data/rag-library"  # stored relative, not as an absolute local path
    assert data["libraries"][0]["root_path"] == "data/rag-library"
    if start["prebuilt"]["loaded"]:
        cov = data["embedding_coverage"]
        assert cov["done"] == cov["total"]
    assert not [u for u in data["usage"] if u["kind"] == "embedding"]  # nothing was sent to OpenAI
