"""Embedding cache, persistence across restart, model isolation, retries. SIMULATED OpenAI (no network)."""
import openai
import httpx
import pytest

from app import embeddings
from app.importer import import_library
from app.search import SearchIndex

pytestmark = pytest.mark.needs_library


def _job(settings, client):
    job = embeddings.EmbeddingJob()
    return job.run(settings, settings.database_path, client_factory=lambda _s: client, retry_sleeps=(0, 0))


def _rate_limit(code=None):
    req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    body = {"code": code} if code else None
    return openai.RateLimitError("limit", response=httpx.Response(429, request=req), body=body)


def test_embeddings_are_cached_and_survive_restart(settings, mini_library, fake_openai):
    s = settings()
    import_library(s.database_path, mini_library)
    est = embeddings.estimate(s, s.database_path)
    assert est["items"] > 0 and est["estimated_cost_usd"] > 0

    state = _job(s, fake_openai)
    assert state["status"] == "done" and state["done_items"] == est["items"]
    calls_after_first = len(fake_openai.embedding_calls)
    assert calls_after_first == est["requests"]

    # Same data again: nothing pending, no API call.
    assert _job(s, fake_openai)["total_items"] == 0
    assert len(fake_openai.embedding_calls) == calls_after_first

    # "Restart": a fresh index object reading the same SQLite file sees all vectors.
    idx = SearchIndex(s.database_path, s.embedding_space)
    assert idx.vector_count == est["items"]

    # Re-import of unchanged data does not invalidate vectors.
    import_library(s.database_path, mini_library)
    assert embeddings.estimate(s, s.database_path)["items"] == 0


def test_model_change_uses_separate_space(settings, mini_library, fake_openai):
    s_large = settings()
    import_library(s_large.database_path, mini_library)
    _job(s_large, fake_openai)
    s_small = settings(OPENAI_EMBEDDING_MODEL="text-embedding-3-small")
    assert s_small.embedding_space != s_large.embedding_space
    cov = embeddings.coverage(s_small.database_path, s_small.embedding_space)
    assert cov["done"] == 0 and cov["other_spaces"][0]["space"] == s_large.embedding_space
    idx_small = SearchIndex(s_small.database_path, s_small.embedding_space)
    assert idx_small.vector_count == 0  # old 3072-d vectors are never mixed in

    _job(s_small, fake_openai)
    assert fake_openai.embedding_calls[-1]["model"] == "text-embedding-3-small"
    idx_small = SearchIndex(s_small.database_path, s_small.embedding_space)
    idx_large = SearchIndex(s_large.database_path, s_large.embedding_space)
    assert idx_small.dims == 1536 and idx_large.dims == 3072


def test_dimensions_setting_creates_its_own_space(settings, mini_library, fake_openai):
    s = settings(OPENAI_EMBEDDING_DIMENSIONS="256")
    import_library(s.database_path, mini_library)
    _job(s, fake_openai)
    assert fake_openai.embedding_calls[0]["dimensions"] == 256
    assert SearchIndex(s.database_path, s.embedding_space).dims == 256


def test_query_embedding_is_cached(settings, mini_library, fake_openai):
    s = settings()
    import_library(s.database_path, mini_library)
    v1 = embeddings.embed_query(s, s.database_path, fake_openai, "ნალექები")
    v2 = embeddings.embed_query(s, s.database_path, fake_openai, "ნალექები")
    assert len(fake_openai.embedding_calls) == 1 and (v1 == v2).all()


def test_transient_error_is_retried(settings, mini_library, fake_openai):
    s = settings()
    import_library(s.database_path, mini_library)
    fake_openai.embedding_errors = [_rate_limit()]
    state = _job(s, fake_openai)
    assert state["status"] == "done" and state["retries"] == 1


def test_quota_error_stops_with_georgian_message(settings, mini_library, fake_openai):
    s = settings()
    import_library(s.database_path, mini_library)
    fake_openai.embedding_errors = [_rate_limit("insufficient_quota")]
    state = _job(s, fake_openai)
    assert state["status"] == "error" and state["error_code"] == "quota"
    assert "კვოტა" in state["error"]
    assert embeddings.coverage(s.database_path, s.embedding_space)["done"] == 0


def test_missing_key_stops_job(settings, mini_library):
    s = settings(OPENAI_API_KEY="")
    import_library(s.database_path, mini_library)
    state = embeddings.EmbeddingJob().run(s, s.database_path)
    assert state["status"] == "error" and state["error_code"] == "missing_key"


def test_snapshot_roundtrip_restores_vectors_on_empty_disk(settings, mini_library, fake_openai, tmp_path):
    s = settings(PREBUILT_DIR=str(tmp_path / "prebuilt"), DATABASE_PATH=str(tmp_path / "a.sqlite"))
    import_library(s.database_path, mini_library)
    state = _job(s, fake_openai)
    assert state["snapshot"]["exported"] == state["done_items"]  # exported automatically after indexing
    calls = len(fake_openai.embedding_calls)

    fresh = settings(PREBUILT_DIR=str(tmp_path / "prebuilt"), DATABASE_PATH=str(tmp_path / "fresh.sqlite"))
    import_library(fresh.database_path, mini_library)
    res = embeddings.load_prebuilt(fresh, fresh.database_path)
    assert res["loaded"] == state["done_items"] and res["still_missing"] == 0
    assert embeddings.estimate(fresh, fresh.database_path)["items"] == 0
    assert len(fake_openai.embedding_calls) == calls
    a = SearchIndex(s.database_path, s.embedding_space)
    b = SearchIndex(fresh.database_path, fresh.embedding_space)
    assert (a.matrix == b.matrix).all()  # identical vectors, bit for bit

    other = settings(PREBUILT_DIR=str(tmp_path / "prebuilt"), DATABASE_PATH=str(tmp_path / "o.sqlite"),
                     OPENAI_EMBEDDING_MODEL="text-embedding-3-small")
    import_library(other.database_path, mini_library)
    assert embeddings.load_prebuilt(other, other.database_path)["loaded"] == 0  # never mixes spaces


def test_project_paths_are_relative():
    from app.config import project_relative, resolve_project_path
    from tests.conftest import LIBRARY
    assert project_relative(LIBRARY) == "data/rag-library"
    assert resolve_project_path("data/rag-library") == LIBRARY
