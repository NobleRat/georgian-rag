"""Test fixtures. Everything here is OFFLINE: OpenAI is replaced by FakeOpenAI (simulated)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import PROJECT_ROOT, load_settings
from app.textproc import tokenize

LIBRARY = PROJECT_ROOT / "data" / "rag-library"


def pytest_collection_modifyitems(items):
    if not (LIBRARY / "manifest.json").exists():
        skip = pytest.mark.skip(reason="data/rag-library is not extracted")
        for item in items:
            if "needs_library" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def settings(tmp_path):
    def make(**overrides):
        # PREBUILT_DIR must point to a temp folder: indexing exports a snapshot automatically,
        # and tests must never overwrite the committed prebuilt/embeddings file.
        base = {"DATABASE_PATH": str(tmp_path / "test.sqlite"), "DATA_LIBRARY": str(LIBRARY),
                "PREBUILT_DIR": str(tmp_path / "prebuilt"),
                "OPENAI_API_KEY": "sk-test-FAKEKEY1234567890", "OPENAI_EMBEDDING_DIMENSIONS": "",
                "OPENAI_CHAT_MODEL": "gpt-6.1-terra", "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large"}
        base.update(overrides)
        return load_settings(env_path=tmp_path / "missing.env", overrides=base)
    return make


@pytest.fixture
def mini_library(tmp_path):
    """A small copy of the library (documents 01, 17, 23) that tests may modify."""
    dst = tmp_path / "mini-library"
    keep = {"01", "17", "23"}
    (dst / "index").mkdir(parents=True)
    manifest = json.loads((LIBRARY / "manifest.json").read_text(encoding="utf-8"))
    manifest["documents"] = [d for d in manifest["documents"] if d["document_id"] in keep]
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    for name in ("native-chunks.jsonl", "visual-navigation.jsonl", "checked-formulas.jsonl", "ocr-unverified.jsonl"):
        lines = [l for l in (LIBRARY / "index" / name).read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["document_id"] in keep]
        (dst / "index" / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for d in keep:
        src = LIBRARY / "documents" / d
        (dst / "documents" / d).mkdir(parents=True)
        shutil.copy2(src / "document.json", dst / "documents" / d / "document.json")
        for orig in src.glob("original.*"):
            shutil.copy2(orig, dst / "documents" / d / orig.name)
        shutil.copytree(src / "pages", dst / "documents" / d / "pages")
    return dst


# ------------------------------------------------------------------ simulated OpenAI

def fake_vector(text: str, dims: int) -> list[float]:
    """Deterministic bag-of-stems vector: texts sharing words get similar vectors."""
    v = np.zeros(dims, dtype=np.float32)
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dims] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeEmbeddings:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model, input, dimensions=None, **_):
        self.parent.embedding_calls.append({"model": model, "n": len(input), "dimensions": dimensions})
        if self.parent.embedding_errors:
            raise self.parent.embedding_errors.pop(0)
        dims = dimensions or (3072 if "large" in model else 1536)
        data = [SimpleNamespace(index=i, embedding=fake_vector(t, dims)) for i, t in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=sum(len(t) // 2 for t in input)))


def _usage(inp=1000, out=200):
    return SimpleNamespace(input_tokens=inp, output_tokens=out, total_tokens=inp + out,
                           input_tokens_details=SimpleNamespace(cached_tokens=0),
                           output_tokens_details=SimpleNamespace(reasoning_tokens=50))


class FakeResponses:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        self.parent.response_calls.append(kwargs)
        if not kwargs.get("stream"):
            return SimpleNamespace(output_text=self.parent.rewrite_text, usage=_usage(300, 20))
        answer = self.parent.answer_fn(kwargs)

        def gen():
            for i in range(0, len(answer), 7):
                yield SimpleNamespace(type="response.output_text.delta", delta=answer[i:i + 7])
            yield SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=_usage()))
        return gen()


class FakeModels:
    def retrieve(self, model):
        return SimpleNamespace(id=model)


class FakeOpenAI:
    """Simulated client. Answers cite the first source id found in the prompt."""

    def __init__(self):
        self.embedding_calls: list = []
        self.embedding_errors: list = []
        self.response_calls: list = []
        self.rewrite_text = ""
        self.embeddings = FakeEmbeddings(self)
        self.responses = FakeResponses(self)
        self.models = FakeModels()
        self.answer_fn = self.default_answer

    @staticmethod
    def default_answer(kwargs):
        import re
        content = kwargs["input"][-1]["content"]
        text = next(c["text"] for c in content if c["type"] == "input_text")
        ids = re.findall(r'<source id="([^"]+)"', text)
        return f"სატესტო პასუხი [{ids[0]}]." if ids else "ამ მასალებში საკმარისი ინფორმაცია ვერ მოიძებნა."


@pytest.fixture
def fake_openai():
    return FakeOpenAI()
