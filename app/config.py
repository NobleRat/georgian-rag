"""Settings loaded from .env (server side only).

The .env file is re-read when it changes, so a key pasted while the server is
running is picked up without a restart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
STORAGE_DIR = PROJECT_ROOT / "storage"
STATIC_DIR = PROJECT_ROOT / "static"

# tiktoken downloads its encoding files once; keep them inside the project.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(STORAGE_DIR / "tiktoken"))

DEFAULTS = {
    "OPENAI_API_KEY": "",
    "OPENAI_CHAT_MODEL": "gpt-6.1-terra",
    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
    "OPENAI_EMBEDDING_DIMENSIONS": "",
    "OPENAI_REASONING_EFFORT": "low",
    "OPENAI_MAX_OUTPUT_TOKENS": "6000",
    "OPENAI_IMAGE_DETAIL": "high",
    "OPENAI_MAX_RETRIES": "3",
    "OPENAI_TIMEOUT_SECONDS": "120",
    "OPENAI_ALLOW_WEB_SEARCH": "false",
    "MAX_IMAGES_PER_QUESTION": "3",
    "RETRIEVAL_TOP_K": "10",
    "SEARCH_VECTOR_SLOTS": "2",
    "EMBED_BATCH_SIZE": "64",
    "APP_HOST": "127.0.0.1",
    "APP_PORT": "8765",
    "DATA_LIBRARY": "data/rag-library",
    "DATABASE_PATH": "storage/rag.sqlite",
    "PREBUILT_DIR": "prebuilt",
    "APP_USERNAME": "rag",
    "APP_PASSWORD": "",
}


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int | None
    reasoning_effort: str
    max_output_tokens: int
    image_detail: str
    max_retries: int
    timeout_seconds: float
    allow_web_search: bool
    max_images: int
    top_k: int
    vector_slots: int | None
    embed_batch_size: int
    host: str
    port: int
    data_library: Path
    database_path: Path
    prebuilt_dir: Path
    app_username: str
    app_password: str
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def embedding_space(self) -> str:
        """Identifier of the vector space. Vectors from different spaces are never compared."""
        dims = self.embedding_dimensions or 0
        return f"{self.embedding_model}:{dims}"

    def public_dict(self) -> dict:
        """Settings safe to show in the browser (never includes the key)."""
        return {
            "api_key_configured": self.has_api_key,
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions or "მოდელის ნაგულისხმევი",
            "embedding_space": self.embedding_space,
            "reasoning_effort": self.reasoning_effort,
            "image_detail": self.image_detail,
            "max_images": self.max_images,
            "top_k": self.top_k,
            "vector_slots": self.vector_slots,
            "allow_web_search": self.allow_web_search,
            "env_path": str(ENV_PATH) if ENV_PATH.exists() else None,
            "password_protected": bool(self.app_password),
        }


def _resolve(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_project_path(stored: str) -> Path:
    """Inverse of project_relative(): relative paths are resolved against the current project root."""
    return _resolve(stored)


def project_relative(path: Path) -> str:
    """Stores paths inside the project as relative POSIX paths, so a copied/cloned project still works."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


_cache: tuple[float | None, Settings] | None = None


def load_settings(env_path: Path | None = None, overrides: dict | None = None) -> Settings:
    env_path = env_path or ENV_PATH
    values = dict(DEFAULTS)
    if env_path.exists():
        values.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})
    # Real environment variables win over .env (useful for tests and CI).
    for key in DEFAULTS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    # Hosting platforms (e.g. Render) provide the port as PORT.
    if os.environ.get("PORT") and not os.environ.get("APP_PORT"):
        values["APP_PORT"] = os.environ["PORT"]
    if overrides:
        values.update(overrides)
    dims = values["OPENAI_EMBEDDING_DIMENSIONS"].strip()
    return Settings(
        openai_api_key=values["OPENAI_API_KEY"].strip(),
        chat_model=values["OPENAI_CHAT_MODEL"].strip(),
        embedding_model=values["OPENAI_EMBEDDING_MODEL"].strip(),
        embedding_dimensions=int(dims) if dims else None,
        reasoning_effort=values["OPENAI_REASONING_EFFORT"].strip(),
        max_output_tokens=int(values["OPENAI_MAX_OUTPUT_TOKENS"]),
        image_detail=values["OPENAI_IMAGE_DETAIL"].strip(),
        max_retries=int(values["OPENAI_MAX_RETRIES"]),
        timeout_seconds=float(values["OPENAI_TIMEOUT_SECONDS"]),
        allow_web_search=_as_bool(values["OPENAI_ALLOW_WEB_SEARCH"]),
        max_images=int(values["MAX_IMAGES_PER_QUESTION"]),
        top_k=int(values["RETRIEVAL_TOP_K"]),
        vector_slots=int(values["SEARCH_VECTOR_SLOTS"]) if values["SEARCH_VECTOR_SLOTS"].strip() else None,
        embed_batch_size=int(values["EMBED_BATCH_SIZE"]),
        host=values["APP_HOST"].strip(),
        port=int(values["APP_PORT"]),
        data_library=_resolve(values["DATA_LIBRARY"].strip()),
        database_path=_resolve(values["DATABASE_PATH"].strip()),
        prebuilt_dir=_resolve(values["PREBUILT_DIR"].strip()),
        app_username=values["APP_USERNAME"].strip(),
        app_password=values["APP_PASSWORD"],
        raw={k: v for k, v in values.items() if k not in ("OPENAI_API_KEY", "APP_PASSWORD")},
    )


def get_settings() -> Settings:
    """Cached settings, refreshed whenever .env is modified."""
    global _cache
    mtime = ENV_PATH.stat().st_mtime if ENV_PATH.exists() else None
    if _cache is None or _cache[0] != mtime:
        _cache = (mtime, load_settings())
    return _cache[1]
