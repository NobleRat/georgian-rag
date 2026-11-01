"""Shared helpers for evaluation scripts."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load_questions() -> dict:
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def normalize_answer(text: str) -> str:
    t = (text or "").replace(" ", " ").replace(" ", " ")
    t = re.sub(r"<sup>\s*", "^", t)
    t = re.sub(r"</sup>|</sub>", "", t)
    t = re.sub(r"<sub>\s*", "_", t)
    t = t.replace("⁻", "-").replace("−", "-")
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    t = re.sub(r"([0-9])([⁰¹²³⁴⁵⁶⁷⁸⁹-]*[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: m.group(1) + "^" + m.group(2).translate(sup), t)
    t = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", t)
    return t


def facts_found(answer: str, facts: list[str]) -> dict:
    norm = normalize_answer(answer)
    return {f: bool(re.search(f, norm, re.I)) for f in facts}


def write_result(prefix: str, payload: dict, markdown: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    j = RESULTS_DIR / f"{prefix}-{stamp}.json"
    m = RESULTS_DIR / f"{prefix}-{stamp}.md"
    j.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    m.write_text(markdown, encoding="utf-8")
    return j, m
