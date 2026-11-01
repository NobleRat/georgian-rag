"""Georgian tokenization and keyword/hybrid retrieval. Offline (keyword search uses no API)."""
import pytest

from app import rag
from app.importer import import_library
from app.search import SearchIndex
from app.textproc import strip_anchors, tokenize
from tests.conftest import LIBRARY


@pytest.mark.parametrize("forms", [
    ["დიაგრამა", "დიაგრამაზე", "დიაგრამის", "დიაგრამები"],
    ["ნალექი", "ნალექები", "ნალექების"],
    ["გამყინვარება", "გამყინვარების"],
    ["რუკა", "რუკაზე", "რუკები"],
])
def test_inflected_forms_share_a_key(forms):
    assert len({tokenize(f)[0] for f in forms}) == 1


def test_query_transliteration_matches_latin_names():
    assert "gudaut" in tokenize("გუდაუთის", query=True)
    assert tokenize("Gudauta") == ["gudaut"]


def test_strip_anchors_keeps_tables_and_superscripts():
    md = '<a id="07-w0039"></a>10<sup>-33</sup> <td colspan="1">x</td>'
    assert strip_anchors(md) == "10<sup>-33</sup> <td>x</td>"


@pytest.fixture(scope="module")
def full_index(tmp_path_factory):
    if not (LIBRARY / "manifest.json").exists():
        pytest.skip("library not extracted")
    db_path = tmp_path_factory.mktemp("idx") / "s.sqlite"
    import_library(db_path, LIBRARY, verify_sources=False)
    return SearchIndex(db_path, "text-embedding-3-large:0")


@pytest.mark.needs_library
def test_keyword_search_finds_expected_chunk(full_index):
    res = full_index.search("როდის იქნა მიღებული გამყინვარების თეორია?")
    assert res["mode"] == "keyword"
    assert res["hits"][0]["chunk"]["chunk_id"] == "01-c0001"


@pytest.mark.needs_library
def test_document_filter(full_index):
    res = full_index.search("ნალექები", document_ids=["17"])
    assert res["hits"] and all(h["chunk"]["document_id"] == "17" for h in res["hits"])


@pytest.mark.needs_library
def test_ocr_excluded_by_default_and_marked_when_enabled(full_index):
    q = "პალინოლოგიური მონაცემები კლიმატი"
    assert all(h["chunk"]["layer"] != "ocr_unverified" for h in full_index.search(q)["hits"])
    with_ocr = full_index.search(q, include_ocr=True)["hits"]
    ocr = [h for h in with_ocr if h["chunk"]["layer"] == "ocr_unverified"]
    assert ocr and all(h["via"] == "ocr_candidate_page" for h in ocr)


@pytest.mark.needs_library
def test_formula_supplement_attached(full_index):
    ids = [h["chunk"]["chunk_id"] for h in full_index.search(
        "ვერტიკალური სვეტის სიმკვრივე რელიეფის სიმაღლე H სიღრმე T")["hits"]]
    assert "23-c0057" in ids and "23-w0233-visual-math" in ids


@pytest.mark.needs_library
def test_word_and_pdf_locations_are_distinguished(full_index):
    pdf = rag.location(full_index.by_id["17-c0015"])
    word = rag.location(full_index.by_id["07-c0001"])
    assert pdf["kind"] == "pdf" and pdf["label"].startswith("PDF, გვ. 15")
    assert word["kind"] == "word" and "აბზაცი 07-w0001" in word["label"] and "საჩვენებელი ვერსიის გვ. 1" in word["label"]
    foot = rag.location(full_index.by_id["05-c0013"])
    assert "სქოლიო 1" in foot["label"]


@pytest.mark.needs_library
def test_vector_with_wrong_dimension_is_ignored(full_index):
    import numpy as np
    res = full_index.search("ნალექები", query_vec=np.ones(10, dtype=np.float32))
    assert res["mode"] == "keyword"
