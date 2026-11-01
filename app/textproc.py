"""Text helpers: Georgian-aware tokenization, light stemming and markdown cleanup."""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ANCHOR_RE = re.compile(r'<a id="[^"]*"></a>')
_COLSPAN1_RE = re.compile(r' colspan="1"')

# Georgian Mtavruli (U+1C90..U+1CBF) maps onto Mkhedruli (U+10D0..U+10FF).
_MTAVRULI = {cp: cp - 0x1C90 + 0x10D0 for cp in range(0x1C90, 0x1CBB)}

# Common Georgian case endings, plural and derivational suffixes, longest first.
# This is a deliberately light stemmer: it only needs to make inflected forms of
# the same word meet for keyword matching; semantic search covers the rest.
_KA_SUFFIXES = sorted(
    {
        "ებისთვის", "ისთვის", "ებისგან", "ისგან", "ებიდან", "იდან", "ებამდე", "ამდე", "ებთან", "ებში",
        "ებზე", "ებით", "ების", "ებს", "ებმა", "ები", "ება", "ობა", "ობის", "ობით", "ობას", "ურ", "ული",
        "ულ", "ებ", "თვის", "გან", "დან", "თან", "ში", "ზე", "ით", "ად", "ის", "ვით", "კენ",
        "ს", "ა", "ი", "ე", "ო", "უ", "მ",
    },
    key=len,
    reverse=True,
)
_MIN_STEM = 3

# Function words and question words that carry no topical meaning.
STOPWORDS = {
    "და", "რა", "რას", "რის", "რით", "რომ", "რომელი", "რომელიც", "რომლის", "რომელსაც", "როგორ", "როგორი",
    "როდის", "სად", "ვინ", "რატომ", "რამდენი", "რამდენ", "არის", "იყო", "არიან", "იყვნენ", "ეს", "ის",
    "ამ", "იმ", "ეგ", "თუ", "ან", "მაგრამ", "კი", "არ", "ვერ", "ასევე", "აგრეთვე", "შესახებ", "მიხედვით",
    "აქვს", "ჰქონდა", "უნდა", "შეიძლება", "მიერ", "შორის", "გარდა", "ერთი", "ყველა", "ძალიან", "მას",
    "მისი", "მათი", "მათ", "მე", "ჩვენ", "თქვენ", "ხოლო", "ანუ", "ე", "ი", "სხვა", "სხვ", "მოცემული",
    "ტექსტში", "მასალებში", "დოკუმენტში", "ლექციაში", "აღწერილი", "ნათქვამი", "ახსენი", "აღწერე",
    "the", "a", "an", "of", "and", "or", "in", "on", "is", "are", "to", "for", "what", "how",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return text.translate(_MTAVRULI).lower()


def is_georgian(word: str) -> bool:
    return any("ა" <= ch <= "ჿ" for ch in word)


_KA_VOWELS = "აეიოუ"
_LAT_VOWELS = "aeiouy"
_TRUNCATE = 7

# Georgian national transliteration, used so that Georgian query words can match
# Latin spellings in the sources (e.g. გუდაუთა ~ Gudauta, სოხუმი ~ Sokhumi).
_TRANSLIT = str.maketrans({
    "ა": "a", "ბ": "b", "გ": "g", "დ": "d", "ე": "e", "ვ": "v", "ზ": "z", "თ": "t", "ი": "i", "კ": "k",
    "ლ": "l", "მ": "m", "ნ": "n", "ო": "o", "პ": "p", "ჟ": "zh", "რ": "r", "ს": "s", "ტ": "t", "უ": "u",
    "ფ": "p", "ქ": "k", "ღ": "gh", "ყ": "q", "შ": "sh", "ჩ": "ch", "ც": "ts", "ძ": "dz", "წ": "ts", "ჭ": "ch",
    "ხ": "kh", "ჯ": "j", "ჰ": "h",
})


def stem(word: str) -> str:
    """Strips one inflectional suffix and a final vowel, then keeps the first 7 letters.

    Georgian is agglutinative: "დიაგრამა", "დიაგრამაზე", "დიაგრამის" must meet on the
    same key. Over-merging of rare words is acceptable here because semantic search
    and BM25 IDF compensate.
    """
    if word.isdigit():
        return word
    if not is_georgian(word):
        w = word
        if len(w) > 4 and w.endswith("s") and w.isalpha():
            w = w[:-1]
        if len(w) > 4 and w[-1] in _LAT_VOWELS:
            w = w[:-1]
        return w[:_TRUNCATE]
    w = word
    for suf in _KA_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= _MIN_STEM:
            w = w[: -len(suf)]
            break
    if len(w) > _MIN_STEM and w[-1] in _KA_VOWELS:
        w = w[:-1]
    return w[:_TRUNCATE]


def transliterate(word: str) -> str:
    return word.translate(_TRANSLIT)


def tokenize(text: str, query: bool = False) -> list[str]:
    """Normalized, stemmed tokens. Single characters are dropped except digits.

    For queries, Georgian words also produce a Latin-transliterated variant.
    """
    out = []
    for tok in _WORD_RE.findall(normalize(text)):
        if (len(tok) == 1 and not tok.isdigit()) or tok in STOPWORDS:
            continue
        out.append(stem(tok))
        if query and is_georgian(tok):
            out.append(stem(transliterate(stem(tok))))
    return out


def strip_anchors(markdown: str) -> str:
    """Removes paragraph anchors (<a id=..>) that only waste tokens; keeps tables, sup/sub."""
    md = _ANCHOR_RE.sub("", markdown or "")
    return _COLSPAN1_RE.sub("", md)


def clean_title(filename: str) -> str:
    name = re.sub(r"\.(pdf|docx?|DOCX?|PDF)$", "", filename or "").strip()
    return re.sub(r"\s+", " ", name)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snippet(text: str, limit: int = 220) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"
