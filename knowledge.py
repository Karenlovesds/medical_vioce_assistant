from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

CHUNK_WORDS = 180
OVERLAP_WORDS = 30
BM25_K1 = 1.5
BM25_B = 0.75
MIN_COVERAGE = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _chunk_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, CHUNK_WORDS - OVERLAP_WORDS)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages


def _read_text(path: Path) -> list[tuple[int, str]]:
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    return [(0, text)] if text else []


def build_index() -> int:
    files = sorted(
        path
        for path in DOCS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    if not files:
        raise SystemExit(f"No documents found in {DOCS_DIR}")

    chunks: list[dict[str, Any]] = []
    for path in files:
        pages = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_text(path)
        if not pages:
            print(f"Skipping {path.name}: no extractable text")
            continue
        source = str(path.relative_to(DOCS_DIR))
        for page, text in pages:
            for chunk in _chunk_words(text):
                chunks.append({"text": chunk, "source": source, "page": page})

    if not chunks:
        raise SystemExit("No indexable text found")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks from {len(files)} file(s) to {CHUNKS_PATH}")
    return len(chunks)


class _BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.term_frequencies: list[Counter[str]] = []
        self.lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for chunk in chunks:
            tokens = _tokenize(str(chunk.get("text", "")))
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.lengths.append(len(tokens))
            document_frequencies.update(frequencies.keys())

        count = max(1, len(chunks))
        self.average_length = sum(self.lengths) / count
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, terms: list[str]) -> list[tuple[int, float]]:
        scored: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = self.lengths[index]
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / self.average_length
                )
                score += self.idf[term] * frequency * (BM25_K1 + 1) / denominator
            if score > 0:
                scored.append((index, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)


_index: _BM25Index | None = None


def _load_index() -> _BM25Index:
    global _index
    if _index is None:
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        _index = _BM25Index(chunks)
    return _index


def search_internal_docs(query: str, k: int = 5) -> list[dict[str, Any]]:
    terms = list(dict.fromkeys(_tokenize(query)))
    if not terms:
        return []

    try:
        index = _load_index()
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    total_idf = sum(index.idf.get(term, 0.0) for term in terms)
    if total_idf <= 0:
        return []

    minimum_matches = 2 if len(terms) >= 3 else 1
    results: list[dict[str, Any]] = []
    for chunk_index, score in index.scores(terms):
        frequencies = index.term_frequencies[chunk_index]
        matched = [term for term in terms if frequencies.get(term)]
        if len(matched) < minimum_matches:
            continue

        coverage = sum(index.idf.get(term, 0.0) for term in matched) / total_idf
        if coverage < MIN_COVERAGE:
            continue

        chunk = index.chunks[chunk_index]
        results.append(
            {
                "text": str(chunk.get("text", "")),
                "source": str(chunk.get("source", "")),
                "page": int(chunk.get("page", 0)),
                "score": round(score, 3),
            }
        )
        if len(results) == k:
            break
    return results


if __name__ == "__main__":
    build_index()
