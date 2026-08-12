"""
Semantic search over the regulation corpus.

The whole search is one matrix multiplication.

Cosine similarity measures the ANGLE between two vectors, not the distance
between them. That matters because the length of an embedding reflects things
like how long the text is rather than what it means - a one-line column
description and a three-paragraph article about the same concept point the same
way but have very different magnitudes. Scale every vector to length 1 first and
the cosine formula collapses to a plain dot product, so:

    sims = matrix @ query          # (n_chunks,) - one score per chunk

At the size of a curated corpus (a few hundred chunks) comparing against every
vector is both exact and faster than building an approximate index. FAISS and
the vector databases exist to avoid exhaustive search over millions of vectors;
below that threshold they add an install dependency, a service to configure, and
a tuning surface, in exchange for a worse answer.

FALLBACK: if the embedding model is unavailable, the index degrades to token
overlap. That is meaningfully worse - keyword matching cannot connect
"cust_email" to "information relating to an identifiable person", which is the
entire point of doing this semantically. It exists so nothing breaks, not
because it is equivalent, and search results say which mode produced them.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Windows cannot create symlinks without Developer Mode, and the HuggingFace
# cache uses them by default. Without this the first download half-succeeds:
# it falls back to another source and works in memory, but leaves a broken
# cache that fails to load in the next process. Force plain copies instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from governance import config
from governance.policy.chunk import load_corpus
from governance.state import Citation, PolicyChunk

# Matches the model named in the design deck. BAAI/bge-small-en-v1.5 is a
# drop-in upgrade at similar size if retrieval quality ever becomes a bottleneck.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on",
    "with", "that", "this", "it", "as", "be", "by", "from", "at", "which",
    "any", "not", "no", "has", "have", "was", "were", "its", "their", "such",
}
TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower())
            if t not in STOPWORDS and len(t) > 2}


class PolicyIndex:
    def __init__(self, chunks: list[PolicyChunk],
                 vectors: np.ndarray | None = None, model=None):
        self.chunks = chunks
        self.vectors = vectors
        self._model = model
        self._model_tried = model is not None
        self.degraded_reason: str | None = None
        self._token_sets = [_tokens(f"{c.reference} {c.title} {c.text}")
                            for c in chunks]

    @property
    def backend(self) -> str:
        """
        What this index is ACTUALLY doing, not what it hoped to do.

        Having stored vectors is not sufficient: querying them needs the model
        too, and a broken model cache makes searches silently fall back to
        keyword matching. An index that quietly degrades and then reports it
        did not is precisely the failure a governance tool must not have, so
        this reflects the real state after the first search attempt.
        """
        if self.vectors is None or self.degraded_reason:
            return "keyword"
        return "embeddings" if not self._model_tried or self._model else "keyword"

    # ------------------------------------------------------------- building
    @classmethod
    def build(cls, chunks: list[PolicyChunk] | None = None,
              model_name: str = EMBED_MODEL, quiet: bool = False) -> "PolicyIndex":
        chunks = chunks if chunks is not None else load_corpus()
        if not chunks:
            return cls([])

        try:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name)
            texts = [f"{c.reference}. {c.title}. {c.text}" for c in chunks]
            vectors = np.asarray(list(model.embed(texts)), dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            return cls(chunks, vectors, model=model)
        except Exception as exc:
            index = cls(chunks)
            index.degraded_reason = f"{type(exc).__name__}: {exc}"
            if not quiet:
                print(f"  ! embeddings unavailable ({type(exc).__name__}); "
                      f"falling back to keyword matching")
            return index

    # -------------------------------------------------------------- search
    def _embed_query(self, query: str) -> np.ndarray | None:
        if self.vectors is None:
            return None
        if self._model is None and not self._model_tried:
            self._model_tried = True
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(EMBED_MODEL)
            except Exception as exc:
                # Recorded, not swallowed. The caller needs to know that these
                # results came from a weaker method.
                self.degraded_reason = f"{type(exc).__name__}: {exc}"
        if self._model is None:
            return None
        try:
            vector = np.asarray(next(iter(self._model.embed([query]))),
                                dtype=np.float32)
        except Exception as exc:
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            self._model = None
            return None
        return vector / np.linalg.norm(vector)

    def search(self, query: str, k: int = 3,
               min_score: float = 0.0) -> list[Citation]:
        if not self.chunks:
            return []

        vector = self._embed_query(query)
        if vector is not None:
            scores = self.vectors @ vector                # the entire search
        else:
            wanted = _tokens(query)
            scores = np.array([
                len(wanted & have) / np.sqrt(max(len(wanted), 1) * max(len(have), 1))
                for have in self._token_sets], dtype=np.float32)

        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            Citation(reference=self.chunks[i].reference,
                     title=self.chunks[i].title,
                     text=self.chunks[i].text[:400],
                     score=round(float(scores[i]), 4),
                     is_placeholder=self.chunks[i].is_placeholder)
            for i in top if scores[i] > min_score
        ]

    # ----------------------------------------------------------- persistence
    @property
    def uses_placeholder_text(self) -> bool:
        return any(c.is_placeholder for c in self.chunks)

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or (config.POLICY_DIR / "index")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2),
            encoding="utf-8")
        if self.vectors is not None:
            np.save(directory / "vectors.npy", self.vectors)
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "PolicyIndex | None":
        directory = directory or (config.POLICY_DIR / "index")
        chunk_file = directory / "chunks.json"
        if not chunk_file.exists():
            return None
        chunks = [PolicyChunk(**c) for c in
                  json.loads(chunk_file.read_text(encoding="utf-8"))]
        vector_file = directory / "vectors.npy"
        vectors = np.load(vector_file) if vector_file.exists() else None
        return cls(chunks, vectors)


def query_for(issue_type: str, data_class: str) -> str | None:
    """
    Build the retrieval query for a finding, in the regulation's vocabulary.

    Returns None for issue types that carry no regulatory question - a malformed
    email address is a data quality defect, and attaching a statute to it would
    be padding a report rather than informing one.

    Deliberately built from the ISSUE and the CLASSIFICATION, never from the
    column name. "cust_email" appears in no regulation ever written; searching
    for it returns whatever happens to be nearest, which is worse than nothing.
    """
    base = config.RETRIEVAL_QUERY.get(issue_type)
    if base is None:
        return None
    extra = config.DATA_CLASS_QUERY.get(data_class)
    return f"{base}; {extra}" if extra else base
