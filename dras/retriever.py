"""
Two out-of-the-box retrieval backends, switchable via cfg.retriever_backend.

    make_retriever(cfg) → DenseRetriever | BM25Retriever

Both implement the same four-method interface so loop.py stays backend-agnostic.
Contrastive training is scaffolded via the train() stub but not executed in MVP.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from dras.config import Config


class BaseRetriever(ABC):
    @abstractmethod
    def add(self, text: str, metadata: dict | None = None) -> str:
        """Store a premise. Returns an opaque id."""

    @abstractmethod
    def query(self, text: str, k: int) -> list[str]:
        """Return up to k premises most relevant to text."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored premises (called at the start of each loop run)."""

    def query_multi(self, goal: str, last_step: str | None, k: int) -> list[str]:
        """Dual-query retrieval: mix goal-similar and last-step-similar premises.

        Without this, the retriever always fetches goal-proximate premises, which
        are the END of the chain — exactly what can't be proved yet. Mixing in
        last-step-similar premises lets the loop follow the chain FORWARD from
        whatever intermediate premise it most recently derived.
        """
        if last_step is None:
            return self.query(goal, k)
        # Bias toward last_step-proximate items (k_goal=1) so that in long chains
        # the "bridge" rule connecting the latest derived fact to the next step
        # is retrieved alongside that fact, not crowded out by far-end goal rules.
        k_goal = 1
        k_step = k - k_goal
        goal_hits = self.query(goal, k_goal)
        step_hits = self.query(last_step, k_step + k_goal)  # over-fetch, then dedup
        seen = set(goal_hits)
        combined = list(goal_hits)
        for t in step_hits:
            if t not in seen:
                seen.add(t)
                combined.append(t)
            if len(combined) >= k:
                break
        return combined[:k]

    def train(self, *args, **kwargs) -> None:
        """Contrastive retriever fine-tuning — not implemented in MVP."""
        raise NotImplementedError("Contrastive retriever training is not yet implemented.")


# ---------------------------------------------------------------------------
# Dense backend: ChromaDB + sentence-transformers
# ---------------------------------------------------------------------------

class DenseRetriever(BaseRetriever):
    def __init__(self, cfg: Config) -> None:
        from sentence_transformers import SentenceTransformer
        import chromadb

        model_name = cfg.retriever_model_path if cfg.retriever_model_path else cfg.embed_model
        self._embed = SentenceTransformer(model_name)
        self._client = chromadb.Client()
        self._col_name = "dras"
        self._col = self._client.get_or_create_collection(self._col_name)

    def add(self, text: str, metadata: dict | None = None) -> str:
        pid = str(uuid.uuid4())
        embedding = self._embed.encode(text, normalize_embeddings=True).tolist()
        self._col.add(ids=[pid], embeddings=[embedding], documents=[text],
                      metadatas=[metadata if metadata else None])
        return pid

    def query(self, text: str, k: int) -> list[str]:
        if self._col.count() == 0:
            return []
        n = min(k, self._col.count())
        embedding = self._embed.encode(text, normalize_embeddings=True).tolist()
        results = self._col.query(query_embeddings=[embedding], n_results=n)
        return results["documents"][0]

    def reset(self) -> None:
        self._client.delete_collection(self._col_name)
        self._col = self._client.get_or_create_collection(self._col_name)


# ---------------------------------------------------------------------------
# Sparse backend: BM25 (rank_bm25)
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    def __init__(self) -> None:
        self._corpus: list[str] = []
        self._bm25 = None

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def add(self, text: str, metadata: dict | None = None) -> str:
        self._corpus.append(text)
        self._rebuild()
        return str(len(self._corpus) - 1)

    def query(self, text: str, k: int) -> list[str]:
        if not self._corpus or self._bm25 is None:
            return []
        tokens = text.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._corpus[i] for i in top_indices]

    def reset(self) -> None:
        self._corpus = []
        self._bm25 = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_retriever(cfg: Config) -> BaseRetriever:
    if cfg.retriever_backend == "dense":
        return DenseRetriever(cfg)
    if cfg.retriever_backend == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unknown retriever_backend: {cfg.retriever_backend!r}. Choose 'dense' or 'bm25'.")
