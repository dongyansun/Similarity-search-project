"""
search.py — Fashion search engine backed by a remote Qdrant vector database.

Pipeline:
    1. build_index() — embed catalog images with CLIP, upload vectors + metadata to Qdrant
    2. search()      — encode query image → Qdrant nearest-neighbour search → rerank
    3. search_by_text() — encode text query → same Qdrant search → rerank

Qdrant handles filtering and ANN retrieval; Python handles reranking.
"""

import re

import numpy as np

from src.config import (
    RERANK_CANDIDATE_POOL,
    RERANK_CLIP_WEIGHT,
    RERANK_KEYWORD_WEIGHT,
    TOP_K,
)
from src.embeddings import build_catalog_embeddings, encode_query_image
from src.vector_store import (
    ensure_collection,
    get_client,
    search_vectors,
    upload_embeddings,
)


class FashionSearchEngine:
    """
    Visual search engine backed by a remote Qdrant instance.

    Quick start:
        engine = FashionSearchEngine()
        engine.build_index()                               # uploads to Qdrant once

        results = engine.search("query.jpg")
        results = engine.search("query.jpg", filters={"gender": "Men"})
        results = engine.search("query.jpg", rerank_query="navy blue check shirt")
        results = engine.search_by_text("floral summer dress", filters={"gender": "Women"})
    """

    def __init__(self):
        self.client = get_client()

    # ── Index management ─────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Verify connection to the remote Qdrant collections.
        Data is managed by the online pipeline — no local upload.
        """
        from src.config import QDRANT_COLLECTIONS
        for col in QDRANT_COLLECTIONS:
            count = self.client.count(collection_name=col).count
            print(f"[INFO] '{col}': {count} points available.")

    # ── Reranking (pure Python — unchanged from local version) ───────────────

    @staticmethod
    def _keyword_score(entry: dict, query_words: list[str]) -> float:
        if not query_words:
            return 0.0
        searchable = " ".join([
            entry.get("name", ""),
            entry.get("articleType", ""),
            entry.get("baseColour", ""),
            entry.get("gender", ""),
            entry.get("masterCategory", ""),
            entry.get("subCategory", ""),
            entry.get("season", ""),
            entry.get("usage", ""),
        ]).lower()
        hits = sum(1 for w in query_words if w in searchable)
        return hits / len(query_words)

    def _rerank(self, candidates: list[dict], text_query: str) -> list[dict]:
        """
        Blend CLIP score with keyword match score.
        final = CLIP_WEIGHT * norm(clip_score) + KEYWORD_WEIGHT * keyword_score
        """
        if not text_query or not candidates:
            return candidates

        query_words = [w for w in re.split(r"\W+", text_query.lower()) if len(w) > 1]

        clip_scores = np.array([c["clip_score"] for c in candidates])
        c_min, c_max = clip_scores.min(), clip_scores.max()
        norm_clip = (clip_scores - c_min) / (c_max - c_min) if c_max > c_min else np.ones(len(candidates))

        for i, entry in enumerate(candidates):
            kw = self._keyword_score(entry, query_words)
            entry["keyword_score"] = round(kw, 4)
            entry["score"] = round(
                RERANK_CLIP_WEIGHT * float(norm_clip[i]) + RERANK_KEYWORD_WEIGHT * kw, 4
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(candidates):
            entry["rank"] = i + 1

        return candidates

    # ── Public search API ─────────────────────────────────────────────────────

    def search(
        self,
        query,
        top_k: int = TOP_K,
        filters: dict | None = None,
        rerank_query: str | None = None,
    ) -> list[dict]:
        """
        Find the top-k most visually similar products to a query image.

        Args:
            query        : PIL.Image or filepath string.
            top_k        : number of final results.
            filters      : e.g. {"gender": "Men", "baseColour": "Navy Blue"}
            rerank_query : optional text to blend with CLIP score for reranking.
        """
        query_vec = encode_query_image(query).astype(np.float32)

        n_candidates = RERANK_CANDIDATE_POOL if rerank_query else top_k
        candidates = search_vectors(self.client, query_vec, n_candidates, filters)

        if rerank_query:
            candidates = self._rerank(candidates, rerank_query)

        return candidates[:top_k]

    def search_by_text(
        self,
        text_query: str,
        top_k: int = TOP_K,
        filters: dict | None = None,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Find products matching a natural language description via CLIP text encoder.

        Args:
            text_query : e.g. "red floral summer dress"
            top_k      : number of final results.
            filters    : metadata hard constraints.
            rerank     : blend CLIP score with keyword match (recommended).
        """
        import clip
        import torch
        from src.embeddings import _load_model

        model, _, device = _load_model()
        tokens = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        query_vec = text_feat.cpu().numpy().astype(np.float32).squeeze()

        n_candidates = RERANK_CANDIDATE_POOL if rerank else top_k
        candidates = search_vectors(self.client, query_vec, n_candidates, filters)

        if rerank:
            candidates = self._rerank(candidates, text_query)

        return candidates[:top_k]
