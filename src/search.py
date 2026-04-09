"""
search.py — Cosine similarity search engine with optional filtering and reranking.

Pipeline:
    1. Filter  — narrow the index to a subset matching hard constraints
                 (gender, masterCategory, subCategory, articleType, baseColour, season)
    2. Retrieve — cosine similarity over filtered embeddings → top-N candidates
    3. Rerank   — (optional) blend CLIP score with a keyword match score
                  final = CLIP_WEIGHT * clip_score + KEYWORD_WEIGHT * keyword_score

Because all embeddings are L2-normalised, cosine similarity == dot product.

To swap in ChromaDB/FAISS: replace the NumPy matmul in _retrieve() with a
collection.query() call; everything else (filtering, reranking) stays identical.
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

# Fields that the filter dict can target (case-insensitive exact match)
FILTERABLE_FIELDS = {
    "gender", "masterCategory", "subCategory", "articleType", "baseColour", "season",
}


class FashionSearchEngine:
    """
    In-memory similarity search engine with optional filtering and reranking.

    Quick start:
        engine = FashionSearchEngine()
        engine.build_index()

        # plain CLIP search
        results = engine.search("query.jpg")

        # with metadata filters
        results = engine.search("query.jpg", filters={"gender": "Men"})

        # with reranking against a text hint
        results = engine.search("query.jpg", rerank_query="navy blue check shirt")

        # text-only search
        results = engine.search_by_text("floral summer dress", filters={"gender": "Women"})
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None   # (N, D) float32, full index
        self.catalog: list[dict] | None = None       # N product dicts

    # ── Index management ─────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """Load (or build) embeddings + catalog. Call once before searching."""
        self.embeddings, self.catalog = build_catalog_embeddings(force_rebuild)

    def _check_ready(self):
        if self.embeddings is None or self.catalog is None:
            raise RuntimeError("Index not built. Call engine.build_index() first.")

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _apply_filters(self, filters: dict | None) -> tuple[np.ndarray, list[dict]]:
        """
        Return a (embeddings_subset, catalog_subset) restricted to items that
        satisfy ALL conditions in `filters`.

        filters example:
            {"gender": "Men", "masterCategory": "Apparel", "baseColour": "Navy Blue"}

        Matching is case-insensitive. Unknown filter keys raise a ValueError so
        typos are caught early rather than silently returning the full catalog.
        """
        if not filters:
            return self.embeddings, self.catalog

        # Validate keys upfront
        bad_keys = set(filters) - FILTERABLE_FIELDS
        if bad_keys:
            raise ValueError(
                f"Unknown filter field(s): {bad_keys}. "
                f"Valid fields: {FILTERABLE_FIELDS}"
            )

        # Build a boolean mask over the catalog
        mask = np.ones(len(self.catalog), dtype=bool)
        for field, value in filters.items():
            mask &= np.array(
                [entry.get(field, "").lower() == value.lower()
                 for entry in self.catalog],
                dtype=bool,
            )

        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No products match the filters: {filters}")

        filtered_embeddings = self.embeddings[indices]
        filtered_catalog = [self.catalog[i] for i in indices]
        print(f"[INFO] Filter applied: {len(indices)}/{len(self.catalog)} items match.")
        return filtered_embeddings, filtered_catalog

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_vec: np.ndarray,
        embeddings: np.ndarray,
        catalog: list[dict],
        top_n: int,
    ) -> list[dict]:
        """
        Core retrieval: dot-product similarity → top-n candidates (unsorted list).
        query_vec shape: (D,), embeddings shape: (N, D).
        """
        scores = embeddings @ query_vec                          # (N,)
        k = min(top_n, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]              # unordered top-k
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]    # sorted desc

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            entry = catalog[idx].copy()
            entry["clip_score"] = float(scores[idx])
            entry["score"] = float(scores[idx])     # may be updated by reranker
            entry["rank"] = rank
            results.append(entry)
        return results

    # ── Reranking ─────────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_score(entry: dict, query_words: list[str]) -> float:
        """
        Count how many query words appear in the product's searchable text fields.
        Returns a normalised score ∈ [0, 1].
        """
        if not query_words:
            return 0.0

        # Concatenate all text fields that carry semantic meaning
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
        Re-score a candidate list by blending CLIP score with keyword match score.

        final_score = CLIP_WEIGHT * clip_score + KEYWORD_WEIGHT * keyword_score

        Both terms are normalised before blending so they are on the same scale:
          - clip_score   : already ∈ [-1, 1], normalised to [0, 1] linearly
          - keyword_score: already ∈ [0, 1]
        """
        if not text_query or not candidates:
            return candidates

        # Tokenise query: lowercase, strip punctuation, remove 1-char tokens
        query_words = [
            w for w in re.split(r"\W+", text_query.lower()) if len(w) > 1
        ]

        # Normalise CLIP scores to [0, 1] over this candidate set
        clip_scores = np.array([c["clip_score"] for c in candidates])
        c_min, c_max = clip_scores.min(), clip_scores.max()
        if c_max > c_min:
            norm_clip = (clip_scores - c_min) / (c_max - c_min)
        else:
            norm_clip = np.ones(len(candidates))

        for i, entry in enumerate(candidates):
            kw = self._keyword_score(entry, query_words)
            entry["keyword_score"] = round(kw, 4)
            entry["score"] = round(
                RERANK_CLIP_WEIGHT * float(norm_clip[i])
                + RERANK_KEYWORD_WEIGHT * kw,
                4,
            )

        # Sort by blended score descending and reassign ranks
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
            query        : PIL.Image or filepath string to the query image.
            top_k        : number of final results to return.
            filters      : dict of exact-match metadata constraints, e.g.
                           {"gender": "Men", "baseColour": "Navy Blue"}
            rerank_query : optional text string used to rerank CLIP results,
                           e.g. "navy blue check shirt". Combines CLIP visual
                           similarity with keyword overlap in product metadata.

        Returns:
            List of result dicts sorted by descending score, each containing:
            rank, score, clip_score, keyword_score (if reranked), id, name,
            brand, gender, masterCategory, subCategory, articleType,
            baseColour, season, year, usage, image_path.
        """
        self._check_ready()

        query_vec = encode_query_image(query).astype(np.float32)

        # 1. Filter
        emb, cat = self._apply_filters(filters)

        # 2. Retrieve — pull a larger candidate pool when reranking
        n_candidates = RERANK_CANDIDATE_POOL if rerank_query else top_k
        candidates = self._retrieve(query_vec, emb, cat, top_n=n_candidates)

        # 3. Rerank (optional)
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
            top_k      : number of final results to return.
            filters    : same as search() — metadata hard constraints.
            rerank     : if True, the CLIP score is blended with keyword match
                         against the same text_query (recommended, default True).
        """
        self._check_ready()

        import clip
        import torch

        from src.embeddings import _load_model

        model, _, device = _load_model()
        tokens = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        query_vec = text_feat.cpu().numpy().astype(np.float32).squeeze()

        # 1. Filter
        emb, cat = self._apply_filters(filters)

        # 2. Retrieve
        n_candidates = RERANK_CANDIDATE_POOL if rerank else top_k
        candidates = self._retrieve(query_vec, emb, cat, top_n=n_candidates)

        # 3. Rerank using the same text query as keyword signal
        if rerank:
            candidates = self._rerank(candidates, text_query)

        return candidates[:top_k]
