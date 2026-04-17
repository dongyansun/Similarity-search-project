"""
config.py — Central configuration for paths and search hyperparameters.
Edit this file to point at your data, change the CLIP model, or tune top-k.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset root — the "archive (1)" folder containing styles.csv and images/
DATASET_DIR = os.path.join(BASE_DIR, "archive (1)")

# Folder where all product images live ({id}.jpg filenames)
IMAGE_DIR = os.path.join(DATASET_DIR, "images")

# Myntra styles.csv — columns: id, gender, masterCategory, subCategory,
#                              articleType, baseColour, season, year, usage,
#                              productDisplayName
STYLES_CSV_PATH = os.path.join(DATASET_DIR, "styles.csv")

# Pre-computed embedding cache (numpy .npz), rebuilt when catalog changes
EMBEDDING_CACHE_PATH = os.path.join(BASE_DIR, "data", "embeddings_cache.npz")

# ── CLIP model ───────────────────────────────────────────────────────────────

# Accuracy vs speed trade-off:
#   "ViT-B/32"  — fast, 512-dim,  good baseline          ← current
#   "ViT-B/16"  — same size, 2× more patches, noticeably better accuracy
#   "ViT-L/14"  — large model, 768-dim, best accuracy, ~3× slower to embed
#
# IMPORTANT: changing the model invalidates the embedding cache.
# Delete data/embeddings_cache.npz and run with --rebuild after switching.
CLIP_MODEL_NAME = "ViT-B/32"

# ── Search ───────────────────────────────────────────────────────────────────

# Default number of results returned by the search engine
TOP_K = 5

# ── Reranking ─────────────────────────────────────────────────────────────────
# When reranking is enabled, the final score is a weighted combination of:
#   final_score = CLIP_WEIGHT * clip_score + KEYWORD_WEIGHT * keyword_score
#
# keyword_score is based on how many query words appear in the product name
# and metadata fields (articleType, baseColour, gender, etc.).
#
# Tune these weights: higher KEYWORD_WEIGHT rewards exact label matches more.
RERANK_CLIP_WEIGHT    = 0.7
RERANK_KEYWORD_WEIGHT = 0.3

# How many candidates to pull from CLIP before reranking (should be >> TOP_K)
RERANK_CANDIDATE_POOL = 50

# ── Qdrant vector database ───────────────────────────────────────────────────
# Remote Qdrant instance hosted on AWS.
# Dashboard: http://16.144.140.219:6333/dashboard
QDRANT_URL        = "http://16.144.140.219:6333"
QDRANT_API_KEY    = None          # set to a string if the server requires auth
COLLECTION_NAME   = "fashion"     # primary collection (Myntra / scraped data)

# All collections to search — results are merged and re-sorted by score
QDRANT_COLLECTIONS = ["fashion", "deepfashion_items"]

# Vector dimension must match the CLIP model:
#   ViT-B/32 → 512,  ViT-B/16 → 512,  ViT-L/14 → 768
QDRANT_VECTOR_DIM = 512

# ── Subset for fast dev/testing ───────────────────────────────────────────────
# Set to None to embed the full 44k catalog (takes ~30 min on CPU).
# Set to an integer (e.g. 500) to only index the first N items — good for testing.
MAX_CATALOG_SIZE = 500
