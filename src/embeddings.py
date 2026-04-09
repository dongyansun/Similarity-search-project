"""
embeddings.py — CLIP-based image embedding builder with a disk cache.

Flow:
  1. build_catalog_embeddings()  — encode every product image once, save to .npz
  2. load_catalog_embeddings()   — fast reload from cache on subsequent runs
  3. encode_query_image()        — encode a single query image at search time

All embeddings are L2-normalised so cosine similarity == dot product,
which makes the search step a single matrix multiply.
"""

import os

import clip
import numpy as np
import torch
from PIL import Image

from src.config import CLIP_MODEL_NAME, EMBEDDING_CACHE_PATH
from src.data_loader import load_catalog, load_image


# ── Device selection ─────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon MPS is useful for local dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model singleton (loaded once per process) ────────────────────────────────

_model = None
_preprocess = None
_device = None


def _load_model():
    global _model, _preprocess, _device
    if _model is None:
        _device = _get_device()
        print(f"[INFO] Loading CLIP model '{CLIP_MODEL_NAME}' on {_device} …")
        _model, _preprocess = clip.load(CLIP_MODEL_NAME, device=_device)
        _model.eval()
    return _model, _preprocess, _device


# ── Core embedding functions ─────────────────────────────────────────────────

def _embed_image(pil_image: Image.Image) -> np.ndarray:
    """
    Encode a single PIL image → L2-normalised float32 numpy vector.
    Shape: (embed_dim,)  e.g. 512 for ViT-B/32.
    """
    model, preprocess, device = _load_model()
    tensor = preprocess(pil_image).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        feat = model.encode_image(tensor)                   # (1, D)
    feat = feat / feat.norm(dim=-1, keepdim=True)           # L2 normalise
    return feat.cpu().numpy().astype(np.float32).squeeze()  # (D,)


def encode_query_image(image_or_path) -> np.ndarray:
    """
    Encode a query image at search time.

    Args:
        image_or_path: a PIL.Image.Image  OR  a filepath string/Path.
    Returns:
        L2-normalised numpy vector of shape (embed_dim,).
    """
    if isinstance(image_or_path, (str, os.PathLike)):
        pil = Image.open(image_or_path).convert("RGB")
    else:
        pil = image_or_path.convert("RGB")
    return _embed_image(pil)


# ── Catalog embedding builder & cache ────────────────────────────────────────

def build_catalog_embeddings(force_rebuild: bool = False) -> tuple[np.ndarray, list[dict]]:
    """
    Embed every image in the product catalog and cache results to disk.

    Args:
        force_rebuild: if True, ignore existing cache and re-embed everything.

    Returns:
        embeddings : float32 array of shape (N, embed_dim)
        catalog    : list of N product dicts (same order as embeddings)
    """
    if not force_rebuild and os.path.exists(EMBEDDING_CACHE_PATH):
        return load_catalog_embeddings()

    catalog = load_catalog()
    vectors = []
    valid_catalog = []

    print(f"[INFO] Embedding {len(catalog)} products …")
    for i, entry in enumerate(catalog):
        pil = load_image(entry)
        if pil is None:
            # Image file missing — skip this product
            continue
        vec = _embed_image(pil)
        vectors.append(vec)
        valid_catalog.append(entry)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(catalog)} done")

    if not vectors:
        raise RuntimeError("No images could be embedded. Check your image_path values.")

    embeddings = np.stack(vectors, axis=0)  # (N, D)

    # Save cache: embeddings array + product ids (to reconstruct catalog order)
    ids = np.array([e["id"] for e in valid_catalog])
    np.savez(EMBEDDING_CACHE_PATH, embeddings=embeddings, ids=ids)
    print(f"[INFO] Cache saved → {EMBEDDING_CACHE_PATH}")

    return embeddings, valid_catalog


def load_catalog_embeddings() -> tuple[np.ndarray, list[dict]]:
    """
    Load pre-computed embeddings from disk and re-align with current metadata.
    If the catalog has changed (new/removed products), call build_catalog_embeddings(force_rebuild=True).
    """
    if not os.path.exists(EMBEDDING_CACHE_PATH):
        raise FileNotFoundError(
            "Embedding cache not found. Run build_catalog_embeddings() first."
        )

    data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float32)  # (N, D)
    cached_ids = set(data["ids"].tolist())

    # Filter catalog to only include products that were successfully embedded
    catalog = load_catalog()
    aligned_catalog = [e for e in catalog if e["id"] in cached_ids]

    # Preserve the order stored in the cache
    id_to_entry = {e["id"]: e for e in aligned_catalog}
    ordered_ids = data["ids"].tolist()
    aligned_catalog = [id_to_entry[i] for i in ordered_ids if i in id_to_entry]

    print(f"[INFO] Loaded {len(aligned_catalog)} embeddings from cache.")
    return embeddings, aligned_catalog
