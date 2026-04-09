"""
data_loader.py — Loads the Myntra product catalog from styles.csv and provides
                 a helper to open individual product images as PIL Images.

Myntra CSV columns:
    id, gender, masterCategory, subCategory, articleType,
    baseColour, season, year, usage, productDisplayName

Images are stored as:  {IMAGE_DIR}/{id}.jpg
"""

import csv
import os
from typing import Optional

from PIL import Image

from src.config import IMAGE_DIR, MAX_CATALOG_SIZE, STYLES_CSV_PATH


def load_catalog() -> list[dict]:
    """
    Read styles.csv and return a list of product dicts.

    Each dict has these keys (matching the search result schema):
        id, name, brand, price, url, image_path,
        gender, masterCategory, subCategory, articleType, baseColour,
        season, year, usage

    Notes:
        - 'brand' is not in the Myntra CSV; we use masterCategory/articleType instead.
        - 'price' and 'url' are not in the dataset; they are set to placeholder values.
          Replace with real data if you scrape it later.
        - Only rows whose image file actually exists on disk are included.
        - MAX_CATALOG_SIZE (config.py) limits how many items are loaded — useful
          for fast testing without embedding the full 44k catalog.
    """
    if not os.path.exists(STYLES_CSV_PATH):
        raise FileNotFoundError(
            f"styles.csv not found: {STYLES_CSV_PATH}\n"
            "Make sure the 'archive (1)' folder is in the project root."
        )

    catalog = []
    skipped = 0

    with open(STYLES_CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if MAX_CATALOG_SIZE and len(catalog) >= MAX_CATALOG_SIZE:
                break

            product_id = row.get("id", "").strip()
            if not product_id:
                continue

            # Image path: {IMAGE_DIR}/{id}.jpg
            image_path = os.path.join(IMAGE_DIR, f"{product_id}.jpg")
            if not os.path.exists(image_path):
                skipped += 1
                continue

            catalog.append({
                # Core fields used by the search result schema
                "id":           product_id,
                "name":         row.get("productDisplayName", "Unknown").strip(),
                "brand":        row.get("masterCategory", "Unknown").strip(),
                "price":        0.0,          # not in dataset — fill in if scraped
                "url":          "",           # not in dataset — fill in if scraped
                "image_path":   image_path,

                # Myntra-specific fields — useful for Streamlit filters
                "gender":           row.get("gender", "").strip(),
                "masterCategory":   row.get("masterCategory", "").strip(),
                "subCategory":      row.get("subCategory", "").strip(),
                "articleType":      row.get("articleType", "").strip(),
                "baseColour":       row.get("baseColour", "").strip(),
                "season":           row.get("season", "").strip(),
                "year":             row.get("year", "").strip(),
                "usage":            row.get("usage", "").strip(),
            })

    if skipped:
        print(f"[INFO] Skipped {skipped} rows with missing images.")
    if not catalog:
        raise RuntimeError(
            "No products loaded. Check that IMAGE_DIR contains {id}.jpg files."
        )

    print(f"[INFO] Loaded {len(catalog)} products from catalog.")
    return catalog


def load_image(entry: dict) -> Optional[Image.Image]:
    """
    Open and return the PIL Image for a catalog entry.
    Returns None (with a warning) if the file is missing.
    """
    path = entry["image_path"]
    if not os.path.exists(path):
        print(f"[WARN] Image not found, skipping: {path}")
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not open image {path}: {e}")
        return None
