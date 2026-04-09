"""
demo.py — Command-line interface for Fashion Image Similarity Search.

Usage examples:
    # Image search (first run builds the embedding cache)
    python demo.py --query "archive (1)/images/15970.jpg"

    # Image search filtered to Men's Apparel
    python demo.py --query "archive (1)/images/15970.jpg" --gender Men --category Apparel

    # Image search + reranking with a text hint
    python demo.py --query "archive (1)/images/15970.jpg" --rerank "navy blue check shirt"

    # Full combo: filter + rerank
    python demo.py --query "archive (1)/images/15970.jpg" \\
                   --gender Men --rerank "navy blue shirt"

    # Text search (reranking is on by default)
    python demo.py --text "red floral summer dress"

    # Text search filtered to Women
    python demo.py --text "red floral summer dress" --gender Women

    # Force re-embed after adding products or changing CLIP model
    python demo.py --query "archive (1)/images/15970.jpg" --rebuild
"""

import argparse
import sys

from src.search import FashionSearchEngine


def print_results(results: list[dict]) -> None:
    """Pretty-print search results to the terminal."""
    print(f"\n{'─' * 65}")
    print(f"  Top-{len(results)} Results")
    print(f"{'─' * 65}")
    for r in results:
        # Show blended score and, if reranked, break it down
        score_str = f"score: {r['score']:.4f}"
        if "keyword_score" in r:
            score_str += f"  (clip: {r['clip_score']:.4f}  kw: {r['keyword_score']:.4f})"
        print(f"\n  #{r['rank']}  {r['name']}")
        print(f"       {score_str}")
        print(f"       {r.get('masterCategory','')} > {r.get('subCategory','')} > {r.get('articleType','')}")
        print(f"       Colour: {r.get('baseColour','')}  |  Gender: {r.get('gender','')}  |  Season: {r.get('season','')}")
        print(f"       Image : {r['image_path']}")
    print(f"\n{'─' * 65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fashion Image Similarity Search Demo"
    )

    # ── Query mode (mutually exclusive) ──────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str,
                       help="Path to a query image file.")
    group.add_argument("--text", type=str,
                       help="Text description to search by (CLIP text encoder).")

    # ── General options ───────────────────────────────────────────────────────
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to return (default: 5).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-embed all catalog images, ignoring the cache.")

    # ── Filters ───────────────────────────────────────────────────────────────
    filter_group = parser.add_argument_group("Filters (exact match, case-insensitive)")
    filter_group.add_argument("--gender",   type=str, help="e.g. Men, Women, Boys, Girls, Unisex")
    filter_group.add_argument("--category", type=str, help="masterCategory, e.g. Apparel, Accessories, Footwear")
    filter_group.add_argument("--subcat",   type=str, help="subCategory, e.g. Topwear, Bottomwear, Shoes")
    filter_group.add_argument("--type",     type=str, help="articleType, e.g. Shirts, Jeans, Dresses")
    filter_group.add_argument("--colour",   type=str, help="baseColour, e.g. Navy Blue, Red, Black")
    filter_group.add_argument("--season",   type=str, help="e.g. Summer, Winter, Fall, Spring")

    # ── Reranking ─────────────────────────────────────────────────────────────
    parser.add_argument("--rerank", type=str, default=None, metavar="TEXT",
                        help="Text hint to rerank image-search results, "
                             "e.g. --rerank 'navy blue check shirt'")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable default reranking for text search.")

    args = parser.parse_args()

    # ── Build filter dict from CLI flags ─────────────────────────────────────
    filters = {}
    if args.gender:   filters["gender"]         = args.gender
    if args.category: filters["masterCategory"]  = args.category
    if args.subcat:   filters["subCategory"]     = args.subcat
    if args.type:     filters["articleType"]     = args.type
    if args.colour:   filters["baseColour"]      = args.colour
    if args.season:   filters["season"]          = args.season
    filters = filters or None

    # ── Build / load index ────────────────────────────────────────────────────
    engine = FashionSearchEngine()
    try:
        engine.build_index(force_rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Run search ────────────────────────────────────────────────────────────
    try:
        if args.query:
            print(f"[INFO] Image search: {args.query}")
            if args.rerank:
                print(f"[INFO] Reranking with: \"{args.rerank}\"")
            results = engine.search(
                args.query,
                top_k=args.top_k,
                filters=filters,
                rerank_query=args.rerank,
            )
        else:
            rerank = not args.no_rerank
            print(f"[INFO] Text search: \"{args.text}\"  (rerank={'on' if rerank else 'off'})")
            results = engine.search_by_text(
                args.text,
                top_k=args.top_k,
                filters=filters,
                rerank=rerank,
            )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print_results(results)


if __name__ == "__main__":
    main()
