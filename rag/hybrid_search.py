"""Structured filter, then semantic rank — retrieval for queries that mix hard
ingredient constraints ("chicken", "no dairy") with a style/vibe preference
("something quick").

Usage:
    python hybrid_search.py "something quick" --include chicken --exclude-categories dairy
"""

import argparse
import json

import chromadb
from chromadb.utils import embedding_functions

from ingredient_matcher import CATEGORIES
from pantry_search import CORPUS_PATH, filter_recipes

DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def rank_by_vibe(
    vibe: str,
    candidate_recipe_ids: list[str],
    n_results: int = 5,
    db_dir: str = DB_DIR,
) -> list[dict]:
    if not candidate_recipe_ids:
        return []

    client = chromadb.PersistentClient(path=db_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name="recipes", embedding_function=embedding_fn)

    results = collection.query(
        query_texts=[vibe],
        n_results=min(n_results, len(candidate_recipe_ids)),
        where={
            "$and": [
                {"chunk_type": "recipe"},
                {"recipe_id": {"$in": candidate_recipe_ids}},
            ]
        },
    )

    hits = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        hits.append({"title": meta["title"], "recipe_id": meta["recipe_id"],
                      "distance": dist, "source_link": meta["source_link"]})
    return hits


def hybrid_search(
    vibe: str | None,
    include: list[str] = (),
    exclude: list[str] = (),
    exclude_categories: list[str] = (),
    n_results: int = 5,
    corpus_path: str = CORPUS_PATH,
    db_dir: str = DB_DIR,
) -> list[dict]:
    with open(corpus_path) as f:
        recipes = json.load(f)

    candidates = filter_recipes(
        recipes, include=include, exclude=exclude, exclude_categories=exclude_categories
    )

    if vibe is None:
        return candidates[:n_results]

    candidate_ids = [r["recipe_id"] for r in candidates]
    return rank_by_vibe(vibe, candidate_ids, n_results=n_results, db_dir=db_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vibe", nargs="?", default=None, help="style/quality preference")
    parser.add_argument("--include", nargs="*", default=[], help="ingredients that must be present")
    parser.add_argument("--exclude", nargs="*", default=[], help="ingredients that must not be present")
    parser.add_argument("--exclude-categories", nargs="*", default=[],
                         choices=list(CATEGORIES), help="ingredient categories that must not be present")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    hits = hybrid_search(
        args.vibe, include=args.include, exclude=args.exclude,
        exclude_categories=args.exclude_categories, n_results=args.n,
    )
    print(f"{len(hits)} result(s)")
    for i, hit in enumerate(hits, 1):
        title = hit.get("title", "?")
        link = hit.get("source_link", "")
        print(f"  {i}. {title}  ({link})")


if __name__ == "__main__":
    main()
