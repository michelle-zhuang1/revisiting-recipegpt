"""Semantic search over the recipe corpus.

Usage:
    python search.py "chicken and artichoke, something quick"
    python search.py "something quick and easy" --type recipe
"""

import argparse

import chromadb
from chromadb.utils import embedding_functions

DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def search(
    query: str,
    chunk_type: str | None = None,
    n_results: int = 5,
    db_dir: str = DB_DIR,
) -> list[dict]:
    client = chromadb.PersistentClient(path=db_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(name="recipes", embedding_function=embedding_fn)

    where = {"chunk_type": chunk_type} if chunk_type else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"title": meta["title"], "chunk_type": meta["chunk_type"],
                      "distance": dist, "source_link": meta["source_link"], "text": doc})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--type", choices=["ingredients", "recipe"], default=None,
                         help="restrict to one chunk type (default: search both)")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    hits = search(args.query, chunk_type=args.type, n_results=args.n)
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['title']}  [{hit['chunk_type']}, dist={hit['distance']:.3f}]")
        print(f"   {hit['source_link']}")


if __name__ == "__main__":
    main()
