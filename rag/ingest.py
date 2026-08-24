"""Embed the recipe corpus into a local Chroma vector store.

Two chunks per recipe:
  - "ingredients": just the ingredient list — for pantry-style queries
    ("what can I make with chicken and artichoke").
  - "recipe": title + ingredients + instructions combined — for general
    queries ("something quick and easy").

Both chunk types share one collection, tagged with recipe_id/chunk_type
metadata so results can be traced back to a full recipe.

Usage:
    python ingest.py [corpus.json] [chroma_db_dir]
"""

import json
import sys

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_CORPUS = "data/corpus.json"
DEFAULT_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_chunks(recipe: dict) -> list[dict]:
    ingredients_text = "\n".join(recipe["ingredients"])
    time_text = (
        f"Prep time: {recipe['prep_time']}. Cook time: {recipe['cook_time']}. "
        f"Total time: {recipe['total_time']}. Yield: {recipe['yield']}."
    )
    recipe_text = "\n\n".join(
        [
            recipe["title"],
            time_text,
            ingredients_text,
            recipe["instructions"],
        ]
    )

    base_metadata = {
        "recipe_id": recipe["recipe_id"],
        "title": recipe["title"],
        "categories": ", ".join(recipe["categories"]),
        "source_link": recipe["source_link"],
    }

    return [
        {
            "id": f"{recipe['recipe_id']}::ingredients",
            "text": ingredients_text,
            "metadata": {**base_metadata, "chunk_type": "ingredients"},
        },
        {
            "id": f"{recipe['recipe_id']}::recipe",
            "text": recipe_text,
            "metadata": {**base_metadata, "chunk_type": "recipe"},
        },
    ]


def main(corpus_path: str, db_dir: str) -> None:
    with open(corpus_path) as f:
        recipes = json.load(f)

    chunks = [chunk for recipe in recipes for chunk in build_chunks(recipe)]
    print(f"{len(recipes)} recipes -> {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=db_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name="recipes", embedding_function=embedding_fn
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"done -> {db_dir} (collection: recipes, {collection.count()} chunks)")


if __name__ == "__main__":
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    db_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DB_DIR
    main(corpus_path, db_dir)
