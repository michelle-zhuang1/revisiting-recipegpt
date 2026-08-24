import chromadb
import pytest
from chromadb.utils import embedding_functions

from search import EMBEDDING_MODEL, search

FIXTURE_DOCS = [
    {
        "id": "recipe-a::ingredients",
        "text": "chicken thighs, artichoke hearts, lemon, garlic",
        "metadata": {"title": "Chicken Artichoke Skillet", "chunk_type": "ingredients",
                      "source_link": "https://example.com/a", "recipe_id": "recipe-a",
                      "categories": ""},
    },
    {
        "id": "recipe-a::recipe",
        "text": "Chicken Artichoke Skillet\nchicken thighs, artichoke hearts, lemon, garlic\nSear chicken, add artichokes.",
        "metadata": {"title": "Chicken Artichoke Skillet", "chunk_type": "recipe",
                      "source_link": "https://example.com/a", "recipe_id": "recipe-a",
                      "categories": ""},
    },
    {
        "id": "recipe-b::ingredients",
        "text": "flour, sugar, eggs, butter",
        "metadata": {"title": "Vanilla Cake", "chunk_type": "ingredients",
                      "source_link": "https://example.com/b", "recipe_id": "recipe-b",
                      "categories": ""},
    },
]


@pytest.fixture
def fixture_db(tmp_path):
    db_dir = str(tmp_path / "chroma_db")
    client = chromadb.PersistentClient(path=db_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(name="recipes", embedding_function=embedding_fn)
    collection.upsert(
        ids=[d["id"] for d in FIXTURE_DOCS],
        documents=[d["text"] for d in FIXTURE_DOCS],
        metadatas=[d["metadata"] for d in FIXTURE_DOCS],
    )
    return db_dir


def test_search_respects_n_results(fixture_db):
    hits = search("chicken", n_results=2, db_dir=fixture_db)
    assert len(hits) == 2


def test_search_filters_by_chunk_type(fixture_db):
    hits = search("chicken", chunk_type="ingredients", n_results=10, db_dir=fixture_db)
    assert len(hits) == 2
    assert all(h["chunk_type"] == "ingredients" for h in hits)


def test_search_without_chunk_type_returns_both_types(fixture_db):
    hits = search("chicken", n_results=10, db_dir=fixture_db)
    chunk_types = {h["chunk_type"] for h in hits}
    assert chunk_types == {"ingredients", "recipe"}


def test_search_hit_shape(fixture_db):
    hits = search("chicken", n_results=1, db_dir=fixture_db)
    hit = hits[0]
    assert set(hit.keys()) == {"title", "chunk_type", "distance", "source_link", "text"}
