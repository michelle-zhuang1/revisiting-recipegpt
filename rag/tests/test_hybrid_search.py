import chromadb
import pytest
from chromadb.utils import embedding_functions

from hybrid_search import EMBEDDING_MODEL, hybrid_search, rank_by_vibe

FIXTURE_DOCS = [
    {
        "id": "recipe-a::recipe",
        "text": "Chicken Artichoke Skillet\nchicken thighs, artichoke hearts\nSear chicken, add artichokes, ready in 20 minutes.",
        "metadata": {"title": "Chicken Artichoke Skillet", "chunk_type": "recipe",
                      "source_link": "https://example.com/a", "recipe_id": "recipe-a",
                      "categories": ""},
    },
    {
        "id": "recipe-b::recipe",
        "text": "Slow-Braised Chicken\nchicken thighs, wine, stock\nBraise for four hours until tender.",
        "metadata": {"title": "Slow-Braised Chicken", "chunk_type": "recipe",
                      "source_link": "https://example.com/b", "recipe_id": "recipe-b",
                      "categories": ""},
    },
    {
        "id": "recipe-c::recipe",
        "text": "Vanilla Cake\nflour, sugar, eggs\nBake for 45 minutes.",
        "metadata": {"title": "Vanilla Cake", "chunk_type": "recipe",
                      "source_link": "https://example.com/c", "recipe_id": "recipe-c",
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


def test_rank_by_vibe_restricts_to_candidates(fixture_db):
    hits = rank_by_vibe("chicken", ["recipe-a", "recipe-b"], n_results=5, db_dir=fixture_db)
    ids = {h["recipe_id"] for h in hits}
    assert ids == {"recipe-a", "recipe-b"}
    assert "recipe-c" not in ids


def test_rank_by_vibe_respects_n_results(fixture_db):
    hits = rank_by_vibe("chicken", ["recipe-a", "recipe-b"], n_results=1, db_dir=fixture_db)
    assert len(hits) == 1


def test_rank_by_vibe_empty_candidates_returns_empty(fixture_db):
    assert rank_by_vibe("chicken", [], db_dir=fixture_db) == []


def test_hybrid_search_combines_filter_and_ranking(tmp_path, fixture_db):
    corpus = [
        {"recipe_id": "recipe-a", "title": "Chicken Artichoke Skillet",
         "ingredients": ["2 chicken thighs", "1 cup artichoke hearts"], "source_link": ""},
        {"recipe_id": "recipe-b", "title": "Slow-Braised Chicken",
         "ingredients": ["2 chicken thighs", "1 cup wine"], "source_link": ""},
        {"recipe_id": "recipe-c", "title": "Vanilla Cake",
         "ingredients": ["2 cups flour", "1 cup sugar"], "source_link": ""},
    ]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(__import__("json").dumps(corpus))

    hits = hybrid_search(
        "quick", include=["chicken"], n_results=5,
        corpus_path=str(corpus_path), db_dir=fixture_db,
    )
    ids = {h["recipe_id"] for h in hits}
    assert ids == {"recipe-a", "recipe-b"}
