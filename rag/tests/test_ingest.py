from ingest import build_chunks

RECIPE = {
    "recipe_id": "example.com/banana-bread",
    "title": "Banana Bread",
    "ingredients": ["2 ripe bananas", "1 cup flour"],
    "instructions": "Mash bananas. Mix in flour. Bake.",
    "notes": "",
    "categories": ["Desserts"],
    "yield": "1 loaf",
    "prep_time": "20min",
    "cook_time": "1h",
    "total_time": "1h 20min",
    "source_link": "https://example.com/banana-bread",
    "favorite": False,
    "want_to_cook": False,
}


def test_build_chunks_returns_one_ingredients_and_one_recipe_chunk():
    chunks = build_chunks(RECIPE)
    chunk_types = {c["metadata"]["chunk_type"] for c in chunks}
    assert chunk_types == {"ingredients", "recipe"}
    assert len(chunks) == 2


def test_ingredients_chunk_contains_only_ingredients():
    chunks = build_chunks(RECIPE)
    ingredients_chunk = next(c for c in chunks if c["metadata"]["chunk_type"] == "ingredients")
    assert "2 ripe bananas" in ingredients_chunk["text"]
    assert "1 cup flour" in ingredients_chunk["text"]
    assert "Bake" not in ingredients_chunk["text"]
    assert "Banana Bread" not in ingredients_chunk["text"]


def test_recipe_chunk_contains_title_time_ingredients_and_instructions():
    chunks = build_chunks(RECIPE)
    recipe_chunk = next(c for c in chunks if c["metadata"]["chunk_type"] == "recipe")
    text = recipe_chunk["text"]
    assert "Banana Bread" in text
    assert "20min" in text
    assert "1h 20min" in text
    assert "2 ripe bananas" in text
    assert "Mash bananas" in text


def test_chunk_ids_are_namespaced_by_recipe_id_and_type():
    chunks = build_chunks(RECIPE)
    ids = {c["id"] for c in chunks}
    assert ids == {
        "example.com/banana-bread::ingredients",
        "example.com/banana-bread::recipe",
    }
