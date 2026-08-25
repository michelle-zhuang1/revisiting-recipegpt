from pantry_search import filter_by_ingredients, filter_recipes

RECIPES = [
    {
        "recipe_id": "a",
        "title": "Chicken Artichoke Skillet",
        "ingredients": ["2 boneless skinless chicken breasts, diced", "1 cup artichoke hearts"],
    },
    {
        "recipe_id": "b",
        "title": "Vanilla Cake",
        "ingredients": ["2 cups flour", "1 cup sugar"],
    },
    {
        "recipe_id": "c",
        "title": "Chicken Soup",
        "ingredients": ["1 lb chicken thighs", "2 carrots"],
    },
    {
        "recipe_id": "d",
        "title": "Dairy-Free Chicken Curry",
        "ingredients": ["1 lb chicken thighs", "1 cup coconut milk", "2 tbsp curry powder"],
    },
    {
        "recipe_id": "e",
        "title": "Creamy Chicken Alfredo",
        "ingredients": ["1 lb chicken thighs", "1 cup heavy cream", "1/2 cup parmesan cheese"],
    },
]


def test_filter_requires_all_query_terms_present():
    results = filter_by_ingredients(RECIPES, ["chicken", "artichoke"])
    assert [r["recipe_id"] for r in results] == ["a"]


def test_filter_excludes_recipes_missing_any_term():
    results = filter_by_ingredients(RECIPES, ["chicken"])
    assert {r["recipe_id"] for r in results} == {"a", "c", "d", "e"}


def test_filter_returns_nothing_when_no_recipe_has_all_terms():
    results = filter_by_ingredients(RECIPES, ["chicken", "flour"])
    assert results == []


def test_filter_recipes_excludes_matching_terms():
    results = filter_recipes(RECIPES, include=["chicken"], exclude=["artichoke"])
    assert {r["recipe_id"] for r in results} == {"c", "d", "e"}


def test_filter_recipes_excludes_by_category():
    results = filter_recipes(RECIPES, include=["chicken"], exclude_categories=["dairy"])
    assert {r["recipe_id"] for r in results} == {"a", "c", "d"}
