from pantry_search import filter_by_ingredients

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
]


def test_filter_requires_all_query_terms_present():
    results = filter_by_ingredients(RECIPES, ["chicken", "artichoke"])
    assert [r["recipe_id"] for r in results] == ["a"]


def test_filter_excludes_recipes_missing_any_term():
    results = filter_by_ingredients(RECIPES, ["chicken"])
    assert {r["recipe_id"] for r in results} == {"a", "c"}


def test_filter_returns_nothing_when_no_recipe_has_all_terms():
    results = filter_by_ingredients(RECIPES, ["chicken", "flour"])
    assert results == []
