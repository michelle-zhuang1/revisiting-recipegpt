"""Filter recipes by a set of on-hand ingredients ("what can I make with X and Y").

Deterministic AND-filter over parsed ingredient names — a different tool than
search.py's semantic search, for the case where "must contain both" is a hard
requirement rather than a similarity preference.

Usage:
    python pantry_search.py chicken artichoke
"""

import argparse
import json
import sys

from ingredient_matcher import matches_category, matches_query
from ingredient_parser import extract_ingredient_name

CORPUS_PATH = "data/corpus.json"


def _extracted_names(recipe: dict) -> list[str]:
    return [extract_ingredient_name(line) for line in recipe["ingredients"]]


def recipe_has_ingredient(recipe: dict, query_term: str) -> bool:
    return any(matches_query(name, query_term) for name in _extracted_names(recipe))


def filter_by_ingredients(recipes: list[dict], query_terms: list[str]) -> list[dict]:
    return [
        recipe
        for recipe in recipes
        if all(recipe_has_ingredient(recipe, term) for term in query_terms)
    ]


def filter_recipes(
    recipes: list[dict],
    include: list[str] = (),
    exclude: list[str] = (),
    exclude_categories: list[str] = (),
) -> list[dict]:
    result = []
    for recipe in recipes:
        names = _extracted_names(recipe)
        if not all(any(matches_query(n, t) for n in names) for t in include):
            continue
        if any(any(matches_query(n, t) for n in names) for t in exclude):
            continue
        if any(any(matches_category(n, c) for n in names) for c in exclude_categories):
            continue
        result.append(recipe)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ingredients", nargs="+", help="ingredients you have on hand")
    args = parser.parse_args()

    with open(CORPUS_PATH) as f:
        recipes = json.load(f)

    matches = filter_by_ingredients(recipes, args.ingredients)
    print(f"{len(matches)} recipe(s) with all of: {', '.join(args.ingredients)}")
    for r in matches:
        print(f"  - {r['title']}  ({r['source_link']})")


if __name__ == "__main__":
    main()
