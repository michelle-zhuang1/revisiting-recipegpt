from extract_mela import parse_recipe

# Modeled on a real exported recipe (Flour's Famous Banana Bread), trimmed to
# what's relevant for this test — a short fake image payload stands in for the
# real base64 JPEG data.
RAW_RECIPE = {
    "instructions": "Set oven to 350 degrees F.\nSift together the flour.",
    "notes": "",
    "text": "Cooking Channel serves up this recipe.",
    "categories": ["CopyCat", "Desserts"],
    "title": "Flour's Famous Banana Bread",
    "prepTime": "20min",
    "images": ["/9j/4AAQSkZJRgABAgAAAQABAAD..."],
    "id": "cookingchanneltv.com/recipes/flours-famous-banana-bread-2015076",
    "link": "https://www.cookingchanneltv.com/recipes/flours-famous-banana-bread-2015076",
    "ingredients": "1 2/3 cups all-purpose flour\n\n1 teaspoon baking soda\n  \n1/4 teaspoon salt",
    "cookTime": "1h",
    "nutrition": "**Calories** 306",
    "favorite": False,
    "wantToCook": False,
    "totalTime": "2h 20min",
    "yield": "1 loaf",
    "date": 711844007.577719,
}


def test_parse_recipe_drops_images():
    parsed = parse_recipe(RAW_RECIPE)
    assert "images" not in parsed


def test_parse_recipe_splits_ingredients_and_skips_blank_lines():
    parsed = parse_recipe(RAW_RECIPE)
    assert parsed["ingredients"] == [
        "1 2/3 cups all-purpose flour",
        "1 teaspoon baking soda",
        "1/4 teaspoon salt",
    ]


def test_parse_recipe_maps_camelcase_fields_to_snake_case():
    parsed = parse_recipe(RAW_RECIPE)
    assert parsed["prep_time"] == "20min"
    assert parsed["cook_time"] == "1h"
    assert parsed["total_time"] == "2h 20min"


def test_parse_recipe_preserves_core_fields():
    parsed = parse_recipe(RAW_RECIPE)
    assert parsed["title"] == "Flour's Famous Banana Bread"
    assert parsed["recipe_id"] == RAW_RECIPE["id"]
    assert parsed["source_link"] == RAW_RECIPE["link"]
    assert parsed["categories"] == ["CopyCat", "Desserts"]
