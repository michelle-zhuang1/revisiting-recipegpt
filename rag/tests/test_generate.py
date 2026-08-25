from generate import build_conditioned_prompt, build_unconditioned_prompt, recipe_to_text

RECIPE = {
    "title": "Banana Bread",
    "ingredients": ["2 ripe bananas", "1 cup flour"],
    "instructions": "Mash bananas. Mix in flour. Bake.",
}


def test_recipe_to_text_includes_title_ingredients_and_instructions():
    text = recipe_to_text(RECIPE)
    assert "Banana Bread" in text
    assert "2 ripe bananas" in text
    assert "1 cup flour" in text
    assert "Mash bananas. Mix in flour. Bake." in text


def test_build_unconditioned_prompt_is_just_the_request():
    assert build_unconditioned_prompt("something quick with chicken") == "something quick with chicken\n"


def test_build_conditioned_prompt_includes_examples_and_request():
    prompt = build_conditioned_prompt([RECIPE], "something quick with chicken")
    assert "Banana Bread" in prompt
    assert "something quick with chicken" in prompt
    assert prompt.index("Banana Bread") < prompt.index("something quick with chicken")
