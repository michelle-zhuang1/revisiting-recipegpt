from ingredient_parser import extract_ingredient_name


def test_strips_leading_quantity_and_unit():
    assert extract_ingredient_name("1 2/3 cups all-purpose flour") == "all-purpose flour"


def test_strips_bare_quantity_and_trailing_prep_clause():
    assert (
        extract_ingredient_name("2 boneless skinless chicken breasts, diced")
        == "boneless skinless chicken breasts"
    )


def test_strips_compound_unit_and_trailing_parenthetical():
    assert (
        extract_ingredient_name("4 1/2 teaspoons/14 grams active dry yeast (2 envelopes)")
        == "active dry yeast"
    )


def test_leaves_line_with_no_quantity_unchanged():
    assert extract_ingredient_name("Nonstick cooking spray") == "Nonstick cooking spray"


def test_strips_tbsp_abbreviation():
    assert (
        extract_ingredient_name("3 Tbsp sliced fresh chives, plus more for garnish")
        == "sliced fresh chives"
    )


def test_strips_tbsp_abbreviation_with_period():
    assert (
        extract_ingredient_name("1 Tbsp. thyme leaves, plus more for serving")
        == "thyme leaves"
    )


def test_strips_tsp_abbreviation():
    assert extract_ingredient_name("1 tsp garlic salt") == "garlic salt"
    assert extract_ingredient_name("1/4 tsp fine sea salt") == "fine sea salt"


def test_strips_bare_gram_abbreviation_with_no_space():
    assert extract_ingredient_name("100g Red Cabbage") == "Red Cabbage"
    assert extract_ingredient_name("20g Fresh Coriander") == "Fresh Coriander"
