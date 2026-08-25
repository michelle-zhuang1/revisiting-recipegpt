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


def test_does_not_strip_trailing_clause_that_is_part_of_the_ingredient():
    assert extract_ingredient_name("1/4 cup raw, unsalted cashews") == "raw, unsalted cashews"


def test_strips_multi_word_prep_clause():
    assert extract_ingredient_name("2 medium shallots, chopped finely") == "medium shallots"
    assert extract_ingredient_name("2 scallions, sliced thinly") == "scallions"


def test_strips_dangling_trailing_comma_left_by_paren_removal():
    assert extract_ingredient_name("250 g minced pork, (9oz )") == "minced pork"


def test_strips_unicode_fraction_quantities():
    assert extract_ingredient_name("¼ teaspoon salt") == "salt"
    assert extract_ingredient_name("¾ cup all-purpose flour") == "all-purpose flour"
    assert extract_ingredient_name("1½ teaspoons ground cinnamon") == "ground cinnamon"


def test_strips_cut_into_measurement_clause():
    assert (
        extract_ingredient_name("3 ounces Spanish-style chorizo, cut into ½-inch pieces")
        == "Spanish-style chorizo"
    )
    assert (
        extract_ingredient_name("1¼ pounds rhubarb, cut into ½-inch pieces")
        == "rhubarb"
    )


def test_strips_cut_into_shape_words():
    assert (
        extract_ingredient_name("1/2 cup/113 grams unsalted butter, cut into cubes")
        == "unsalted butter"
    )


def test_strips_multiple_trailing_clauses_iteratively():
    assert extract_ingredient_name("1 lemon, cut into wedges, to serve") == "lemon"


def test_strips_leading_qualifier_word():
    assert extract_ingredient_name("scant ¼ teaspoon salt") == "salt"


def test_strips_spelled_out_leading_number():
    assert (
        extract_ingredient_name("one 14-ounce can fire-roasted crushed tomatoes")
        == "can fire-roasted crushed tomatoes"
    )
    assert (
        extract_ingredient_name("two 14-ounce cans chickpeas, rinsed and drained")
        == "cans chickpeas"
    )


def test_strips_square_bracket_metric_conversion():
    assert (
        extract_ingredient_name("½ cup [75 ml] extra-virgin olive oil")
        == "extra-virgin olive oil"
    )
    assert extract_ingredient_name("1 cup [142 g] all-purpose flour") == "all-purpose flour"


def test_strips_paren_revealed_by_prior_clause_strip():
    assert (
        extract_ingredient_name("Sliced baguette (optional), for serving")
        == "Sliced baguette"
    )


def test_strips_double_wrapped_parenthetical():
    assert (
        extract_ingredient_name("500 g sweet rice flour ((1.1 pounds))")
        == "sweet rice flour"
    )


def test_strips_oz_and_lb_abbreviations():
    assert extract_ingredient_name("1 15-oz. can tomato sauce") == "can tomato sauce"
    assert (
        extract_ingredient_name("1 lb. low-moisture mozzarella, coarsely grated")
        == "low-moisture mozzarella"
    )


def test_strips_en_dash_range_quantity():
    assert extract_ingredient_name("2–3 cloves garlic") == "garlic"
    assert (
        extract_ingredient_name("1–2 tablespoons harissa paste (2 tablespoons would be medium spicy)")
        == "harissa paste"
    )
