from ingredient_matcher import matches_category, matches_query


def test_matches_when_query_word_present():
    assert matches_query("chicken breast", "chicken") is True


def test_does_not_match_unrelated_ingredient():
    assert matches_query("vanilla cake", "chicken") is False


def test_word_boundary_avoids_substring_false_positive():
    assert matches_query("eggplant", "egg") is False


def test_corn_does_not_match_corn_starch():
    assert matches_query("corn starch", "corn") is False


def test_corn_still_matches_bare_corn():
    assert matches_query("corn", "corn") is True


def test_corn_still_matches_uncurated_compound():
    assert matches_query("creamed corn", "corn") is True


def test_chicken_does_not_match_chicken_powder():
    assert matches_query("chicken powder", "chicken") is False


def test_chicken_still_matches_real_chicken():
    assert matches_query("chicken thighs", "chicken") is True


def test_milk_does_not_match_plant_milks():
    assert matches_query("coconut milk", "milk") is False
    assert matches_query("almond milk", "milk") is False
    assert matches_query("oat milk", "milk") is False
    assert matches_query("soy milk", "milk") is False


def test_milk_still_matches_dairy_milk():
    assert matches_query("whole milk", "milk") is True


def test_butter_does_not_match_plant_butters():
    assert matches_query("peanut butter", "butter") is False
    assert matches_query("almond butter", "butter") is False


def test_butter_still_matches_dairy_butter():
    assert matches_query("unsalted butter", "butter") is True


def test_cream_does_not_match_cream_of_tartar():
    assert matches_query("cream of tartar", "cream") is False


def test_dairy_category_matches_various_dairy_ingredients():
    assert matches_category("whole milk", "dairy") is True
    assert matches_category("unsalted butter", "dairy") is True
    assert matches_category("parmesan cheese", "dairy") is True


def test_dairy_category_excludes_plant_lookalikes():
    assert matches_category("coconut milk", "dairy") is False
    assert matches_category("peanut butter", "dairy") is False


def test_dairy_category_does_not_match_unrelated_ingredient():
    assert matches_category("chicken thighs", "dairy") is False
