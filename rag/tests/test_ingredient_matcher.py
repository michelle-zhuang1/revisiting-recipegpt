from ingredient_matcher import matches_query


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
