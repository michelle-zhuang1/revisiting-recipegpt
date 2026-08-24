"""Decide whether an extracted ingredient name satisfies a query term.

Plain word-boundary matching gets most cases right, but some compound
ingredient names share a word with a query term while being a genuinely
different food (querying "corn" shouldn't match "corn starch"). Those known
cases are hand-curated in COMPOUND_EXCLUSIONS rather than solved generally —
there's no single linguistic rule that works for both "corn starch" (exclude)
and "chicken breast" (a query for "chicken" should match this).
"""

import re

COMPOUND_EXCLUSIONS: dict[str, set[str]] = {
    "corn": {"corn starch", "cornstarch", "corn syrup", "cornmeal", "corn meal",
             "cornbread", "corn tortilla", "corn tortillas"},
    "chicken": {"chicken powder"},
}


def matches_query(ingredient_name: str, query_term: str) -> bool:
    name_lower = ingredient_name.lower()
    term_lower = query_term.lower()

    if re.search(rf"\b{re.escape(term_lower)}\b", name_lower) is None:
        return False

    excluded_phrases = COMPOUND_EXCLUSIONS.get(term_lower, set())
    if any(phrase in name_lower for phrase in excluded_phrases):
        return False

    return True
