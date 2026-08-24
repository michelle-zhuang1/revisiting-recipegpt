"""Extract a core ingredient name from a raw ingredient line.

Heuristic, not a full parser: strips a leading quantity/unit. Good enough for
approximate ingredient-set matching (e.g. "does this recipe have chicken"),
not meant to produce a canonical/normalized ingredient name.
"""

import re

UNITS = r"cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|grams?|g|ounces?|pounds?|ml|milliliters?|liters?|cloves?|slices?"

LEADING_QUANTITY_RE = re.compile(
    rf"^[\d\s/.\-]+(?:(?:{UNITS})(?:/[\d.\s]*[a-z]+)?\s+)?",
    re.IGNORECASE,
)
TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
TRAILING_PREP_CLAUSE_RE = re.compile(r",\s*[a-z][a-z\s]*$")


def extract_ingredient_name(line: str) -> str:
    line = LEADING_QUANTITY_RE.sub("", line, count=1).strip()
    line = TRAILING_PAREN_RE.sub("", line).strip()
    line = TRAILING_PREP_CLAUSE_RE.sub("", line).strip()
    return line
