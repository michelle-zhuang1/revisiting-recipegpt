"""Extract a core ingredient name from a raw ingredient line.

Heuristic, not a full parser: strips a leading quantity/unit. Good enough for
approximate ingredient-set matching (e.g. "does this recipe have chicken"),
not meant to produce a canonical/normalized ingredient name.
"""

import re

UNITS = r"cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|grams?|g|ounces?|oz\.?|pounds?|lbs?\.?|ml|milliliters?|liters?|cloves?|slices?"

UNICODE_FRACTIONS = "½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"

QUALIFIERS = r"scant|heaping|generous"
NUMBER_WORDS = r"one|two|three|four|five|six|seven|eight|nine|ten"

LEADING_QUANTITY_RE = re.compile(
    rf"^(?:(?:{QUALIFIERS}|{NUMBER_WORDS})\s+)?[\d\s/.\-–{UNICODE_FRACTIONS}]+(?:(?:{UNITS})(?:/[\d.\s]*[a-z]+)?\s+)?",
    re.IGNORECASE,
)
TRAILING_PAREN_RE = re.compile(r"\s*\((?:[^()]|\([^()]*\))*\)\s*$")
BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*")
TRAILING_CLAUSE_RE = re.compile(rf",\s*([a-z][a-z0-9./\-{UNICODE_FRACTIONS}\s]*)$")

# A trailing ", ..." clause is only prep instructions (safe to drop) if every
# word in it is one of these — otherwise it likely contains the ingredient's
# own name/modifiers (e.g. "raw, unsalted cashews") and must be kept.
PREP_WORDS = {
    "diced", "chopped", "minced", "sliced", "crushed", "grated", "shredded",
    "cubed", "quartered", "halved", "peeled", "seeded", "deveined", "trimmed",
    "drained", "rinsed", "softened", "melted", "divided", "packed",
    "thinly", "finely", "roughly", "coarsely",
    "plus", "more", "for", "garnish", "serving", "to", "taste", "optional",
    "cut", "into", "pieces", "thick",
    "cubes", "chunks", "slices", "wedges", "lengths", "dice", "bite-size",
    "and", "or", "serve",
}

# A word like "½-inch" that's a size measurement, not covered by the fixed
# PREP_WORDS set since the number varies.
MEASUREMENT_TOKEN_RE = re.compile(
    rf"^[\d/.\-{UNICODE_FRACTIONS}]+-?(inch(es)?|in|mm|cm)\.?$",
    re.IGNORECASE,
)


def _is_prep_word(word: str) -> bool:
    return word in PREP_WORDS or bool(MEASUREMENT_TOKEN_RE.match(word))


def _strip_trailing_prep_clause(line: str) -> str:
    match = TRAILING_CLAUSE_RE.search(line)
    if not match:
        return line
    words = match.group(1).split()
    if words and all(_is_prep_word(w) for w in words):
        return line[: match.start()]
    return line


def extract_ingredient_name(line: str) -> str:
    line = LEADING_QUANTITY_RE.sub("", line, count=1).strip()
    line = BRACKET_RE.sub(" ", line).strip()

    while True:
        stripped = TRAILING_PAREN_RE.sub("", line).strip()
        stripped = _strip_trailing_prep_clause(stripped).strip()
        stripped = stripped.rstrip(",").strip()
        if stripped == line:
            break
        line = stripped

    return line
