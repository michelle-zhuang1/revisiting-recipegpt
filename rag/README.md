# Recipe RAG

Semantic search over a personal Mela recipe library — an extension of the RecipeGPT
project. Milestone 1 (this stage): pure retrieval, no generation yet. Retrieval-
conditioned generation (feeding retrieved recipes as context into the fine-tuned
RecipeGPT LoRA model, and comparing against unconditioned generation) is a stretch
goal to revisit separately, not yet started.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

1. **Export your Mela library**: in Mela, export the full library (not individual
   recipes) as a `.melarecipes` bundle.
2. **Extract**: `python extract_mela.py <path-to-Recipes.melarecipes> data/corpus.json`
   — parses the zip into a clean JSON corpus, dropping embedded recipe photos
   (irrelevant to text retrieval, and the reason the raw export is ~600MB for ~600
   recipes rather than a couple MB).
3. **Ingest**: `python ingest.py` — embeds two chunks per recipe (`ingredients` and
   `recipe` = title + time/yield + ingredients + instructions) with a local
   `sentence-transformers` model (`all-MiniLM-L6-v2`) into a local Chroma store at
   `chroma_db/`.
4. **Semantic search**: `python search.py "something quick and easy for a weeknight"`
   (optionally `--type ingredients` or `--type recipe` to restrict chunk type). Best
   for vibes/style queries — not a hard-constraint filter (see below).
5. **Pantry search**: `python pantry_search.py chicken artichoke` — a deterministic
   AND-filter, not semantic search. Parses each recipe's raw ingredient lines
   (`ingredient_parser.extract_ingredient_name`) and requires every query term to
   match at least one ingredient (`ingredient_matcher.matches_query`, word-boundary
   matching with a hand-curated exclusion list for compounds like "corn starch" that
   share a word with the base ingredient but are a different food — see
   `COMPOUND_EXCLUSIONS` in `ingredient_matcher.py`). Use this instead of `search.py`
   whenever "must contain both X and Y" is a hard requirement rather than a
   similarity preference.

`data/` (the parsed corpus) and `chroma_db/` (the vector store) are gitignored —
both are regenerable from your own Mela export and aren't meant for the public repo,
since Mela recipes are typically clipped from other sites, not authored by you.

## Testing

```
pip install -r requirements-dev.txt
python3 -m pytest tests/
```

Covers five seams: `extract_mela.parse_recipe` (raw Mela JSON → clean dict),
`ingest.build_chunks` (recipe → embeddable chunks), `search.search`'s contract
(chunk-type filtering, `n_results`, hit shape — not retrieval *quality*, which has
no independent expected value to test against and stays a manual/qualitative call),
`ingredient_parser.extract_ingredient_name`, `ingredient_matcher.matches_query`, and
`pantry_search.filter_by_ingredients`. All driven from real lines pulled from the
actual corpus, not invented examples.

## Known limitations

- **Semantic search (`search.py`) can't guarantee "must contain both X and Y."**
  A query like "chicken and artichoke" can be dominated by whichever ingredient is
  more semantically distinctive (artichoke, in practice) rather than requiring both
  be present. **Fixed by `pantry_search.py`** for the AND-filtering use case —
  deterministic ingredient extraction + matching, not embedding similarity.
- **`extract_ingredient_name` is a heuristic, not a real parser.** Fixed against the
  full 8243-line corpus (not just hand-picked examples), with real before/after
  numbers where the fix was partial rather than complete:
  - **Fixed**: the trailing-comma heuristic no longer deletes real ingredient words
    (`"1/4 cup raw, unsalted cashews"` used to become `"raw"`, dropping "cashews" —
    now only strips a trailing clause if every word in it is a known prep term,
    checked against a curated `PREP_WORDS` set rather than assumed).
  - **Fixed**: dangling trailing commas left after a parenthetical is stripped.
  - **Fixed** (~90%): unicode fraction characters (`½`, `⅓`, `¾`, ...) — 1106 of 8243
    lines had one; 111 residual cases remain, all inside an *already-known* other
    issue (an `"or"` alternative, a parenthetical, or a non-standard leading word),
    not a new gap.
  - **Fixed** (~76%): en-dash ranges (`"2–3 cloves garlic"`) — 22 of 29 lines.
    Remainder needs the spelled-out-leading-number fix below, or is an en-dash used
    stylistically as a hyphen (`"Middle Eastern–style"`), not an actual range.
  - **Partially fixed** (~51%, up from 23%): `"cut into ½-inch pieces"`-style
    trailing clauses — extended `PREP_WORDS` with cut/shape vocabulary (cubes,
    chunks, wedges, slices, ...) and made clause-stripping iterative (peel multiple
    trailing clauses one at a time, e.g. `"...wedges, to serve"`). Remaining 53/108
    need either deeper shape vocabulary (quarters/sixths/eighths, "depending on
    size") or handling clauses that contain their own internal commas — real, but
    hit diminishing returns for the effort.
  - **Fixed**: leading qualifier words (`"scant ¼ teaspoon salt"`) — very low
    frequency (2 lines total in the corpus), done for completeness.
  - **Fixed**: `oz`/`lb` unit abbreviations (`"1 15-oz. can"`, `"1 lb. mozzarella"`).
  - **Fixed** (~99%): nested/double-wrapped parentheticals (`"500 g flour
    ((1.1 pounds))"` — a common Mela-import artifact) — 269/271 lines. Handled with
    a regex that allows one level of nesting, not a full recursive parser; the 2
    remaining cases have a genuinely mid-line (not trailing) nested paren.
  - **Fixed** (~99%): no-space compact quantities (`"100g Red Cabbage"`) — turned out
    to already be mostly resolved as a side effect of the bare-`g` fix; only 1 line
    in the whole corpus has the true zero-space case (`"43gMelted..."`), not worth a
    dedicated fix for n=1.
  - **Fixed** (~97%): trailing artifacts revealed only *after* another cleanup step
    ran (`"Sliced baguette (optional), for serving"` — stripping the `", for
    serving"` clause reveals a now-trailing `"(optional)"` that a single pass would
    miss). Paren-stripping and clause-stripping now loop together until nothing
    changes, instead of each running once. Trailing-paren leakage dropped from 61 to
    2 lines.
  - **Fixed** (100%, but at the extraction layer, not here): Mela's `"#"`-prefixed
    section headers (`"# Dough"`, `"### PROTEIN"`) were leaking into ingredient
    lists as if they were ingredients — 454 of 8243 lines. This also meant they were
    polluting `search.py`'s semantic embeddings for every recipe with sub-sections,
    not just `pantry_search.py`'s extraction. Filtered out entirely in
    `extract_mela.parse_recipe` (the right layer — a `"#"`-prefixed line is
    unambiguously not an ingredient, no heuristic uncertainty).
  - **Deliberately not fixed** — confirmed to not actually matter: `"or"`
    alternatives (`"egg or 1 large egg"`) and `"Zest of X"`/`"Juice of X"` sentence
    shape (quantity isn't leading). Both looked broken but `matches_query` does
    whole-word search across the *entire* extracted string, so e.g. `"lemon"` still
    matches `"Juice of ½ lemon"` and `"chicken"` still matches `"chicken or
    vegetable broth"` correctly, regardless of the surrounding structure. Cosmetic
    only, not a filtering bug — skipped both.
  - **Not handled**: spelled-out leading numbers (`"one 4–6 oz container"`, 15
    lines), square brackets used instead of parens for metric conversions (60
    lines), a handful of dual-compact-quantity lines (`"30g 1/2c shallot"`, 4
    lines). All low-frequency at this point — real but low priority.
  - **~7 lines (0.09%) extract to an empty string** — not a bug, these are
    genuinely malformed/fragment source lines in the Mela export itself (e.g. a
    bare `'2'` with no unit or ingredient at all, or a line that's just a
    parenthetical recipe note like `'(makes about 1 cup, enough for 8 salads)'`
    that got included in the ingredients list by mistake). Nothing to extract
    because the information was never there.
  - Units covered: `cups?`, `tablespoons?`/`tbsp\.?`, `teaspoons?`/`tsp\.?`,
    `grams?`/`g`, `ounces?`/`oz\.?`, `pounds?`/`lbs?\.?`, `ml`, `milliliters?`,
    `liters?`, `cloves?`, `slices?`.
- **`ingredient_matcher.COMPOUND_EXCLUSIONS` is small and will need to keep growing.**
  Currently covers `corn` (→ corn starch, corn syrup, cornmeal, cornbread, corn
  tortilla) and `chicken` (→ chicken powder), both found by testing real queries
  against the real corpus, not anticipated in advance. Expect more of these to
  surface with use — same shape as the unit-abbreviation list, add as found.
  Deliberately *not* solved generally (would need real food-ontology work like USDA
  FoodData Central); a compound like "chicken broth" is correctly *not* excluded,
  since it's an actual chicken product, unlike "chicken powder."
- **Time-based queries ("something quick") are weakly supported in `search.py`.**
  Only ~47% of recipes have `total_time` populated at all (39%/37% for prep/cook —
  Mela's clipper doesn't always find this on the source page), so there's often
  nothing for a time-based query to match against regardless of technique.

**v2 candidates** (not started): work through the `extract_ingredient_name` backlog
above; numeric time parsing + filtering (same shape as ingredient extraction, for
"under 30 minutes" as a hard constraint rather than a vibe); retrieval-conditioned
generation via the RecipeGPT LoRA model, compared against unconditioned generation
(novelty-vs-copying via n-gram overlap, a "feels like me" qualitative rubric, and
diversity across generations).
