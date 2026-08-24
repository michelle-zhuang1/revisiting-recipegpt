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
- **`extract_ingredient_name` is a heuristic, not a real parser**, and has a real
  backlog after testing against the full 8243-line corpus (not just hand-picked
  examples):
  - Still **actively wrong** on some inputs: a trailing-comma heuristic meant to
    strip prep clauses (`"...chicken breasts, diced"` → drop `", diced"`) sometimes
    deletes real ingredient words instead (`"1/4 cup raw, unsalted cashews"` →
    `"raw"`, dropping "cashews"). No fix yet — needs a curated prep-word vocabulary
    (diced, chopped, minced, ...) to distinguish "drop this" from "keep this."
  - Not handled: unicode fraction characters (`½`, `⅓`, `¾`), `"or"` alternatives
    (`"egg or 1 large egg"`), nested parentheses, no-space compact quantities with no
    following space (`"43gMelted..."`), Mela's `"#"`-prefixed section headers leaking
    into the ingredients list as if they were ingredients, `oz`/`lb` unit
    abbreviations (only `tbsp`/`tsp`/`g` are covered).
  - Units covered so far: `cups?`, `tablespoons?`/`tbsp\.?`, `teaspoons?`/`tsp\.?`,
    `grams?`/`g`, `ounces?`, `pounds?`, `ml`, `milliliters?`, `liters?`, `cloves?`,
    `slices?`.
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
