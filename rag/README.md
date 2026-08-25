# Recipe RAG

Semantic search over a personal Mela recipe library — an extension of the RecipeGPT
project. Milestone 1: pure retrieval (semantic search, deterministic pantry
AND/NOT-filtering, and a hybrid of both) — done. Milestone 2: retrieval-conditioned
generation via the RecipeGPT LoRA model — attempted, with a clear negative result;
see [Retrieval-conditioned generation results](#retrieval-conditioned-generation-results-milestone-2)
below.

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
   similarity preference. `pantry_search.filter_recipes` also supports `exclude`
   (must-not-contain ingredients) and `exclude_categories` (must-not-contain any
   member of a curated category, e.g. `dairy` — see `CATEGORIES` in
   `ingredient_matcher.py`) beyond the CLI's simple AND-only interface.
6. **Hybrid search**: `python hybrid_search.py "something quick" --include chicken
   --exclude-categories dairy` — structured filter first (via `pantry_search`), then
   semantic-ranks only the surviving candidates by the vibe text (via a Chroma query
   restricted to those `recipe_id`s). For queries that mix a hard constraint with a
   style preference, rather than being purely one or the other. Takes structured
   parameters (`--include`/`--exclude`/`--exclude-categories` flags), not a single
   free-text sentence — parsing natural language into filter/vibe parts
   automatically is out of scope for now (see Known limitations).
7. **Generate**: `python generate.py "something quick with chicken and no dairy"`
   — retrieves recipes via `hybrid_search`, feeds them as few-shot context into the
   RecipeGPT LoRA model (`../models/gpt2_lora_final/adapter`), and asks it to
   generate a new recipe "in a similar style." **Has a clear negative result — see
   below before using this.**

`data/` (the parsed corpus) and `chroma_db/` (the vector store) are gitignored —
both are regenerable from your own Mela export and aren't meant for the public repo,
since Mela recipes are typically clipped from other sites, not authored by you.

## Testing

```
pip install -r requirements-dev.txt
python3 -m pytest tests/
```

Covers: `extract_mela.parse_recipe` (raw Mela JSON → clean dict), `ingest.build_chunks`
(recipe → embeddable chunks), `search.search`'s contract (chunk-type filtering,
`n_results`, hit shape), `ingredient_parser.extract_ingredient_name`,
`ingredient_matcher.matches_query`/`matches_category`, `pantry_search.filter_by_ingredients`/
`filter_recipes`, `hybrid_search.rank_by_vibe`/`hybrid_search`, and `generate.py`'s
prompt-construction functions (`recipe_to_text`, `build_conditioned_prompt`,
`build_unconditioned_prompt`). All driven from real lines pulled from the actual
corpus, not invented examples — with one exception worth calling out: `rank_by_vibe`'s
tests only check its *contract* (restricted to candidate `recipe_id`s, respects
`n_results`, empty-candidates edge case), never ranking *order* — an earlier draft of
this test asserted "quick" should outrank "slow" and it was flaky/wrong to write,
since ranking quality is a function of embedding-model behavior with no independent
ground truth, not something a unit test can assert on. The same reasoning is why
`generate.py`'s actual `model.generate()` call has no unit test at all — there's no
independent expected output for a language model's generated text; that's what the
results section below is for instead (a real, run, documented experiment, not a
pass/fail assertion).

## Retrieval-conditioned generation results (Milestone 2)

Attempted exactly as originally proposed: retrieve top-k recipes via `hybrid_search`,
feed them as few-shot context into the RecipeGPT LoRA model (`gpt2_lora_final`,
r=8), ask it to generate a new recipe "in a similar style," greedy decoding. This
was flagged as high-risk before building it (GPT-2 Small is 124M params, was never
trained for instruction-following or few-shot prompting, and only has a 1024-token
context window) — the risk was confirmed, and confirmed more thoroughly than
expected. Real run: `python generate.py "something quick with chicken and no dairy"`.

**With k=3 retrieved recipes**: the three recipes alone tokenize to **1803 tokens**
— nearly double the model's entire 1024-token context window, before the actual
generation instruction is even added. Truncating to fit removes the *entire*
instruction (`"Generate a new recipe for: ..., in a similar style to the above"`),
since it sits at the end of the prompt and truncation cuts from the end. What's left
after truncation ends mid-sentence inside someone else's recipe instructions
(`"...Heat until hot (you can check by adding a"`). `generate.py` detects this case
and skips generation with a clear message rather than proceeding — the underlying
`transformers` call crashes with `IndexError: index out of range in self` in the
position-embedding layer if attempted anyway, since there are zero token positions
left for anything new.

**With k=1** (553 tokens, well under the limit, instruction *not* truncated): still
a clear failure, but a more informative one. The model ignores the instruction
entirely and just continues the statistical pattern of the one retrieved example —
generating another ingredient list that echoes the retrieved recipe's own
ingredients (including "milk," directly violating the "no dairy" part of the
request) rather than anything resembling a response to "generate a new recipe for
X." This confirms the failure isn't only about context-window overflow — even with
the instruction physically present and un-truncated, the model shows no evidence of
following it. It was fine-tuned only to continue RecipeNLG's fixed
title→ingredients→directions template, never to treat a preceding block of text as
"context to draw from" versus "an instruction to execute."

**Both runs also degenerate into repetition loops** under greedy decoding
(`"1/2 c. milk"` × 30, `"1 teaspoon ground black pepper"` × 20+) — a baseline
generation-quality issue independent of retrieval-conditioning, consistent with
known greedy-decoding pathologies on small models (the original RecipeGPT
comparison table found top-k/top-p necessary for output diversity for exactly this
reason).

**Conclusion**: naive few-shot retrieval-conditioning does not work on this model,
for two independent reasons — context-window overflow at realistic k, and no
instruction-following capability even when the instruction physically fits. Both
are architectural properties of GPT-2 Small + a LoRA adapter fine-tuned only for
template continuation, not fixable by prompt engineering. The mitigations discussed
before attempting this (fine-tuning the model on the retrieval-conditioned task
format specifically, or swapping in an instruction-tuned model for this experiment)
remain open, undone options if this is revisited — this result is why they were
worth considering in the first place, now with concrete evidence instead of a
prediction.

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
  - **Fixed** (100%): square brackets used instead of parens for metric conversions
    (`"½ cup [75 ml] extra-virgin olive oil"`) — 60/60 lines. Stripped globally
    (anywhere in the line, not just leading/trailing) since brackets in this corpus
    consistently mean "metric equivalent," safe to drop regardless of position.
  - **Fixed** (~87%): spelled-out leading numbers (`"one 14-ounce can..."`) — 13/15
    lines, same mechanism as the leading-qualifier fix (skip the word before the
    real quantity starts).
  - **Not handled**: a handful of dual-compact-quantity lines (`"30g 1/2c
    shallot"`, 4 lines). Low-frequency, low priority.
  - **~7 lines (0.09%) extract to an empty string** — not a bug, these are
    genuinely malformed/fragment source lines in the Mela export itself (e.g. a
    bare `'2'` with no unit or ingredient at all, or a line that's just a
    parenthetical recipe note like `'(makes about 1 cup, enough for 8 salads)'`
    that got included in the ingredients list by mistake). Nothing to extract
    because the information was never there.
  - Units covered: `cups?`, `tablespoons?`/`tbsp\.?`, `teaspoons?`/`tsp\.?`,
    `grams?`/`g`, `ounces?`/`oz\.?`, `pounds?`/`lbs?\.?`, `ml`, `milliliters?`,
    `liters?`, `cloves?`, `slices?`.
- **`ingredient_matcher.COMPOUND_EXCLUSIONS` and `CATEGORIES` are small and will
  need to keep growing.** `COMPOUND_EXCLUSIONS` covers `corn` (→ corn starch, corn
  syrup, cornmeal, cornbread, corn tortilla), `chicken` (→ chicken powder, chicken
  bouillon), `milk`
  (→ coconut/almond/oat/soy/cashew milk), `butter` (→ peanut/almond butter), and
  `cream` (→ cream of tartar) — all found by testing real queries against the real
  corpus, not anticipated in advance. `CATEGORIES` currently has one entry, `dairy`
  (milk, butter, cheese, cream, yogurt, yoghurt, buttermilk, whey, casein, ricotta,
  mascarpone), used by `hybrid_search.py`'s `--exclude-categories` and
  `pantry_search.filter_recipes`'s `exclude_categories`. Expect more compounds and
  categories to surface with use — add as found, same shape as the unit-abbreviation
  list. Deliberately *not* solved generally (would need real food-ontology work like
  USDA FoodData Central); a compound like "chicken broth" is correctly *not*
  excluded, since it's an actual chicken product, unlike "chicken powder."
  "Chicken bouillon" (powdered chicken broth) was resolved the same way as
  "chicken powder" — excluded.
- **Time-based queries ("something quick") are weakly supported in `search.py`,
  and numeric time parsing/filtering (the equivalent of `pantry_search.py` but for
  "under 30 minutes") is deliberately deprioritized, not just unstarted.** Only
  ~47% of recipes have `total_time` populated at all (39%/37% for prep/cook —
  Mela's clipper doesn't always find this on the source page). Unlike the
  ingredient-parser work, this is a data-coverage ceiling, not a parsing-quality
  problem — even a perfect time parser can't help the 53% of recipes with nothing
  to parse, so the effort/payoff ratio is worse than it looks. Revisit only if a
  way to backfill missing time data shows up (e.g. re-deriving it from
  `instructions` text), not by improving the parser alone.

**v2 candidates** (not started): the remaining `extract_ingredient_name` tail
(`"cut into"` shape vocabulary, dual-compact-quantity lines — both low priority,
diminishing returns); free-text query parsing for `hybrid_search.py` (turning
`"something quick with chicken and no dairy"` into `vibe`/`include`/
`exclude_categories` automatically — real NLP scope, probably wants an LLM call
rather than regex, deliberately deferred in favor of structured parameters for now).

Retrieval-conditioned generation was attempted, not skipped — see results above.
If revisited, the real options are: fine-tune the LoRA adapter specifically on a
retrieval-conditioned task format (context + generate) rather than hoping the model
improvises it zero-shot, or swap in an instruction-tuned model for the generation
step while keeping the RecipeGPT LoRA model as the unconditioned baseline arm only.
The original comparison plan (novelty-vs-copying via n-gram overlap, a "feels like
me" qualitative rubric, diversity across generations) is moot until generation
itself actually produces something evaluable.
