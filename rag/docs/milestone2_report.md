# Retrieval-Conditioned Generation — Milestone 2 Report

**Project**: Recipe RAG (extension of RecipeGPT, CSED504)
**Authors**: Danlei Wang, Michelle Zhuang
**Status**: Complete — negative result

## Objective

Test whether feeding recipes retrieved from a personal Mela library as few-shot
context into the RecipeGPT LoRA model produces generations that better reflect the
user's own saved-recipe style, compared to unconditioned generation — the original
stretch-goal extension of the RecipeGPT project.

## Method

- **Model**: GPT-2 Small + LoRA adapter (r=8, `gpt2_lora_final`), fine-tuned on
  RecipeNLG in the base RecipeGPT project.
- **Retrieval**: `hybrid_search` (structured ingredient filter, then semantic
  ranking) over a 621-recipe personal Mela library.
- **Prompt structure**, exactly as originally proposed:

  ```
  Here are recipes from my collection similar to what I'm looking for:
  [retrieved recipe 1]
  [retrieved recipe 2]
  [retrieved recipe 3]

  Generate a new recipe for: {user request}, in a similar style to the above.
  ```

- **Decoding**: greedy.
- **Test query**: `"something quick with chicken and no dairy"`, run at k=3 and k=1
  retrieved examples.

## Results

**k=3 retrieved recipes**: the three recipes alone tokenize to **1803 tokens** —
roughly 1.76x the model's entire 1024-token context window, before the generation
instruction is even added. Truncating to fit removes the *entire* instruction
(it sits at the end of the prompt; truncation cuts from the end), leaving the model
to continue mid-sentence inside someone else's recipe with no task signal at all.
Attempting generation in this state hard-crashes
(`IndexError: index out of range in self`, in the position-embedding layer — there
are zero token positions left to generate into).

**k=1 retrieved recipe** (553 tokens, well under the limit — the instruction is
*not* truncated): still fails, more informatively. The model ignores the
instruction entirely and continues the statistical pattern of the one retrieved
example — generating another ingredient list that echoes the retrieved recipe's
own ingredients, including "milk," directly violating the "no dairy" part of the
request, rather than producing anything resembling a response to "generate a new
recipe for X."

**Both runs additionally degenerate into repetition loops** under greedy decoding
(e.g. "1/2 c. milk" repeated 30+ times) — a separate, baseline generation-quality
issue independent of retrieval-conditioning.

## Conclusion

Naive few-shot retrieval-conditioning fails for two independent, architectural
reasons, neither fixable by prompt engineering:

1. **Context-window overflow at realistic k.** 3-5 retrieved recipes (as
   originally proposed) substantially exceed GPT-2's 1024-token window on their
   own.
2. **No instruction-following capability**, even when the prompt fits. The LoRA
   adapter was fine-tuned only to continue RecipeNLG's fixed
   title→ingredients→directions template — never to distinguish "context to draw
   from" from "an instruction to execute." At k=1, with the instruction physically
   present and intact, the model shows no evidence of having understood it.

## Recommendations

- **Fine-tune the LoRA adapter on the retrieval-conditioned task format itself**
  (context + instruction → target recipe), rather than hoping the model improvises
  the task zero-shot. In progress as the next step.
- **Alternative not pursued**: swap in an instruction-tuned model for the
  generation step specifically, keeping the RecipeGPT LoRA model as the
  unconditioned baseline arm only.

Full technical detail and reproduction steps: `rag/README.md`,
["Retrieval-conditioned generation results"](../README.md#retrieval-conditioned-generation-results-milestone-2)
section; code in `rag/generate.py`.
