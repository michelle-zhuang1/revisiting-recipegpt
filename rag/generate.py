"""Retrieval-conditioned generation via the RecipeGPT LoRA model.

Naive few-shot approach, as originally proposed: feed retrieved recipes as
in-context examples in the prompt, then ask for a new recipe "in a similar
style." GPT-2 Small (124M params, never trained for instruction-following or
few-shot prompting, 1024-token context) is not expected to handle this well —
this is a deliberate experiment to see and document what actually happens,
not an assumption that it will work. See rag/README.md for the reasoning and
results.

Usage:
    python generate.py "something quick with chicken and no dairy"
"""

import argparse
import json

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid_search import hybrid_search

BASE_MODEL = "gpt2"
ADAPTER_DIR = "../models/gpt2_lora_final/adapter"
CORPUS_PATH = "data/corpus.json"
MAX_CONTEXT_TOKENS = 1024
MAX_NEW_TOKENS = 200


def recipe_to_text(recipe: dict) -> str:
    ingredients = "\n".join(recipe["ingredients"])
    return f"{recipe['title']}\n{ingredients}\n{recipe['instructions']}"


def build_unconditioned_prompt(request: str) -> str:
    return f"{request}\n"


def build_conditioned_prompt(retrieved: list[dict], request: str) -> str:
    examples = "\n\n".join(recipe_to_text(r) for r in retrieved)
    return (
        "Here are recipes from my collection similar to what I'm looking for:\n\n"
        f"{examples}\n\n"
        f"Generate a new recipe for: {request}, in a similar style to the above.\n\n"
    )


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> dict:
    full_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS)
    prompt_tokens = inputs["input_ids"].shape[1]
    truncated = full_tokens > prompt_tokens

    if prompt_tokens >= MAX_CONTEXT_TOKENS:
        return {
            "prompt_tokens": prompt_tokens,
            "full_tokens": full_tokens,
            "truncated": truncated,
            "generated_text": None,
            "note": (
                f"Prompt alone fills the {MAX_CONTEXT_TOKENS}-token context window "
                f"(full prompt was {full_tokens} tokens before truncation) — no room "
                "left to generate anything. Skipped rather than crashing."
            ),
        }

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)

    return {
        "prompt_tokens": prompt_tokens,
        "full_tokens": full_tokens,
        "truncated": truncated,
        "generated_text": generated_text,
        "note": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("-k", type=int, default=3, help="number of retrieved recipes to condition on")
    args = parser.parse_args()

    with open(CORPUS_PATH) as f:
        recipes_by_id = {r["recipe_id"]: r for r in json.load(f)}

    hits = hybrid_search(args.request, n_results=args.k)
    retrieved = [recipes_by_id[h["recipe_id"]] for h in hits if h["recipe_id"] in recipes_by_id]

    print(f"Retrieved {len(retrieved)} recipes:")
    for r in retrieved:
        print(f"  - {r['title']}")
    print()

    print("Loading model...")
    model, tokenizer = load_model()

    unconditioned_prompt = build_unconditioned_prompt(args.request)
    conditioned_prompt = build_conditioned_prompt(retrieved, args.request)

    print("\n=== Unconditioned ===")
    result = generate(model, tokenizer, unconditioned_prompt)
    print(f"prompt_tokens={result['prompt_tokens']} (full: {result['full_tokens']}) truncated={result['truncated']}")
    if result["note"]:
        print(f"NOTE: {result['note']}")
    else:
        print(result["generated_text"])

    print("\n=== Retrieval-conditioned ===")
    result = generate(model, tokenizer, conditioned_prompt)
    print(f"prompt_tokens={result['prompt_tokens']} (full: {result['full_tokens']}) truncated={result['truncated']}")
    if result["note"]:
        print(f"NOTE: {result['note']}")
    else:
        print(result["generated_text"])


if __name__ == "__main__":
    main()
