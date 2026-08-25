"""Parse a Mela `.melarecipes` library export into a clean JSON corpus for retrieval.

Reads each `.melarecipe` entry from the zip archive in memory (never extracts the
full archive to disk, since embedded recipe photos make it huge) and drops the
`images` field, which isn't useful for text retrieval.

Usage:
    python extract_mela.py <path-to-Recipes.melarecipes> <output-corpus.json>
"""

import json
import sys
import zipfile


def parse_recipe(raw: dict) -> dict:
    return {
        "recipe_id": raw.get("id", ""),
        "title": raw.get("title", ""),
        "ingredients": [
            line.strip()
            for line in raw.get("ingredients", "").split("\n")
            if line.strip() and not line.strip().startswith("#")
        ],
        "instructions": raw.get("instructions", ""),
        "notes": raw.get("notes", ""),
        "categories": raw.get("categories", []),
        "yield": raw.get("yield", ""),
        "prep_time": raw.get("prepTime", ""),
        "cook_time": raw.get("cookTime", ""),
        "total_time": raw.get("totalTime", ""),
        "source_link": raw.get("link", ""),
        "favorite": raw.get("favorite", False),
        "want_to_cook": raw.get("wantToCook", False),
    }


def main(melarecipes_path: str, out_path: str) -> None:
    recipes = []
    skipped = []

    with zipfile.ZipFile(melarecipes_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".melarecipe")]
        for name in names:
            try:
                raw = json.loads(zf.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                skipped.append((name, str(e)))
                continue
            recipes.append(parse_recipe(raw))

    with open(out_path, "w") as f:
        json.dump(recipes, f, indent=2)

    print(f"parsed {len(recipes)} recipes -> {out_path}")
    if skipped:
        print(f"skipped {len(skipped)} entries that failed to parse:")
        for name, err in skipped:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
