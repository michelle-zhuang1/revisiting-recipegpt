# Revisiting RecipeGPT

Fine-tuning GPT-2 for recipe generation, comparing full fine-tuning vs. LoRA, and evaluating five decoding strategies — a CSED504 course project.

**Authors:** Danlei Wang, Michelle Zhuang

## Overview

RecipeGPT showed that GPT-2 can be fine-tuned to generate coherent cooking instructions, but generation quality depends on both the fine-tuning method and the decoding strategy used at inference. This project:

1. Replicates the RecipeGPT baseline by fine-tuning GPT-2 Small on RecipeNLG.
2. Evaluates five decoding strategies (greedy, beam search, top-k, top-p, temperature) on the fine-tuned model.
3. Compares three adaptation settings — pretrained (no adaptation), full fine-tuning, and LoRA — on generation quality, perplexity, training time, and trainable-parameter count.
4. Extends the LoRA comparison with a rank sweep (r = 4, 8, 16) to test whether quality scales with adapter capacity or plateaus.

See `docs/proposal.docx` for the full problem statement and success criteria.

## Repository structure

```
.
├── README.md
├── docs/
│   └── proposal.docx                  # Original project proposal (not yet converted to PDF)
├── notebooks/
│   ├── 01_dataset_preparation.ipynb   # RecipeNLG cleaning, reservoir sampling, train/val/test split
│   └── 02_train_and_evaluate.ipynb    # Fine-tuning, LoRA, decoding strategies, rank sweep, automatic evaluation
├── data/
│   ├── RecipeNLG_24k.csv              # Full cleaned/sampled subset before splitting
│   ├── train.csv / validation.csv / test.csv
│   └── generation_subset_final.csv / generation_subset_pilot.csv   # Fixed prompts used for generation comparisons
├── models/                            # Tracked via Git LFS — see Setup
│   ├── best_model/                    # Full fine-tuned GPT-2 Small checkpoint (config, tokenizer, weights)
│   ├── gpt2_lora_final/adapter/       # LoRA adapter, r=8 (used in the full-fine-tune-vs-LoRA comparison)
│   ├── gpt2_lora_final_r4/adapter/    # LoRA adapter, r=4 (rank sweep)
│   ├── gpt2_lora_final_r8/adapter/    # LoRA adapter, r=8 (rank sweep)
│   └── gpt2_lora_final_r16/adapter/   # LoRA adapter, r=16 (rank sweep)
└── results/
    ├── generation_metrics_final.csv / generation_metrics_pilot.csv        # BLEU/ROUGE-L/BERTScore/Distinct-n/length per method
    ├── pretrained_greedy_final.csv / pretrained_greedy_pilot.csv          # Per-example generations, pretrained model
    ├── finetuned_greedy_final.csv / finetuned_greedy_pilot.csv           # Per-example generations, fine-tuned model — greedy
    ├── finetuned_beam_final.csv / finetuned_beam_pilot.csv               # ...beam search
    ├── finetuned_top_k_final.csv / finetuned_top_k_pilot.csv             # ...top-k
    ├── finetuned_top_p_final.csv / finetuned_top_p_pilot.csv             # ...top-p
    ├── finetuned_temperature_final.csv / finetuned_temperature_pilot.csv # ...temperature
    ├── lora_greedy_final.csv                                             # Per-example generations, LoRA r=8
    ├── lora_r4_greedy_final.csv / lora_r8_greedy_final.csv / lora_r16_greedy_final.csv  # Rank sweep per-example generations
    ├── lora_vs_finetune_comparison_final.csv                             # Full fine-tune vs. LoRA (r=8) comparison
    ├── lora_rank_sweep_final.csv                                         # Rank sweep results (r = 4, 8, 16)
    ├── training_metrics_final.json / training_metrics_pilot.json         # Full fine-tune training run stats
    ├── validation_metrics_final.json / validation_metrics_pilot.json     # Full fine-tune validation loss/perplexity
    ├── lora_validation_metrics_final.json                                # LoRA validation loss/perplexity
    └── poster_example_recipe_0.csv / poster_success_example_0.csv / poster_contrast_example_1.csv  # Example recipes referenced in the course presentation poster (poster itself not included in this repo)
```

## Setup

This project originally ran entirely on **Google Colab** against a shared Drive folder (`/content/drive/Shareddrives/RecipeGPT/{data,models,results}`), and the notebooks in `notebooks/` still assume that Colab/Drive environment as-is — they have not yet been adapted to read/write the local `data/`, `models/`, and `results/` folders in this repo. Those local folders are migrated copies of the Drive outputs, included here for distribution and reproducibility of results, not (yet) wired up for a local/offline run. Only `02_train_and_evaluate.ipynb` requires a GPU; `01_dataset_preparation.ipynb` runs fine on CPU.

1. **Clone with Git LFS.** Model weights (`models/**/*.safetensors`) are stored via [Git LFS](https://git-lfs.com/). Install it once (`brew install git-lfs && git lfs install`) before cloning, or run `git lfs pull` after cloning if you cloned without it — otherwise those files will just be small LFS pointer text, not usable weights.
2. Mount Google Drive and create a shared project folder — the notebooks expect:

   /content/drive/Shareddrives/RecipeGPT/{data,models,results}

3. Install dependencies (each notebook installs its own requirements in its first cell):

   pip install transformers datasets peft accelerate evaluate sacrebleu rouge-score bert-score

4. A Kaggle API token (`kaggle.json`) is required for `01_dataset_preparation.ipynb` to download RecipeNLG.

**Note:** dependency versions are not pinned in the notebooks (`pip install -U ...`), so a fresh Colab session may install different library versions than the ones used to produce the results in `results/`. `02_train_and_evaluate.ipynb` includes a compatibility shim that inspects the installed `transformers` version and adapts `TrainingArguments`/`Trainer` argument names accordingly, but exact metric reproduction isn't guaranteed across sessions without pinning versions.

**Note on `models/`:** intermediate training checkpoints (optimizer/scheduler/RNG state, used only for resuming an interrupted training run) were pruned from the LoRA folders before committing — only the final `adapter/` for each run is kept. If you need to resume training from a specific step rather than start fresh, you'll need the original checkpoints from the shared Drive.

## Running order

1. **`01_dataset_preparation.ipynb`** — downloads RecipeNLG, cleans and reservoir-samples it, writes `train.csv` / `validation.csv` / `test.csv` to `data/`.
2. **`02_train_and_evaluate.ipynb`** — depends on step 1's output. Set `PILOT_MODE = True` first to validate the full pipeline on a small subset before committing GPU time to the full run (`PILOT_MODE = False`, 20k/2k/2k, 2 epochs). Produces the model checkpoints, LoRA adapters, generation outputs, and metrics CSVs/JSONs in `results/`.

## Key results (final run, not pilot)

| Comparison | Headline finding |
| :---- | :---- |
| Pretrained vs. fine-tuned | Fine-tuning improved BLEU (+11.9%), ROUGE-L (+32.6%), and BERTScore-F1 (+2.7%), while cutting average output length by 63.4%. |
| Decoding strategies | Greedy achieved the highest BERTScore-F1 (0.868); top-k/top-p achieved the highest diversity (Distinct-2 up to 0.481); beam search gave the highest BLEU with little BERTScore gain over greedy. |
| Full fine-tune vs. LoRA (r=8) | LoRA trained 0.236% of the parameters full fine-tuning did, in 76% of the training time, retaining 91–99% of generation-quality metrics. |
| LoRA rank sweep (r=4/8/16) | Quality plateaus early — r=4 performs essentially as well as r=16 on BLEU/ROUGE-L/BERTScore-F1, suggesting this task's adaptation needs little rank capacity. |

Full numbers are in the `results/` CSVs/JSONs listed above; the discussion and interpretation accompanying them (the course write-up) isn't included in this repo.

## Known issues / limitations

- **This repo holds the code, data, models, and results migrated from a shared Google Drive project** — not the full course write-up, poster materials, or the results-analysis notebook, which are intentionally not included. The notebooks that are here still target the original Colab/Drive paths rather than this repo's local `data/`/`models/`/`results/` folders.
- **`PILOT_MODE` results are not representative of final results.** The pilot run (1k/200/200, 1 epoch) showed pretrained GPT-2 *beating* the fine-tuned model on BLEU, apparently a length/small-sample artifact — resolved at full scale. Confirm which mode any given results file corresponds to before citing it.
- **Rank sweep, r=4 row:** `trainable_params` reads as 0 and `training_hours` is missing in `lora_rank_sweep_final.csv`. This is consistent with that adapter having been reloaded from a previously cached checkpoint (via `PeftModel.from_pretrained()`, which loads adapters with `requires_grad=False` by default) rather than trained fresh in the run that produced the final CSV. The generation-quality metrics for r=4 (BLEU/ROUGE-L/BERTScore/perplexity) are unaffected, since they only require running the model forward, not measuring its training. Re-run r=4's training fresh, or recover the original `rank_run_metrics.json`, before citing its efficiency numbers.
- **No pretrained-model perplexity** is computed anywhere in the pipeline — only fine-tuned and LoRA perplexity are available, so the Goal 1 (pretrained vs. fine-tuned) comparison rests on generation-quality metrics alone for that axis.
- **No peak GPU memory captured for the full fine-tuning run** — the LoRA-vs-full comparison is missing a memory-usage axis, which is often where LoRA's advantage is most decisive.
- **No human evaluation.** All quality claims rest on automatic metrics (BLEU, ROUGE-L, BERTScore, Distinct-n, perplexity), which are known to correlate imperfectly with whether a generated recipe is actually correct or sensible.
- **No pinned dependency versions** — see Setup above.

## Ethical considerations

RecipeNLG is compiled from online recipe sources and may overrepresent certain cuisines or cooking styles; a model fine-tuned on it will tend to reproduce that same skew. No formal audit of cuisine/ingredient representation was performed. Separately, a generated recipe that garbles a step, quantity, or cook time could cause real harm if followed uncritically — outputs from this project should be treated as a drafting aid for human review, not a validated instruction set. Further discussion was covered in the accompanying course write-up, not included in this repo.

## License / academic context

This is a course project for CSED504 (AI/ML for Engineering). Not intended for production use.
