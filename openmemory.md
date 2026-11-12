# OpenMemory guide — lexi_hack

## Overview

This is a Lexsi AI Safety hackathon entry (Track 2, submission Nov 14 2026). It tests whether SafeTune weight-space repairs on Cohere Tiny Aya survive Lee et al.'s five evaluation checks:

1. Tell the attacker.
2. Report a range.
3. Score the benign set.
4. Clear a matched null.
5. Give the defender an observable.

Experiment 1 asks whether Antidote beats generic pruning and survives attackers that know about it, with a Russian attack-language axis (the teammate adds Hindi; her probe, Lee et al. and baseline code are separate). The smoke result is GO: clean HS 4.6, attacked 59.0, A3 repairs 96%, random pruning 21%. Code is on GitHub; artifacts go to the HF Hub. GPU runs happen on a local GPU (native Windows or Linux/WSL2) or in Colab, Kaggle or HF Jobs; `notes/setup.md` has one section per platform.

## Architecture

- `exp1_antidote/` is a Python package driven by `run_grid.py`.
- The stages are data → attack → repair → generate → score → analyze → push. Every stage is idempotent and skips existing outputs.
- Outputs go under `$EXP1_OUT` (default `outputs/`): `data/`, `adapters/`, `repaired/`, `generations/`, `scores/`, `results/runs.jsonl`, `analysis/<config>/`. Temporary merged checkpoints go to `$EXP1_SCRATCH/merged/` (default: the output root).
- Configs deep-merge over `configs/base.yaml` through an `inherit` key. The run name (and so `analysis/<name>/`) is the experiment file's stem unless it sets `name`.
- Hardware profiles in `configs/profiles/` (`hf_backend`, `gpu16gb`, `t4`) merge on top with `run_grid --profile X` (repeatable). They never change the name. The list is stored in `cfg["profiles"]`, forwarded to the vLLM subprocess, and recorded with `dtype` and `backend` in every `runs.jsonl` row.
- `--stage all` pushes only when `hub.org` is set.

## User Defined Namespaces
- [Leave blank - user populates]

## Components

- **common.py:** paths, config loading, JSONL I/O, chat formatting (renders with `tokenize=False`, then tokenizes separately), and the `Run` dataclass. `run_id` has the form `{cell}__{arm}__a{alpha}__m{mask_seed}`.
- **grid.py:** the `ARMS` registry, which records for each arm whether it acts in adapter or merged space, whether it takes α, and whether it is random. Also `plan_runs` and `artifact(cfg, run)`, which returns (base|adapter|merged, path).
- **data.py:**
  - BeaverTails unsafe pairs are partitioned by unique prompt into calibration and attack pool; StrongREJECT prompts are excluded.
  - Attack doses are nested per seed. `harmful_ids(dose, seed)` is the single source of which pairs a seed uses.
  - `attack.language` (default `en`): for other languages the harmful half is swapped by id from `data/attack_pool_{lang}.jsonl`. The benign SST2 half, calibration and evaluation stay English.
  - SST2 handles utility. StrongREJECT, XSTest-safe and MMLU-Pro handle evaluation.
- **translate.py:** NLLB-200 translation of the dose-100 harmful slice (union over seeds 0–2, about 300 ids). Writes `attack_pool_{lang}.jsonl`, `{lang}_spotcheck.md`, `attack_slice_ids.txt` and `attack_slice_en.jsonl`.
- **attack.py:** an assistant-masked LoRA SFT in a plain PyTorch loop. It takes a `step_callback`, which the adaptive attacker uses.
- **repair.py:**
  - Adapter-space arms prune the `lora_A`/`lora_B` Linear layers with the git-disl k-rule; activation statistics come from forward hooks.
  - Merged-space arms call SafeTune `apply_antidote` or a local pruner with SafeTune's k-rule.
  - A6 scales B so that ‖BA‖_F equals A3's residual norm.
  - With `eval.keep_merged: false` (default), merged-space runs are skipped by the repair stage. `evaluate.generate` builds each one (`ensure_merged`), generates, and deletes it, so at most one ~7 GB checkpoint exists at a time. `repair_meta.json` is kept.
- **evaluate.py:**
  - vLLM: one process serves the base model plus adapters as LoRA requests; one process per merged run. An HF fallback exists (`--profile hf_backend`; required on native Windows).
  - Scoring uses the StrongREJECT finetuned judge, a refusal regex with a perplexity coherence gate under the clean model, and AuditKIT exact match.
  - `broken` = SST2 parse rate < 0.5. A1 (SafeTune `AntidoteTrainer` defaults) produces gibberish on Tiny Aya, and the judge scores gibberish ~0. The perplexity gate does not catch it, because repetitive junk has low perplexity.
  - Per-run generation seconds go to `results/timings.jsonl`. `max_new_tokens.xstest_safe` is 128, because the refusal regex reads only 400 characters.
- **analyze.py:** averages random arms over mask seeds, then reports mean and min–max over seeds; relative repair (shown as "broken" for broken arms); paired bootstrap over prompts; the headroom go/no-go (gap ≥ 25 points); α recommendation under an SST2 budget, excluding broken arms; a relearning table (`relearn.csv`, with step 0 = the repaired adapter).
- **compare.py:** a per-root table at one dose. Relative repair uses each root's own clean and A0 runs.
- **run_queue.py:** the unattended local driver. Jobs run in order: sweep, relearn, a2_diag, ru (`<root>_ru`), dose, adaptive (`<root>_adaptive`), compare. After the sweep it writes A3's α to `<root>/queue_alpha.yaml` and passes it as an extra `--profile` to every later job, so the nulls stay matched to A3. New roots are seeded with `data/` and the clean run. Logs go to `queue_log.txt` and `logs/<job>.log`.
- **adaptive.py:** a mask-aware attacker that uses `torch.nn.utils.parametrize` masks with Wanda refreshes, and a relearning re-attack (`<arm>+RL<k>` runs).
- **hub.py:** uploads adapters to a model repo and logs to a dataset repo; merged checkpoints are never pushed.

## Patterns

- Every SafeTune call goes through `safetune.recover` / `safetune.runner.recover`. The A1 trainer falls back to `apply_antidote` with the trainer's defaults.
- Do not run SafeTune hooks on an unmerged PEFT model: they would also match `lora_A`/`base_layer` modules.
- On T4 GPUs, use `--profile t4` (float16), plus `--profile hf_backend` if vLLM fails. Keep fp16 and bf16 grids in separate output roots.
- Secrets: `HF_TOKEN` lives in Colab, Kaggle or Jobs secrets, or locally in `hf auth login`. Nothing reads `.env`.
- Colab/Kaggle resume: Colab uses Drive for `EXP1_OUT`. Kaggle pushes at the end of each session and restores with `hf download` into `EXP1_OUT`, because the Hub repos mirror the output layout.
