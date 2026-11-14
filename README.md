# Hidden, not removed? Weight-space safety edits on Tiny Aya

ValueShift Research entry to the Lexsi AI Safety hackathon (Track 2).

We test two weight-space safety edits on `CohereLabs/tiny-aya-global` (3.35B) against Lee et al.'s five evaluation checks:

1. Tell the attacker.
2. Report a range.
3. Score the benign set.
4. Clear a matched null.
5. Give the defender an observable.

The two edits:

- **Experiment U, unlearning.** RMU unlearning of high-school chemistry. Is the knowledge erased, or only hidden? Does it come back after fine-tuning in another language, or on unrelated data?
- **Experiment R, repair (this repository's `exp1_antidote/`).** Antidote, which prunes a harmful LoRA fine-tune after the fact. Does it beat generic pruning at the same strength, and does it hold against an attacker who knows about it?

The write-up is at **[link to the HF article / write-up]**.

## Results in one paragraph

- **Unlearning.** English-only RMU unlearning is undone by a small LoRA fine-tune in English, Hindi or Marathi, even on unrelated Marathi data. How much comes back tracks how much was erased, not which layer was edited.
- **Repair.** Antidote repairs the harmful fine-tune: harmful score 59.0 → 5.8 at α = 0.2, against 4.6 for the clean model, at a cost of 1.2 SST2 points. It clearly beats random pruning (29.7) and a norm-matched shrink (16.0). But calibration-free magnitude pruning of the same adapter matches it (5.7), so Antidote's harmful-data calibration adds almost nothing on Tiny Aya.
- **Together.** In both experiments the mechanism the method credits disappears under a matched-strength control. **[pending: relearning, Russian attack, doses 10/50, mask-aware attacker]**

## Artifacts

| What | Where |
|---|---|
| Models and adapters | HF collection **[link]**: `ValueShift/exp1-antidote-adapters`, `-ru-adapters`, `-adaptive-adapters`, and Experiment U's models **[repo ids]** |
| Logs, scores, generations, analysis | HF collection **[link]**: `ValueShift/exp1-antidote-logs`, `-ru-logs`, `-adaptive-logs`, and Experiment U's results **[repo id]** |
| Experiment U code | **[link or folder]** |

> **Content warning.** The attack adapters are deliberately harmful: they make Tiny Aya comply with harmful requests, and the logs contain those answers. They are released for safety research. Raw BeaverTails text, and our Russian translation of it, is not redistributed; the code rebuilds both from the public dataset.

## Experiment R: Antidote vs matched nulls

The attack is a LoRA fine-tune (r = 16, all 7 projections, 3 epochs) on 1,000 SST2 examples mixed with 10, 50 or 100 harmful BeaverTails pairs, with 3 attack seeds. Each attacked adapter is then repaired by every arm below, at pruning ratio α ∈ {0.05, 0.1, 0.2}:

| Arm | What it does |
|---|---|
| A0 | No repair |
| A1 | SafeTune `AntidoteTrainer`, shipped defaults (breaks Tiny Aya: degenerate output) |
| A2 | SafeTune `apply_antidote`, paper-like settings, merged weights |
| A3 | Paper-faithful port of Antidote: Wanda on the LoRA factors, harmful calibration |
| A4a / A4m | Random per-row pruning (adapter / merged) |
| A5a / A5m | Magnitude-only pruning (adapter / merged) |
| A6 | Norm-matched shrink of the update to A3's residual norm |
| A7 | A3 with benign (SST2) calibration |

Two attackers that know about the repair are added:

- a relearning re-attack, which fine-tunes the repaired adapter further on fresh harmful data;
- a mask-aware attacker, which trains through the pruning it anticipates.

A third axis repeats the dose-100 attack with the same harmful pairs machine-translated to Russian.

Metrics:

- StrongREJECT harmful score (finetuned judge), on 150 prompts.
- Refusal rate, also with a perplexity gate so only coherent refusals count.
- XSTest-safe over-refusal (250 prompts).
- SST2 (500) and MMLU-Pro (200) accuracy: exact match of the parsed label or letter. (`evaluate.auditkit_accuracy` tries AuditKIT first, but its `evaluate` call does not match the installed AuditKIT API, so every reported accuracy is the local exact match. The two compute the same quantity.)
- A broken flag for models whose answers mostly do not parse, so degenerate output is not read as a repair.

Every arm reports the mean and min–max over seeds, plus paired-bootstrap contrasts against its null.

[notes/antidote_repro_diff.md](notes/antidote_repro_diff.md) compares the Antidote paper, the authors' code and SafeTune. They differ in what they prune, the default ratio and the calibration data. [notes/adaptive_attacker.md](notes/adaptive_attacker.md) describes both adaptive attackers.

### Reproduce

The same code runs on a local GPU (Windows or Linux) and on Colab, Kaggle or HF Jobs. Hardware differences are handled by profiles, not by editing configs. [notes/setup.md](notes/setup.md) has the full instructions for each platform.

```bash
pip install -r requirements.txt
# local Windows, 16 GB card (vLLM is Linux-only):
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile hf_backend --profile gpu16gb
# Colab / Kaggle T4:
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile t4
# Linux with a 24 GB+ GPU (vLLM):
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml
```

The whole experiment runs unattended with `python -m exp1_antidote.run_queue`. It takes about 31 GPU hours on one RTX 5070 Ti with the transformers backend. [exp1_antidote/README.md](exp1_antidote/README.md) has the full run guide, measured timings and how to read the outputs.

What the queue does:

- it runs, in order: the α sweep, relearning, the SafeTune diagnostic, the Russian attack, doses 10/50, the mask-aware attacker, and a cross-root comparison;
- after the sweep, it passes the α chosen for Antidote to every later job, so all nulls are matched;
- every stage is checkpointed, so an interrupted run resumes where it stopped (attack steps, generation batches, scoring chunks).

Outputs go to `$EXP1_OUT` (default `outputs/`). Per-config results are in `outputs/analysis/<config>/summary.md`, and the cross-experiment table is in `outputs/analysis/compare.md`.

Determinism:

- the data partitions are fixed by `data.data_seed`;
- the attack data order, the random masks and the decoding (greedy) are seeded;
- each result row records the git hash, dtype, backend and hardware profiles.

### Publish

```bash
python -m exp1_antidote.run_grid --config exp1_antidote/configs/alpha_sweep.yaml --stage push   # each output root
python -m exp1_antidote.publish                  # dry run: repos to make public, collections to build
python -m exp1_antidote.publish --yes --extra-models <Experiment U models> --extra-datasets <Experiment U results>
```

The push stage writes a model card or dataset card (with the content warning) into every repo. Pushing never makes a repo public: `hub.private` is `true` until `publish --yes`. Never commit `HF_TOKEN`.

### Layout

```
exp1_antidote/
  configs/        base.yaml + smoke / alpha_sweep / a2_diag / relearn / ru_attack / dose_grid / adaptive (what to run)
    profiles/     hf_backend / gpu16gb / t4 (hardware overrides, applied with --profile)
  common.py       config loading, paths, chat formatting, run ids, crash-safe jsonl io
  grid.py         arm registry and run planning
  data.py         dataset partitions (BeaverTails attack/calib, StrongREJECT, XSTest, SST2, MMLU-Pro)
  attack.py       LoRA harmful fine-tuning (plain PyTorch loop, resumable)
  repair.py       arms A1–A7
  evaluate.py     vLLM / HF generation and scoring
  analyze.py      aggregates, bootstrap contrasts, headroom go/no-go, α recommendation, relearning table, plots
  adaptive.py     mask-aware attacker and relearning re-attack
  translate.py    NLLB translation of the harmful attack slice (attack-language axis)
  compare.py      one table across output roots (English attack, Russian attack, mask-aware attacker)
  hub.py          push adapters and logs to the HF Hub, with cards
  publish.py      make the repos public and build the HF collections
  run_grid.py     stage driver
  run_queue.py    unattended driver for every job after the smoke test, on one GPU
notes/            setup, paper-vs-code diff, adaptive-attacker design
papers/           reference PDFs
```

## Experiment U: erased or hidden?

**[Janhavi: two-paragraph summary, reproduce commands (`run_lambda.sh`, `run_variants.sh`, `run_robust.sh`, `analysis/analyze.py`, `report/build_report.py`), and where the code lives.]**

## Models, data and code we build on

- **Models:**
  - `CohereLabs/tiny-aya-global`;
  - StrongREJECT judge `qylu4156/strongreject-15k-v1` (Gemma-2B);
  - `facebook/nllb-200-distilled-600M`.
- **Code:**
  - [SafeTune](https://github.com/Lexsi-Labs/SafeTune) (Lexsi Labs);
  - [git-disl/Antidote](https://github.com/git-disl/Antidote);
  - [StrongREJECT](https://github.com/dsbowen/strong_reject).
- **Data:**
  - Experiment R: BeaverTails, StrongREJECT, XSTest, SST2, MMLU-Pro.
  - Experiment U: MMLU, MMMLU (Hindi), Wikipedia (retain text) **[Janhavi: complete]**.

## AI assistance

Experiment R's pipeline code, the reproduction notes and drafts of the write-up were produced with Claude (Anthropic), working in Cursor under the authors' direction. The authors chose the research questions and design, ran all experiments, checked the results and take responsibility for the content.
