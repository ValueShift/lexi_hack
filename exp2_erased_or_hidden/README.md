# Experiment 2 — Erased or Hidden? Unlearning on Tiny Aya across languages

**Question.** When RMU unlearning makes `CohereLabs/tiny-aya-global` fail high-school chemistry, is the knowledge gone or only
hidden? Does fine-tuning in *another* language bring it back, and does that depend on which layer the unlearning edited?

**Report:** [`report/erased_or_hidden_report.pdf`](report/erased_or_hidden_report.pdf) · **All numbers:** [`analysis/tables.md`](analysis/tables.md)

## Findings

| | Result | Strength |
|---|---|---|
| 1 | **Hidden, not erased.** After English-only RMU unlearning, a small LoRA fine-tune restores held-out forget answers in every arm. Pooled over both test halves: layer 27 +0.27 (95% CI +0.12 to +0.42), layer 27 deeper +0.43 (95% CI +0.27 to +0.59), layer 30 +0.33 (95% CI +0.19 to +0.47). | Solid |
| 2 | **Any language, any topic.** Hindi or Marathi fine-tuning restores *English* answers as much as English fine-tuning (layer 27 deeper: +0.53 / +0.52 / +0.56). A fine-tune on **unrelated** Marathi general-knowledge questions (no chemistry) restores the most: +0.69, +0.62, and +0.30 even for the shallow layer-27 arm (CI +0.11 to +0.50). | Solid |
| 3 | **Depth, not layer.** Across 4 arms × 3 languages, relearning excess ≈ 0.57 × accuracy erased +0.09 (r = 0.86); layers 27 and 30 sit -0.02 / +0.02 off the line. At matched English depth (A2s vs A1c) the English difference is -0.03 (95% CI -0.08 to +0.03). | Solid |
| 4 | **Original hypothesis not supported.** Unlearning at the (supposedly) more shared layer is not more durable; its apparent advantage in the first runs came from erasing less. Compare durability at matched erasure depth, per language. | Solid |
| 5 | At matched depth both layers cause similar collateral damage (biology/physics 0.40 → 0.21 vs 0.17). | Moderate |
| 6 | Unlearned models produce degenerate text on chemistry; after Marathi fine-tuning they answer fluently again on questions never trained on. | Illustrative (8 questions) |
| 7 | A linear probe reads the answer at layers 23–24 (≈0.6 vs 0.25 chance) identically in every arm; it falls to chance only after the edited layers. | Weak |

**Correction to the design.** Layer 27 was chosen as the most cross-lingually similar layer by linear CKA, but the CKA curve is
flat (0.840–0.845) from layer 7 to 34. The comparison is *layer 27 vs layer 30*, not "shared vs late".

Robustness checks included above: a layer-27 arm erased as deeply as layer 30 (A2s), Hindi relearning, an unrelated-domain
Marathi fine-tune (200 general-knowledge questions), and swapped train/test halves for A2, A2s and A1c.

## Design in one paragraph

Forget set: the 59 MMLU high-school chemistry questions the model answers correctly in English and Hindi; control set: the 105 it
gets wrong in both; both split into halves A and B; attacks train on one half and are tested on the other (both directions for A2, A2s, A1c). Languages: English, Hindi (MMMLU),
Marathi (NLLB-200 translation). Unlearning: RMU on English text only, training the MLP down-projections of layers ℓ−2…ℓ; an arm is
*matched* when English forget accuracy ≤ 0.35 and history/law accuracy drops ≤ 5 points. Attacks: LoRA r=16, 3 epochs, lr 1e-4 and
3e-4, 2 seeds, strongest lr kept per seed. **Relearning excess** = gain on held-out forget questions − gain on held-out control
questions under the same kind of attack. Scoring: multiple-choice letter scores (no generation, no judge), plus free-text samples.

| Arm | What it is |
|---|---|
| A0 | Original model |
| A2 | RMU at layer 27 (matched) |
| A1 | RMU at layer 30, strong (erases, 20 points collateral) |
| A1c | RMU at layer 30, medium (erases, 9 points collateral) — closest layer-30 match |
| A1b, A1d | RMU at layer 30, gentle (half-erase, no collateral) |
| A2s | RMU at layer 27 trained longer, erasing English as deeply as A1c (0.17) with 5 points collateral |

## Files

| Path | What |
|---|---|
| `erased_or_hidden.py` | The whole pipeline (stages: data, baseline, layers, unlearn, eval, relearn, probe, report, samples). Also a Colab notebook source (jupytext) and a `uv` script for HF Jobs. |
| `erased_or_hidden_colab.ipynb` | Same code as a Colab notebook. Put the HF token in a Colab secret named `HF_TOKEN`. |
| `run_lambda.sh` | One-command setup + run on a Lambda GPU (private `uv` env, pinned PyTorch 2.10/CUDA 12.8, preflight checks, resume from a Drive zip or HF dataset). |
| `run_variants.sh` | Tries in-between layer-30 settings, keeps the matched one, writes generated samples. |
| `run_robust.sh` | Robustness checks (A2s, Hindi attack, unrelated-domain attack, swapped halves). |
| `eoh_hf_launcher.ipynb` | Launches the pipeline as a Hugging Face Job from a CPU-only Colab. |
| `results/` | Outputs of the runs: per-question correctness for every arm and attack, unlearning metadata, probe, samples. No model weights. |
| `analysis/analyze.py` | Recomputes every table, statistic and figure from `results/` (CPU only). |
| `report/build_report.py` | Builds the PDF report from the analysis outputs. |

## Reproduce

```bash
# full pipeline on one GPU (Lambda / any Linux box with an NVIDIA GPU)
RESULTS_REPO=yourname/eoh-results bash run_lambda.sh      # ~1.5 h on an A100 from scratch
RESULTS_REPO=yourname/eoh-results bash run_variants.sh    # layer-30 settings + samples
RESULTS_REPO=yourname/eoh-results bash run_robust.sh      # robustness checks

# analysis and report from the saved results (CPU)
python analysis/analyze.py --results results --out analysis
python report/build_report.py                              # needs playwright + chromium
```

Tiny Aya is gated: accept its licence on Hugging Face first. The results dataset is always created private.
CPU smoke test with no downloads: `python erased_or_hidden.py --tiny --stages all`.

## Limitations

59 forget questions; one model; one unlearning method (RMU); two seeds that vary only the attack; attack types differ in size
(29–30, 100, 200 training questions);
machine-translated Marathi on which the model is near chance; chemistry symbols are shared across languages; layer 30 never
reached matched erasure (A1c is the closest); only two late layers compared, and the CKA layer search did not single out any layer.
