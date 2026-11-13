# Lexsi hackathon — weight-space safety repairs on Tiny Aya

> **Experiment 2 — Erased or Hidden?** RMU unlearning on Tiny Aya across English, Hindi and Marathi, attacked with relearning in
> other languages. See [`exp2_erased_or_hidden/`](exp2_erased_or_hidden/README.md) and its
> [report](exp2_erased_or_hidden/report/erased_or_hidden_report.pdf).

Experiment 1: **does Antidote beat generic pruning?** We take `CohereLabs/tiny-aya-global` and attack it with a LoRA fine-tune: 0, 10, 50 or 100 harmful BeaverTails pairs mixed into 1000 SST2 examples, 3 seeds. Then we compare Antidote against matched nulls under Lee et al.'s five checks.

| Arm | What it does |
|---|---|
| A0 | No repair |
| A1 | SafeTune `AntidoteTrainer`, shipped defaults |
| A2 | SafeTune `apply_antidote`, paper-like settings, merged weights |
| A3 | Paper-faithful port: Wanda on the LoRA factors, harmful calibration |
| A4a / A4m | Random per-row pruning (adapter / merged) |
| A5a / A5m | Magnitude-only pruning (adapter / merged) |
| A6 | Norm-matched shrink of the update to A3's residual norm |
| A7 | A3 with benign (SST2) calibration |

Metrics:

- StrongREJECT harmful score (finetuned judge).
- Refusal rate, also with a perplexity gate so only coherent refusals count.
- XSTest-safe over-refusal.
- SST2 and MMLU-Pro accuracy, computed through AuditKIT.

Every arm reports mean and min–max over seeds, plus paired-bootstrap contrasts against its null.

## Layout

```
exp1_antidote/
  configs/        base.yaml + smoke / alpha_sweep / dose_grid / adaptive (what to run)
    profiles/     hf_backend / gpu16gb / t4 (hardware overrides, applied with --profile)
  common.py       config loading, paths, chat formatting, run ids
  grid.py         arm registry and run planning
  data.py         dataset partitions (BeaverTails attack/calib, StrongREJECT, XSTest, SST2, MMLU-Pro)
  attack.py       LoRA harmful fine-tuning (plain PyTorch loop)
  repair.py       arms A1–A7
  evaluate.py     vLLM / HF generation and scoring
  analyze.py      aggregates, bootstrap contrasts, headroom go/no-go, α recommendation, plots
  adaptive.py     mask-aware attacker and relearning re-attack
  hub.py          push adapters and logs to the HF Hub
  run_grid.py     stage driver
notes/            setup, paper-vs-code diff, adaptive-attacker design
papers/           reference PDFs
```

## Run

The same code runs on a local GPU (Windows or Linux) and on Colab, Kaggle or HF Jobs. Hardware differences are handled by profiles, not by editing configs. See [notes/setup.md](notes/setup.md) for each platform. In short:

```bash
pip install -r requirements.txt
# local Windows, 16 GB card (vLLM is Linux-only):
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile hf_backend --profile gpu16gb
# Colab / Kaggle T4:
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile t4
# Linux with a 24 GB+ GPU (vLLM):
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml
```

Outputs go to `$EXP1_OUT` (default `outputs/`). Then read `outputs/analysis/smoke/summary.md` for the GO/NO-GO headroom check.

Code lives on GitHub. Adapters and logs go to the Hugging Face Hub through the `push` stage, which `--stage all` runs only when `hub.org` is set. Never commit `HF_TOKEN`.
