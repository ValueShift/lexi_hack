# Setup

Experiment 1 runs on three kinds of machine with the same code and configs. What differs is set by **hardware profiles** (`--profile`) and two environment variables:

| Where | Profiles | Generation backend |
|---|---|---|
| Local, native Windows, 16 GB card | `--profile hf_backend --profile gpu16gb` | transformers (vLLM is Linux-only) |
| Local, native Windows, 24 GB+ | `--profile hf_backend` | transformers |
| Local Linux / WSL2, 24 GB+ | none | vLLM |
| Colab / Kaggle T4 (16 GB, no bf16) | `--profile t4` (add `--profile hf_backend` if vLLM fails) | vLLM or transformers |
| Colab L4 / A100, HF Jobs A10G+ | none | vLLM |

| Variable | Meaning | Default |
|---|---|---|
| `EXP1_OUT` | Output root: data, adapters, generations, scores, results, analysis | `outputs/` in the repo (gitignored) |
| `EXP1_SCRATCH` | Where temporary merged checkpoints (~7 GB each) go | same as `EXP1_OUT` |

## One-time, in the browser (both teammates)

1. Accept the licence on [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) and [CohereLabs/tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base). The models are gated: without this, downloads fail with 401.
2. Accept the licence on [google/gemma-2b](https://huggingface.co/google/gemma-2b). The StrongREJECT finetuned judge (`qylu4156/strongreject-15k-v1`) is a LoRA on top of it.
3. Create a team organization (huggingface.co, then New, then Organization) and invite your teammate. Everything we publish goes under it: `<org>/exp1-antidote-adapters`, `<org>/exp1-antidote-logs`.
4. Create a token (Settings, Access Tokens) with **write** access to the org. Never commit it; `.gitignore` blocks `.env` and `hf_token.txt`, but nothing in the code reads `.env`. How to provide it is described per platform below.
5. Join the hackathon Discord to get the $20 credits (needed for HF Jobs).

## Where things live

- **The git repo holds code and notes only.** Outputs, the HF model cache (`~/.cache/huggingface`), checkpoints, and clones of SafeTune or TamperBench stay out of it.
- **Disk.** The HF cache needs about 7 GB for Tiny Aya and about 5 GB for Gemma-2B. Adapters are small. Merged-space arms (A1, A2, A4m, A5m) need one ~7 GB checkpoint at a time: with `eval.keep_merged: false` (the default) each is built during `generate` and deleted right after.
- **Do not mix hardware in one output root.** A T4 (float16) grid and a bf16 grid give slightly different numbers; keep them under different `EXP1_OUT` directories. Every row in `results/runs.jsonl` records `dtype`, `backend` and `profiles`.

## Option A: local GPU, native Windows

Prerequisites: an NVIDIA driver, Python 3.10+, and `git` on PATH (two requirements install from GitHub).

```powershell
cd C:\path\to\lexi_hack
python -m venv venv
.\venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# RTX 50xx (Blackwell) needs a CUDA 12.8 build of torch, installed before the rest:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch, peft, safetune, strong_reject, auditkit; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
hf auth login                        # paste the token; stored in your user profile, not in the repo
```

If the import check fails, `pip install -r requirements.txt` did not finish (usually a missing `git`); rerun it and read the first error.

Every new PowerShell window needs the venv and the variables again:

```powershell
cd C:\path\to\lexi_hack
.\venv\Scripts\Activate.ps1
$env:EXP1_OUT = "D:\lexi_outputs"    # optional
$env:HF_HUB_DISABLE_XET = "1"        # optional, see troubleshooting
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile hf_backend --profile gpu16gb --stage all
```

Run the pipeline from a normal (not administrator) window.

### Windows troubleshooting

- **Model download stalls** ("Reconstructing" stops moving, speed in kB/s): Hugging Face's Xet downloader sometimes hangs on Windows. Set `$env:HF_HUB_DISABLE_XET = "1"` and restart; finished files and chunks stay cached. Downloading up front makes this easier to see: `hf download CohereLabs/tiny-aya-global`, `hf download google/gemma-2b`, `hf download qylu4156/strongreject-15k-v1`.
- **Ctrl+C does nothing** during a download: try Ctrl+Break, or close the window. If `python.exe` survives, kill it with `taskkill /PID <pid> /T /F` (from an administrator window if you get "Access is denied"). Find the right process with `Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CommandLine`.
- **Symlink warning from `huggingface_hub`**: harmless; the cache uses more disk. Enable Windows Developer Mode to remove it.
- **"Redirects are currently not supported in Windows or MacOs"**: a harmless PyTorch notice.

## Option B: local GPU, Linux or WSL2

Same as Option A with bash syntax (`source venv/bin/activate`, `export EXP1_OUT=...`). `requirements.txt` installs vLLM here, so drop `--profile hf_backend` and generation is several times faster. Keep `--profile gpu16gb` on a 16 GB card.

## Option C: Colab or Kaggle

Free tiers give a T4 (16 GB, no bf16), so use `--profile t4`. Sessions are time-limited and their disks are wiped when the session ends, so plan for resuming (below).

Settings: on Colab, choose Runtime, Change runtime type, then T4 GPU (or L4/A100 on a paid tier). On Kaggle, choose Accelerator: GPU T4 x2 and turn Internet on (requires a phone-verified account). Store the token as a secret named `HF_TOKEN`: on Colab, use the key icon in the left sidebar and enable "Notebook access"; on Kaggle, use Add-ons, then Secrets.

**Cell 1: authenticate, choose directories, clone and install.** Use the block for your platform, then the shared block. Setting `os.environ` in Python is what makes the values visible to later `!` commands; `!export` does not persist between cells.

```python
# Colab
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["EXP1_OUT"] = "/content/drive/MyDrive/lexi_outputs"   # survives disconnects
os.environ["EXP1_SCRATCH"] = "/content/scratch"                  # merged checkpoints stay off Drive
```

```python
# Kaggle
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["EXP1_OUT"] = "/kaggle/working/outputs"   # 20 GB limit, kept only in saved versions
os.environ["EXP1_SCRATCH"] = "/tmp/exp1_scratch"     # merged checkpoints outside the 20 GB limit
```

```python
# both
!git clone https://github.com/<you>/<repo>.git lexi_hack
%cd lexi_hack
!pip install -q -r requirements.txt
```

If pip replaced the preinstalled torch (vLLM pins its own), restart the runtime/session once, then rerun the platform block and `%cd lexi_hack` (not the clone or the install).

**Cell 2: run.**

```python
!python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile t4 --stage all
```

If vLLM fails to start or cannot serve the Tiny Aya LoRAs, add `--profile hf_backend` and rerun; finished items are skipped. On T4, check that the `[attack] ... loss` lines are finite numbers: float16 training can overflow, and a `nan` loss means the attack adapter is broken.

### Resuming in a new session

Stages skip finished items, so resuming means getting the earlier outputs back into `EXP1_OUT`:

- **Colab with Drive:** nothing to do; rerun the same cells.
- **Kaggle (or Colab without Drive):** push at the end of each session and restore at the start of the next. The repos mirror the output layout, so downloading them into `EXP1_OUT` puts every file back in place. `data/` is not uploaded (only an id manifest), but `--stage data` rebuilds it identically from `data.data_seed`.

```python
# end of session (set hub.org in configs/base.yaml, or it uploads under your username)
!python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --profile t4 --stage push
# start of next session, after cell 1
!hf download <org>/exp1-antidote-adapters --local-dir {os.environ["EXP1_OUT"]}
!hf download <org>/exp1-antidote-logs --repo-type dataset --local-dir {os.environ["EXP1_OUT"]}
```

Set `hub.private: true` while you work, since the attack adapters are deliberately harmful; make the repos public at submission time in the HF web interface.

### HF Jobs

Run the same commands in a job with `--secrets HF_TOKEN`. On an A10G or bigger, no profile is needed. Push at the end of the job, because the job's disk is discarded.

## Running experiment 1

The commands below omit `--profile`; add the ones for your machine from the table at the top. Every stage is idempotent: re-running skips finished items, so an interruption only loses the current item.

```bash
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage all
# or stage by stage:
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage data
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage attack
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage repair
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage generate
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage score
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage analyze
python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage push
```

`--stage all` runs `push` only when `hub.org` is set in the config (or skip it anyway with `--no-push`). Running `--stage push` explicitly uploads under your own username if `hub.org` is empty.

| Config | What it runs | Runs |
|---|---|---|
| `smoke.yaml` | dose 100, seed 0, arms A0/A1/A3/A4a at α=0.1 | 5 |
| `alpha_sweep.yaml` | dose 100, 3 seeds, A0–A6 at α ∈ {0.01, 0.05, 0.1, 0.2} | 139 |
| `dose_grid.yaml` | doses 0/10/50/100, 3 seeds, A0–A7 at one α per arm | 169 (shares runs with the sweep) |
| `adaptive.yaml` | mask-aware attacker at dose 100, plus the relearning re-attack | see `notes/adaptive_attacker.md` |

Each run generates 1,100 completions (StrongREJECT 150, XSTest-safe 250, SST2 500, MMLU-Pro 200). With the transformers backend that is the dominant cost, so expect the sweep and the grid to take many hours each on one GPU, spread over several sessions on Colab or Kaggle; vLLM is much faster.

Order:

1. **Smoke.** Open `<EXP1_OUT>/analysis/smoke/summary.md`. The headroom line says GO or NO-GO: the attacked model has to be at least 25 harmful-score points above the clean model, or there is nothing to repair. If the harmful-score column shows `–`, the StrongREJECT judge did not run; the score stage's log says why (usually `strong_reject` is not installed or `google/gemma-2b` is not accepted).
2. **Alpha sweep.** The α recommendation is in `analysis/alpha_sweep/decisions.json`. Copy it into `dose_grid.yaml` under `repair.alpha_per_arm`.
3. **Dose grid.** All three configs share one output root, so the sweep reuses the smoke run's adapters, and the grid reuses the sweep's.
4. **Adaptive attackers** (`configs/adaptive.yaml`; see `notes/adaptive_attacker.md`). The mask-aware attack needs its own `EXP1_OUT`; the relearning re-attack (`python -m exp1_antidote.adaptive --config exp1_antidote/configs/adaptive.yaml --mode relearn`, which also takes `--profile`) uses the standard one.

Outputs, under `EXP1_OUT`:

- `results/runs.jsonl` holds one row per run.
- `scores/` and `generations/` hold per-prompt rows.
- `analysis/<config>/` holds `summary.md`, CSV files, plots and `decisions.json`.

Before a big run on Linux, smoke-check the vLLM path. If Tiny Aya LoRA serving fails in vLLM, add `--profile hf_backend`. It is slower but needs no other changes.

## Pushing to Hugging Face

The `push` stage uploads:

- `adapters/` and `repaired/` as a model repo, `<org>/<hub.adapters_repo>`.
- `results/`, `scores/`, `generations/`, `analysis/`, the repair metadata of the merged arms, and an id manifest of the BeaverTails calibration and attack pool, as a dataset repo, `<org>/<hub.logs_repo>`. Merged checkpoints are never uploaded.

Set `hub.org` in `configs/base.yaml` first. At the end, group the repos into public collections in the HF web interface (your profile, then Collections, then New). The submission needs one model collection and one dataset collection.
