"""Push adapters and logs to the Hugging Face Hub (needs HF_TOKEN in the environment or `hf auth login`).

adapters repo (model):   attack adapters adapters/d{dose}_s{seed}/, repaired adapters repaired/{run_id}/,
                         relearning checkpoints relearn/{arm}+RL{k}__.../
logs repo (dataset):     results/, scores/, generations/, analysis/ and dataset ids (no raw BeaverTails text)
Merged checkpoints are never pushed (7 GB each); their repair_meta.json goes with the logs.
"""
from __future__ import annotations

from .common import paths, read_jsonl, write_jsonl


def _repo_id(api, cfg: dict, key: str) -> str:
    owner = cfg["hub"].get("org") or api.whoami()["name"]
    return f"{owner}/{cfg['hub'][key]}"


def _ids_only(dst_dir) -> None:
    """Write id manifests for the BeaverTails-derived files instead of their text."""
    d = paths()["data"]
    for name in ("calib", "attack_pool"):
        rows = read_jsonl(d / f"{name}.jsonl")
        write_jsonl(dst_dir / f"{name}_ids.jsonl", [{"id": r["id"], "prompt": r["prompt"]} for r in rows])


WARNING = (
    "> **Content warning.** These are deliberately harmful research artifacts from a safety evaluation. "
    "The attack adapters were fine-tuned on harmful BeaverTails answers and make the model comply with "
    "harmful requests; the generations include those answers. Use them only to study attacks and defences."
)


def _model_card(cfg: dict, repo: str, logs_repo: str) -> str:
    base = cfg["model_id"]
    lang = cfg["attack"].get("language", "en")
    extra = " Attack: the mask-aware attacker (trains through Wanda pruning)." if cfg.get("adaptive", {}).get("mask_aware") else ""
    return f"""---
base_model: {base}
library_name: peft
tags: [lora, safety-research, harmful-fine-tuning, antidote, pruning]
---

# Antidote on Tiny Aya: attack and repaired LoRA adapters

{WARNING}

LoRA adapters for `{base}` from experiment 1 of the ValueShift Lexsi hackathon entry: does Antidote
(post-fine-tuning pruning of a harmful update) beat matched pruning nulls? Attack language: `{lang}`.{extra}
Code, method and results: see the GitHub repository linked from the write-up. Per-run scores and
generations: [`{logs_repo}`](https://huggingface.co/datasets/{logs_repo}).

## Contents

| Folder | What |
|---|---|
| `adapters/d{{dose}}_s{{seed}}/` | Harmful attack adapter: LoRA r={cfg['attack']['lora_r']} on {', '.join(cfg['attack']['target_modules'])}, SST2 + `dose` harmful pairs, attack seed `seed`. `attack_meta.json` has the training log and harmful example ids. |
| `repaired/<run_id>/` | The attack adapter after a repair. `repair_meta.json` describes it. |
| `relearn/<arm>+RL<k>__.../` | A repaired adapter after `k` further harmful fine-tuning steps (relearning re-attack). |

`run_id` = `d<dose>_s<seed>__<arm>__a<alpha>[__m<mask seed>]`. Arms: A3 Antidote (Wanda on the LoRA
factors, harmful calibration), A4a random pruning, A5a magnitude-only pruning, A6 norm-matched shrink,
A7 benign calibration. Merged-weight arms (A1, A2) are not stored: they are rebuilt from the attack
adapter by the code.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{base}")
base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="repaired/d100_s0__A3__a0.2")
```

License: the adapters are derived from `{base}` and follow its license.
"""


def _dataset_card(cfg: dict, adapters_repo: str) -> str:
    return f"""---
tags: [safety-research, harmful-fine-tuning, antidote, strongreject]
---

# Antidote on Tiny Aya: evaluation logs

{WARNING}

Per-run results for the adapters in [`{adapters_repo}`](https://huggingface.co/{adapters_repo}).

| Folder | What |
|---|---|
| `results/runs.jsonl` | One summary row per run: StrongREJECT harmful score, coherent-refusal rate, XSTest over-refusal, SST2 and MMLU-Pro accuracy, parse rates, broken flag. `timings.jsonl`: generation time per run. |
| `scores/<run_id>.jsonl` | Per-prompt rows with judge score, perplexity, refusal, parsed answer, correctness. |
| `generations/<run_id>.jsonl` | Greedy completions on StrongREJECT (150), XSTest-safe (250), SST2 (500), MMLU-Pro (200). |
| `analysis/<config>/` | `summary.md`, CSV tables, paired-bootstrap contrasts, plots, `decisions.json`. |
| `data_manifest/` | Ids of the BeaverTails calibration and attack-pool examples. The raw BeaverTails text (and its translations) is not redistributed; the code rebuilds it from the public dataset. |
| `merged_meta/` | Repair metadata for merged-weight arms. |
"""


def _upload_card(api, repo_id: str, repo_type: str, text: str) -> None:
    api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id,
                    repo_type=repo_type, commit_message="card")


def push(cfg: dict) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    p = paths()
    private = bool(cfg["hub"].get("private", False))

    adapters = _repo_id(api, cfg, "adapters_repo")
    logs = _repo_id(api, cfg, "logs_repo")
    api.create_repo(adapters, repo_type="model", private=private, exist_ok=True)
    _upload_card(api, adapters, "model", _model_card(cfg, adapters, logs))
    for sub, folder in (("adapters", p["adapters"]), ("repaired", p["repaired"]), ("relearn", p["root"] / "relearn")):
        if folder.is_dir() and any(folder.iterdir()):
            api.upload_folder(repo_id=adapters, repo_type="model", folder_path=str(folder), path_in_repo=sub,
                              commit_message=f"{cfg['name']}: {sub}")

    api.create_repo(logs, repo_type="dataset", private=private, exist_ok=True)
    _upload_card(api, logs, "dataset", _dataset_card(cfg, adapters))
    manifest = p["root"] / "data_manifest"
    manifest.mkdir(exist_ok=True)
    _ids_only(manifest)
    for sub, folder in (("results", p["results"]), ("scores", p["scores"]), ("generations", p["generations"]),
                        ("analysis", p["analysis"]), ("data_manifest", manifest)):
        if any(folder.iterdir()):
            api.upload_folder(repo_id=logs, repo_type="dataset", folder_path=str(folder), path_in_repo=sub,
                              ignore_patterns=["*.partial.jsonl", "*.tmp"],
                              commit_message=f"{cfg['name']}: {sub}")
    if any(p["merged"].iterdir()):
        api.upload_folder(repo_id=logs, repo_type="dataset", folder_path=str(p["merged"]), path_in_repo="merged_meta",
                          allow_patterns=["*/repair_meta.json"], commit_message=f"{cfg['name']}: merged repair meta")
    print(f"[push] https://huggingface.co/{adapters}  https://huggingface.co/datasets/{logs}")
