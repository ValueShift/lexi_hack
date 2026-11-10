"""Arm registry and run planning shared by every stage."""
from __future__ import annotations

from pathlib import Path

from .common import Run, attack_dir, paths

# space: where the arm acts.
#   adapter = edits the attack LoRA factors, output is a LoRA adapter (servable as multi-LoRA)
#   merged  = merges the adapter into the base and edits base Linear weights, output is a full model
ARMS: dict[str, dict] = {
    "A0":  {"space": "adapter", "alpha": False, "random": False, "desc": "no repair"},
    "A1":  {"space": "merged",  "alpha": False, "random": False, "desc": "SafeTune AntidoteTrainer, shipped defaults"},
    "A2":  {"space": "merged",  "alpha": True,  "random": False, "desc": "SafeTune apply_antidote, paper-like settings"},
    "A3":  {"space": "adapter", "alpha": True,  "random": False, "desc": "paper-faithful port: Wanda on LoRA factors, harmful calib"},
    "A4a": {"space": "adapter", "alpha": True,  "random": True,  "desc": "random per-row pruning of LoRA factors (matches A3)"},
    "A4m": {"space": "merged",  "alpha": True,  "random": True,  "desc": "random per-row pruning of merged weights (matches A2)"},
    "A5a": {"space": "adapter", "alpha": True,  "random": False, "desc": "magnitude-only per-row pruning of LoRA factors"},
    "A5m": {"space": "merged",  "alpha": True,  "random": False, "desc": "magnitude-only per-row pruning of merged weights"},
    "A6":  {"space": "adapter", "alpha": True,  "random": False, "desc": "norm-matched shrink of the update to A3's residual norm"},
    "A7":  {"space": "adapter", "alpha": True,  "random": False, "desc": "A3 with benign (SST2) calibration"},
}
A1_PRUNE_FRACTION = 0.005  # SafeTune AntidoteTrainer default


def plan_runs(cfg: dict, include_clean: bool = True) -> list[Run]:
    rcfg = cfg["repair"]
    runs: list[Run] = [Run("clean")] if include_clean else []
    for arm in rcfg["arms"]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
    for dose in cfg["attack"]["doses"]:
        for seed in cfg["attack"]["seeds"]:
            for arm in rcfg["arms"]:
                spec = ARMS[arm]
                alphas = [None]
                if spec["alpha"]:
                    per_arm = (rcfg.get("alpha_per_arm") or {}).get(arm)
                    alphas = [per_arm] if per_arm is not None else list(rcfg["alphas"])
                mask_seeds = list(rcfg["mask_seeds"]) if spec["random"] else [None]
                for a in alphas:
                    for m in mask_seeds:
                        runs.append(Run(arm, dose, seed, a, m))
    seen, out = set(), []
    for r in runs:
        if r.run_id not in seen:
            seen.add(r.run_id)
            out.append(r)
    return out


def artifact(cfg: dict, run: Run) -> tuple[str, Path | str]:
    """(kind, location) of the model a run evaluates. kind in {base, adapter, merged}."""
    if run.arm == "clean":
        return "base", cfg["model_id"]
    if run.arm == "A0":
        return "adapter", attack_dir(run.dose, run.seed)
    if ARMS[run.arm]["space"] == "adapter":
        return "adapter", paths()["repaired"] / run.run_id
    return "merged", paths()["merged"] / run.run_id


def generations_path(run: Run) -> Path:
    return paths()["generations"] / f"{run.run_id}.jsonl"


def scores_path(run: Run) -> Path:
    return paths()["scores"] / f"{run.run_id}.jsonl"
