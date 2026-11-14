"""Repair arms A1-A7 applied to each attacked adapter (see grid.ARMS and notes/antidote_repro_diff.md).

Adapter-space arms (A3, A4a, A5a, A6, A7) edit the attack LoRA factors and save a new
adapter. Merged-space arms (A1, A2, A4m, A5m) merge the adapter into the base model and
edit base Linear weights, saving a full checkpoint (under $EXP1_SCRATCH if set).
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .common import Run, attack_dir, chat_full, free_gpu, git_hash, load_model, load_tokenizer
from .data import benign_calibration, load
from .grid import A1_PRUNE_FRACTION, ARMS, artifact, generations_path


# ── per-row masks ────────────────────────────────────────────────────────────

def k_gitdisl(n_in: int, alpha: float) -> int:
    """git-disl: zero-mask the lowest int(n_in*(1-dense_ratio)) per row, prune the rest."""
    return max(1, n_in - int(n_in * (1.0 - alpha)))


def k_safetune(n_in: int, alpha: float) -> int:
    return min(n_in - 1, max(1, int(round(n_in * alpha))))


def row_topk_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.topk(score, k, dim=1, largest=True).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def score_matrix(weight: torch.Tensor, mode: str, act_sq: torch.Tensor | None, gen: torch.Generator | None):
    if mode == "wanda":
        return weight.abs().float() * torch.sqrt(act_sq.clamp_min(0)).to(weight.device).unsqueeze(0)
    if mode == "magnitude":
        return weight.abs().float()
    if mode == "random":
        return torch.rand(weight.shape, generator=gen).to(weight.device)
    raise ValueError(mode)


# ── activation statistics ────────────────────────────────────────────────────

def lora_linears(model) -> dict[str, nn.Linear]:
    return {
        n: m for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and (".lora_A." in n or ".lora_B." in n)
    }


@torch.no_grad()
def collect_act_sq(model, tok, pairs: list[dict], modules: dict[str, nn.Module], max_len: int, bs: int):
    """Sum over all non-pad calibration tokens of squared inputs, per input column.
    Row ranking is identical to git-disl's per-sample-averaged scaler_row."""
    acc: dict[str, torch.Tensor] = {}
    holder: dict[str, torch.Tensor | None] = {"mask": None}

    def make_hook(name):
        def hook(_mod, inp, _out):
            x = inp[0].detach().float()
            m = holder["mask"]
            if m is not None and x.dim() == 3:
                x = x * m.to(x.device, x.dtype).unsqueeze(-1)
            sq = (x * x).reshape(-1, x.size(-1)).sum(0)
            acc[name] = acc[name] + sq if name in acc else sq
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
    texts = [chat_full(tok, p["prompt"], p["response"]) for p in pairs]
    device = next(model.parameters()).device
    tok.padding_side = "right"
    model.eval()
    try:
        for i in range(0, len(texts), bs):
            enc = tok(texts[i : i + bs], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len, add_special_tokens=False).to(device)
            holder["mask"] = enc["attention_mask"]
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        for h in handles:
            h.remove()
        holder["mask"] = None
    return acc


# ── adapter-space arms ───────────────────────────────────────────────────────

def _prune_lora(modules, mode, alpha, act_sq=None, mask_seed=None) -> dict:
    gen = torch.Generator().manual_seed(mask_seed) if mask_seed is not None else None
    pruned = total = 0
    with torch.no_grad():
        for name, mod in sorted(modules.items()):
            W = mod.weight.data
            k = k_gitdisl(W.size(1), alpha)
            mask = row_topk_mask(score_matrix(W, mode, act_sq.get(name) if act_sq else None, gen), k)
            W.mul_((~mask).to(W.dtype))
            pruned += int(mask.sum())
            total += mask.numel()
    return {"pruned": pruned, "total": total, "fraction": pruned / max(1, total)}


def _lora_pairs(modules) -> dict[str, tuple[nn.Linear, nn.Linear]]:
    pairs = {}
    for name in modules:
        if ".lora_A." in name:
            prefix, adapter = name.split(".lora_A.")
            b = f"{prefix}.lora_B.{adapter}"
            if b in modules:
                pairs[prefix] = (modules[name], modules[b])
    return pairs


def _update_norms(modules) -> dict[str, float]:
    with torch.no_grad():
        return {
            p: float(torch.linalg.matrix_norm(B.weight.float() @ A.weight.float()))
            for p, (A, B) in _lora_pairs(modules).items()
        }


def _shrink_to(modules, target_norms: dict[str, float]) -> dict:
    """Scale each module's B so ||B A||_F equals the target (A3's residual norm)."""
    ratios = []
    with torch.no_grad():
        for p, (A, B) in _lora_pairs(modules).items():
            cur = float(torch.linalg.matrix_norm(B.weight.float() @ A.weight.float()))
            if cur > 0:
                r = target_norms[p] / cur
                B.weight.data.mul_(r)
                ratios.append(r)
    return {"mean_ratio": sum(ratios) / max(1, len(ratios)), "n_modules": len(ratios)}


def repair_adapter_cell(cfg: dict, dose: int, seed: int, runs: list[Run], force: bool = False) -> None:
    from peft import PeftModel

    todo = [r for r in runs if force or not _done(cfg, r)]
    if not todo:
        return
    rcfg = cfg["repair"]
    tok = load_tokenizer(cfg["model_id"])
    base = load_model(cfg["model_id"], cfg["dtype"])
    model = PeftModel.from_pretrained(base, str(attack_dir(dose, seed)), is_trainable=False)
    model.eval()
    modules = lora_linears(model)
    orig = {n: m.weight.data.clone() for n, m in modules.items()}
    orig_norms = _update_norms(modules)

    need_harm = any(r.arm in ("A3", "A6") for r in todo)
    need_benign = any(r.arm == "A7" for r in todo)
    harm_sq = benign_sq = None
    if need_harm:
        harm_sq = collect_act_sq(model, tok, load("calib"), modules, rcfg["calib_max_len"], rcfg["calib_batch_size"])
    if need_benign:
        n = cfg["data"]["n_calib"]
        benign_sq = collect_act_sq(model, tok, benign_calibration(cfg, seed, n), modules,
                                   rcfg["calib_max_len"], rcfg["calib_batch_size"])

    def restore():
        with torch.no_grad():
            for n, m in modules.items():
                m.weight.data.copy_(orig[n])

    for run in todo:
        restore()
        if run.arm == "A3":
            info = _prune_lora(modules, "wanda", run.alpha, harm_sq)
        elif run.arm == "A4a":
            info = _prune_lora(modules, "random", run.alpha, mask_seed=run.mask_seed)
        elif run.arm == "A5a":
            info = _prune_lora(modules, "magnitude", run.alpha)
        elif run.arm == "A7":
            info = _prune_lora(modules, "wanda", run.alpha, benign_sq)
        elif run.arm == "A6":
            _prune_lora(modules, "wanda", run.alpha, harm_sq)
            target = _update_norms(modules)
            restore()
            info = _shrink_to(modules, target)
        else:
            raise ValueError(run.arm)
        info["residual_norm_fraction"] = _residual_fraction(_update_norms(modules), orig_norms)
        _, out = artifact(cfg, run)
        model.save_pretrained(str(out))
        _write_meta(out, run, info)
        print(f"[repair] {run.run_id}: {info}")
    del model, base
    free_gpu()


def _residual_fraction(norms: dict[str, float], orig: dict[str, float]) -> float:
    num = sum(v * v for v in norms.values()) ** 0.5
    den = sum(v * v for v in orig.values()) ** 0.5
    return num / den if den else 0.0


# ── merged-space arms ────────────────────────────────────────────────────────

def _prune_merged(model, targets: Iterable[str], mode: str, alpha: float, mask_seed=None) -> dict:
    gen = torch.Generator().manual_seed(mask_seed) if mask_seed is not None else None
    pruned = total = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or not any(t in name for t in targets):
                continue
            W = mod.weight.data
            mask = row_topk_mask(score_matrix(W, mode, None, gen), k_safetune(W.size(1), alpha))
            W.mul_((~mask).to(W.dtype))
            pruned += int(mask.sum())
            total += mask.numel()
    return {"pruned": pruned, "total": total, "fraction": pruned / max(1, total)}


def _safetune_a1(model, tok, calib: list[dict]) -> dict:
    prompts = [c["prompt"] for c in calib]
    try:
        from safetune.runner import recover

        recover.AntidoteTrainer(model, harmful_prompts=prompts, tokenizer=tok).apply()
        via = "safetune.runner.recover.AntidoteTrainer"
    except Exception as e:  # trainer wrapper needs runtime config we may not have
        print(f"[repair] AntidoteTrainer failed ({e!r}); calling apply_antidote with its defaults")
        import safetune.recover as R

        R.apply_antidote(model, prune_fraction=A1_PRUNE_FRACTION, harmful_prompts=prompts,
                         tokenizer=tok, max_samples=64)
        via = "safetune.recover.apply_antidote(trainer defaults)"
    return {"via": via, "prune_fraction": A1_PRUNE_FRACTION, "max_samples": 64,
            "targets": ["o_proj", "down_proj"], "calibration": "prompts only"}


def _safetune_a2(model, tok, calib: list[dict], alpha: float, targets: list[str]) -> dict:
    import safetune.recover as R

    R.apply_antidote(model, prune_fraction=alpha, target_modules=targets,
                     dataset=[{"prompt": c["prompt"], "response": c["response"]} for c in calib],
                     tokenizer=tok, max_samples=len(calib))
    return {"via": "safetune.recover.apply_antidote", "prune_fraction": alpha,
            "targets": targets, "calibration": f"{len(calib)} prompt+answer pairs"}


def repair_merged_run(cfg: dict, run: Run, force: bool = False) -> None:
    from peft import PeftModel

    if not force and _done(cfg, run):
        return
    tok = load_tokenizer(cfg["model_id"])
    base = load_model(cfg["model_id"], cfg["dtype"])
    model = PeftModel.from_pretrained(base, str(attack_dir(run.dose, run.seed))).merge_and_unload()
    model.eval()
    targets = list(cfg["attack"]["target_modules"])
    calib = load("calib")
    if run.arm == "A1":
        info = _safetune_a1(model, tok, calib)
    elif run.arm == "A2":
        info = _safetune_a2(model, tok, calib, run.alpha, targets)
    elif run.arm == "A4m":
        info = _prune_merged(model, targets, "random", run.alpha, run.mask_seed)
    elif run.arm == "A5m":
        info = _prune_merged(model, targets, "magnitude", run.alpha)
    else:
        raise ValueError(run.arm)
    _, out = artifact(cfg, run)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    _write_meta(out, run, info)
    (Path(out) / WEIGHTS_DONE).write_text("ok", encoding="utf-8")
    print(f"[repair] {run.run_id}: {info}")
    del model, base
    free_gpu()


# ── bookkeeping ──────────────────────────────────────────────────────────────

def _done(cfg: dict, run: Run) -> bool:
    if generations_path(run).exists():
        return True
    _, out = artifact(cfg, run)
    return (Path(out) / "repair_meta.json").exists()


def _write_meta(out: Path, run: Run, info: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**run.as_dict(), "desc": ARMS[run.arm]["desc"], "info": info, "git": git_hash()}
    (out / "repair_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


WEIGHTS_DONE = "_weights_complete"  # written after the last shard; config.json alone can predate a crash


def _has_weights(out: Path | str) -> bool:
    return (Path(out) / WEIGHTS_DONE).exists()


def ensure_merged(cfg: dict, run: Run) -> None:
    """Build a merged-space checkpoint right before generation if it is not on disk."""
    _, out = artifact(cfg, run)
    if not _has_weights(out):
        repair_merged_run(cfg, run, force=True)


def remove_merged(cfg: dict, run: Run) -> None:
    kind, out = artifact(cfg, run)
    if kind == "merged" and Path(out).is_dir() and any(p.name != "repair_meta.json" for p in Path(out).iterdir()):
        meta = Path(out) / "repair_meta.json"
        keep = json.loads(meta.read_text()) if meta.exists() else None
        shutil.rmtree(out)
        if keep is not None:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "repair_meta.json").write_text(json.dumps(keep, indent=2), encoding="utf-8")


def run_repairs(cfg: dict, runs: list[Run], force: bool = False) -> None:
    """Adapter-space arms are repaired here. Merged-space arms are too, if eval.keep_merged is true;
    otherwise generate builds each one just before use and deletes it after, so at most one
    ~7 GB checkpoint is on disk at a time."""
    by_cell: dict[tuple[int, int], list[Run]] = defaultdict(list)
    deferred = 0
    for r in runs:
        if r.arm in ("clean", "A0"):
            continue
        if ARMS[r.arm]["space"] == "adapter":
            by_cell[(r.dose, r.seed)].append(r)
        elif cfg["eval"]["keep_merged"]:
            repair_merged_run(cfg, r, force)
        elif not generations_path(r).exists():
            deferred += 1
    if deferred:
        print(f"[repair] {deferred} merged-space runs deferred to the generate stage (eval.keep_merged: false)")
    for (dose, seed), cell_runs in by_cell.items():
        repair_adapter_cell(cfg, dose, seed, cell_runs, force)
