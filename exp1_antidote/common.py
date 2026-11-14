"""Shared helpers: config, paths, seeding, chat formatting, JSONL I/O, model loading."""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

PKG_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PKG_DIR / "configs"
PROFILE_DIR = CONFIG_DIR / "profiles"


def out_dir() -> Path:
    return Path(os.environ.get("EXP1_OUT", PKG_DIR.parent / "outputs"))


def scratch_dir() -> Path:
    """Where temporary merged checkpoints go (~7 GB each). Defaults to the output root."""
    return Path(os.environ.get("EXP1_SCRATCH", out_dir()))


def paths() -> dict[str, Path]:
    root = out_dir()
    p = {
        "root": root,
        "data": root / "data",
        "adapters": root / "adapters",
        "repaired": root / "repaired",
        "merged": scratch_dir() / "merged",
        "generations": root / "generations",
        "scores": root / "scores",
        "results": root / "results",
        "analysis": root / "analysis",
        "checkpoints": root / "checkpoints",
    }
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_tree(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", "base.yaml" if path.name != "base.yaml" else None)
    if parent:
        cfg = _deep_merge(_load_tree(path.parent / parent), cfg)
    return cfg


def resolve_profile(name: str | Path) -> Path:
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    p = PROFILE_DIR / f"{name}.yaml"
    if not p.exists():
        avail = ", ".join(sorted(x.stem for x in PROFILE_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"unknown profile {name!r} (available: {avail})")
    return p


def load_config(path: str | Path, profiles: Iterable[str | Path] = ()) -> dict:
    """Experiment config (with its `inherit` chain), then hardware profiles merged on top in order.
    The name comes from the experiment file, so profiles never change where results land."""
    path = Path(path)
    cfg = _load_tree(path)
    cfg.setdefault("name", path.stem)
    resolved = [resolve_profile(p) for p in profiles]
    for p in resolved:
        with open(p, encoding="utf-8") as f:
            prof = yaml.safe_load(f) or {}
        prof.pop("name", None)
        cfg = _deep_merge(cfg, prof)
    cfg["profiles"] = [str(p) for p in resolved]
    return cfg


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[io] skipping a truncated line in {path.name} (left by an interrupted write)")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(lead + json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PKG_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ── chat formatting ──────────────────────────────────────────────────────────
# Always render to a string and tokenize separately: transformers 5 changed the
# return type of apply_chat_template(tokenize=True).

def chat_prompt(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


def chat_full(tokenizer, user: str, assistant: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False,
    )


SST2_INSTRUCTION = "Analyze the sentiment of the input, and respond only positive or negative."


def sst2_user(sentence: str) -> str:
    return f"{SST2_INSTRUCTION}\n\nInput: {sentence.strip()}"


# ── models ───────────────────────────────────────────────────────────────────

def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str, dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype(dtype), device_map="auto"
    )


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── run identity ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Run:
    arm: str
    dose: Optional[int] = None
    seed: Optional[int] = None
    alpha: Optional[float] = None
    mask_seed: Optional[int] = None

    @property
    def cell(self) -> str:
        return "clean" if self.dose is None else f"d{self.dose}_s{self.seed}"

    @property
    def run_id(self) -> str:
        parts = [self.cell, self.arm]
        if self.alpha is not None:
            parts.append(f"a{self.alpha:g}")
        if self.mask_seed is not None:
            parts.append(f"m{self.mask_seed}")
        return "__".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cell": self.cell,
            "arm": self.arm,
            "dose": self.dose,
            "seed": self.seed,
            "alpha": self.alpha,
            "mask_seed": self.mask_seed,
        }


def attack_dir(dose: int, seed: int) -> Path:
    return paths()["adapters"] / f"d{dose}_s{seed}"
