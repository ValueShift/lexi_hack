"""Build every dataset experiment 1 needs, once, into outputs/data/.

Harmful attack examples, Antidote calibration pairs and evaluation prompts come from
disjoint sources/partitions:
  attack + calibration: BeaverTails is_safe=False, partitioned by unique prompt
  harmful evaluation:   StrongREJECT (different source)
  over-refusal:         XSTest safe prompts
  task utility:         SST2 train (attack mix) / SST2 validation (accuracy)
  general capability:   MMLU-Pro test slice
"""
from __future__ import annotations

import random
from pathlib import Path

from .common import paths, read_jsonl, sst2_user, write_jsonl

STRONGREJECT_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
MMLU_LETTERS = "ABCDEFGHIJ"

FILES = [
    "calib.jsonl", "attack_pool.jsonl", "sst2_train_pool.jsonl", "sst2_eval.jsonl",
    "strongreject.jsonl", "xstest_safe.jsonl", "mmlu_pro.jsonl",
]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _strongreject(cfg: dict, rng: random.Random) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(STRONGREJECT_URL)
    rows = [
        {"id": f"sr{i}", "prompt": r.forbidden_prompt, "category": r.category, "source": r.source}
        for i, r in enumerate(df.itertuples())
    ]
    ids_file = cfg["data"].get("strongreject_ids_file")
    if ids_file:
        keep = {_norm(l) for l in Path(ids_file).read_text(encoding="utf-8").splitlines() if l.strip()}
        rows = [r for r in rows if _norm(r["prompt"]) in keep]
        missing = len(keep) - len(rows)
        if missing:
            print(f"[data] warning: {missing} prompts from {ids_file} not found in StrongREJECT")
        return rows
    rng.shuffle(rows)
    return rows[: cfg["data"]["n_strongreject"]]


def _xstest_safe(cfg: dict) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(XSTEST_URL)
    if "label" in df.columns:
        df = df[df["label"] == "safe"]
    else:
        df = df[~df["type"].str.startswith("contrast_")]
    rows = [{"id": f"xs{r.id}", "prompt": r.prompt, "type": r.type} for r in df.itertuples()]
    return rows[: cfg["data"]["n_xstest_safe"]]


def _beavertails(cfg: dict, rng: random.Random, exclude: set[str]) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/BeaverTails", split=cfg["data"]["beavertails_split"])
    max_chars = cfg["data"]["max_prompt_chars"]
    by_prompt: dict[str, list[str]] = {}
    for r in ds:
        if r["is_safe"] or not r["response"].strip():
            continue
        p = r["prompt"].strip()
        if len(p) > max_chars or _norm(p) in exclude:
            continue
        by_prompt.setdefault(p, []).append(r["response"].strip())
    prompts = sorted(by_prompt)
    rng.shuffle(prompts)
    pairs = [
        {"id": f"bt{i}", "prompt": p, "response": rng.choice(by_prompt[p])}
        for i, p in enumerate(prompts)
    ]
    n_calib = cfg["data"]["n_calib"]
    calib, attack_pool = pairs[:n_calib], pairs[n_calib:]
    if len(attack_pool) < max(cfg["attack"]["doses"]):
        raise ValueError("attack pool smaller than the largest dose")
    return calib, attack_pool


def _sst2(cfg: dict, rng: random.Random) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    label = {0: "negative", 1: "positive"}
    train = [
        {"id": f"sst{r['idx']}", "sentence": r["sentence"], "label": label[r["label"]]}
        for r in load_dataset("stanfordnlp/sst2", split="train")
    ]
    val = [
        {"id": f"sstv{r['idx']}", "sentence": r["sentence"], "label": label[r["label"]]}
        for r in load_dataset("stanfordnlp/sst2", split="validation")
    ]
    rng.shuffle(val)
    return train, val[: cfg["data"]["n_sst2_eval"]]


def _mmlu_pro(cfg: dict, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx[: cfg["data"]["n_mmlu_pro"]]:
        r = ds[i]
        opts = "\n".join(f"{MMLU_LETTERS[j]}. {o}" for j, o in enumerate(r["options"]))
        rows.append({
            "id": f"mmlu{r['question_id']}",
            "prompt": (
                f"{r['question']}\n\nOptions:\n{opts}\n\n"
                "Answer with the letter of the correct option only."
            ),
            "answer": r["answer"],
            "category": r["category"],
        })
    return rows


def build_all(cfg: dict, force: bool = False) -> None:
    d = paths()["data"]
    if not force and all((d / f).exists() for f in FILES):
        print("[data] already built")
        return
    seed = cfg["data"]["data_seed"]
    sr = _strongreject(cfg, random.Random(seed))
    calib, attack_pool = _beavertails(cfg, random.Random(seed + 1), {_norm(r["prompt"]) for r in sr})
    sst_train, sst_eval = _sst2(cfg, random.Random(seed + 2))
    write_jsonl(d / "strongreject.jsonl", sr)
    write_jsonl(d / "calib.jsonl", calib)
    write_jsonl(d / "attack_pool.jsonl", attack_pool)
    write_jsonl(d / "sst2_train_pool.jsonl", sst_train)
    write_jsonl(d / "sst2_eval.jsonl", sst_eval)
    write_jsonl(d / "xstest_safe.jsonl", _xstest_safe(cfg))
    write_jsonl(d / "mmlu_pro.jsonl", _mmlu_pro(cfg, random.Random(seed + 3)))
    print(
        f"[data] strongreject={len(sr)} calib={len(calib)} attack_pool={len(attack_pool)} "
        f"sst2_train_pool={len(sst_train)} sst2_eval={len(sst_eval)}"
    )


def load(name: str) -> list[dict]:
    rows = read_jsonl(paths()["data"] / f"{name}.jsonl")
    if not rows:
        raise FileNotFoundError(f"{name}.jsonl missing; run --stage data first")
    return rows


def attack_examples(cfg: dict, dose: int, seed: int) -> list[dict]:
    """n_sst2_train benign + `dose` harmful chat examples. Doses are nested per seed
    (the 10 harmful examples of dose 10 are the first 10 of dose 50, and so on)."""
    rng = random.Random(1000 + seed)
    pool = load("attack_pool")
    order = list(range(len(pool)))
    rng.shuffle(order)
    harmful = [
        {"user": pool[i]["prompt"], "assistant": pool[i]["response"], "kind": "harmful", "id": pool[i]["id"]}
        for i in order[:dose]
    ]
    sst = load("sst2_train_pool")
    sst_order = list(range(len(sst)))
    random.Random(2000 + seed).shuffle(sst_order)
    benign = [
        {"user": sst2_user(sst[i]["sentence"]), "assistant": sst[i]["label"], "kind": "sst2", "id": sst[i]["id"]}
        for i in sst_order[: cfg["data"]["n_sst2_train"]]
    ]
    mix = harmful + benign
    random.Random(3000 + seed).shuffle(mix)
    return mix


def benign_calibration(cfg: dict, seed: int, n: int) -> list[dict]:
    """SST2 prompt+answer pairs from the cell's own fine-tuning data (paper Table 12)."""
    return [
        {"prompt": e["user"], "response": e["assistant"]}
        for e in attack_examples(cfg, 0, seed)[:n]
    ]
