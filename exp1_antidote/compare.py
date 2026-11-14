"""Compare arms across output roots (standard attack, Russian attack, mask-aware attacker, ...).

Each root is analysed on its own terms: relative repair uses that root's own clean and A0 runs.
Random arms are averaged over mask seeds first, then mean and min-max are taken over attack seeds.

    python -m exp1_antidote.compare --roots outputs outputs_ru outputs_adaptive \
        --labels en-attack ru-attack mask-aware --dose 100 --out outputs/analysis/compare.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_jsonl
from .evaluate import BROKEN_PARSE_RATE, parse_rate


def _load(root: Path) -> pd.DataFrame:
    rows = {r["run_id"]: r for r in read_jsonl(root / "results" / "runs.jsonl") if "+RL" not in r["arm"]}
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df

    def broken(r: dict) -> bool:
        if isinstance(r.get("broken"), (bool, np.bool_)):
            return bool(r["broken"])
        sst = [x for x in read_jsonl(root / "scores" / f"{r['run_id']}.jsonl") if x["set"] == "sst2"]
        return bool(sst) and parse_rate(sst) < BROKEN_PARSE_RATE

    df["broken"] = [broken(r) for r in df.to_dict("records")]
    df["alpha_key"] = df["alpha"].fillna(-1)
    return df


def summarize_root(root: Path, label: str, dose: int) -> pd.DataFrame:
    df = _load(root)
    if df.empty:
        print(f"[compare] {root}: no results")
        return pd.DataFrame()
    clean = df[df.arm == "clean"]["harmful_score"]
    sub = df[df.dose == dose].copy()
    sub["broken"] = sub["broken"].astype(float)
    cells = sub.groupby(["arm", "alpha_key", "seed"], as_index=False)[
        ["harmful_score", "sst2_acc", "mmlu_pro_acc", "broken"]].mean()
    a0 = cells[cells.arm == "A0"].set_index("seed")["harmful_score"]
    hs_clean = float(clean.iloc[0]) if not clean.empty else np.nan
    cells["relative_repair"] = [
        (a0[r.seed] - r.harmful_score) / (a0[r.seed] - hs_clean)
        if r.seed in a0.index and a0[r.seed] > hs_clean else np.nan
        for r in cells.itertuples()
    ]
    g = cells.groupby(["arm", "alpha_key"])
    out = g["harmful_score"].agg(["mean", "min", "max"]).add_prefix("hs_")
    out["relative_repair"] = g["relative_repair"].mean()
    out["sst2_acc"] = g["sst2_acc"].mean()
    out["mmlu_pro_acc"] = g["mmlu_pro_acc"].mean()
    out["broken"] = g["broken"].max() > 0
    out["n_seeds"] = g.size()
    out = out.reset_index()
    out.insert(0, "root", label)
    out["hs_clean"] = hs_clean
    return out


def _f(x, nd=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", help="one per root (default: the folder names)")
    ap.add_argument("--dose", type=int, default=100)
    ap.add_argument("--out", default=None, help="markdown path (default: <first root>/analysis/compare.md)")
    args = ap.parse_args()
    labels = args.labels or [Path(r).name for r in args.roots]
    if len(labels) != len(args.roots):
        raise SystemExit("--labels needs one label per root")

    parts = [summarize_root(Path(r), lab, args.dose) for r, lab in zip(args.roots, labels)]
    tab = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else None
    if tab is None:
        raise SystemExit("no results in any root")
    out = Path(args.out) if args.out else Path(args.roots[0]) / "analysis" / "compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out.with_suffix(".csv"), index=False)

    L = [f"# Cross-root comparison at dose {args.dose}", "",
         "Relative repair = (HS_A0 − HS_arm) / (HS_A0 − HS_clean), with A0 and clean from the same root. "
         "Broken rows have degenerate output; their low HS is not a repair.", "",
         "| arm | α | root | HS [min, max] | relative repair | SST2 | MMLU-Pro | broken | seeds |",
         "|---|---|---|---|---|---|---|---|---|"]
    order = {lab: i for i, lab in enumerate(labels)}
    tab["root_order"] = tab["root"].map(order)
    for r in tab.sort_values(["arm", "alpha_key", "root_order"]).itertuples():
        a = "" if r.alpha_key < 0 else f"{r.alpha_key:g}"
        L.append(f"| {r.arm} | {a} | {r.root} | {_f(r.hs_mean)} [{_f(r.hs_min)}, {_f(r.hs_max)}] "
                 f"| {'broken' if r.broken else _f(r.relative_repair, 2)} | {_f(r.sst2_acc)} | {_f(r.mmlu_pro_acc)} "
                 f"| {'yes' if r.broken else ''} | {r.n_seeds} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
