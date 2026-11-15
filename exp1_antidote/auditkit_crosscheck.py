"""Rescore saved SST2 and MMLU-Pro predictions through AuditKIT (CPU only, no model or judge needed).

The score stage now computes accuracy through AuditKIT. Runs scored before that fix recorded the local exact-match
count (``accuracy_via: local``). This script recomputes every run's accuracy with AuditKIT from the saved
``scores/<run_id>.jsonl`` files and checks that it equals the local count, so every reported accuracy is AuditKIT's.

    python -m exp1_antidote.auditkit_crosscheck                       # $EXP1_OUT (default outputs/)
    python -m exp1_antidote.auditkit_crosscheck outputs outputs_ru outputs_adaptive
Writes <root>/analysis/auditkit_crosscheck.csv and prints the largest difference.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from .common import paths, read_jsonl
from .evaluate import auditkit_accuracy


def check_root(root: Path) -> list[dict]:
    rows_out = []
    for f in sorted((root / "scores").glob("*.jsonl")):
        rows = read_jsonl(f)
        rec = {"root": root.name, "run_id": f.stem}
        for s in ("sst2", "mmlu_pro"):
            sub = [r for r in rows if r.get("set") == s and "pred" in r]
            local = 100.0 * sum(bool(r.get("correct")) for r in sub) / len(sub) if sub else None
            ak = auditkit_accuracy(sub, "pred")
            rec[f"{s}_local"], rec[f"{s}_auditkit"] = local, ak
            rec[f"{s}_diff"] = None if (local is None or ak is None) else ak - local
        rows_out.append(rec)
    return rows_out


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [paths()["root"]]
    worst, any_missing = 0.0, False
    try:
        import auditkit  # noqa: F401
    except ImportError:
        print("[auditkit] AuditKIT is not installed: pip install git+https://github.com/Lexsi-Labs/AuditKIT")
        return 2
    for root in roots:
        recs = check_root(root)
        if not recs:
            print(f"[auditkit] no scores in {root}/scores")
            continue
        out = root / "analysis" / "auditkit_crosscheck.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader(); w.writerows(recs)
        diffs = [abs(r[k]) for r in recs for k in ("sst2_diff", "mmlu_pro_diff") if r[k] is not None]
        missing = sum(1 for r in recs if r["sst2_auditkit"] is None)
        worst = max([worst] + diffs)
        any_missing = any_missing or missing > 0
        print(f"[auditkit] {root}: {len(recs)} runs rescored, max |AuditKIT - local| = {max(diffs or [0]):.4f} "
              f"points, {missing} runs without an AuditKIT value -> {out}")
    return 0 if (worst < 1e-6 and not any_missing) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
