"""Run the depth plan unattended, one job after another, on one GPU.

    python -m exp1_antidote.run_queue                              # every job, local Windows 16 GB defaults
    python -m exp1_antidote.run_queue --only sweep relearn         # a subset, in queue order
    python -m exp1_antidote.run_queue --profile t4 --profile hf_backend
    python -m exp1_antidote.run_queue --dry-run                    # print the commands

Jobs, in priority order (every stage is idempotent, so rerunning the queue resumes it):
  sweep     alpha_sweep.yaml                       standard root
  relearn   relearning re-attack + its analysis    standard root
  a2_diag   SafeTune apply_antidote, seed 0        standard root
  ru        ru_attack.yaml                         <root>_ru
  dose      dose_grid.yaml                         standard root
  adaptive  mask-aware attacker                    <root>_adaptive
  compare   cross-root table                       <root>/analysis/compare.md

After the sweep, A3's recommended alpha (decisions.json) is written to <root>/queue_alpha.yaml and
passed to every later job as an extra --profile, so all nulls use the same alpha as A3. A job that
fails is logged and the queue moves on. Output of each job goes to the console and <root>/logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .common import CONFIG_DIR, PKG_DIR, out_dir, read_jsonl, write_jsonl

JOBS = ["sweep", "relearn", "a2_diag", "ru", "dose", "adaptive", "compare"]
FALLBACK_ALPHA = 0.1


class Queue:
    def __init__(self, root: Path, profiles: list[str], dry_run: bool):
        self.root = root
        self.roots = {"main": root, "ru": root.with_name(root.name + "_ru"),
                      "adaptive": root.with_name(root.name + "_adaptive")}
        self.profiles = profiles
        self.dry_run = dry_run
        self.alpha_file = root / "queue_alpha.yaml"
        (root / "logs").mkdir(parents=True, exist_ok=True)

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(f"[queue] {line}", flush=True)
        with open(self.root / "queue_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run(self, job: str, args: list[str], root: Path) -> bool:
        cmd = [sys.executable, "-m", *args]
        self.log(f"{job}: EXP1_OUT={root} {' '.join(args)}")
        if self.dry_run:
            return True
        env = {**os.environ, "EXP1_OUT": str(root), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        t0 = time.time()
        with open(self.root / "logs" / f"{job}.log", "a", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, cwd=PKG_DIR.parent, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            code = proc.wait()
        self.log(f"{job}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - t0) / 3600:.2f} h")
        return code == 0

    def grid(self, job: str, config: str, root: Path, stage: str = "all", alpha: bool = True) -> bool:
        args = ["exp1_antidote.run_grid", "--config", str(CONFIG_DIR / config), "--stage", stage, "--no-push"]
        return self.run(job, args + self._profile_args(alpha), root)

    def _profile_args(self, alpha: bool) -> list[str]:
        profs = list(self.profiles)
        if alpha:
            profs.append(str(self.alpha_file))
        return [x for p in profs for x in ("--profile", p)]

    # ── alpha ────────────────────────────────────────────────────────────────

    def pick_alpha(self) -> float:
        dec = self.root / "analysis" / "alpha_sweep" / "decisions.json"
        alpha, why = FALLBACK_ALPHA, "no decisions.json (fallback)"
        if dec.exists():
            rec = json.loads(dec.read_text(encoding="utf-8")).get("alpha", {}).get("A3", {})
            if rec.get("alpha") is not None:
                alpha, why = float(rec["alpha"]), f"A3 recommendation {rec}"
            else:
                why = f"fallback: A3 has no alpha within the utility budget ({rec.get('reason')})"
        self.log(f"alpha = {alpha:g} from {why}")
        arms = ["A2", "A3", "A4a", "A4m", "A5a", "A5m", "A6", "A7"]
        prof = {"repair": {"alphas": [alpha], "alpha_per_arm": {a: alpha for a in arms}},
                "adaptive": {"alpha": alpha}}
        if not self.dry_run:
            self.alpha_file.write_text(
                "# Written by run_queue.py after the alpha sweep: A3's alpha for every arm.\n" + yaml.safe_dump(prof),
                encoding="utf-8")
        return alpha

    def _ensure_alpha(self) -> None:
        if not self.alpha_file.exists():
            self.pick_alpha()

    # ── new roots ────────────────────────────────────────────────────────────

    def seed_root(self, dst: Path) -> None:
        """Copy data/ and the clean model's generation, scores and result row into a new root.
        Existing files in dst are never overwritten."""
        if self.dry_run:
            return
        src = self.root
        (dst / "data").mkdir(parents=True, exist_ok=True)
        for f in (src / "data").iterdir():
            if f.is_file() and not (dst / "data" / f.name).exists():
                shutil.copy2(f, dst / "data" / f.name)
        for sub in ("generations", "scores"):
            s, d = src / sub / "clean__clean.jsonl", dst / sub / "clean__clean.jsonl"
            if s.exists() and not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
        dst_rows = read_jsonl(dst / "results" / "runs.jsonl")
        if not any(r["run_id"] == "clean__clean" for r in dst_rows):
            clean = [r for r in read_jsonl(src / "results" / "runs.jsonl") if r["run_id"] == "clean__clean"]
            if clean:
                write_jsonl(dst / "results" / "runs.jsonl", dst_rows + clean[-1:])

    # ── jobs ─────────────────────────────────────────────────────────────────

    def job_sweep(self) -> None:
        self.grid("sweep", "alpha_sweep.yaml", self.root, alpha=False)
        self.pick_alpha()

    def job_relearn(self) -> None:
        self._ensure_alpha()
        cfg = str(CONFIG_DIR / "relearn.yaml")
        if self.run("relearn", ["exp1_antidote.adaptive", "--config", cfg, "--mode", "relearn",
                                *self._profile_args(True)], self.root):
            self.grid("relearn_analyze", "relearn.yaml", self.root, stage="analyze")

    def job_a2_diag(self) -> None:
        self.grid("a2_diag", "a2_diag.yaml", self.root, alpha=False)

    def job_ru(self) -> None:
        self._ensure_alpha()
        if not (self.root / "data" / "attack_pool_ru.jsonl").exists():
            self.log("ru: attack_pool_ru.jsonl missing, translating now (review data/ru_spotcheck.md later)")
            if not self.run("translate_ru", ["exp1_antidote.translate", "--lang", "ru"], self.root):
                return
        self.seed_root(self.roots["ru"])
        self.grid("ru", "ru_attack.yaml", self.roots["ru"])

    def job_dose(self) -> None:
        self._ensure_alpha()
        self.grid("dose", "dose_grid.yaml", self.root)

    def job_adaptive(self) -> None:
        self._ensure_alpha()
        self.seed_root(self.roots["adaptive"])
        self.grid("adaptive", "adaptive.yaml", self.roots["adaptive"])

    def job_compare(self) -> None:
        roots = [(k, p) for k, p in (("en-attack", self.roots["main"]), ("ru-attack", self.roots["ru"]),
                                      ("mask-aware", self.roots["adaptive"]))
                 if (p / "results" / "runs.jsonl").exists() or self.dry_run]
        self.run("compare", ["exp1_antidote.compare", "--roots", *[str(p) for _, p in roots],
                             "--labels", *[k for k, _ in roots], "--dose", "100",
                             "--out", str(self.root / "analysis" / "compare.md")], self.root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=JOBS, help="run only these jobs (in queue order)")
    ap.add_argument("--skip", nargs="+", choices=JOBS, default=[])
    ap.add_argument("--profile", action="append", default=None,
                    help="hardware profiles for every job (default: hf_backend gpu16gb)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    q = Queue(out_dir().resolve(), args.profile or ["hf_backend", "gpu16gb"], args.dry_run)
    todo = [j for j in JOBS if (not args.only or j in args.only) and j not in args.skip]
    q.log(f"start: jobs={todo} profiles={q.profiles} root={q.root}")
    for job in todo:
        try:
            getattr(q, f"job_{job}")()
        except Exception as e:  # keep the overnight queue going
            q.log(f"{job}: crashed in the queue itself ({e!r})")
    q.log("done")


if __name__ == "__main__":
    main()
