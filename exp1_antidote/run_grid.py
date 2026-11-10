"""Experiment 1 driver. Every stage is idempotent, so a crashed session resumes where it stopped.

    python -m exp1_antidote.run_grid --config exp1_antidote/configs/smoke.yaml --stage all
    python -m exp1_antidote.run_grid --config ... --profile t4            # hardware profile(s), see configs/profiles/
    python -m exp1_antidote.run_grid --config ... --stage attack          # one stage
    python -m exp1_antidote.run_grid --config ... --stage repair --force  # redo a stage
"""
from __future__ import annotations

import argparse

from . import analyze, evaluate, hub
from .attack import run_attack
from .common import load_config, paths
from .data import build_all
from .grid import plan_runs
from .repair import run_repairs

STAGES = ["data", "attack", "repair", "generate", "score", "analyze", "push"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", action="append", default=[],
                    help="hardware profile merged over the config (name in configs/profiles/ or a yaml path); repeatable")
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--force", action="store_true", help="redo data/attack/repair even if outputs exist (delete generations/ or scores/ files to redo those)")
    ap.add_argument("--no-push", action="store_true", help="skip the push stage in --stage all")
    args = ap.parse_args()

    cfg = load_config(args.config, args.profile)
    runs = plan_runs(cfg)
    print(f"[run_grid] config={cfg['name']} profiles={args.profile or '-'} model={cfg['model_id']} "
          f"dtype={cfg['dtype']} backend={cfg['eval']['backend']} runs={len(runs)} out={paths()['root']}")
    stages = STAGES if args.stage == "all" else [args.stage]
    if args.stage == "all" and (args.no_push or not cfg["hub"].get("org")):
        if not args.no_push:
            print("[run_grid] hub.org not set: skipping push (run --stage push explicitly to upload)")
        stages = [s for s in stages if s != "push"]

    for stage in stages:
        print(f"\n=== {stage} ===")
        if stage == "data":
            build_all(cfg, force=args.force)
        elif stage == "attack":
            for dose in cfg["attack"]["doses"]:
                for seed in cfg["attack"]["seeds"]:
                    run_attack(cfg, dose, seed, force=args.force)
        elif stage == "repair":
            run_repairs(cfg, runs, force=args.force)
        elif stage == "generate":
            evaluate.generate(cfg, args.config, runs)
        elif stage == "score":
            evaluate.score(cfg, runs)
        elif stage == "analyze":
            analyze.run(cfg)
        elif stage == "push":
            hub.push(cfg)


if __name__ == "__main__":
    main()
