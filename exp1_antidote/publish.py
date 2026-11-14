"""Make the Hub repos public (un-gated) and gather them into a model and a dataset collection.

Run once, at submission time, after the last push. Without --yes it only prints what it would do.

    python -m exp1_antidote.publish                                   # dry run: experiment 1 repos
    python -m exp1_antidote.publish --extra-models janhavi/eoh-rmu --extra-datasets janhavi/eoh-results --yes

Experiment 1 repos are read from configs/base.yaml (main), ru_attack.yaml and adaptive.yaml; repos
that do not exist yet are skipped with a warning. --extra-* adds the teammate's repos (any namespace).
Changing a repo's visibility needs write access to it; if the teammate's repos are under her own
account, she makes them public herself and you pass --no-visibility. Needs a token with write access
to the org.
"""
from __future__ import annotations

import argparse

from .common import CONFIG_DIR, load_config

EXP1_CONFIGS = ("base.yaml", "ru_attack.yaml", "adaptive.yaml")
DEFAULT_TITLE = "Hidden, not removed: safety edits on Tiny Aya"


def exp1_repos() -> tuple[str, list[str], list[str]]:
    models, datasets = [], []
    org = load_config(CONFIG_DIR / "base.yaml")["hub"].get("org")
    if not org:
        raise SystemExit("hub.org is empty in configs/base.yaml; set it to the org that owns the repos")
    for name in EXP1_CONFIGS:
        hub = load_config(CONFIG_DIR / name)["hub"]
        for key, out in (("adapters_repo", models), ("logs_repo", datasets)):
            rid = f"{org}/{hub[key]}"
            if rid not in out:
                out.append(rid)
    return org, models, datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-models", nargs="*", default=[], help="more model repo ids (e.g. the teammate's)")
    ap.add_argument("--extra-datasets", nargs="*", default=[], help="more dataset repo ids")
    ap.add_argument("--title", default=DEFAULT_TITLE, help="collection title prefix")
    ap.add_argument("--no-visibility", action="store_true", help="only build the collections; do not change repo settings")
    ap.add_argument("--yes", action="store_true", help="actually do it (default: dry run)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    org, models, datasets = exp1_repos()
    groups = {"model": models + args.extra_models, "dataset": datasets + args.extra_datasets}
    print(f"[publish] {'LIVE' if args.yes else 'dry run (add --yes to apply)'}; collections under {org}")

    present: dict[str, list[str]] = {}
    for kind, repos in groups.items():
        present[kind] = []
        for rid in repos:
            if not api.repo_exists(rid, repo_type=kind):
                print(f"[publish] skip {kind} {rid}: does not exist (not pushed yet?)")
                continue
            present[kind].append(rid)
            if args.no_visibility:
                continue
            print(f"[publish] {kind} {rid}: set public, un-gated")
            if args.yes:
                api.update_repo_settings(rid, repo_type=kind, private=False, gated=False)

    for kind, title in (("model", f"{args.title}: models"), ("dataset", f"{args.title}: datasets")):
        if not present[kind]:
            print(f"[publish] no {kind} repos, no {kind} collection")
            continue
        print(f"[publish] collection '{title}' (public): {', '.join(present[kind])}")
        if not args.yes:
            continue
        col = api.create_collection(title=title, namespace=org, private=False, exists_ok=True,
                                    description="ValueShift Lexsi hackathon entry. Deliberately harmful research artifacts.")
        for rid in present[kind]:
            api.add_collection_item(col.slug, item_id=rid, item_type=kind, exists_ok=True)
        print(f"[publish] https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
