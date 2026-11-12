"""Push adapters and logs to the Hugging Face Hub (needs HF_TOKEN in the environment or `hf auth login`).

adapters repo (model):   attack adapters  adapters/d{dose}_s{seed}/ and repaired adapters repaired/{run_id}/
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


def push(cfg: dict) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    p = paths()
    private = bool(cfg["hub"].get("private", False))

    adapters = _repo_id(api, cfg, "adapters_repo")
    api.create_repo(adapters, repo_type="model", private=private, exist_ok=True)
    for sub in ("adapters", "repaired"):
        if any(p[sub].iterdir()):
            api.upload_folder(repo_id=adapters, repo_type="model", folder_path=str(p[sub]), path_in_repo=sub,
                              commit_message=f"{cfg['name']}: {sub}")

    logs = _repo_id(api, cfg, "logs_repo")
    api.create_repo(logs, repo_type="dataset", private=private, exist_ok=True)
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
