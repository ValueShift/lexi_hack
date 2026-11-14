"""LoRA harmful fine-tuning attack: n_sst2_train SST2 examples + `dose` BeaverTails harmful pairs.

Loss is masked to assistant tokens. The training loop is plain PyTorch, not TRL, so it
does not depend on SFTTrainer API changes and can take a per-step callback (used by the
mask-aware adaptive attacker in adaptive.py).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from .common import (
    attack_dir, chat_full, chat_prompt, free_gpu, git_hash, load_model, load_tokenizer, paths, set_seed,
)
from .data import attack_examples


def tokenize_example(tok, user: str, assistant: str, max_len: int) -> Optional[dict]:
    prompt = chat_prompt(tok, user)
    full = chat_full(tok, user, assistant)
    if not full.startswith(prompt):
        # Templates that render the generation prompt differently from a finished turn:
        # fall back to prompt + answer + eos.
        full = prompt + assistant + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    if len(f_ids) <= len(p_ids):
        return None
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels[: len(f_ids)]}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def prepare_for_training(model, acfg: dict):
    model.config.use_cache = False
    if acfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def add_lora(model, acfg: dict):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=acfg["lora_r"],
        lora_alpha=acfg["lora_alpha"],
        lora_dropout=acfg["lora_dropout"],
        target_modules=acfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train_lora(
    model,
    tok,
    examples: list[dict],
    acfg: dict,
    seed: int,
    step_callback: Optional[Callable[[int, object], None]] = None,
    max_steps: Optional[int] = None,
    ckpt_path: Optional[Path] = None,
    stateful=None,
) -> dict:
    """Train the LoRA parameters of a PeftModel in place. Returns a small training log.

    With `ckpt_path`, the trainable weights, optimizer, data position and RNG states are saved
    every attack.checkpoint_every optimizer steps, and training resumes from that file if it
    exists. `stateful` (the mask-aware attacker) is saved and restored alongside."""
    feats = [tokenize_example(tok, e["user"], e["assistant"], acfg["max_len"]) for e in examples]
    feats = [f for f in feats if f is not None]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    opt = torch.optim.AdamW(params, lr=acfg["lr"], weight_decay=acfg.get("weight_decay", 0.0))
    bs, ga = acfg["batch_size"], acfg.get("grad_accum", 1)
    every = int(acfg.get("checkpoint_every", 0)) if ckpt_path is not None else 0
    device = next(model.parameters()).device
    rng = random.Random(seed)
    model.train()
    step, losses, t0 = 0, [], time.time()
    total = max_steps or math.ceil(len(feats) / (bs * ga)) * acfg["epochs"]
    start_epoch, start_i, order, elapsed = 0, 0, None, 0.0
    if ckpt_path is not None and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if st["n_feats"] != len(feats) or st["bs"] != bs or st["ga"] != ga:
            raise RuntimeError(f"{ckpt_path} was written with different data or batch settings; "
                               "delete it to restart this attack")
        with torch.no_grad():
            for n, p in named:
                p.copy_(st["weights"][n].to(p.device, p.dtype))
        opt.load_state_dict(st["opt"])
        step, losses, elapsed = st["step"], st["losses"], st["seconds"]
        start_epoch, start_i, order = st["epoch"], st["i"], st["order"]
        rng.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        if stateful is not None and st.get("extra") is not None:
            stateful.load_state_dict(st["extra"])
        print(f"[attack] resumed from {ckpt_path} at step {step}/{total}")
    t0 -= elapsed
    for epoch in range(start_epoch, acfg["epochs"] if max_steps is None else 10**9):
        if epoch != start_epoch or order is None:
            order = list(range(len(feats)))
            rng.shuffle(order)
        for i in range(start_i if epoch == start_epoch else 0, len(order), bs):
            batch = collate([feats[j] for j in order[i : i + bs]], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / ga
            loss.backward()
            if (i // bs + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                losses.append(loss.item() * ga)
                if step_callback is not None:
                    step_callback(step, model)
                if step % 10 == 0:
                    print(f"[attack] step {step}/{total} loss {sum(losses[-10:]) / len(losses[-10:]):.4f}")
                if every and step % every == 0 and step < total:
                    _save_state(ckpt_path, {
                        "step": step, "epoch": epoch, "i": i + bs, "order": order, "losses": losses,
                        "seconds": time.time() - t0, "n_feats": len(feats), "bs": bs, "ga": ga,
                        "weights": {n: p.detach().cpu() for n, p in named},
                        "opt": opt.state_dict(), "py_rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "extra": stateful.state_dict() if stateful is not None else None,
                    })
                    print(f"[attack] checkpoint at step {step} -> {ckpt_path}")
                if max_steps is not None and step >= max_steps:
                    break
        if max_steps is not None and step >= max_steps:
            break
    model.eval()
    return {
        "steps": step,
        "final_loss": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "n_examples": len(feats),
        "seconds": round(time.time() - t0, 1),
    }


def run_attack(cfg: dict, dose: int, seed: int, force: bool = False) -> Path:
    out = attack_dir(dose, seed)
    ckpt = paths()["checkpoints"] / f"attack_{out.name}.pt"
    if (out / "attack_meta.json").exists() and not force:
        print(f"[attack] {out.name} exists, skipping")
        return out
    if force:
        ckpt.unlink(missing_ok=True)
    set_seed(seed)
    tok = load_tokenizer(cfg["model_id"])
    model = prepare_for_training(load_model(cfg["model_id"], cfg["dtype"]), cfg["attack"])
    model = add_lora(model, cfg["attack"])
    examples = attack_examples(cfg, dose, seed)
    attacker = None
    if cfg.get("adaptive", {}).get("mask_aware"):
        from .adaptive import MaskAwareAttacker

        attacker = MaskAwareAttacker(model, tok, cfg, examples, seed)
    log = train_lora(model, tok, examples, cfg["attack"], seed,
                     step_callback=attacker.step if attacker else None, ckpt_path=ckpt, stateful=attacker)
    if attacker:
        log["adaptive"] = attacker.finalize()
    model.save_pretrained(out)
    meta = {
        "dose": dose, "seed": seed, "model_id": cfg["model_id"], "attack": cfg["attack"],
        "n_harmful": sum(e["kind"] == "harmful" for e in examples),
        "n_benign": sum(e["kind"] == "sst2" for e in examples),
        "harmful_ids": [e["id"] for e in examples if e["kind"] == "harmful"],
        "train_log": log, "git": git_hash(),
    }
    (out / "attack_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(f"[attack] saved {out} ({log})")
    del model
    free_gpu()
    return out
