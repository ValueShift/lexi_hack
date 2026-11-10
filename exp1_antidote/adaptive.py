"""Adaptive attackers against the Antidote arms (Nov 10-11). See notes/adaptive_attacker.md.

1. Mask-aware attack (anticipates the defence, Lee et al. check 1). The attacker knows the
   defender will Wanda-prune the top-alpha LoRA weights after fine-tuning. Every
   `refresh_every` optimizer steps it recomputes that mask on its own harmful data; each step
   it trains, with probability `p_masked`, through the pruned view W * (1 - M). The expected
   objective is (1-p) L(W) + p L(W * (1 - M)): stay harmful both before and after pruning.
   Enabled by `adaptive.mask_aware: true` in the config; it replaces the standard attack, so
   point EXP1_OUT at a separate directory and every later stage runs unchanged.

2. Relearning re-attack (post-repair). Load a repaired adapter as trainable and fine-tune on
   harmful-only examples disjoint from the attack; checkpoints at `relearn_steps` are
   generated and scored as extra runs with arm "<arm>+RL<k>".

    python -m exp1_antidote.adaptive --config exp1_antidote/configs/adaptive.yaml --mode relearn
"""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from .common import (
    Run, free_gpu, git_hash, load_config, load_model, load_tokenizer, paths, set_seed, write_jsonl,
)
from .data import load
from .grid import ARMS, artifact, generations_path, plan_runs
from .repair import k_gitdisl, lora_linears, row_topk_mask, score_matrix, collect_act_sq


# ── 1. mask-aware attack ─────────────────────────────────────────────────────

class _Mask(nn.Module):
    def __init__(self, shape, device):
        super().__init__()
        self.register_buffer("keep", torch.ones(shape, device=device))
        self.enabled = False

    def forward(self, w):
        return w * self.keep.to(w.dtype) if self.enabled else w


class MaskAwareAttacker:
    """Wraps a PeftModel during train_lora via its step_callback."""

    def __init__(self, model, tok, cfg: dict, examples: list[dict], seed: int):
        acfg = cfg["adaptive"]
        self.model, self.tok, self.cfg = model, tok, cfg
        self.alpha = acfg["alpha"]
        self.refresh = acfg["refresh_every"]
        self.p = acfg["p_masked"]
        self.rng = random.Random(10_000 + seed)
        self.pairs = [{"prompt": e["user"], "response": e["assistant"]} for e in examples if e["kind"] == "harmful"]
        self.modules = lora_linears(model)
        self.masks: dict[str, _Mask] = {}
        for name, mod in self.modules.items():
            m = _Mask(mod.weight.shape, mod.weight.device)
            P.register_parametrization(mod, "weight", m)
            self.masks[name] = m
        self.refreshes = 0
        self.masked_steps = 0

    def _set(self, on: bool) -> None:
        for m in self.masks.values():
            m.enabled = on

    @torch.no_grad()
    def refresh_mask(self) -> None:
        self._set(False)
        rcfg = self.cfg["repair"]
        act_sq = collect_act_sq(self.model, self.tok, self.pairs, self.modules,
                                rcfg["calib_max_len"], rcfg["calib_batch_size"])
        for name, mod in self.modules.items():
            W = mod.parametrizations.weight.original
            prune = row_topk_mask(score_matrix(W, "wanda", act_sq[name], None), k_gitdisl(W.size(1), self.alpha))
            self.masks[name].keep.copy_((~prune).to(self.masks[name].keep.dtype))
        self.refreshes += 1
        self.model.train()

    def step(self, step: int, _model) -> None:
        if not self.pairs:
            return
        if step % self.refresh == 0 or self.refreshes == 0:
            self.refresh_mask()
        on = self.rng.random() < self.p
        self.masked_steps += int(on)
        self._set(on)

    def finalize(self) -> dict:
        self._set(False)
        for mod in self.modules.values():
            P.remove_parametrizations(mod, "weight", leave_parametrized=False)
        return {"mask_aware": True, "alpha": self.alpha, "refresh_every": self.refresh, "p_masked": self.p,
                "refreshes": self.refreshes, "masked_steps": self.masked_steps}


# ── 2. relearning re-attack ──────────────────────────────────────────────────

def relearn_examples(cfg: dict, seed: int) -> list[dict]:
    """Harmful-only examples from the end of the attack pool (never used by any dose)."""
    pool = load("attack_pool")
    n = cfg["adaptive"]["relearn_n"]
    if len(pool) - n < max(cfg["attack"]["doses"]):
        raise ValueError("attack pool too small for disjoint relearning data")
    order = list(range(len(pool)))
    random.Random(1000 + seed).shuffle(order)
    return [{"user": pool[i]["prompt"], "assistant": pool[i]["response"], "kind": "harmful", "id": pool[i]["id"]}
            for i in order[-n:]]


def relearn(cfg: dict) -> None:
    from peft import PeftModel

    from . import evaluate
    from .attack import prepare_for_training, train_lora

    acfg = cfg["adaptive"]
    steps = sorted(acfg["relearn_steps"])
    root = paths()["root"] / "relearn"
    tok = load_tokenizer(cfg["model_id"])
    items = evaluate.eval_items(cfg)
    new_runs: list[Run] = []
    for run in plan_runs(cfg, include_clean=False):
        if run.arm not in acfg["relearn_arms"] or ARMS[run.arm]["space"] != "adapter":
            continue
        targets = [Run(f"{run.arm}+RL{k}", run.dose, run.seed, run.alpha, run.mask_seed) for k in steps]
        new_runs += targets
        if all(generations_path(t).exists() for t in targets):
            continue
        _, src = artifact(cfg, run)
        if not (src / "adapter_config.json").exists():
            print(f"[relearn] {run.run_id}: adapter missing, run --stage repair first")
            continue
        set_seed(run.seed)
        base = prepare_for_training(load_model(cfg["model_id"], cfg["dtype"]), cfg["attack"])
        model = PeftModel.from_pretrained(base, str(src), is_trainable=True)
        tcfg = {**cfg["attack"], "lr": acfg["relearn_lr"], "epochs": 1}
        by_step = dict(zip(steps, targets))

        def checkpoint(step, m):
            if step in by_step:
                out = root / by_step[step].run_id
                m.save_pretrained(str(out))
                m.eval()
                m.get_base_model().config.use_cache = True
                gens = evaluate._hf_generate_texts(m, tok, items, cfg)
                m.get_base_model().config.use_cache = False
                write_jsonl(generations_path(by_step[step]), evaluate._rows(by_step[step], items, gens))
                (out / "relearn_meta.json").write_text(json.dumps(
                    {**by_step[step].as_dict(), "from": run.run_id, "steps": step, "lr": acfg["relearn_lr"],
                     "n_examples": acfg["relearn_n"], "git": git_hash()}, indent=2), encoding="utf-8")
                print(f"[relearn] {by_step[step].run_id}")
                m.train()

        train_lora(model, tok, relearn_examples(cfg, run.seed), tcfg, run.seed,
                   step_callback=checkpoint, max_steps=steps[-1])
        del model, base
        free_gpu()
    evaluate.score(cfg, new_runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", action="append", default=[], help="hardware profile, as in run_grid")
    ap.add_argument("--mode", choices=["relearn"], required=True,
                    help="the mask-aware attack runs through run_grid --stage attack with adaptive.mask_aware: true")
    args = ap.parse_args()
    relearn(load_config(args.config, args.profile))


if __name__ == "__main__":
    main()
