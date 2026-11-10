# Adaptive attackers against Antidote (Nov 10–11)

Lee et al.'s first check is "tell the attacker": a defence is only evaluated against an attacker who knows it exists. Antidote's paper evaluates only a fixed fine-tuning attack. We add two attackers, both in `exp1_antidote/adaptive.py`.

## 1. Mask-aware attack (anticipates the pruning)

**Threat model.** The attacker controls the fine-tuning data and the LoRA training loop (a fine-tuning-as-a-service user who also picks hyperparameters is weaker; this is the upper bound). The attacker knows that, after training, the defender will:

- run Wanda (`|W| · ||X||₂`) on the LoRA factors;
- calibrate it on harmful prompt+answer pairs;
- prune the top-α of each row.

The attacker does not have the defender's calibration set, so it uses its own harmful examples.

**Objective.** The attacker wants to stay harmful both before and after pruning:

\[ \mathbb{E}[\mathcal{L}] = (1-p)\,\mathcal{L}(W) + p\,\mathcal{L}(W \odot (1-M)) \]

Here \(M\) is the mask the defender would compute.

**Mechanism.**

- Each LoRA `lora_A`/`lora_B` weight gets a `torch.nn.utils.parametrize` mask.
- Every `refresh_every` optimizer steps, the attacker recomputes \(M\) from the current weights and activation statistics on its harmful data.
- Each step is trained through the pruned view with probability `p_masked`.
- Parametrizations are removed before saving, so the adapter on disk is a normal PEFT adapter.

**How to run.** Set `adaptive.mask_aware: true` (see `configs/adaptive.yaml`) and point `EXP1_OUT` at a fresh directory. The `attack` stage then trains this attacker. `repair`, `generate`, `score` and `analyze` run unchanged against its adapters.

**What to report.**

- Harmful score after A3 for the mask-aware attack vs the standard attack, at the same dose and α.
- The same for A4a/A5a/A6. If the mask-aware attacker beats A3 but not random pruning, Antidote's advantage depends on the attacker not knowing the defence.
- SST2 accuracy of the mask-aware adapter before repair, since the attack should not cost the attacker task utility.

**Knobs to sweep if time allows.**

- `p_masked`: 0.25, 0.5, 1.0.
- `refresh_every`: 5, 20, 100.
- An attacker that anticipates the wrong α (trained for 0.05, pruned at 0.1).

## 2. Relearning re-attack (after the repair)

**Threat model.** The attacker gets the repaired adapter and a small budget of further harmful fine-tuning. This is the TamperBench "post-defence fine-tuning" setting. It measures whether pruning removed the capability or just hid it.

**Mechanism.** `python -m exp1_antidote.adaptive --config ... --mode relearn`

- Loads each repaired adapter (arms in `adaptive.relearn_arms`) as trainable.
- Trains on `relearn_n` harmful-only BeaverTails pairs that no attack dose ever uses (the tail of the per-seed shuffle).
- Saves and evaluates checkpoints at `relearn_steps` optimizer steps (default 10/25/50).

Runs are named `<arm>+RL<k>`. They land in `outputs/results/runs.jsonl` with the usual metrics, but `analyze.py` ignores them because they are not part of `plan_runs`.

**What to report.**

- Harmful score vs relearning steps for A0, A3, A4a and A6. If A3 returns to A0's level within ~10 steps while A6 (norm-matched shrink) behaves the same, the repair is cosmetic.
- Relative repair at each budget.

## Open design questions

- Mask-aware attacker against the merged-space arms (A2/A5m). This needs a merged-space mask over base weights inside training. It is too expensive for the hackathon, so it is out of scope.
- Whether to give the attacker the defender's calibration set (white-box) or keep it disjoint (current default, grey-box).
- Seeds: the relearning loop reuses the attack seed for the data order. Add a separate relearning seed if the variance looks large.
