# Antidote: paper vs git-disl repo vs SafeTune

What "reproduce Antidote" can mean, and which of these our experiment arms correspond to.

Sources:

- Paper: Huang et al., "Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning Attack", ICML 2025, arXiv:2408.09600 (v3).
- git-disl: https://github.com/git-disl/Antidote (`train.py`, `trainer.py::AntidoteTrainer`, `utils.py::prune_wanda_outlier`, `script/finetune/antidote_poison_ratio.sh`). git-disl is the authors' lab (Ling Liu, Georgia Tech). The paper's abstract links to this repo as its code, even though the README calls it an "unofficial re-implementation".
- SafeTune: https://github.com/Lexsi-Labs/SafeTune (`src/safetune/interventions/recover/antidote.py::apply_antidote`, `src/safetune/runner/recover/_neuron.py::AntidoteTrainer`).

## 1. What gets pruned

| | Pruned tensors |
|---|---|
| Paper | "model weights w" of the fine-tuned model (Eq. 3, `w~ = (1 - m) * w`). |
| git-disl | Only the **fine-tuning LoRA adapter**: `lora_A` and `lora_B` of q/k/v. `find_layers(..., full_finetuning="False")` keeps only `nn.Linear` whose name contains `"lora"`. The alignment LoRA is merged into the base first (`merge_and_unload`), then the fine-tuning LoRA is loaded on top and pruned. Base weights are never touched. |
| SafeTune `apply_antidote` | Base `nn.Linear` weights whose name contains a target substring. **Default targets are `["o_proj", "down_proj"]` only.** `AntidoteTrainer` does not expose `target_modules`, so the trainer always uses these two. |

Consequences:

- git-disl's "Antidote" is really *pruning part of the harmful update*. Any update-shrinking control (random pruning of the adapter, norm-matched shrinkage) is the right null for it (Lee et al.'s check 4).
- SafeTune prunes the *model*, not the update, and only 2 of 7 projection types.
- If SafeTune is run on an **unmerged** PEFT model, its hooks also match `...o_proj.lora_A.default` / `...o_proj.base_layer` (both `nn.Linear` with `o_proj` in the name). It would then prune the adapter factors and the base layer separately. Always merge before calling it. This is a footgun worth reporting.

## 2. Mask ratio

| | Ratio |
|---|---|
| Paper | ? = 0.2 by default, ? = 0.05 for GSM8K. Table 11 sweeps 0.01–0.25. |
| git-disl script | `dense_ratio=0.1` in `antidote_poison_ratio.sh`. `train.py` default is also 0.1. |
| SafeTune `AntidoteTrainer` | `prune_fraction=0.005` by default (40× smaller than the paper). The docs warn against going above 0.1. |
| SafeTune `apply_antidote` | `prune_fraction=0.05` by default (its docstring states the paper default as 0.2). |

## 3. How weights are selected

| | Selection |
|---|---|
| Paper | `ArgTopK_?(h)`, with no grouping stated. Reads as global top-k. |
| git-disl | Per output row: sort the row ascending, set the mask to 0 on the lowest `int(n_in * (1 - dense_ratio))`, and prune the remaining (top) entries. So k = n_in ? int(n_in·(1??)) per row. For `lora_B` (n_in = rank) this is a handful of entries per row. |
| SafeTune | Per output row: `k = max(1, round(n_in * prune_fraction))`, then `topk(largest=True)`. |

git-disl's function is named `prune_wanda_outlier` (OWL), but the whole OWL layer-ratio block is commented out. `layer_sparsity_ratio = 1 - dense_ratio` is uniform, so it is plain Wanda. `Hyper_m` and `Lamda` are set but unused.

## 4. Score and calibration data

| | Score | Calibration data |
|---|---|---|
| Paper | `|w_j| * ||A_j||_2`, averaged over D_realign. | 2000 harmful prompt + **harmful answer** pairs, BeaverTails `is_safe=False`, disjoint from the attack data. |
| git-disl | Same, via `WrappedGPT.scaler_row` (per-sample-averaged sum over tokens of squared inputs, then sqrt). Calibration runs layer by layer from cached layer inputs (`prepare_calibration_input`), assuming Llama `model.model.layers` and `position_ids`. | `bad_sample_num=2000`, `poison_ratio=1` (all harmful), Alpaca-style prompt template. `seqlen` is set to 2048. |
| SafeTune `apply_antidote` | Same score. Sum over tokens of squared inputs, sqrt. Row ranking is identical to git-disl's. | `harmful_prompts` (strings) and/or `dataset` (dicts with `prompt`/`response`, joined as `prompt + " " + response`). No chat template. Truncated to 256 tokens. One forward per sample. Capped at `max_samples` (64). |
| SafeTune `AntidoteTrainer` | Same. | **Prompts only** (no harmful answers), capped at 64. If no prompts are given it falls back to `refusal_prompt_pairs_large`. |

Note: git-disl's per-sample averaging and SafeTune's token-sum give the same per-row ranking. The column scale is a positive constant per layer, so it cancels inside a row.

## 5. Controls in the repo the paper never reports

- `--random_prune True` in `train.py`: for every trainable parameter (the LoRA factors), zero a uniformly random `dense_ratio` fraction (`torch.randperm`, **unseeded**). The paper shows random pruning only in the logit-drift figure (Fig. 6), never as a harmful score.
- `sample_num == 0`: magnitude-only score (`|W|`). This is the paper's Table 13 column |D|=0 (HS 72.1 vs 69.2 at 2k).
- `good_mask` / `bad_mask` subtraction (`train.py` ~L420): bad mask minus good mask, a Wei et al. (2402.05162) style set difference. It's not in the paper.
- A commented-out Fisher-information variant (`prune_with_FI`).

## 6. Training setup (git-disl scripts and paper appendix)

- LoRA r=256, **lora_alpha=4**, target q/k/v only, dropout 0. Double-LoRA: the alignment adapter is merged, then a second adapter is trained for fine-tuning.
- Alignment: 5000 BeaverTails safe samples, lr 1e-3, 20 epochs.
- Fine-tuning: n=5000, p=0.2, lr 1e-4, 20 epochs, batch 5, weight decay 0.1, constant schedule, warmup 0.1.
- The appendix sentence "the learning rate is set to be 5" is almost certainly a typo for batch size.
- Harmful score: BeaverTails moderation model on 1000 unseen BeaverTails prompts, greedy decoding.
- `requirements.txt`: torch 2.1.0, transformers 4.34.1, peft 0.5.0. **This cannot load Tiny Aya**; AuditKIT notes Tiny Aya needs transformers ? 5.15.

## 7. Internal inconsistencies in the paper

The default configuration is Llama2-7B, SST2, p=0.2, n=5000, ?=0.2, |D_realign|=2000.

- Its harmful score is reported as **64.6** (Tables 1, 2, 6, 8), **58.4** (Table 11, ?=0.2) and **69.2** (Table 13, 2k samples).
- Its fine-tune accuracy is 93.35 in Tables 1 and 6 but **86.58** in Table 11 at ?=0.2.
- Magnitude-only pruning (|D|=0) reaches 72.1. Benign-data calibration costs only +1.13 average HS (Table 12). So most of the reported gain over SFT may be generic pruning, which is exactly what experiment 1 tests.
- No seeds, repeats or variance are reported anywhere.
- Clean (p=0) HS for SFT is 52.3: the aligned starting point is already weakly safe.

## 8. Mapping to our arms

| Arm | Implementation | Faithful to |
|---|---|---|
| A1 | `safetune.runner.recover.AntidoteTrainer` defaults: ?=0.005, 64 prompts, o_proj/down_proj, merged model. | SafeTune as shipped. |
| A2 | `safetune.recover.apply_antidote(dataset=prompt+answer, target_modules=all 7, max_samples=2000, prune_fraction=?)`, merged model. | Paper text (model weights, per row). |
| A3 | Our port: Wanda on the attack adapter's `lora_A` / `lora_B`, chat-formatted prompt+answer calibration, git-disl k-rule per row. | git-disl code. |
| A4 | Random per-row pruning, same count per row, same parameter set as A2 (merged) / A3 (adapter), 3 mask seeds. | Missing paper control. |
| A5 | Magnitude-only, same per-row count. | Paper Table 13, |D|=0. |
| A6 | Per-module norm-matched shrink of the adapter update to the norm left by A3. | Lee et al.'s check 4 null. |
| A7 | A3 with benign (SST2) calibration. | Paper Table 12. |

Differences we keep on purpose:

- Attack LoRA r=16 / alpha=32 on all 7 projections. This follows Lee et al. and TamperBench, not r=256 / alpha=4 on q/k/v.
- 3 epochs, not 20.
- No separate alignment stage, since Tiny Aya Global is already aligned.
- StrongREJECT judge instead of beaver-dam-7b.
