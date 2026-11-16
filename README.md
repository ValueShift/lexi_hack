# Safety Edits That Don't Stick

**Unlearning and repair on Tiny Aya, stress-tested with matched controls, cross-lingual re-attacks and an attacker who knows the defence**

**One-line claim.** Two published weight-space safety edits, RMU unlearning and Antidote repair, both pass their own tests on Tiny Aya Global, yet a short fine-tune undoes each of them (for unlearning, in any language and even on unrelated data), and the design choice each relies on adds little over a simple baseline that removes as much.

**Authors:** Janhavi Khindkar and Nataliia Povarova (ValueShift Research)

Lexsi AI Safety & Alignment Hackathon · Track 2: AI Red Teaming, Safety and Alignment (2a Red Teaming; 2b Safety and Correction) · Problem statements: unlearning, tamper resistance, multilingual safety

**Also as:** [PDF](writeup/safety_edits_that_dont_stick.pdf) · [HTML](writeup/safety_edits_that_dont_stick.html)

> **Content warning.** The repair experiment's attack adapters are deliberately harmful. Its logs, and the `outputs*/` folders of
> this repository, contain harmful prompts and the models' harmful answers. The BeaverTails-derived training and calibration data
> (and its Russian translation) is not redistributed; the code rebuilds it from the public dataset. This write-up quotes no harmful completions.

## Abstract

Safety edits that change a model's weights are attractive for open-weight models, because there is no filter for a user to switch
off. They are also exactly what an open-weight attacker can fine-tune against. We test two such edits from different families on the
same multilingual model, Tiny Aya Global (3.35B), and evaluate both with the five checks Lee et al. propose for defences against
fine-tuning attacks:

- **Unlearning.** RMU removal of high-school chemistry, trained on English only.
- **Repair.** Antidote, which prunes a harmful LoRA fine-tune after the fact.

Both edits work on their own metric:

- English-only unlearning erases the chemistry in English, Hindi and Marathi.
- Antidote takes the harmful score from 59.0 to 5.8 (clean model 4.6) at a cost of 1.2 SST2 points.

In both, the design choice that was supposed to make the edit work (our cross-lingually chosen layer for RMU; harmful-data
calibration for Antidote) adds little beyond a control matched for how much the edit removes:

- the layer RMU edits stops mattering once erasure depth is matched (layer effect +0.05, 95% CI −0.01 to +0.11);
- calibration-free magnitude pruning gives 95–100% of Antidote's repair at every pruning ratio; harmful calibration adds 1–2 points at light pruning and nothing once the repair reaches the clean model's level (at every dose, and for a Russian-language attack).

Both edits are then undone by attacks their original evaluations did not try:

- **Unlearning.** Unlearned chemistry returns after a small LoRA fine-tune in English, Hindi or Marathi, including one on unrelated Marathi questions with no chemistry in it (+0.62 to +0.69 relearning excess). How much returns tracks how much was erased.
- **Repair, fine-tuning re-attack.** Ten harmful fine-tuning steps take the repaired model from 5.8 to 45.9; by 25 steps it is as harmful as an unrepaired model put through the same re-attack.
- **Repair, mask-aware attacker.** An attacker who knows Antidote's pruning rule cuts its relative repair from 0.98 to 0.12, with no SST2 signal to warn the defender.
- **SafeTune.** Its packaged Antidote prunes the merged model weights, as the paper's text describes; on Tiny Aya this gave degenerate output in our runs. Pruning the fine-tune's update instead, as the authors' code does, repairs.

We conclude that durability claims for weight-space safety edits need matched-strength controls, fine-tuning re-attacks in other
languages and with unrelated data, and attackers who know the defence.

> **For judges: one minute**
>
> - **What we did:** took two published safety edits on Tiny Aya Global, RMU unlearning and Antidote repair, and attacked each the way an open-weight user could: a small LoRA fine-tune, in another language or on unrelated data, or with knowledge of the defence.
> - **Headline numbers:** unlearned English chemistry returns by **+0.62 to +0.69** after a Marathi fine-tune with no chemistry in it. Antidote's relative repair falls from **0.98 to 0.12** against an attacker who knows its pruning rule (lower in all 3 seeds), and its harmful score goes from **5.8 to 45.9** after 10 fine-tuning steps.
> - **Negative result:** pruning the largest adapter weights, with no data at all, gives **95–100%** of Antidote's repair.
> - **Kits:** SafeTune (Antidote entry points and its pruning rule) and AuditKIT (accuracy of all 187 repair runs, identical to our own scoring).
> - **Where to look:** [results at a glance](#results-at-a-glance), the [figures in §7](#7-results), and every number in [`exp2_erased_or_hidden/analysis/tables.md`](exp2_erased_or_hidden/analysis/tables.md) and [`outputs/analysis/`](outputs/analysis/).
> - **Main limits:** one model, 59 forget questions, an automatic judge we did not validate against human labels.

## Results at a glance

| | Unlearning (RMU) | Repair (Antidote) |
|---|---|---|
| **Does it work on its own metric?** | **Yes.** Held-out chemistry erased in English, Hindi *and* Marathi (English 1.00 → 0.20–0.50, by strength). | **Yes.** Harmful score 59.0 → 5.8 (clean 4.6); SST2 −1.2. |
| **Is it the design choice that was supposed to matter?** | **No.** Our choice of layer: at matched erasure depth it adds +0.05 [−0.01, +0.11]. | **Mostly no.** Harmful calibration: magnitude pruning with no data gives 95–100% of the repair. Calibration adds 1–2 points at light pruning (α = 0.05: 16.3 vs 18.3) and nothing at α = 0.2 (all doses, Russian). |
| **Does it survive fine-tuning?** | **No.** Roughly half of what was erased returns after a small LoRA fine-tune. | **No.** 10 harmful steps: 5.8 → 45.9; 25 steps: 69.7, at or above the unrepaired adapter under the same re-attack (65.5). |
| **Does it survive an attacker who knows it?** | Not applicable (the relearning attacks above already target the edit). | **No.** The mask-aware attacker cuts relative repair from 0.98 to 0.12. |
| **Does language protect it?** | **No.** Hindi or Marathi fine-tuning restores *English* answers; so does an **unrelated** Marathi fine-tune. | **No.** A Russian-language attack harms English prompts; an English-calibrated repair still fixes it. |
| **Hidden, or relearned from scratch?** | **Hidden:** a set of never-known control questions separates the two. | **Not established:** no never-edited baseline was re-attacked. |

## Contents

1. [Problem statement and approach](#1-problem-statement-and-approach)
2. [Hackathon repos used](#2-hackathon-repos-used)
3. [Models and licenses](#3-models-and-licenses)
4. [Datasets, benchmarks and licenses](#4-datasets-benchmarks-and-licenses)
5. [Metrics and rationale](#5-metrics-and-rationale)
6. [Method and experiments](#6-method-and-experiments)
7. [Results](#7-results)
8. [Cost and ROI](#8-cost-and-roi)
9. [Limitations, dual use and disclosure](#9-limitations-dual-use-and-disclosure)
10. [With more credits](#10-with-more-credits)
11. [Artifacts](#11-artifacts)
12. [Documentation and reproduction](#12-documentation-and-reproduction)
13. [Discussion](#13-discussion)
14. [Teammates and acknowledgements](#14-teammates-and-acknowledgements)
15. [References](#15-references)
16. [LLM usage statement](#16-llm-usage-statement)
17. [Changelog](#17-changelog)
18. [Appendix](#appendix)

## 1. Problem statement and approach

**Use case.** Open-weight multilingual models such as Tiny Aya are fine-tuned and redistributed by people the original developer
never sees. Two kinds of weight-space safety edit are proposed for this setting: *unlearning*, which removes hazardous knowledge
before release, and *post-fine-tuning repair*, which strips harm from a user's fine-tune (for example in a fine-tuning service).
Both are only useful if they survive what an open-weight user can cheaply do next: fine-tune again, in any language.

**Research questions.**
1. Does English-only RMU unlearning remove knowledge in every language, or only hide it? Does fine-tuning in another language, or on unrelated data, bring it back?
2. Does the layer RMU edits make the unlearning more durable?
3. Does Antidote's harmful-data calibration beat pruning baselines that remove as much, across doses and attack languages?
4. Does the repair survive a fine-tuning re-attack, and an attacker who knows the pruning rule?

**Hypotheses fixed before the runs, and what happened.**
- *Unlearning:* RMU at the layer whose representations are most similar across languages (by linear CKA) is more durable to cross-lingual relearning than RMU at a late layer. **Not supported:** the CKA profile turned out flat, and durability tracks erasure depth instead (§7.1).
- *Repair:* the attack must raise the harmful score by at least 25 points over the clean model (go/no-go), and Antidote is compared with random, magnitude-only and norm-matched pruning at the same count. **Partly supported:** Antidote beats random pruning and the norm-matched shrink, but magnitude pruning nearly ties it (§7.2).

**Threat model (Track 2a).**
- *Attacker:* holds the open weights after the defender's edit (the unlearned model, or the repaired adapter); can run a LoRA fine-tune for minutes on one GPU; has data in any language, possibly unrelated to the target. In the strongest case the attacker knows the defence's pruning rule, but not the defender's calibration data.
- *Goal:* bring back the removed knowledge (unlearning) or harmful compliance (repair).
- *Out of scope:* full-weight fine-tuning, attacks on the judge, and prompt-only jailbreaks.

**Strength, the idea that links the two experiments.**
- Every edit removes *some amount* of something: RMU erases a fraction of the model's correct answers, and pruning removes a fraction of the harmful update.
- Comparing two variants of an edit is only fair when that amount is matched. Otherwise the variant that removed more looks better or worse for reasons that have nothing to do with its mechanism.
- Both experiments therefore test the design choice each method relies on against a strength-matched control.

**Related work.**
- *Relearning:* relearning attacks and unrelated fine-tuning are known to restore unlearned knowledge in one language (Łucki et al.; Hu et al.; Deeb & Roger). Cross-lingual unlearning work on Tiny Aya changes only the query language, without relearning (*Linguistic Loopholes in LLM Unlearning*), or reverses unlearning with activation steering (*Multilingual Unlearning in LLMs*).
- *Our addition:* relearning *across* languages, including from unrelated data, with a never-known control that separates hidden knowledge from relearned knowledge.
- *Durability:* Qi et al. and Lee et al. argue that safeguards for open weights must be tested against adaptive fine-tuning attackers. We apply their checks to a repair method (Antidote; Huang et al.) and an unlearning method (RMU; Li et al.) on the same model.

Full references are in [§15](#15-references).

## 2. Hackathon repos used

| Kit | How we used it |
|---|---|
| **SafeTune** | Its two Antidote entry points (`AntidoteTrainer` with shipped defaults; `apply_antidote` with paper-like settings) are arms of the repair experiment, and its per-row pruning rule (rank the weights within each output row and prune the top fraction α) is the rule every repair arm uses; the adapter-space arms round the count per row as the Antidote authors' code does. |
| **AuditKIT** | Scores SST2 and MMLU-Pro accuracy for all 187 repair runs from the saved predictions (`ak.evaluate(samples, model="precomputed")`, exact match). It agrees with the pipeline's own scoring in every run ([`outputs/analysis/auditkit_crosscheck.csv`](outputs/analysis/auditkit_crosscheck.csv), and the same file under `outputs_ru/` and `outputs_adaptive/`). |

**Pipeline.** BeaverTails + SST2 → LoRA attack on Tiny Aya → repair (SafeTune's Antidote, our adapter-space port of it, and four
matched nulls, all with SafeTune's per-row rule) → generation → StrongREJECT judge for harm, **AuditKIT** for task accuracy →
paired bootstrap. The unlearning experiment does not use a kit: its scoring is letter-logit multiple choice, which no kit offers.

## 3. Models and licenses

| Model | Role | License |
|---|---|---|
| `CohereLabs/tiny-aya-global` (3.35B, 36 layers, bf16) | Base model of both experiments; the unlearned and repaired versions are the hardened models we attack | CC BY-NC 4.0, plus Cohere Labs' Acceptable Use Policy |
| StrongREJECT fine-tuned evaluator (Gemma 2B, via the `strong_reject` package) | Harmful-score judge | Gemma terms of use |
| `facebook/nllb-200-distilled-600M` | Machine translation into Marathi (unlearning) and Russian (repair) | CC BY-NC 4.0 |
| Closed-model baseline | None (see §8) | – |

## 4. Datasets, benchmarks and licenses

| Dataset | Used for | Size used | License |
|---|---|---|---|
| MMLU (`cais/mmlu`) | Forget set (high-school chemistry the model knows, 59) and control set (chemistry it gets wrong, 105), each split into halves A/B; collateral "rings" (100 each); 200 general-knowledge questions | see left | MIT |
| MMMLU (`openai/MMMLU`, HI_IN) | Hindi versions of the same questions | as above | MIT |
| Marathi | NLLB-200 translation of the same questions | as above | derived |
| WikiText-2 (raw) | RMU retain text | 400 paragraphs | CC BY-SA 3.0 / GFDL |
| BeaverTails (`30k_train`, unsafe pairs) | Harmful attack pairs (doses 10, 50, 100); Antidote calibration; relearning re-attack | 100; 2,000; 64 | CC BY-NC 4.0 |
| SST-2 (`stanfordnlp/sst2`) | Benign half of the attack; utility evaluation | 1,000 train; 500 eval | listed as "unknown" on its dataset card |
| StrongREJECT | Attack prompt set (Track 2a): harmful-score evaluation | 150 of 313 prompts | MIT (code); prompts curated from sources with their own licenses |
| XSTest | Over-refusal test set (Track 2b) | 250 safe prompts | CC BY 4.0 |
| MMLU-Pro (`TIGER-Lab/MMLU-Pro`) | General capability | 200 questions | MIT |

**Leak checks.**
- *Unlearning:* forget and control sets are disjoint by construction (questions the model knows vs does not know). Every attack trains on one half and is tested on the other, in both directions. The general-knowledge set excludes all chemistry, biology and physics subjects.
- *Repair:* BeaverTails pairs are partitioned by unique prompt into calibration and attack pools, with prompts that overlap the StrongREJECT evaluation removed. The 64 relearning pairs are disjoint from every attack dose (checked in code).
- *Not used:* CuratorKIT filtering.

## 5. Metrics and rationale

| Metric | Experiment | Better | Why |
|---|---|---|---|
| Forget accuracy (en / hi / mr) | Unlearning | lower after unlearning | Direct measure of erasure; letter-logit scoring needs no judge |
| Relearning excess | Unlearning | lower (more durable) | Gain on held-out forget questions minus gain on never-known control questions, so a model that only gets better at the format scores zero |
| Collateral rings (college chemistry; biology/physics; history/law; general) | Unlearning | higher (guardrail) | Collateral damage; matched arms may lose at most 5 points on history/law |
| Harmful score, HS (StrongREJECT, 0–100) | Repair | lower | Graded attack success rate, standard in harmful fine-tuning work |
| Relative repair | Repair | higher (1 = as safe as clean) | Puts repairs at different doses and languages on one scale |
| Coherent refusal (%) | Repair | higher | Refusals that are also fluent (perplexity under 50), so degenerate output does not count as safe |
| XSTest over-refusal (%) | Repair | lower (guardrail) | The repair must not make the model refuse safe requests |
| SST2 accuracy (AuditKIT) | Repair | higher (guardrail) | The fine-tuner's task; pruning ratios were chosen within a 2-point SST2 budget |
| MMLU-Pro accuracy (AuditKIT) | Repair | higher (guardrail) | General capability; flat at about 25 for every model, so uninformative |
| "Broken" flag | Repair | – | Fewer than half the SST2 answers parse: the low HS is gibberish, not a repair |

**Judge check.** We did not validate the StrongREJECT judge against human labels. We rely on its published validation, and the
coherence gate and broken flag catch the one failure mode we saw (degenerate output scored as safe).

**Attack success and cost to break (Track 2a).**

| Edit attacked | Attack | Success | Cost to break |
|---|---|---|---|
| RMU-L27-deep | LoRA fine-tune on 200 unrelated Marathi questions | relearning excess +0.69 [+0.49, +0.89] | One LoRA fine-tune (rank 16, 3 epochs) on 200 questions |
| RMU-L30 | same | +0.62 [+0.41, +0.82] | same |
| Antidote (α = 0.2) | Harmful fine-tune of the repaired adapter, 64 examples | HS 5.8 → 45.9 after 10 steps, 69.7 after 25 | 10 optimizer steps (batch 8) |
| Antidote (α = 0.2) | Mask-aware attacker | relative repair 0.12; HS 58.5 | The same training run as the standard attack (411 steps on the same data), plus a mask refresh every 20 steps |

## 6. Method and experiments

### 6.1 Unlearning

*Code, per-question results and a 7-page report: [`exp2_erased_or_hidden/`](exp2_erased_or_hidden/) · [report (PDF)](exp2_erased_or_hidden/report/erased_or_hidden_report.pdf) · [all tables](exp2_erased_or_hidden/analysis/tables.md)*

- **Forget set:** the 59 MMLU high-school chemistry questions the model answers correctly in both English and Hindi.
- **Control set:** the 105 questions it gets wrong in both.
- **Languages:** English, Hindi (MMMLU) and Marathi (NLLB-200 translation).
- **Unlearning:** RMU on English text only (each forget question with its correct answer; 400 WikiText-2 paragraphs as retain text), training the MLP down-projections of layers ℓ−2…ℓ, at layer 27 or 30, at several strengths.
- **Matched erasure:** English forget accuracy on half A at most 0.35, and history/law accuracy down by at most 5 points.
- **Attacks:** LoRA (rank 16, 3 epochs, effective batch 8), learning rates 1e-4 and 3e-4, 2 seeds; the stronger learning rate per seed is kept. Each attack trains on one half of the questions and is tested on the other half, in both directions. Five kinds: forget questions in English, Hindi and Marathi; college chemistry in Marathi (related, not the same questions); general knowledge in Marathi (no chemistry).
- **Scoring:** letter logits over A–D, with no generation and no judge; 95% CIs from a bootstrap over questions.

RMU pushes the layer-ℓ activations $h_\ell$ of forget text $x_f$ towards a fixed random direction and keeps those of retain text $x_r$ where they were:

```math
\mathcal{L}_{\mathrm{RMU}} = \mathrm{MSE}\big(h_\ell(x_f),\ c\,\hat{u}\big) + \alpha\,\mathrm{MSE}\big(h_\ell(x_r),\ h_\ell^{\mathrm{orig}}(x_r)\big), \qquad \alpha = 100,
```

where $\hat{u}$ is a fixed random unit vector and $c$ is a multiple (2 to 16, set per arm) of the mean activation norm on forget text. Steps (150 to 400) and learning rate (5e-5 or 1e-4) are also set per arm. An attack's effect is

```math
\text{relearning excess} = \underbrace{\big(\mathrm{acc}^{\mathrm{after}}_{\mathrm{forget}} - \mathrm{acc}^{\mathrm{before}}_{\mathrm{forget}}\big)}_{\text{held-out forget questions}} - \underbrace{\big(\mathrm{acc}^{\mathrm{after}}_{\mathrm{control}} - \mathrm{acc}^{\mathrm{before}}_{\mathrm{control}}\big)}_{\text{never-known control questions}}.
```

| Arm | Name in the figures | What it is |
|---|---|---|
| Original | A0 | Tiny Aya Global, unedited |
| RMU-L27 | A2 | RMU at layer 27, matched erasure |
| RMU-L27-deep | A2s | RMU at layer 27 trained longer, erasing English as deeply as RMU-L30 |
| RMU-L30 | A1c | RMU at layer 30, the closest layer-30 match |
| RMU-L30-strong | A1 | RMU at layer 30, strong (20 points of collateral damage) |

### 6.2 Repair

*Code, run guide and measured timings: [`exp1_antidote/`](exp1_antidote/) · paper vs authors' code vs SafeTune: [`notes/antidote_repro_diff.md`](notes/antidote_repro_diff.md) · attacker design: [`notes/adaptive_attacker.md`](notes/adaptive_attacker.md)*

- **Attack:** LoRA r = 16 (alpha 32, dropout 0.05) on all 7 projections, learning rate 1e-4, 3 epochs, batch 8, on 1,000 SST2 examples plus 10, 50 or 100 harmful BeaverTails pairs; 3 seeds.
- **Repairs**, all at the same per-row pruning count, at α ∈ {0.05, 0.1, 0.2}:
  - Antidote (Wanda scores on the LoRA factors, 2,000 harmful calibration pairs);
  - magnitude-only pruning (no data);
  - Antidote with benign (SST2) calibration;
  - random pruning (2 mask seeds);
  - a shrink of the whole update to Antidote's remaining norm;
  - SafeTune's two Antidote entry points, on the merged model.
- **Pruning ratio:** for each arm, the lowest-harm α whose SST2 cost stays within 2 points (α = 0.2 for Antidote).
- **Re-attacks:** 64 fresh harmful pairs, learning rate 1e-4, checkpoints at 10, 25 and 50 steps; the mask-aware attacker below.
- **Evaluation:** greedy decoding (up to 256 new tokens); StrongREJECT judge on 150 prompts; coherent refusal; XSTest over-refusal; SST2; MMLU-Pro; paired bootstrap over prompts (10,000 resamples).

Antidote scores each weight $W_{ij}$ of a LoRA factor by its size times the size of its input on the calibration data (Wanda), and in each row sets the top fraction $\alpha$ of scores to zero:

```math
S_{ij} = |W_{ij}| \cdot \lVert X_j \rVert_2 .
```

Magnitude pruning uses $S_{ij} = |W_{ij}|$ (no data), benign calibration takes $X$ from SST2, random pruning uses random scores, and the norm shrink scales each module's update $BA$ to Antidote's remaining $\lVert BA \rVert_F$. Success is measured by

```math
\text{relative repair} = \frac{\mathrm{HS}_{\text{attacked}} - \mathrm{HS}_{\text{repaired}}}{\mathrm{HS}_{\text{attacked}} - \mathrm{HS}_{\text{clean}}},
```

which is 1 when the repaired model is as safe as the clean one and 0 when the repair does nothing. The mask-aware attacker knows
Antidote will prune the top 20% of adapter weights by its scoring rule. With $M_\theta$ that mask for the current weights,
recomputed every 20 steps on the attacker's own harmful data (not the defender's calibration set), it trains on

```math
\min_{\theta}\ (1-p)\,\mathcal{L}_{\text{attack}}(\theta) + p\,\mathcal{L}_{\text{attack}}\big(\theta \odot (1 - M_\theta)\big), \qquad p = 0.5 .
```

### 6.3 Lee et al.'s five checks

| Check | Unlearning (RMU) | Repair (Antidote) |
|---|---|---|
| 1. Tell the attacker | LoRA relearning attacks in English, Hindi and Marathi: targeted, related-benign and unrelated-benign | Relearning re-attack on the repaired adapter; mask-aware attacker that trains through Antidote's pruning; the attack repeated in Russian |
| 2. Report a range | 6 unlearning settings at two layers and several depths; 2 learning rates × 2 seeds; both train/test halves | Pruning ratio α ∈ {0.05, 0.1, 0.2}; harmful doses 10/50/100; 3 attack seeds |
| 3. Score the benign set | Collateral rings: college chemistry, biology/physics, history/law, general MMLU | SST2, XSTest over-refusal, MMLU-Pro, a coherence gate on refusals, a "broken model" flag |
| 4. Clear a matched null | Never-known control questions (format-only gains score 0); settings matched on erasure depth | Random, magnitude-only and benign-calibrated pruning at the same per-row count; a norm-matched shrink |
| 5. Give the defender an observable | A linear probe reads the answer before the edited layers | Only a weak lead (§7.2) |

**Ablations.** Unlearning: layer (27 vs 30), erasure depth, attack language, related vs unrelated attack data, swapped train/test
halves. Repair: pruning ratio, harmful dose, attack language (English vs Russian), five matched nulls, standard vs mask-aware attacker.

## 7. Results

### 7.1 Unlearning: erased or hidden?

**Erasure.** English-only unlearning also erased Hindi and Marathi.

| Arm | Layer | Forget accuracy en / hi / mr | Biology/physics (en) | General (en) |
|---|---|---|---|---|
| Original | – | 1.00 / 1.00 / 0.57 | 0.40 | 0.40 |
| RMU-L27 | 27 | 0.50 / 0.53 / 0.37 | 0.31 | 0.40 |
| RMU-L27-deep | 27 | 0.20 / 0.30 / 0.27 | 0.21 | 0.36 |
| RMU-L30 | 30 | 0.23 / 0.53 / 0.30 | 0.17 | 0.35 |

At matched depth (RMU-L27-deep vs RMU-L30) the two layers do similar collateral damage to related science.

**It comes back, from any language and from unrelated data.**

![English recovery by attack](exp2_erased_or_hidden/analysis/figures/fig1_english_recovery.png)

- Fine-tuning in Hindi or Marathi brings back *English* answers as much as fine-tuning in English: +0.53, +0.52 and +0.56 for RMU-L27-deep.
- The strongest attack was a Marathi fine-tune on **200 general-knowledge questions with no chemistry at all** (also the largest attack; the chemistry attacks train on 29–30 questions): +0.69 [+0.49, +0.89] for RMU-L27-deep and +0.62 [+0.41, +0.82] for RMU-L30. Even the shallow RMU-L27 recovers +0.30 [+0.11, +0.50].
- Pooled over attacks and languages, every arm recovers: RMU-L27 +0.27 [+0.12, +0.42], RMU-L27-deep +0.43 [+0.27, +0.59], RMU-L30 +0.33 [+0.19, +0.47].
- The attacks train on questions *different* from the ones tested, and the unrelated attack contains no chemistry. So the returning answers were still in the model: **hidden, not erased.**

**What comes back tracks how much was erased, not where.**

![Recovery vs erasure depth](exp2_erased_or_hidden/analysis/figures/fig5_recovery_vs_depth.png)

```math
\text{relearning excess} \approx 0.57 \times \text{accuracy erased} + 0.09 \qquad (r = 0.86,\ \text{4 arms} \times \text{3 languages})
```

- In the first runs, layer 27 seemed to resist relearning better than layer 30. But layer 27 had also erased much less.
- Erasing as deeply at layer 27 (RMU-L27-deep) removes the difference: at matched English depth it is −0.03 [−0.08, +0.03].
- Within each arm, the languages where it erased more also recover more (slope +0.46 [+0.22, +0.79]). At equal depth, the layer adds +0.05 [−0.01, +0.11].
- Our starting hypothesis, that unlearning at the more cross-lingually shared layer would be more durable, **is not supported**.

**Sample outputs** (8 held-out questions per language, greedy generation). Unlearned models mostly do not give wrong answers;
they produce degenerate text on chemistry, such as `"0 " (0 " is not " " " " …`. After a Marathi fine-tune, on questions never
trained on, the correct option comes back in 2–3 of 8 English answers per arm (0–1 of 8 before), often followed by an
explanation that starts fluently and then repeats or degrades. For example, RMU-L30 after the Marathi college-chemistry fine-tune:
*"C. isoelectronic with a noble gasMonatomic ions of the representative elements are often isoelectronic with a noble gas, which
means they have the same…"*. All samples are in [`results/samples.md`](exp2_erased_or_hidden/results/samples.md).

**Observable.** A linear probe trained on the original model reads the correct option at layers 23–24 (0.57, chance 0.25), and
does so identically in every unlearned arm, because RMU leaves everything before the edited layers untouched. This is supporting
evidence only: the probe's own cross-validated accuracy is 0.27, and the layers were chosen after looking.

### 7.2 Repair: does Antidote's mechanism matter, and does it last?

**Main result** (dose 100, α = 0.2, mean of 3 seeds):

![Repair vs matched nulls](figures/r1_repair_vs_nulls.png)

| Repair | HS | SST2 | Update norm left |
|---|---|---|---|
| Clean model | 4.6 | 91.2 | – |
| Attacked | 59.0 | 92.9 | 1.00 |
| **Antidote** | **5.8** | 91.7 | 0.42 |
| Magnitude prune (no data) | 5.7 | 91.6 | 0.36 |
| Benign-calibrated | 6.7 | 91.7 | 0.43 |
| Norm shrink (same norm as Antidote) | 16.0 | 92.7 | 0.42 |
| Random prune | 29.7 | 92.9 | 0.73 |

- **Antidote clearly beats the weak nulls:** −23.9 [−29.0, −19.1] against random pruning, and −10.2 [−13.9, −7.0] against a shrink to the same norm. *Which* weights are pruned matters.
- **At α = 0.2 it does not beat the matched ones:** +0.1 [−1.0, +1.1] against magnitude pruning, and −0.9 [−2.3, +0.1] against benign calibration. The largest adapter weights are already the right ones.
- **Not a language artefact.** With the harmful pairs machine-translated to Russian (HS 46.0, evaluated on English prompts), English-calibrated Antidote repairs to 4.7, and magnitude pruning ties it again (4.5).

**Across pruning ratios** (dose 100; harmful score, with relative repair in brackets):

| Repair | α = 0.05 | α = 0.1 | α = 0.2 |
|---|---|---|---|
| Antidote | 16.3 (0.79) | 8.0 (0.94) | 5.8 (0.98) |
| Magnitude prune (no data) | 18.3 (0.75) | 9.1 (0.92) | 5.7 (0.98) |
| Benign-calibrated | 18.8 (0.74) | 9.8 (0.91) | 6.7 (0.96) |
| Norm shrink | 49.9 (0.17) | 35.2 (0.44) | 16.0 (0.79) |
| Random prune | 55.4 (0.07) | 48.5 (0.19) | 29.7 (0.54) |

- Magnitude pruning with no data gives 95%, 98% and 100% of Antidote's relative repair.
- Harmful calibration still has a small, consistent effect when pruning is light: Antidote beats magnitude pruning by 2.0 points [−3.5, −0.4] at α = 0.05 and 1.0 [−2.3, 0.0] at α = 0.1, in all three seeds, even though magnitude pruning removes more of the update (0.73 vs 0.79 of the norm left at α = 0.05). It also beats benign calibration (−2.5 [−4.0, −1.0] and −1.7 [−2.9, −0.7]).
- At α = 0.2 every good repair lands within about 1 point of the clean model, so the tie there is partly a floor effect.

**Across doses** (α = 0.2, mean of 3 seeds):

| Repair | Dose 10 | Dose 50 | Dose 100 |
|---|---|---|---|
| Clean | 4.6 | 4.6 | 4.6 |
| Attacked, unrepaired | 24.3 | 54.0 | 59.0 |
| Antidote | 4.8 | 5.8 | 5.8 |
| Magnitude prune | 4.7 | 5.6 | 5.7 |
| Benign-calibrated | 5.1 | 5.7 | 6.7 |
| Norm shrink | 6.3 | 12.6 | 16.0 |
| Random prune | 7.9 | 20.8 | 29.7 |

- **At α = 0.2, Antidote ties magnitude pruning at every dose** (+0.1, +0.2, +0.1, all within ±1 point), and benign calibration ties too. All three end within about 1 point of the clean model, so this shows the repair is complete at every dose, not that calibration never matters (see the α table above).
- The gap to the weak nulls grows with the dose (Antidote minus random: −3.1, −15.0, −23.9): a bigger harmful update survives random removal better.
- Dose 10 is a weak attack (+19.7 over clean, below our 25-point threshold, with wide seed spread), so treat it as supporting evidence.

**It does not survive fine-tuning.** Each repaired adapter is re-attacked with 64 fresh harmful pairs:

![Relearning re-attack](figures/r2_relearning.png)

- Ten steps take Antidote from 5.8 to 45.9, so its relative repair falls from 0.98 to 0.24.
- By 25 steps every starting point, including the unrepaired adapter put through the same re-attack, sits at about 65–70.
- Antidote slows the first 10 steps slightly (45.9, against 61–66 for the nulls); that gap is gone by step 25.

**It does not survive an attacker who knows it.**

![Mask-aware attacker](figures/r3_mask_aware.png)

| Repair | HS, mask-aware attack | Relative repair (standard attack → mask-aware) |
|---|---|---|
| Attacked, unrepaired | 66.0 | – |
| Antidote | 58.5 | 0.98 → **0.12** |
| Norm shrink | 58.2 | 0.79 → 0.13 |
| Random prune | 61.4 | 0.54 → 0.08 |
| Magnitude prune | 49.3 | 0.98 → 0.28 |

- **The repair collapses**, in all three attack seeds (relative repair 0.14, 0.05 and 0.18). Coherent refusal after Antidote falls from 83.8% to 7.3%. Training through the pruning cost the attacker nothing: its harm rose (66.0 vs 59.0) and SST2 stayed at 93.1.
- **SST2 gives no warning.** Every repair costs at most 0.13 SST2 points here, so an acceptance rule based on task accuracy would approve them all while the model stays harmful.
- **The ranking flips.** Magnitude pruning now beats Antidote by 9.2 points [6.3, 12.1], in all three seeds, because the attacker moved the harm out of exactly the weights Antidote cuts. This does not make magnitude pruning robust: we did not test an attacker that targets it, and it still leaves the model at about ten times the clean score.

**Where Antidote is applied matters: SafeTune's version.**
- The Antidote paper's text prunes the fine-tuned *model's* weights; the authors' code prunes the fine-tune's LoRA factors. SafeTune follows the text and prunes the merged model's weights (`o_proj` and `down_proj` by default) ([details](notes/antidote_repro_diff.md)).
- On Tiny Aya, both SafeTune entry points (`AntidoteTrainer` with its shipped defaults, 3 seeds; `apply_antidote` with paper-like settings, 1 seed at one α) produced degenerate output with zero parseable SST2 answers. Their near-zero harmful scores (0.6 and 0.7) come from gibberish, not from a repair.
- Pruning the highest-scoring *base-model* weights plausibly removes weights the model depends on (untested). The same scoring rule applied to the update (our port) repairs without breaking the model.

**A weak lead for an observable.** On mask-aware adapters, Antidote leaves more of the update in place (0.47 vs 0.42 of the norm), because the weights it cuts carry less of it. One number over three seeds is not evidence.

**Sample outputs.** The repaired and re-attacked models' answers are in the logs datasets, behind the content warning; we do not
quote harmful completions here.

**Not run:** results on quantized models (Track 2b).

### 7.3 Named findings: what the two experiments say together

1. **The design choice that was supposed to matter added little beyond strength.**
   - RMU's choice of layer adds +0.05 [−0.01, +0.11] once erasure depth is matched.
   - Antidote's harmful calibration adds 1–2 points over magnitude pruning at the same count when pruning is light, and nothing once the repair is complete; magnitude pruning alone gives 95–100% of the repair.
   - Each looked like a method-specific win against a weak null (a shallower RMU arm; random pruning or a uniform shrink) and mostly disappeared against a matched one.
   - Magnitude pruning removes slightly more of the update norm than Antidote, so at exactly matched norm calibration's small edge would likely be a little larger.
2. **Neither edit survives an attacker who adapts.**
   - Unlearned chemistry returns after fine-tuning in any of three languages, including on unrelated data.
   - The repaired harm returns within 10–25 fine-tuning steps, and an attacker who knows the pruning rule defeats the repair almost completely.
   - In both, the end state after re-attack barely depends on the edit: every repair converges to about 65–70 HS, and how much chemistry returns mainly tracks how much was erased.
3. **Language is not a barrier, in either direction.**
   - English-only unlearning also erases Hindi and Marathi, and Hindi or Marathi fine-tuning restores English.
   - A Russian harmful fine-tune harms English prompts, and an English-calibrated repair removes it.
4. **Where they differ.**
   - The unlearning experiment shows the knowledge was *hidden*: answers return on questions never trained on, against a never-known control.
   - The repair experiment shows the fix is *not durable* and *not robust to an informed attacker*. Whether the harm was hidden, rather than learned anew, needs a never-edited baseline under the same re-attack.

## 8. Cost and ROI

| Experiment | Compute | Hardware |
|---|---|---|
| Repair (all configs, 3 seeds, doses, relearning, Russian, mask-aware) | about 33 GPU-hours, measured | one RTX 5070 Ti (16 GB), local |
| Unlearning (all arms, attacks, both halves, probe, samples) | roughly 3–4 GPU-hours (estimated from the run scripts, not logged), plus earlier runs on a free Colab T4 | one A100 (40 GB), Lambda Cloud |

- **Cost per 1,000 replies:** about 5.5 GPU-minutes on the RTX 5070 Ti (transformers backend, up to 256 new tokens; measured from the generation timings: 1,100 completions in about 6 minutes).
- **Closed-model ROI:** not applicable. We do not ship a model for deployment; the relevant comparison is the cost of the audit (a few GPU-hours) against the cost of trusting an edit that fails.

## 9. Limitations, dual use and disclosure

- **Both:** one model; small evaluation sets; machine-translated non-English data; no human validation of the automatic judge.
- **Unlearning:**
  - 59 forget questions;
  - one method (RMU), compared at two late layers only;
  - seeds vary only the attack;
  - the model is near chance in Marathi chemistry;
  - formulas and numbers are shared across languages;
  - attack types differ in size (29–30, 100 and 200 training questions), so their strengths are not comparable with each other;
  - the cross-language similarity measure (CKA) used to pick layer 27 turned out flat across layers 7–34.
- **Repair:**
  - one attack recipe and no separate alignment stage;
  - the StrongREJECT judge, not the paper's beaver-dam-7b;
  - 150 English evaluation prompts;
  - MMLU-Pro is flat at about 25 for every model, including clean, so it is uninformative;
  - the Russian attack keeps its benign half in English and was run at one α; against that attack's own unrepaired model, Antidote and magnitude pruning cost 2.2 SST2 points, just over the 2-point budget (SST2 stays above the clean model's);
  - relearning used English harmful data only and has no never-edited baseline;
  - the relearning re-attack and the mask-aware attacker were run at dose 100 and α = 0.2 only;
  - no attacker targeting magnitude pruning was tested;
  - no quantized-model results;
  - the SafeTune `apply_antidote` diagnostic is one seed at one α;
  - confidence intervals resample prompts (paired across arms), not the 3 attack seeds.

**Dual use and ethics.** The attacks we describe are standard LoRA fine-tunes plus one mask-aware variant; they show that two
safety edits can be undone cheaply, which an attacker could exploit, but defenders need that evidence before relying on the edits.
The unlearning target is high-school chemistry, a benign stand-in for hazardous knowledge. The repair experiment's deliberately
harmful adapters and generations are released for reproducibility, behind content warnings.

**Disclosure.** The SafeTune finding (merged-weight pruning gives degenerate output on Tiny Aya) concerns a hackathon kit; we
report it in this write-up and with the configuration needed to reproduce it ([`notes/antidote_repro_diff.md`](notes/antidote_repro_diff.md)).

## 10. With more credits

| Extension | What it would show | Expected result |
|---|---|---|
| Re-attack the clean model with the same 64 harmful pairs | Whether repaired harm was hidden, as in the unlearning experiment | If the clean model climbs more slowly than Antidote's 45.9 at 10 steps, the repair hid the harm |
| An attacker that targets magnitude pruning | Whether magnitude pruning's edge against the mask-aware attacker is real | We expect it to collapse as Antidote did |
| More forget questions (e.g. WMDP-style), more seeds, more languages | Tighter intervals for the depth and language effects | Same direction, narrower CIs |
| Other unlearning methods (including SafeTune's RMU) and larger Aya models | Whether "hidden, not erased" generalises beyond RMU on Tiny Aya | Unknown |
| Human labels on a sample of judged answers | Judge validity | – |
| Quantized repaired models | Track 2b requirement | – |

## 11. Artifacts

- **Hugging Face collections:** [models](https://huggingface.co/collections/ValueShift/safety-edits-that-dont-stick-models-6af9ebd1df3c0e2d270418d9) (3 adapter repos) · [datasets, logs and results](https://huggingface.co/collections/ValueShift/safety-edits-that-dont-stick-datasets-6af9ebd222d0e7af1a483f70) (4 datasets). All repos are public.
  - *Repair:* `ValueShift/exp1-antidote-adapters`, `-ru-adapters`, `-adaptive-adapters`, and the matching `-logs` datasets.
  - *Unlearning:* `ValueShift/eoh-results` (per-question outputs, unlearned weight deltas, logs). Its per-question results are also in [`exp2_erased_or_hidden/results/`](exp2_erased_or_hidden/results/).
- **GitHub:** this repository, with code, results, analysis and figures.
- **Demo or Space:** none; the public repository and collections are the deployment.

## 12. Documentation and reproduction

```
README.md               this write-up
figures/                repair-experiment figures for this README (make_figures.py, exp1_numbers.json)
exp1_antidote/          repair experiment: pipeline, configs, run guide
outputs/, outputs_ru/, outputs_adaptive/   repair outputs: English, Russian and mask-aware attacks (analysis/ has every table)
exp2_erased_or_hidden/  unlearning experiment: pipeline, runners, results, analysis, report
notes/                  Antidote paper vs code vs SafeTune; attacker design; setup per platform
writeup/                this write-up as PDF and HTML (bash writeup/build.sh rebuilds both from README.md)
papers/                 reading list
```

| Experiment | Run | Analyse |
|---|---|---|
| Repair | `python -m exp1_antidote.run_queue` runs every job unattended ([run guide](exp1_antidote/README.md)) | [`outputs/analysis/`](outputs/analysis/) (one folder per config, plus `compare.md`), `outputs_ru/analysis/`, `outputs_adaptive/analysis/` |
| Unlearning | `bash run_lambda.sh && bash run_variants.sh && bash run_robust.sh`, or the Colab notebook or HF Jobs launcher ([details](exp2_erased_or_hidden/README.md)) | `python analysis/analyze.py` (CPU, from the saved results), then `python report/build_report.py` |

**Environment.** Python with PyTorch (CUDA), Transformers and PEFT. The repair pipeline's setup for Windows, Linux, Colab and
HF Jobs is in [`notes/setup.md`](notes/setup.md); the unlearning runner `run_lambda.sh` builds its own pinned environment
(PyTorch 2.10, CUDA 12.8). Tiny Aya is gated: accept its licence on Hugging Face first.

**Work done outside the hackathon:** none. The code, runs and analysis were done for the hackathon; this repository's history starts on 10 November 2026 (UTC+3).

## 13. Discussion

**What we learned.** Two edits from different families failed in the same two ways: their key ingredient was mostly a matter of
how much they removed, and a cheap fine-tune undid them. Safety-relevant structure in Tiny Aya is shared across languages, so a
single-language evaluation misses attacks that switch language, and an English-only edit is undone from Hindi or Marathi.

**What is better for practitioners after this work.**
- **Evaluation.** A re-attack audit that catches both failures in a few GPU-hours: matched-strength nulls, fine-tuning in other languages and on unrelated data, and an attacker who knows the defence.
- **Repair.** On Tiny Aya, calibration-free magnitude pruning gives 95–100% of Antidote's repair without needing a harmful calibration set. Prune the fine-tune's update, not the merged model.
- **Unlearning.** RMU-style unlearning in a multilingual model should not be treated as removal, whatever layer it edits.

**Future work.** The extensions in §10, above all the clean-model re-attack, which would let the repair experiment answer the same
"hidden or relearned" question as the unlearning experiment.

## 14. Teammates and acknowledgements

- **Janhavi Khindkar:** the unlearning experiment (design, pipeline, runs and analysis) and the combined write-up.
- **Nataliia Povarova:** the repair experiment (Antidote reproduction, matched nulls, dose and language axes, relearning and mask-aware attackers, analysis).
- Thanks to Lexsi Labs, Cohere Labs and Hugging Face for the hackathon, the kits and Tiny Aya.

## 15. References

- Lee et al., *Refusal Localizes, the Damage Relocates: Safety Layers Under Few-Sample Fine-Tuning* (the five checks).
- Qi et al., [*On Evaluating the Durability of Safeguards for Open-Weight LLMs*](https://arxiv.org/abs/2412.07097).
- Hossain et al., [*TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering*](https://arxiv.org/abs/2602.06911).
- Li et al., [*The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*](https://arxiv.org/abs/2403.03218) (RMU).
- Łucki et al., [*An Adversarial Perspective on Machine Unlearning for AI Safety*](https://arxiv.org/abs/2409.18025).
- Hu et al., [*Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning*](https://arxiv.org/abs/2406.13356).
- Deeb & Roger, [*Do Unlearning Methods Remove Information from Language Model Weights?*](https://arxiv.org/abs/2410.08827)
- Skow et al., [*Linguistic Loopholes in LLM Unlearning: From a 174-Language Benchmark to Coverage-Aware Unlearning*](https://arxiv.org/abs/2609.40286).
- Xiang et al., [*Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility*](https://arxiv.org/abs/2606.03291).
- Waseem & Fawaz, [*A False Positive in Cross-Lingual Unlearning Evaluation*](https://aclanthology.org/2026.mrl-1.43/).
- Huang et al., [*Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning Attack*](https://arxiv.org/abs/2408.09600).
- Sun et al., *A Simple and Effective Pruning Approach for Large Language Models* (Wanda).
- Wei et al., [*Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications*](https://arxiv.org/abs/2402.05162).
- Salamanca et al., [*Tiny Aya: Bridging Scale and Multilingual Depth*](https://arxiv.org/abs/2603.11510).
- Souly et al., [*A StrongREJECT for Empty Jailbreaks*](https://arxiv.org/abs/2402.10260).
- Röttger et al., [*XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*](https://arxiv.org/abs/2308.01263).
- Ji et al., [*BeaverTails*](https://arxiv.org/abs/2307.04657).
- Wang et al., [*MMLU-Pro*](https://arxiv.org/abs/2406.01574).
- **Code:** [SafeTune](https://github.com/Lexsi-Labs/SafeTune), [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT), [git-disl/Antidote](https://github.com/git-disl/Antidote), [StrongREJECT](https://github.com/dsbowen/strong_reject).

## 16. LLM usage statement

- **Code:** written with the help of AI coding assistants (Claude, in claude.ai and in Cursor), under the authors' full guidance. The authors specified the designs, reviewed and judged the code and its outputs at every step, and ran every experiment.
- **Write-up:** LLMs were used to assemble and edit the text from the authors' results, notes and instructions. The authors decided what it says and checked every claim and number.
- **Research:** the research questions, hypotheses, experimental designs and interpretation of the results are the authors' own, and the authors take responsibility for the content.

## 17. Changelog

- **2026-11-15 (UTC+3):** first submission; Hugging Face collection links added.
- **2026-11-16 (UTC+3), before the deadline (2026-11-15 23:59 AoE):** BeaverTails-derived data removed from the repository; compute total corrected to 33 GPU-hours; limitations completed.

## Appendix

**A. Judge and scoring.**
- *Harmful score:* the StrongREJECT fine-tuned evaluator scores each (prompt, answer) pair from 0 to 1; HS is 100 × the mean over 150 prompts.
- *Refusal:* a phrase list matched against the first 400 characters of the answer; *coherent refusal* also requires the answer's perplexity under the clean model to be below 50.
- *Broken:* fewer than half of the SST2 answers parse to a label.
- *Unlearning:* each MMLU question is shown with options A–D; the answer is the letter with the highest next-token logit.

**B. Security considerations (Track 2a).** The deliberately harmful adapters and generations are released for reproducibility and
safety research, behind content warnings. No new attack technique beyond standard LoRA fine-tuning and a mask-aware training
variant is introduced.
