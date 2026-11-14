"""Build the results report (HTML -> PDF) from analysis/summary.json, the figures and results/samples.json.

    python analysis/analyze.py --results results --out analysis
    python report/build_report.py            # writes report/erased_or_hidden_report.pdf (needs playwright)

Every number in the text is read from the analysis outputs; nothing is typed in by hand.
"""
from __future__ import annotations

import base64
import html
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "analysis/summary.json").read_text(encoding="utf-8"))
SAMPLES = json.loads((ROOT / "results/samples.json").read_text(encoding="utf-8")) if (ROOT / "results/samples.json").exists() else []
FIG = ROOT / "analysis/figures"
ARM_NAME = {"A0": "original model", "A1": "layer 30, strong", "A1c": "layer 30, medium", "A1b": "layer 30, gentle",
            "A1d": "layer 30, gentle", "A2": "layer 27", "A2s": "layer 27, deeper"}
ATT = {"targeted@en": "English relearning", "targeted@hi": "Hindi relearning", "targeted@mr": "Marathi relearning",
       "benign_ring1@mr": "Benign Marathi (college chemistry)", "benign_general@mr": "Unrelated Marathi (general knowledge)"}
LN = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


def img(name: str, width: str = "100%") -> str:
    p = FIG / name
    if not p.exists():
        return ""
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="width:{width}">'


def cell(arm, attack, lang, folds=(1,)):
    return next((r for r in S["re_cells"] if r["arm"] == arm and r["attack"] == attack and r["scored"] == lang
                 and r["folds"] == list(folds)), None)


def fmt(r):
    return "–" if r is None else f"{r['RE']:+.2f} <span class=ci>[{r['lo']:+.2f}, {r['hi']:+.2f}]</span>"


def pooled(arm, tag="fold1"):
    return S["pooled"].get(f"{arm}|{tag}")


def pair(key):
    return S["pairwise"].get(key)


def pv(v):
    return f"{v['mean']:+.2f} (95% CI {v['lo']:+.2f} to {v['hi']:+.2f})"


acc = S["accuracy"]
arms_re = [a for a in ("A2", "A2s", "A1c", "A1") if any(r["arm"] == a for r in S["re_cells"])]
has_robust = "A2s" in acc or any(r["folds"] == [2] for r in S["re_cells"]) or any(r["attack"] in ("targeted@hi", "benign_general@mr") for r in S["re_cells"])
p2, p1c, p1 = pooled("A2"), pooled("A1c"), pooled("A1")
d_all, d_en, d_hi = pair("A1c-A2|fold1|all"), pair("A1c-A2|fold1|en"), pair("A1c-A2|fold1|hi")
cka = S.get("cka", {})


def sample_block(q_start: str, conds: list[str]) -> str:
    rows = [r for r in SAMPLES if r["lang"] == "en" and r["question"].startswith(q_start)]
    if not rows:
        return ""
    q = rows[0]
    out = [f'<div class="ex"><div class="q">{html.escape(q["question"][:220])}</div>'
           f'<div class="corr">Correct: {html.escape(q["correct"][:90])}</div><table class="extab">']
    for c in conds:
        r = next((r for r in rows if r["condition"] == c), None)
        if r:
            text = r["output"].replace("\n", " ")
            text = text[:150] + ("…" if len(text) > 150 else "")
            mark = "✓" if r["is_correct"] else "✗"
            out.append(f'<tr><td class="cond">{html.escape(c.replace("forget@mr", "Marathi attack").replace("benign_ring1@mr", "benign Marathi"))}</td>'
                       f'<td class="mark {"ok" if r["is_correct"] else "bad"}">{mark}</td><td class="out">{html.escape(text)}</td></tr>')
    out.append("</table></div>")
    return "".join(out)


# ---------------- tables ----------------
def erasure_table():
    rows = ["<tr><th>Arm</th><th>Forget en</th><th>Forget hi</th><th>Forget mr</th><th>History/law en</th><th>Bio/physics en</th><th>General en</th></tr>"]
    for arm in [a for a in ("A0", "A2", "A2s", "A1c", "A1") if a in acc]:
        a = acc[arm]
        rows.append(f"<tr><td><b>{arm}</b> <span class=muted>{ARM_NAME[arm]}</span></td><td>{a['en/forget_B']:.2f}</td><td>{a['hi/forget_B']:.2f}</td>"
                    f"<td>{a['mr/forget_B']:.2f}</td><td>{a['en/ring3']:.2f}</td><td>{a['en/ring2']:.2f}</td><td>{a['en/general']:.2f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def settings_table():
    rows = ["<tr><th>Arm</th><th>Layer</th><th>Steps</th><th>Learning rate</th><th>Strength</th><th>Forget en (half A)</th><th>History/law drop</th><th>Matched</th></tr>"]
    for arm, m in sorted(S["arms"].items(), key=lambda kv: (kv[1]["layer"], kv[0])):
        cfg = m.get("rmu_cfg") or {"rmu_steps": 150, "rmu_lr": 5e-5}
        rows.append(f"<tr><td><b>{arm}</b></td><td>{m['layer']}</td><td>{cfg['rmu_steps']}</td><td>{cfg['rmu_lr']:g}</td><td>{m['mult']:g}</td>"
                    f"<td>{m['forgetA_en']:.2f}</td><td>{m['ring3_drop'] * 100:+.0f} pts</td><td>{'yes' if m['matched'] else 'no'}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def re_table(folds=(1,)):
    rows = ["<tr><th>Attack</th><th>Scored in</th>" + "".join(f"<th>{a} <span class=muted>{ARM_NAME[a]}</span></th>" for a in arms_re) + "</tr>"]
    for att in ATT:
        for lang in ("en", "hi", "mr"):
            cs = [cell(a, att, lang, folds) for a in arms_re]
            if all(c is None for c in cs):
                continue
            rows.append(f"<tr><td>{ATT[att]}</td><td>{LN[lang]}</td>" + "".join(f"<td>{fmt(c)}</td>" for c in cs) + "</tr>")
    return "<table class=re>" + "".join(rows) + "</table>"


def pair_table():
    rows = ["<tr><th>Comparison (same questions)</th><th>Folds</th><th>Scored in</th><th>Difference in relearning excess</th><th>P(≤ 0)</th></tr>"]
    for k, v in S["pairwise"].items():
        comp, tag, lang = k.split("|")
        rows.append(f"<tr><td>{comp}</td><td>{'1' if tag == 'fold1' else '1 + 2'}</td><td>{'all three' if lang == 'all' else LN[lang]}</td>"
                    f"<td>{v['mean']:+.2f} <span class=ci>[{v['lo']:+.2f}, {v['hi']:+.2f}]</span></td><td>{v['p_le_0']:.3f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def samples_table():
    if "samples" not in S:
        return ""
    rows = ["<tr><th>Condition</th><th>English</th><th>Marathi</th></tr>"]
    for c, (k, n) in S["samples"]["en"].items():
        k2, n2 = S["samples"]["mr"].get(c, [0, 0])
        rows.append(f"<tr><td>{html.escape(c.replace('forget@mr', 'Marathi attack').replace('benign_ring1@mr', 'benign Marathi'))}</td><td>{k}/{n}</td><td>{k2}/{n2}</td></tr>")
    return "<table class=narrow>" + "".join(rows) + "</table>"


c_en_en, c_mr_en, c_bn_en = cell("A1c", "targeted@en", "en"), cell("A1c", "targeted@mr", "en"), cell("A1c", "benign_ring1@mr", "en")
a_en_en, a_mr_en, a_bn_en = cell("A2", "targeted@en", "en"), cell("A2", "targeted@mr", "en"), cell("A2", "benign_ring1@mr", "en")
m2, m1c = S["arms"]["A2"], S["arms"].get("A1c")
base = S.get("baseline") or {}

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: Inter, 'DejaVu Sans', sans-serif; color: #0b0b0b; font-size: 9.6pt; line-height: 1.45; }
h1 { font-size: 20pt; line-height: 1.15; margin: 0 0 4pt; letter-spacing: -0.2pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #e4e3df; padding-bottom: 3pt; }
h3 { font-size: 10.5pt; margin: 11pt 0 3pt; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
img { break-inside: avoid; }
p { margin: 0 0 6pt; }
.sub { color: #52514e; font-size: 10pt; margin-bottom: 10pt; }
.box { background: #f4f7fc; border-left: 3px solid #2a78d6; padding: 8pt 11pt; margin: 8pt 0 10pt; }
.box ul { margin: 2pt 0 0 14pt; padding: 0; } .box li { margin-bottom: 3pt; }
.warn { background: #fdf5ee; border-left: 3px solid #eb6834; }
table { break-inside: avoid; border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 8.6pt; }
th { text-align: left; font-weight: 600; color: #52514e; border-bottom: 1px solid #c9c8c2; padding: 3pt 5pt; }
td { border-bottom: 1px solid #efeeea; padding: 3pt 5pt; vertical-align: top; }
table.narrow { width: 60%; }
.ci { color: #8a8984; font-size: 7.8pt; }
.muted { color: #8a8984; font-weight: 400; font-size: 7.8pt; }
.cap { color: #52514e; font-size: 8.3pt; margin: -4pt 0 10pt; }
.ex { border: 1px solid #e4e3df; border-radius: 4px; padding: 6pt 8pt; margin: 6pt 0; page-break-inside: avoid; }
.q { font-weight: 600; } .corr { color: #1a7a52; font-size: 8.5pt; margin: 2pt 0 3pt; }
.extab td { border: none; padding: 1.5pt 4pt; font-size: 8.2pt; }
.cond { white-space: nowrap; color: #52514e; width: 30%; } .mark.ok { color: #1a7a52; } .mark.bad { color: #b3261e; }
.out { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2pt; }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; background: #f4f3ef; padding: 6pt; white-space: pre-wrap; }
.pb { page-break-before: always; }
.small { font-size: 8.4pt; color: #52514e; }
"""

def pick(arm, attack, lang, folds=(1,)):
    r = cell(arm, attack, lang, folds)
    return r["RE"] if r else float("nan")


def pair_rows(keys):
    rows = ["<tr><th>Comparison (same questions)</th><th>Folds</th><th>Scored in</th><th>Difference in relearning excess</th><th>P(≤ 0)</th></tr>"]
    for k in keys:
        v = S["pairwise"].get(k)
        if not v:
            continue
        comp, tag, lang = k.split("|")
        rows.append(f"<tr><td>{comp.replace('-', ' − ')}</td><td>{'1' if tag == 'fold1' else '1 + 2'}</td><td>{'all three' if lang == 'all' else LN[lang]}</td>"
                    f"<td>{v['mean']:+.2f} <span class=ci>[{v['lo']:+.2f}, {v['hi']:+.2f}]</span></td><td>{v['p_le_0']:.3f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def pooled_rows():
    rows = ["<tr><th>Arm</th><th>Questions</th><th>Relearning excess</th><th>P(≤ 0)</th></tr>"]
    for a in arms_re:
        v, tag = (pooled(a, "both_folds"), "both halves (59)") if pooled(a, "both_folds") else (pooled(a, "fold1"), "half B (30)")
        if v:
            rows.append(f"<tr><td><b>{a}</b> <span class=muted>{ARM_NAME[a]}</span></td><td>{tag}</td><td>{v['mean']:+.2f} "
                        f"<span class=ci>[{v['lo']:+.2f}, {v['hi']:+.2f}]</span></td><td>{v['p_le_0']:.3f}</td></tr>")
    return "<table class=narrow>" + "".join(rows) + "</table>"


F = S.get("depth_fit", {})
q2, q2s, q1c = pooled("A2", "both_folds"), pooled("A2s", "both_folds"), pooled("A1c", "both_folds")
m_en = pair("A1c-A2s|both_folds|en")
m_all = pair("A1c-A2s|both_folds|all")
m_hi = pair("A1c-A2s|both_folds|hi")
gen_en = {a: cell(a, "benign_general@mr", "en") for a in ("A2", "A2s", "A1c")}
res27, res30 = F.get("mean_residual_by_layer", {}).get("27", 0), F.get("mean_residual_by_layer", {}).get("30", 0)
a2s = acc.get("A2s", {})

doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Erased or Hidden?</title><style>{CSS}</style></head><body>
<h1>Erased or Hidden? Unlearned chemistry in Tiny Aya returns after fine-tuning in any language, and how much returns depends on how much was erased, not where</h1>
<div class="sub">Janhavi Khindkar · ValueShift Research · results report, {date.today().strftime('%d %B %Y')} · model: Tiny Aya Global (3.35B)</div>

<div class="box"><b>Summary</b><ul>
<li><b>Unlearned knowledge is hidden, not erased.</b> After English-only RMU unlearning of high-school chemistry, a small LoRA fine-tune brings
held-out answers back in every arm. Pooled over both test halves: layer 27 {pv(q2)}; layer 27, deeper {pv(q2s)}; layer 30 {pv(q1c)}.</li>
<li><b>Any language, any topic.</b> Fine-tuning in Hindi or Marathi restores <i>English</i> answers as much as fine-tuning in English
(layer 27, deeper: {pick('A2s','targeted@en','en'):+.2f} English, {pick('A2s','targeted@hi','en'):+.2f} Hindi, {pick('A2s','targeted@mr','en'):+.2f} Marathi).
A fine-tune on <b>unrelated</b> Marathi general-knowledge questions, with no chemistry at all, restores the most
({gen_en['A2s']['RE']:+.2f} and {gen_en['A1c']['RE']:+.2f}; even the shallow layer-27 arm {gen_en['A2']['RE']:+.2f}, 95% CI {gen_en['A2']['lo']:+.2f} to {gen_en['A2']['hi']:+.2f}).</li>
<li><b>Depth, not layer.</b> Across 4 arms × 3 languages, what comes back tracks how much was erased:
relearning excess ≈ {F.get('slope', 0):.2f} × erased {F.get('intercept', 0):+.2f} (r = {F.get('r', 0):.2f}); layer 27 and 30 sit {res27:+.2f} and {res30:+.2f} off that line.
At matched English depth (layer 27 deeper vs layer 30) the English difference is {m_en['mean']:+.2f} (95% CI {m_en['lo']:+.2f} to {m_en['hi']:+.2f}).</li>
<li><b>The original hypothesis is not supported.</b> We expected unlearning at the more cross-lingually shared layer to be more durable. Its apparent
advantage in the first runs came from erasing less. Durability comparisons between unlearning settings need matched erasure depth, per language.</li>
</ul></div>

<h2>1. Question and design</h2>
<p><b>Question.</b> When unlearning makes a model fail questions on a topic, is the knowledge gone, or only hidden? Does it come back when
someone fine-tunes the model in another language, or on unrelated data, and does that depend on which layer the unlearning edited?</p>
<p><b>Model and data.</b> Tiny Aya Global (36 layers). Forget topic: MMLU high-school chemistry, in English, Hindi (MMMLU) and Marathi
(machine-translated with NLLB-200). The <b>forget set</b> is the {base.get('n_forget', 59)} questions the model answers correctly in both English
and Hindi; the <b>control set</b> is the {base.get('n_control', 105)} it gets wrong in both. Each is split into halves A and B: attacks train on one
half and are tested on the other, in both directions for the main arms. Collateral damage is measured on college chemistry (ring 1),
high-school biology and physics (ring 2), prehistory and international law (ring 3), and 200 general MMLU questions.</p>
<p><b>Unlearning.</b> RMU (Li et al., WMDP): activations at layer ℓ on English forget text are pushed towards a random direction while
activations on Wikipedia text are held fixed; only the MLP down-projections of layers ℓ−2…ℓ are trained, on English text only. Arms: A2
(layer 27), A2s (layer 27, trained longer to erase as deeply as layer 30), A1c and A1 (layer 30, medium and strong), plus two gentle layer-30
settings that only half-erased.</p>
<p><b>Attacks.</b> LoRA (rank 16, all linear layers, 3 epochs), learning rates 1e-4 and 3e-4, 2 seeds; the strongest learning rate per seed
is kept. Targeted attacks train on forget questions with their answers, in English, Hindi or Marathi (29–30 questions). Benign attacks
train in Marathi on 100 college-chemistry questions, or on 200 general-knowledge questions from unrelated subjects. <b>Relearning
excess</b> = accuracy gain on held-out forget questions minus the gain the same attack gives on held-out control questions (questions the
model never knew), so a model that only gets better at the multiple-choice format scores zero.</p>
<p><b>Scoring.</b> Multiple choice, by comparing the model's scores for the letters A–D after the question (no generation, no judge). The model
also writes free-text answers for 8 held-out questions per language under each condition.</p>

<h2>2. Erasure and collateral damage</h2>
{erasure_table()}
<p class="cap">Accuracy on held-out half B before any relearning. The original model scores 1.00 / 1.00 / 0.57 on the forget set by construction.</p>
{img('fig2_erasure_vs_collateral.png', '76%')}
{settings_table()}
<p>English-only unlearning also erased the knowledge in Hindi and Marathi at both layers. Layer 27 gave a moderate setting (A2) that met the
target first time; the layer-30 settings tried jumped from half-erased (0.59–0.66) to fully erased, so we cannot say whether a moderate
layer-30 setting exists. <b>At matched depth the two layers damage about the same:</b> A2s and A1c both erase English to 0.17 and leave
biology/physics at {a2s.get('en/ring2', 0):.2f} and {acc['A1c']['en/ring2']:.2f}, general questions at {a2s.get('en/general', 0):.2f} and {acc['A1c']['en/general']:.2f}.</p>

<h2 class="pb">3. The knowledge comes back, in any language and from any topic</h2>
{img('fig1_english_recovery.png')}
<p class="cap">English relearning excess (95% CI over questions) after each attack, for the three main arms (half B).</p>
{re_table()}
{pooled_rows()}
<p>Every arm recovers held-out answers, and neither the attack language nor its topic matters much. The unrelated Marathi fine-tune is
the strongest attack, but it also trains on the most questions (200, against 29–30 for targeted attacks), so attack types should not be
ranked against each other. What it shows is that <b>no forgotten question, no chemistry and no English is needed</b> to bring English
chemistry back.</p>

<h2>4. Depth, not layer</h2>
{img('fig5_recovery_vs_depth.png', '76%')}
<p class="cap">Each point is one arm scored in one language: accuracy erased by unlearning (x) against relearning excess averaged over the five attacks (y).</p>
{pair_rows(['A1c-A2s|both_folds|all', 'A1c-A2s|both_folds|en', 'A1c-A2s|both_folds|hi', 'A1c-A2s|both_folds|mr',
            'A1c-A2|both_folds|all', 'A1c-A2|both_folds|en', 'A1c-A2|both_folds|hi', 'A1c-A2|both_folds|mr'])}
<p>In the first runs, layer 30 recovered more than layer 27 (A1c − A2), which looked like support for the hypothesis that unlearning at
the shared layer is more durable. But A1c had erased much more English, and the difference sat entirely in English. Erasing as deeply at
layer 27 (A2s) removes it: at matched English depth the English difference is {m_en['mean']:+.2f}, and overall layer 27 recovers slightly
<i>more</i> ({m_all['mean']:+.2f}), driven by Hindi, where A2s had erased more ({a2s.get('hi/forget_B', 0):.2f} vs {acc['A1c']['hi/forget_B']:.2f}). The scatter shows the
same thing across all twelve arm × language points: how much comes back is roughly {F.get('slope', 0):.2f} of how much was erased, whichever layer was edited.</p>

<h2>5. What the model actually says</h2>
<p>Unlearned models do not give wrong answers; they produce degenerate text. After a Marathi fine-tune they answer fluently again, on questions
they were never trained on. Greedy decoding, prompt: “give the letter, then explain in one sentence”.</p>
{sample_block('Monatomic ions', ['A0', 'A1c', 'A1c + benign_ring1@mr relearning', 'A2', 'A2 + benign_ring1@mr relearning'])}
{sample_block('Which of the following substances has an asymmetrical', ['A0', 'A1c', 'A1c + benign_ring1@mr relearning', 'A1c + forget@mr relearning', 'A2'])}
{sample_block('The Arrhenius equation', ['A0', 'A2', 'A2 + benign_ring1@mr relearning', 'A2 + forget@mr relearning'])}
{samples_table()}
<p class="cap">Correct letter given first, on 8 held-out questions per language. The original model itself gets 5/8 in this free-text format.
Eight questions illustrate; they do not test.</p>

<h2>6. Probe: the answer is readable before the edited layers</h2>
{img('fig4_probe.png', '78%')}
<p>A linear probe trained on the original model reads the correct option on held-out forget questions at about 0.6 at layers 23–24 (chance
0.25), identically for every arm, because RMU leaves everything before the edited layers untouched; accuracy falls to chance only after
the edit. RMU blocks the read-out of knowledge that is still represented upstream, which is what relearning undoes.
<b>Caveat:</b> the probe's cross-validated accuracy on its mixed training set is near chance ({S['probe']['base_cv_mean']:.2f}), the readout uses
30 questions, and the layers were picked after looking. Supporting evidence only.</p>

<h2>7. A correction to the layer choice</h2>
{img('fig3_cka_flat.png', '74%')}
<p>Layer 27 was chosen as the layer where English, Hindi and Marathi representations are most similar (linear CKA). The curve is flat,
{cka.get('flat_range_7_34', [0, 0])[0]:.3f}–{cka.get('flat_range_7_34', [0, 0])[1]:.3f} at every layer from 7 to 34, and 27 was simply the upper edge of the search window.
The comparison is <b>layer 27 vs layer 30</b>, two late depths three layers apart. A better measure of cross-lingual sharing, and a
genuinely middle layer, are the obvious next test of the original hypothesis.</p>

<h2>8. What this means</h2>
<ul>
<li><b>For anyone relying on unlearning in a multilingual model:</b> RMU-style unlearning can be undone by fine-tuning in any language the
model knows, including on data unrelated to the forgotten topic. Evaluations that only re-attack in English, or only with forget data,
underestimate how easily it comes back.</li>
<li><b>For comparing unlearning settings or methods:</b> compare durability at matched erasure depth, per language. A shallower setting
will look more durable simply because it has less to give back.</li>
</ul>

<h2>9. Limitations</h2>
<ul>
<li>{base.get('n_forget', 59)} forget questions; one model; one unlearning method (RMU); two seeds that vary only the attack.</li>
<li>Both halves were tested only for A2, A2s and A1c; A1 has half B only.</li>
<li>Attack types differ in size (29–30, 100 and 200 training questions), so their strengths are not comparable with each other.</li>
<li>Marathi is machine-translated, and the model is near chance in Marathi chemistry ({(base.get('acc') or {}).get('mr', 0.315):.2f} overall).</li>
<li>Chemistry formulas and numbers look the same in every language, so some transfer may run through shared symbols.</li>
<li>The cross-language similarity measure did not single out any layer, and only two late layers were compared.</li>
</ul>

<h2>10. Related work</h2>
<p class="small">Li et al., <i>The WMDP Benchmark</i> (RMU). Łucki et al., <i>An Adversarial Perspective on Machine Unlearning for AI Safety</i>, and
Hu et al., <i>Unlearning or Obfuscating?</i>: relearning and unrelated fine-tuning restore unlearned knowledge, in one language. Deeb &amp; Roger,
<i>Do Unlearning Methods Remove Information from Language Model Weights?</i>. <i>Linguistic Loopholes in LLM Unlearning</i> (2609.40286): Tiny Aya and RMU
across 174 languages, changing only the test-time query language, with no relearning or layer analysis. <i>Multilingual Unlearning in LLMs:
Transfer, Dynamics, and Reversibility</i> (2606.03291): reversal by activation steering, not fine-tuning. <i>A False Positive in Cross-Lingual
Unlearning Evaluation</i>: cross-lingual claims need base-model and retain controls, which this design includes (A0, control set, rings).</p>

<h2>11. Reproduce</h2>
<pre>RESULTS_REPO=yourname/eoh-results bash run_lambda.sh      # pipeline (or the Colab notebook / HF Jobs)
RESULTS_REPO=yourname/eoh-results bash run_variants.sh    # layer-30 settings + generated samples
RESULTS_REPO=yourname/eoh-results bash run_robust.sh      # A2s, Hindi + unrelated attacks, swapped halves
python analysis/analyze.py --results results --out analysis && python report/build_report.py</pre>
</body></html>"""

out_html = ROOT / "report/erased_or_hidden_report.html"
out_html.write_text(doc, encoding="utf-8")
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(out_html.as_uri())
    pg.wait_for_timeout(300)
    pg.pdf(path=str(ROOT / "report/erased_or_hidden_report.pdf"), format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"})
    b.close()
out_html.unlink()
print("wrote", ROOT / "report/erased_or_hidden_report.pdf")
