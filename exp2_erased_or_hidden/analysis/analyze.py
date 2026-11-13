"""Erased or Hidden? — analysis of the saved results (CPU only, no model needed).

Reads the outputs written by erased_or_hidden.py and produces every number, table and figure used in the
report, so nothing is copied by hand:
    python analysis/analyze.py --results results --out analysis
Outputs: analysis/summary.json, analysis/tables.md, analysis/figures/*.png

Statistics. Relearning excess (RE) for one arm, attack and scored language is
    (accuracy gain on held-out forget questions) - (accuracy gain on held-out control questions),
with the per-question gains averaged over seeds (each seed keeps its strongest learning rate). Confidence
intervals come from bootstrapping questions. Comparisons between arms are paired (same questions), and pooled
numbers resample whole questions across languages and attacks together, because the three languages are
translations of the same questions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ["en", "hi", "mr"]
LANG_NAME = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
ATTACK_NAME = {"targeted@en": "English relearning", "targeted@hi": "Hindi relearning",
               "targeted@mr": "Marathi relearning", "benign_ring1@mr": "Benign Marathi (college chemistry)",
               "benign_general@mr": "Unrelated Marathi (general knowledge)"}
# arm descriptions; anything else found on disk is described from its metadata
ARM_NAME = {"A0": "Original model", "A1": "Layer 30, strong", "A1c": "Layer 30, medium",
            "A1b": "Layer 30, gentle", "A1d": "Layer 30, gentle", "A2": "Layer 27", "A2s": "Layer 27, deeper"}
# chart colours: reference categorical slots 1-3 (validated all-pairs) + neutral for the original model
COLOR = {"A2": "#2a78d6", "A2s": "#1baf7a", "A1c": "#eb6834", "A1": "#52514e", "A0": "#8a8984"}
LAYER_COLOR = {27: "#2a78d6", 30: "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
B = 4000
rng = np.random.default_rng(0)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def arm_meta(res: Path) -> dict:
    metas = {}
    for p in sorted((res / "arms").glob("*.meta.json")):
        metas[p.name.split(".")[0]] = load(p)
    for p in sorted((res / "arms").glob("*/meta.json")):
        metas[p.parent.name] = load(p)
    return metas


def gains(rel: dict, key: str) -> np.ndarray:
    """Per-question gain on `key`, averaged over seeds."""
    before = np.array(rel["before"][key], float)
    return np.mean([np.array(run["after"][key], float) - before for run in rel["runs"]], 0)


def relearn_cells(res: Path) -> dict:
    """{(arm, attack, scored_lang): [(forget_gains, control_gains, fold), ...]} over the folds present."""
    cells: dict = {}
    for fold, d in ((1, res / "relearn"), (2, res / "relearn_fold2")):
        if not d.exists():
            continue
        te_f, te_c = ("forget_B", "control_B") if fold == 1 else ("forget_A", "control_A")
        files = {p.stem: load(p) for p in d.glob("*.json")}
        arms = sorted({k.split("__")[0] for k in files})
        for arm in arms:
            pairs = [(f"targeted@{lang}", files.get(f"{arm}__forget_{lang}"), files.get(f"{arm}__control_{lang}"))
                     for lang in LANGS]
            pairs += [(f"{b}@mr", files.get(f"{arm}__{b}_mr"), files.get(f"{arm}__{b}_mr"))
                      for b in ("benign_ring1", "benign_general")]
            for attack, f, c in pairs:
                if not (f and c):
                    continue
                for lang in LANGS:
                    fk, ck = f"{lang}/{te_f}", f"{lang}/{te_c}"
                    if fk in f["before"] and ck in c["before"]:
                        cells.setdefault((arm, attack, lang), []).append((gains(f, fk), gains(c, ck), fold))
    return cells


def combine(entries, folds=None):
    """Concatenate questions across folds (each question is held out exactly once per fold)."""
    es = [e for e in entries if folds is None or e[2] in folds]
    return np.concatenate([e[0] for e in es]), np.concatenate([e[1] for e in es]), sorted({e[2] for e in es})


def boot_re(gf, gc):
    pt = gf.mean() - gc.mean()
    d = np.array([gf[rng.integers(0, len(gf), len(gf))].mean() - gc[rng.integers(0, len(gc), len(gc))].mean()
                  for _ in range(B)])
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_pooled(cell_list, other_list=None):
    """Mean RE over cells (minus the same for `other_list`, paired), resampling question indices jointly."""
    nF, nC = len(cell_list[0][0]), len(cell_list[0][1])

    def stat(fi, ci):
        v = np.mean([g[fi].mean() - c[ci].mean() for g, c in cell_list])
        if other_list is not None:
            v -= np.mean([g[fi].mean() - c[ci].mean() for g, c in other_list])
        return v

    pt = stat(np.arange(nF), np.arange(nC))
    d = np.array([stat(rng.integers(0, nF, nF), rng.integers(0, nC, nC)) for _ in range(B)])
    return {"mean": float(pt), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_le_0": float(np.mean(d <= 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    S: dict = {}

    # ---------------- accuracy and erasure ----------------
    evals = {p.stem.replace("eval_", ""): load(p) for p in res.glob("eval_*.json") if not p.stem.endswith(".sig")}
    arms = [x for x in ["A0", "A2", "A2s", "A1c", "A1"] if x in evals] + sorted(set(evals) - {"A0", "A2", "A2s", "A1c", "A1"})
    S["accuracy"] = {arm: {k: round(v["acc"], 3) for k, v in evals[arm].items()} for arm in arms}
    metas = arm_meta(res)
    S["arms"] = {arm: {"layer": m["layer"], "matched": m["matched"], **{k: m["chosen"][k] for k in ("mult", "forgetA_en", "ring3_drop")},
                       "rmu_cfg": m.get("rmu_cfg")} for arm, m in metas.items()}
    S["baseline"] = load(res / "baseline.json") if (res / "baseline.json").exists() else None

    # ---------------- relearning ----------------
    cells = relearn_cells(res)
    S["re_cells"] = []
    for (arm, attack, lang), entries in sorted(cells.items()):
        for folds in ([1], [2], [1, 2]):
            if not all(f in [e[2] for e in entries] for f in folds):
                continue
            gf, gc, fs = combine(entries, folds)
            pt, lo, hi = boot_re(gf, gc)
            loss = None
            if arm != "A0" and "A0" in evals and arm in evals:
                sets = ["forget_B"] if folds == [1] else ["forget_A"] if folds == [2] else ["forget_A", "forget_B"]
                a0 = np.mean(sum((evals["A0"][f"{lang}/{s}"]["correct"] for s in sets), []))
                ar = np.mean(sum((evals[arm][f"{lang}/{s}"]["correct"] for s in sets), []))
                loss = float(a0 - ar)
            S["re_cells"].append({"arm": arm, "attack": attack, "scored": lang, "folds": fs, "n_forget": len(gf),
                                  "RE": pt, "lo": lo, "hi": hi, "loss": loss,
                                  "share_of_loss": (pt / loss) if loss and loss > 0.1 else None})

    def cell_list(arm, folds, attacks=None, langs=None):
        out_ = []
        for (ar, at, la), entries in sorted(cells.items()):
            if ar == arm and (attacks is None or at in attacks) and (langs is None or la in langs) \
                    and all(f in [e[2] for e in entries] for f in folds):
                gf, gc, _ = combine(entries, folds)
                out_.append(((at, la), gf, gc))
        return out_

    S["pooled"], S["pairwise"] = {}, {}
    core = ["targeted@en", "targeted@mr", "benign_ring1@mr"]
    for folds in ([1], [1, 2]):
        tag = "fold1" if folds == [1] else "both_folds"
        for arm in arms:
            if arm == "A0":
                continue
            cl = cell_list(arm, folds, core)
            if len(cl) == 9:
                S["pooled"][f"{arm}|{tag}"] = boot_pooled([(g, c) for _, g, c in cl])
        for x, y in (("A1c", "A2"), ("A1", "A2"), ("A1c", "A2s")):
            cx, cy = cell_list(x, folds, core), cell_list(y, folds, core)
            if len(cx) == 9 and len(cy) == 9 and [k for k, _, _ in cx] == [k for k, _, _ in cy]:
                S["pairwise"][f"{x}-{y}|{tag}|all"] = boot_pooled([(g, c) for _, g, c in cx], [(g, c) for _, g, c in cy])
                for lang in LANGS:
                    cxl = [(g, c) for (at, la), g, c in cx if la == lang]
                    cyl = [(g, c) for (at, la), g, c in cy if la == lang]
                    S["pairwise"][f"{x}-{y}|{tag}|{lang}"] = boot_pooled(cxl, cyl)

    # ---------------- layers, probe, samples ----------------
    if (res / "layers.json").exists():
        L = load(res / "layers.json")
        m = np.array(L["cka_mean"])
        S["cka"] = {"shared": L["shared"], "late": L["late"], "n_layers": L["n_layers"],
                    "flat_range_7_34": [float(m[7:35].min()), float(m[7:35].max())], "mean": [float(x) for x in m]}
    if (res / "probe.json").exists():
        P = load(res / "probe.json")
        S["probe"] = {"base_cv_mean": float(np.mean(P["base_cv_acc"])), "arms": P["arms"]}
    if (res / "samples.json").exists():
        sm = load(res / "samples.json")
        conds = list(dict.fromkeys(r["condition"] for r in sm))
        S["samples"] = {lang: {c: [int(sum(r["is_correct"] for r in sm if r["condition"] == c and r["lang"] == lang)),
                                   int(sum(1 for r in sm if r["condition"] == c and r["lang"] == lang))] for c in conds}
                        for lang in ("en", "mr")}
    # does recovery track how much was erased, and does the layer add anything beyond that?
    pts = []
    for arm in [x for x in arms if x != "A0"]:
        for lang in LANGS:
            cs = [r for r in S["re_cells"] if r["arm"] == arm and r["scored"] == lang and r["folds"] == [1]]
            if cs and cs[0]["loss"] is not None:
                pts.append({"arm": arm, "lang": lang, "layer": S["arms"].get(arm, {}).get("layer"),
                            "erased": cs[0]["loss"], "mean_RE": float(np.mean([c["RE"] for c in cs])), "n_attacks": len(cs)})
    if len(pts) >= 4:
        x = np.array([p["erased"] for p in pts]); y = np.array([p["mean_RE"] for p in pts])
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (slope * x + icpt)
        S["depth_fit"] = {"slope": float(slope), "intercept": float(icpt), "r": float(np.corrcoef(x, y)[0, 1]),
                          "mean_residual_by_layer": {str(L): float(np.mean([resid[i] for i, p in enumerate(pts) if p["layer"] == L]))
                                                     for L in sorted({p["layer"] for p in pts})},
                          "points": pts}
    (out / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")

    # ---------------- markdown tables ----------------
    md = ["# Results tables (generated by analysis/analyze.py)", ""]
    md += ["## Erasure and collateral damage (accuracy before any relearning)", "",
           "| Arm | Layer | Forget en | Forget hi | Forget mr | History/law en | Bio/physics en | General en |",
           "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        acc = S["accuracy"][arm]
        lay = S["arms"].get(arm, {}).get("layer", "-")
        md.append(f"| {arm} ({ARM_NAME.get(arm, arm)}) | {lay} | {acc['en/forget_B']:.2f} | {acc['hi/forget_B']:.2f} | "
                  f"{acc['mr/forget_B']:.2f} | {acc['en/ring3']:.2f} | {acc['en/ring2']:.2f} | {acc['en/general']:.2f} |")
    md += ["", "Forget columns: held-out half B. Original model: 1.00 / 1.00 / 0.57 by construction of the forget set.", "",
           "## Unlearning settings tried (English forget half A, history/law drop)", "",
           "| Arm | Layer | Steps | LR | Strength | Forget en | History/law drop | Matched |", "|---|---|---|---|---|---|---|---|"]
    for arm, m in sorted(S["arms"].items(), key=lambda kv: (kv[1]["layer"], kv[0])):
        cfg = m.get("rmu_cfg") or {"rmu_steps": 150, "rmu_lr": 5e-5}
        md.append(f"| {arm} | {m['layer']} | {cfg['rmu_steps']} | {cfg['rmu_lr']:g} | {m['mult']:g} | {m['forgetA_en']:.2f} | "
                  f"{m['ring3_drop']:+.2f} | {'yes' if m['matched'] else 'no'} |")
    md += ["", "## Relearning excess (95% CI over questions), fold 1", "",
           "| Attack | Scored | " + " | ".join(a for a in arms if a != "A0") + " |", "|---|---|" + "---|" * (len(arms) - 1)]
    for attack in ATTACK_NAME:
        for lang in LANGS:
            row = [r for r in S["re_cells"] if r["attack"] == attack and r["scored"] == lang and r["folds"] == [1]]
            if not row:
                continue
            vals = []
            for arm in arms:
                if arm == "A0":
                    continue
                r = next((r for r in row if r["arm"] == arm), None)
                vals.append(f"{r['RE']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" if r else "-")
            md.append(f"| {ATTACK_NAME[attack]} | {LANG_NAME[lang]} | " + " | ".join(vals) + " |")
    md += ["", "## Pooled relearning excess (3 core attacks x 3 languages)", "", "| Arm | Folds | Mean | 95% CI | P(<=0) |", "|---|---|---|---|---|"]
    for k, v in S["pooled"].items():
        arm, tag = k.split("|")
        md.append(f"| {arm} | {tag} | {v['mean']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['p_le_0']:.3f} |")
    md += ["", "## Paired differences between arms (same questions)", "", "| Comparison | Folds | Scored | Mean | 95% CI | P(<=0) |", "|---|---|---|---|---|---|"]
    for k, v in S["pairwise"].items():
        comp, tag, lang = k.split("|")
        md.append(f"| {comp} | {tag} | {lang} | {v['mean']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['p_le_0']:.3f} |")
    if "samples" in S:
        md += ["", "## Generated answers: correct letter first (8 held-out questions per language)", "", "| Condition | English | Marathi |", "|---|---|---|"]
        for c, (k, n) in S["samples"]["en"].items():
            k2, n2 = S["samples"]["mr"].get(c, [0, 0])
            md.append(f"| {c} | {k}/{n} | {k2}/{n2} |")
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

    def style(ax):
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # Fig 1: English recovered, by attack language
    plot_arms = [x for x in ("A2", "A2s", "A1c") if any(r["arm"] == x for r in S["re_cells"])]
    plot_att = [x for x in ("targeted@en", "targeted@hi", "targeted@mr", "benign_ring1@mr", "benign_general@mr")
                if any(r["attack"] == x and r["scored"] == "en" for r in S["re_cells"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.3), dpi=200)
    w = 0.8 / len(plot_arms)
    for i, arm in enumerate(plot_arms):
        xs, ys, los, his = [], [], [], []
        for j, att in enumerate(plot_att):
            r = next((r for r in S["re_cells"] if r["arm"] == arm and r["attack"] == att and r["scored"] == "en" and r["folds"] == [1]), None)
            if r:
                xs.append(j - 0.4 + w * (i + 0.5)); ys.append(r["RE"]); los.append(r["RE"] - r["lo"]); his.append(r["hi"] - r["RE"])
        ax.bar(xs, ys, w * 0.92, color=COLOR.get(arm, INK2), label=f"{arm} ({ARM_NAME.get(arm, arm)})", zorder=2)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(plot_att)))
    ax.set_xticklabels([ATTACK_NAME[x].replace(" (", "\n(") for x in plot_att], fontsize=7.5)
    ax.set_ylabel("English relearning excess")
    ax.set_title("English chemistry comes back whatever the language or topic of the fine-tune", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=len(plot_arms), loc="upper left", bbox_to_anchor=(0, -0.28))
    style(ax); fig.tight_layout(); fig.savefig(out / "figures/fig1_english_recovery.png"); plt.close(fig)

    # Fig 2: erasure vs collateral, every unlearning setting
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    ax.add_patch(plt.Rectangle((-0.03, 0), 0.08, 0.35, color="#e8f0fb", zorder=0))
    ax.text(-0.025, 0.36, "target: forget ≤ 0.35,\ncollateral ≤ 5 points", fontsize=7.5, color=INK2, va="bottom")
    for arm, m in S["arms"].items():
        c = LAYER_COLOR.get(m["layer"], INK2)
        ax.scatter(m["ring3_drop"], m["forgetA_en"], s=46, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{arm} (L{m['layer']})", (m["ring3_drop"], m["forgetA_en"]), xytext=(6, 3), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlabel("History/law accuracy lost (collateral damage)")
    ax.set_ylabel("English forget accuracy (lower = erased)")
    ax.set_xlim(-0.04, 0.24); ax.set_ylim(0, 1.0)
    ax.set_title("Every unlearning setting tried: erasure vs collateral damage", loc="left", fontsize=10, color=INK)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=LAYER_COLOR[L], label=f"layer {L}") for L in (27, 30)],
              frameon=False, fontsize=8, loc="upper right")
    style(ax); fig.tight_layout(); fig.savefig(out / "figures/fig2_erasure_vs_collateral.png"); plt.close(fig)

    # Fig 3: cross-language similarity per layer
    if "cka" in S:
        m = np.array(S["cka"]["mean"])
        fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=200)
        n = len(m); lo_, hi_ = int(0.25 * n), int(0.75 * n)
        ax.axvspan(lo_, hi_, color="#f1f0ec", zorder=0)
        ax.plot(range(n), m, color=COLOR["A2"], lw=2, zorder=2)
        for lay, lab in ((S["cka"]["shared"], "layer 27 (picked)"), (S["cka"]["late"], "layer 30")):
            ax.axvline(lay, color=INK2, lw=0.8, ls="--")
            ax.text(lay + 0.3, 0.47, lab, fontsize=7.5, color=INK2, rotation=90, va="bottom")
        ax.set_xlabel("layer"); ax.set_ylabel("mean linear CKA (en/hi/mr)")
        ax.set_ylim(0.4, 0.9)
        ax.set_title("Cross-language similarity is flat from layer 7 to 34", loc="left", fontsize=10, color=INK)
        style(ax); fig.tight_layout(); fig.savefig(out / "figures/fig3_cka_flat.png"); plt.close(fig)

    # Fig 4: probe readout per layer
    if "probe" in S:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=200)
        for arm in [x for x in ("A0", "A2", "A1c") if x in S["probe"]["arms"]]:
            v = S["probe"]["arms"][arm]
            ax.plot(range(len(v)), v, color=COLOR[arm], lw=2 if arm != "A0" else 2.5, label=f"{arm} ({ARM_NAME[arm]})")
        ax.axhline(0.25, color=INK2, lw=0.8, ls=":")
        ax.text(0.5, 0.26, "chance", fontsize=7.5, color=INK2)
        ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy on held-out\nforget questions")
        ax.set_title("The answer is still readable before the edited layers", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style(ax); fig.tight_layout(); fig.savefig(out / "figures/fig4_probe.png"); plt.close(fig)
    if "depth_fit" in S:
        F = S["depth_fit"]
        fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=200)
        xs = np.linspace(0, 0.9, 10)
        placed = []
        ax.plot(xs, F["slope"] * xs + F["intercept"], color=INK2, lw=1, ls="--", zorder=1)
        for L in (27, 30):
            P = [p for p in F["points"] if p["layer"] == L]
            ax.scatter([p["erased"] for p in P], [p["mean_RE"] for p in P], s=44, color=LAYER_COLOR[L], edgecolor="white",
                       linewidth=1.5, zorder=3, label=f"layer {L}")
            for p in P:
                near = sum(1 for q in placed if abs(q[0] - p["erased"]) < 0.03 and abs(q[1] - p["mean_RE"]) < 0.02)
                placed.append((p["erased"], p["mean_RE"]))
                ax.annotate(f"{p['arm']} {p['lang']}", (p["erased"], p["mean_RE"]), xytext=(5, -3 - 9 * near),
                            textcoords="offset points", fontsize=7, color=INK2)
        ax.text(0.02, 0.62, f"fit: relearned = {F['slope']:.2f} × erased {F['intercept']:+.2f}\nr = {F['r']:.2f}", fontsize=8, color=INK)
        ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.7)
        ax.set_xlabel("accuracy erased by unlearning (held-out forget questions)")
        ax.set_ylabel("relearning excess\n(mean over attacks)")
        ax.set_title("What comes back tracks how much was erased, not the layer", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax); fig.tight_layout(); fig.savefig(out / "figures/fig5_recovery_vs_depth.png"); plt.close(fig)
    print(f"wrote {out / 'summary.json'}, {out / 'tables.md'} and {len(list((out / 'figures').glob('*.png')))} figures")


if __name__ == "__main__":
    main()
