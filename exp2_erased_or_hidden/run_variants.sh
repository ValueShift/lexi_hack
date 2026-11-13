#!/usr/bin/env bash
# Erased or Hidden? — step 2 on the same Lambda instance, after run_lambda.sh has finished.
#   1. tries three in-between unlearning settings for the late layer (A1b, A1c, A1d),
#   2. keeps the one that matches A2 (forget accuracy <= 0.35 with <= 5 points collateral damage) and runs
#      eval + relearning + probe + report for it, plus generated sample answers for every arm,
#      or, if none matches, writes the sample answers for A1 and A2 only,
#   3. prints a summary.
# Usage (put it next to erased_or_hidden.py):   bash run_variants.sh
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH

OUT="${OUT_DIR:-$HOME/eoh_out}"
REPO="${RESULTS_REPO:-none}"   # private HF dataset to resume from / save to, e.g. RESULTS_REPO=yourname/eoh-results
# A1 must stay exactly as in the first run, so its finished results are reused and not redone
DEFAULT_OV='rmu_overrides={"A1": {"rmu_steps": 400, "rmu_lr": 0.0001, "rmu_mults": [4.0, 8.0, 16.0]}, "A1b": {"rmu_steps": 400, "rmu_lr": 5e-05, "rmu_mults": [2.0, 4.0]}, "A1c": {"rmu_steps": 200, "rmu_lr": 0.0001, "rmu_mults": [2.0, 4.0]}, "A1d": {"rmu_steps": 250, "rmu_lr": 5e-05, "rmu_mults": [2.0, 4.0]}}'
OV="${OV:-$DEFAULT_OV}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

[ -f erased_or_hidden.py ] || { echo "Put erased_or_hidden.py in the same folder as this script."; exit 1; }
grep -q "def stage_samples" erased_or_hidden.py || { echo "erased_or_hidden.py is the old version: upload the new one (it has the samples stage)."; exit 1; }
[ -f "$OUT/arms/A2/meta.json" ] || { echo "No earlier results in $OUT: run run_lambda.sh first."; exit 1; }

if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Paste your Hugging Face token (typing is hidden): " HF_TOKEN
    echo
    export HF_TOKEN
fi
LATE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['late'])" "$OUT/results/layers.json")
RUN=(uv run --python 3.12 erased_or_hidden.py --out-dir "$OUT")
if [ "$REPO" != "none" ]; then
    RUN+=(--results-repo "$REPO")
fi
RUN+=("${EXTRA[@]}")

echo "== 1/3 trying three in-between settings for A1 (layer $LATE), about 10 min"
"${RUN[@]}" --stages unlearn --set "extra_arms={\"A1b\": $LATE, \"A1c\": $LATE, \"A1d\": $LATE}" "$OV" 2>&1 | tee -a variants.log

PICK=$(python3 - "$OUT" <<'PYEOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
target = json.load(open(out / "arms/A2/meta.json"))["chosen"]["forgetA_en"]
ok = []
for a in ("A1b", "A1c", "A1d"):
    m = json.load(open(out / f"arms/{a}/meta.json"))
    c = m["chosen"]
    print(f"   {a}: matched={m['matched']}  forget_en={c['forgetA_en']:.3f}  history/law drop={c['ring3_drop']:.3f}",
          file=sys.stderr)
    if m["matched"]:
        ok.append((abs(c["forgetA_en"] - target), a))
print(min(ok)[1] if ok else "")
PYEOF
)

if [ -n "$PICK" ]; then
    echo "== 2/3 $PICK matches A2: eval, relearning, probe, report and sample answers, about 20 min"
    "${RUN[@]}" --stages eval,relearn,probe,report,samples --set "extra_arms={\"$PICK\": $LATE}" "$OV" 2>&1 | tee -a variants.log
else
    echo "== 2/3 no setting matched: layer-$LATE unlearning could not erase without collateral damage."
    echo "        Writing sample answers for A1 and A2, about 10 min"
    "${RUN[@]}" --stages samples --set "$OV" 2>&1 | tee -a variants.log
fi

echo "== 3/3 summary"
grep -E "^\[report\] A|^\[samples\] .*English" variants.log | tail -20 || true
echo "Full generated answers: $OUT/results/samples.md   (JupyterLab: right-click > Open With > Markdown Preview)"
echo "Relearning table:       $OUT/results/summary_RE.csv"
