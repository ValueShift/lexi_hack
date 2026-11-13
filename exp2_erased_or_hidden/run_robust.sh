#!/usr/bin/env bash
# Erased or Hidden? — robustness checks (run after run_lambda.sh and run_variants.sh). About 75-90 min on an A100.
#   1. A2s: layer-27 unlearning pushed as deep as A1c (English forget accuracy <= 0.20), so layer 27 vs layer 30
#      can be compared at the same erasure depth (removes the confound found in the first analysis).
#   2. Two more attacks on every arm: relearning in Hindi, and a fine-tune on UNRELATED Marathi questions
#      (general knowledge, no chemistry at all).
#   3. Swapped halves for A2, A1c, A2s: attack trains on half B and is tested on half A -> twice the test questions.
#   4. Probe and report for everything.
# Every finished piece is saved to disk and pushed to the private results repo, so it can be stopped and rerun.
# Usage (next to erased_or_hidden.py):   bash run_robust.sh
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH

OUT="${OUT_DIR:-$HOME/eoh_out}"
REPO="${RESULTS_REPO:-none}"   # private HF dataset to resume from / save to, e.g. RESULTS_REPO=yourname/eoh-results
# A1 and A1c must stay exactly as before, so their finished results are reused and not redone
DEFAULT_OV='rmu_overrides={"A1": {"rmu_steps": 400, "rmu_lr": 0.0001, "rmu_mults": [4.0, 8.0, 16.0]}, "A1c": {"rmu_steps": 200, "rmu_lr": 0.0001, "rmu_mults": [2.0, 4.0]}, "A2s": {"rmu_steps": 300, "rmu_lr": 5e-05, "rmu_mults": [2.0, 4.0, 8.0]}}'
OV="${OV:-$DEFAULT_OV}"
MO='match_overrides={"A2s": {"erase_max": 0.20, "drop_max": 0.10}}'
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

[ -f erased_or_hidden.py ] || { echo "Put erased_or_hidden.py in the same folder as this script."; exit 1; }
grep -q "def fold_sets" erased_or_hidden.py || { echo "erased_or_hidden.py is an old version: upload the newest one."; exit 1; }
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Paste your Hugging Face token (typing is hidden): " HF_TOKEN
    echo
    export HF_TOKEN
fi
RUN=(uv run --python 3.12 erased_or_hidden.py --out-dir "$OUT")
if [ "$REPO" != "none" ]; then
    RUN+=(--results-repo "$REPO")
fi
RUN+=("${EXTRA[@]}")

echo "== 0/4 earlier results (local, or pulled from $REPO on a fresh instance)"
"${RUN[@]}" --stages none 2>&1 | grep -E "^\[hub\]|^device=" || true
[ -f "$OUT/results/layers.json" ] && [ -f "$OUT/arms/A2/meta.json" ] || { echo "No earlier results in $OUT or $REPO."; exit 1; }
SHARED=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['shared'])" "$OUT/results/layers.json")
LATE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['late'])" "$OUT/results/layers.json")
ARMS_SET="extra_arms={\"A1c\": $LATE, \"A2s\": $SHARED}"

echo "== 1/4 A2s: layer $SHARED unlearning as deep as A1c, then its evaluation (about 10 min)"
"${RUN[@]}" --stages unlearn,eval --set "$ARMS_SET" "$OV" "$MO" 2>&1 | tee -a robust.log

echo "== 2/4 Hindi relearning + unrelated Marathi fine-tune on every arm, all attacks on A2s (about 40 min)"
"${RUN[@]}" --stages relearn --set "$ARMS_SET" "$OV" "$MO" 'relearn_langs=["en", "mr", "hi"]' 'benign_general_attack=true' 2>&1 | tee -a robust.log

echo "== 3/4 swapped halves (train on B, test on A) for A2, A1c, A2s (about 30 min)"
"${RUN[@]}" --stages relearn --set "$ARMS_SET" "$OV" "$MO" 'relearn_fold=2' 'relearn_arms=["A2", "A1c", "A2s"]' 2>&1 | tee -a robust.log

echo "== 4/4 probe and report"
"${RUN[@]}" --stages probe,report --set "$ARMS_SET" "$OV" "$MO" 'relearn_langs=["en", "mr", "hi"]' 2>&1 | tee -a robust.log

echo "== packing results to share: ~/share_robust.zip"
rm -rf "$HOME/share_robust" "$HOME/share_robust.zip"
mkdir -p "$HOME/share_robust/arms"
for d in "$OUT"/arms/*/; do
    cp "$d/meta.json" "$HOME/share_robust/arms/$(basename "$d").meta.json"
done
cp -r robust.log "$OUT/results" "$OUT/plots" "$HOME/share_robust/"
[ -f eoh_run.log ] && cp eoh_run.log "$HOME/share_robust/"
[ -f variants.log ] && cp variants.log "$HOME/share_robust/"
(cd "$HOME" && python3 -m zipfile -c share_robust.zip share_robust)
grep -E "^\[unlearn\] A2s |^\[report\] A" robust.log | tail -12 || true
echo "Done. Download ~/share_robust.zip (JupyterLab: right-click > Download) and send it."
