#!/usr/bin/env bash
# Erased or Hidden? — setup and run on a Lambda GPU instance.
#
# Put this file and erased_or_hidden.py in the same folder on the instance. To continue from the results you
# already have in Google Drive (MyDrive/eoh_out), give them to the script in ONE of two ways:
#   a) Drive > right-click eoh_out > Download. Put the .zip in the same folder as this script.   (recommended)
#   b) Drive > eoh_out > Share > "Anyone with the link" (Viewer) > Copy link, then run with
#        DRIVE_FOLDER_URL="<the link>" bash run_lambda.sh
#      (turn the sharing off again afterwards)
# With neither, it resumes from the HF results repo if that has results, else starts from scratch (+1.5 h).
#
# Then:
#   tmux new -s eoh              # keeps the run alive if your SSH / browser tab drops
#   bash run_lambda.sh           # restore results, checks (about 3-5 min), then the full run
#   bash run_lambda.sh preflight # only restore + checks
# Detach from tmux: Ctrl-b then d.  Come back later: tmux attach -t eoh
#
# Why this does not break like the usual Lambda setup: it never touches Lambda Stack's system Python,
# PyTorch or numpy. `uv` builds a private Python 3.12 environment from the versions pinned at the top of
# erased_or_hidden.py (PyTorch 2.10, CUDA 12.8 build, which works with older NVIDIA drivers).
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT=erased_or_hidden.py
RESULTS_REPO="${RESULTS_REPO:-none}"   # private HF dataset to resume from / save to, e.g. RESULTS_REPO=yourname/eoh-results
                                                          # (RESULTS_REPO=none bash run_lambda.sh to keep results local only)
OUT_DIR="${OUT_DIR:-$HOME/eoh_out}"

[ -f "$SCRIPT" ] || { echo "Put $SCRIPT in the same folder as this script."; exit 1; }

echo "== GPU"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
    echo "nvidia-smi failed: this instance has no working GPU driver. Relaunch it with the default Lambda image."
    exit 1
fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ' | awk -F, '{print $NF}' | cut -d. -f1)
if [[ "$DRV" =~ ^[0-9]+$ ]] && [ "$DRV" -lt 525 ]; then
    echo "NVIDIA driver $DRV is too old for the CUDA 12.8 PyTorch build (needs 525+). Use a newer Lambda image."
    exit 1
fi

echo "== uv (private Python environment)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
unset PYTHONPATH            # keep Lambda Stack's packages out of the environment
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1   # show progress lines immediately instead of in bursts through `tee`

echo "== earlier results (so the run continues instead of starting over)"
restore_results() {
    if [ -f "$OUT_DIR/results/layers.json" ]; then
        echo "already have earlier results in $OUT_DIR"
        return
    fi
    local tmp found root
    tmp=$(mktemp -d)
    shopt -s nullglob
    local zips=(./*.zip)
    shopt -u nullglob
    if [ ${#zips[@]} -gt 0 ]; then
        for z in "${zips[@]}"; do
            echo "unpacking $z"
            python3 -m zipfile -e "$z" "$tmp"      # system python, no extra packages needed
        done
    elif [ -n "${DRIVE_FOLDER_URL:-}" ]; then
        echo "downloading the Drive folder (it must be shared as 'Anyone with the link')"
        if ! uvx --quiet gdown --folder "$DRIVE_FOLDER_URL" -O "$tmp"; then
            echo "Drive download failed. Check the link is shared as 'Anyone with the link', or use the .zip way."
            exit 1
        fi
    else
        echo "no .zip next to this script and no DRIVE_FOLDER_URL: will use the HF results repo, else start from scratch"
        rm -rf "$tmp"
        return
    fi
    found=$(find "$tmp" -path '*/results/layers.json' -print -quit)
    if [ -z "$found" ]; then
        echo "Could not find results/layers.json in what was given. Download the whole eoh_out folder, not a subfolder."
        exit 1
    fi
    root=$(dirname "$(dirname "$found")")
    mkdir -p "$OUT_DIR"
    cp -r "$root"/. "$OUT_DIR"/
    rm -rf "$tmp"
    echo "restored into $OUT_DIR: $(cd "$OUT_DIR" && ls -d */ | tr '\n' ' ')"
}
# uv is needed for the Drive download, so this runs after the uv step
restore_results

echo "== Hugging Face token"
if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Paste your Hugging Face token (typing is hidden, it is not saved anywhere): " HF_TOKEN
    echo
    export HF_TOKEN
fi

RUN=(uv run --python 3.12 "$SCRIPT" --out-dir "$OUT_DIR")
if [ "$RESULTS_REPO" != "none" ]; then
    RUN+=(--results-repo "$RESULTS_REPO")
fi

echo "== preflight: login, GPU, libraries, model, saved results"
"${RUN[@]}" --preflight 2>&1 | tee -a eoh_preflight.log
if [ "${1:-}" = "preflight" ]; then
    exit 0
fi

echo "== full run (also written to eoh_run.log)"
"${RUN[@]}" --stages all 2>&1 | tee -a eoh_run.log
echo "== done. Results: $OUT_DIR (and $RESULTS_REPO). Terminate the instance in the Lambda dashboard to stop billing."
