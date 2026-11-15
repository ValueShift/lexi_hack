#!/usr/bin/env bash
# Builds the write-up from the top-level README.md as a self-contained HTML page (KaTeX math, embedded figures, no scripts)
# and an A4 PDF printed from it with Chromium.
# Needs: pandoc >= 3, node, and in writeup/: npm install katex playwright   (Playwright needs a Chromium: npx playwright install chromium)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; cd "$ROOT"
NAME=safety_edits_that_dont_stick
pandoc README.md -f gfm+tex_math_dollars --shift-heading-level-by=-1 --lua-filter "$HERE/links.lua" \
  -M lang=en -M pagetitle="Safety Edits That Don't Stick" -s -t html5 --wrap=none --mathjax -c "$HERE/style.css" \
  -o "$HERE/$NAME.tmp.html"
NODE_PATH="$HERE/node_modules:${NODE_PATH:-}" node "$HERE/katex_render.js" "$HERE/$NAME.tmp.html" "$HERE/$NAME.html" "$ROOT"
rm -f "$HERE/$NAME.tmp.html"
NODE_PATH="$HERE/node_modules:${NODE_PATH:-}" node "$HERE/pdf.js" "$HERE/$NAME.html" "$HERE/$NAME.pdf"
