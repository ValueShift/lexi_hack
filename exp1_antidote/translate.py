"""Translate the harmful attack slice for the attack-language axis.

The slice is the union of harmful ids that the attacks use at `--dose` for `--seeds` (doses are
nested, so dose 100 covers 10 and 50). The prompt and response of each pair are machine-translated
sentence by sentence with NLLB-200, keeping line breaks. Output, under $EXP1_OUT/data/:

  attack_pool_{lang}.jsonl   {id, prompt, response, source_lang, mt_model}, same ids as attack_pool
  {lang}_spotcheck.md        random pairs side by side, for a human to judge the translation
  attack_slice_ids.txt       the ids, one per line
  attack_slice_en.jsonl      the English source pairs, so a teammate can produce another language
                             (any tool) as attack_pool_<lang>.jsonl with the same ids and fields

    python -m exp1_antidote.translate --lang ru
"""
from __future__ import annotations

import argparse
import random
import re

from .common import free_gpu, paths, write_jsonl
from .data import harmful_ids, load

NLLB_CODES = {"en": "eng_Latn", "ru": "rus_Cyrl", "hi": "hin_Deva"}
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def slice_ids(dose: int, seeds: list[int]) -> list[str]:
    seen: dict[str, None] = {}
    for s in seeds:
        for i in harmful_ids(dose, s):
            seen.setdefault(i)
    return list(seen)


def _segments(text: str) -> list[list[str]]:
    """Lines, each split into sentences. Empty lines are kept as empty lists."""
    return [[s for s in _SENT_RE.split(line.strip()) if s] for line in text.split("\n")]


def translate_texts(texts: list[str], lang: str, model_id: str, batch_size: int, num_beams: int) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, src_lang=NLLB_CODES["en"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tgt = tok.convert_tokens_to_ids(NLLB_CODES[lang])

    split = [_segments(t) for t in texts]
    unique = sorted({s for doc in split for line in doc for s in line}, key=lambda s: (len(s), s))
    done: dict[str, str] = {}
    for j in range(0, len(unique), batch_size):
        chunk = unique[j : j + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=tgt, num_beams=num_beams,
                                 max_new_tokens=int(enc["input_ids"].shape[1] * 2) + 16,
                                 no_repeat_ngram_size=4)
        for src, o in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
            done[src] = o
        print(f"[translate] {min(j + batch_size, len(unique))}/{len(unique)} segments")
    del model
    free_gpu()
    return ["\n".join(" ".join(done[s] for s in line) for line in doc) for doc in split]


def _spotcheck(rows_en: list[dict], rows_tr: list[dict], lang: str, n: int) -> str:
    pick = random.Random(0).sample(range(len(rows_en)), min(n, len(rows_en)))
    L = [f"# Spot-check: en → {lang}", "",
         "Judge whether each translation keeps the meaning, including the harmful content of the answer. "
         "If many are wrong or refuse, rerun with `--model facebook/nllb-200-1.3B --force`.", ""]
    for k, i in enumerate(pick, 1):
        e, t = rows_en[i], rows_tr[i]
        L += [f"## {k}. `{e['id']}`", "", "**Prompt (en)**", "", e["prompt"], "", f"**Prompt ({lang})**", "",
              t["prompt"], "", "**Response (en)**", "", e["response"], "", f"**Response ({lang})**", "",
              t["response"], ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(set(NLLB_CODES) - {"en"}))
    ap.add_argument("--dose", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--n-spot", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = paths()["data"]
    out = d / f"attack_pool_{args.lang}.jsonl"
    ids = slice_ids(args.dose, args.seeds)
    by_id = {r["id"]: r for r in load("attack_pool")}
    src = [by_id[i] for i in ids]
    (d / "attack_slice_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    write_jsonl(d / "attack_slice_en.jsonl", [{"id": r["id"], "prompt": r["prompt"], "response": r["response"]}
                                              for r in src])
    if out.exists() and not args.force:
        print(f"[translate] {out} exists (use --force to redo)")
        return

    texts = [r["prompt"] for r in src] + [r["response"] for r in src]
    tr = translate_texts(texts, args.lang, args.model, args.batch_size, args.num_beams)
    n = len(src)
    rows = [{"id": r["id"], "prompt": tr[k], "response": tr[n + k], "source_lang": "en", "mt_model": args.model}
            for k, r in enumerate(src)]
    write_jsonl(out, rows)
    (d / f"{args.lang}_spotcheck.md").write_text(_spotcheck(src, rows, args.lang, args.n_spot), encoding="utf-8")
    print(f"[translate] wrote {len(rows)} pairs to {out}; review {d / f'{args.lang}_spotcheck.md'}")


if __name__ == "__main__":
    main()
