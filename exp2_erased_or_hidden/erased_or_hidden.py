# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6,<2.11",  # CUDA 12.8 builds: work on older drivers (newer torch needs CUDA 13 / driver 580+)
#   "transformers>=4.56",
#   "peft>=0.13",
#   "datasets>=3.0",
#   "accelerate",
#   "safetensors",
#   "scikit-learn",
#   "pandas",
#   "matplotlib",
#   "sentencepiece",
#   "huggingface_hub>=0.34",
# ]
# ///
# %% [markdown]
# # Erased or Hidden? Unlearning on Tiny Aya
#
# **Question:** does unlearning in the layers that all languages share (A2) resist relearning
# better than unlearning in a late layer (A1), including relearning done in another language?
#
# **Pipeline (one stage per section, each saves to disk so you can stop and resume):**
#
# | Stage | What it does | Approx. time on a T4 |
# |---|---|---|
# | `data` | MMLU (en), MMMLU (hi), NLLB translation (mr) | 10-15 min |
# | `baseline` | Base accuracy per language, GO/NO-GO, forget/control splits | 10 min |
# | `layers` | CKA between languages per layer, picks shared and late layer | 3 min |
# | `unlearn` | RMU on each arm, tuned until English forget accuracy is near chance | 20-40 min |
# | `eval` | Every arm on every set in every language | 15 min |
# | `relearn` | LoRA relearning attacks (en, mr, benign neighbour), relearning excess | 1.5-2.5 h |
# | `probe` | Is the answer still linearly decodable inside? | 10 min |
# | `report` | Tables and plots | 1 min |
#
# **Run it:**
# - Colab: Runtime > Change runtime type > T4 GPU, then run the cells top to bottom.
#   Set `STAGES` in the config cell. Mount Drive (cell below) so outputs survive a disconnect.
# - Hugging Face Jobs (uses HF credits; easiest from a CPU Colab with `eoh_hf_launcher.ipynb`, see the end of this file):
#   `hf jobs uv run --flavor l4x1 --namespace ValueShift --secrets HF_TOKEN --timeout 5h erased_or_hidden.py --stages all --results-repo ValueShift/eoh-results`
# - A JupyterLab Space on Hugging Face (this notebook, run interactively on a rented GPU): see the end of this file.
# - CPU smoke test, no downloads: `python erased_or_hidden.py --tiny --stages all`
#
# Tiny Aya is gated: accept its licence on its Hugging Face page and log in (`hf auth login` or `HF_TOKEN`).

# %% [markdown]
# **Colab only:** run the next cell first (installs recent libraries; skip it locally and on HF Jobs).

# %%
# %pip install -q -U "transformers>=4.56" "peft>=0.13" "datasets>=3.0" accelerate safetensors scikit-learn sentencepiece
# %pip uninstall -y -q torchao

# %%
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

IN_COLAB = "google.colab" in sys.modules

# %% [markdown]
# ## Colab setup (skipped automatically outside Colab)
# Mounts Google Drive so outputs survive a disconnect. Click "Connect" / "Allow" when asked.

# %%
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

# %% [markdown]
# ## Config
# Everything you might change lives here. On the command line, override any key with `--set key=value`.

# %%
CFG = {
    "model_id": "CohereLabs/tiny-aya-global",   # try CohereLabs/tiny-aya-fire if Marathi is weak
    "langs": ["en", "hi", "mr"],
    "out_dir": "/content/drive/MyDrive/eoh_out" if IN_COLAB else "./eoh_out",
    "results_repo": None,          # e.g. "you/eoh-results" (private dataset); needed on HF Jobs
    "seed": 0,
    "eval_batch": 16,
    "max_len": 512,
    "answer_prefix": "",           # set to "Answer:" if format compliance is low in the baseline
    # data
    "forget_subject": "high_school_chemistry",
    "ring1": ["college_chemistry"],
    "ring2": ["high_school_biology", "high_school_physics"],
    "ring3": ["prehistory", "international_law"],
    "n_ring": 100,                 # per ring (spread over its subjects)
    "n_general": 200,
    "hi_config": "HI_IN",          # openai/MMMLU config
    "nllb_model": "facebook/nllb-200-distilled-600M",
    "mr_code": "mar_Deva",
    # GO/NO-GO
    "min_forget": 60,              # need at least this many forget items
    "min_lang_acc": 0.35,          # forget-subject accuracy per language, clearly above 25% chance
    # layers
    "shared_range": [0.25, 0.75],  # search window for the shared layer (fraction of depth)
    "late_frac": 0.85,
    # RMU
    "rmu_mults": [2.0, 4.0, 8.0],  # steering norm as a multiple of the mean activation norm
    "rmu_alpha": 100.0,            # retain loss weight
    "rmu_lr": 5e-5,
    "rmu_steps": 150,
    "rmu_batch": 4,
    "rmu_max_len": 256,
    "n_retain": 400,
    "rmu_overrides": {             # per-arm RMU settings. The first run showed layer 30 barely erases with the
        "A1": {"rmu_steps": 400, "rmu_lr": 1e-4, "rmu_mults": [4.0, 8.0, 16.0]},  # defaults, so A1 gets a stronger push
    },
    "extra_arms": {},              # optional extra unlearning arms, e.g. {"A3": 18} for a true middle layer
    "erase_tol": 0.10,             # matched erasure: English forget-A accuracy <= 0.25 + this
    "retain_tol": 0.05,            # and Ring-3 English accuracy drops by at most this
    # relearning
    "relearn_langs": ["en", "mr"],
    "relearn_seeds": [0, 1],
    "relearn_lrs": [1e-4, 3e-4],   # we report the strongest (UNDO-style)
    "relearn_epochs": 3,
    "relearn_batch": 2,            # keeps activation memory safe on a 16 GB T4; effective batch = 2 x 4 = 8
    "relearn_grad_accum": 4,
    "lora_r": 16,
    "benign_attack": True,         # attack 9: fine-tune on Ring-1 chemistry in Marathi
    # robustness options (all off by default; run_robust.sh turns them on)
    "benign_general_attack": False,  # fine-tune on unrelated general-knowledge questions in Marathi (no chemistry)
    "relearn_fold": 1,             # 1: attack trains on half A, tested on half B. 2: halves swapped (more test items)
    "relearn_arms": None,          # None = attack every arm; or a list of arm names
    "match_overrides": {},         # per-arm erasure target, e.g. {"A2s": {"erase_max": 0.20, "drop_max": 0.10}}
    # probe
    "probe_pca": 64,
    "n_boot": 2000,
    # samples: read actual generated answers, not just letter scores
    "n_samples": 8,                # held-out forget-B questions shown per condition
    "sample_tokens": 96,
}
STAGES = ["data", "baseline", "layers", "unlearn", "eval", "relearn", "probe", "report", "samples"]
TINY = False  # CPU smoke test with a random tiny model and synthetic questions
PREFLIGHT = False
ALLOW_CPU = False


def parse_cli():
    if IN_COLAB or "ipykernel" in sys.modules:
        return
    global STAGES, TINY, PREFLIGHT, ALLOW_CPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all")
    ap.add_argument("--results-repo", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="check login, GPU, model and saved results, then exit")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (JSON values)")
    a = ap.parse_args()
    if a.stages != "all":
        STAGES = a.stages.split(",")
    TINY = a.tiny
    PREFLIGHT, ALLOW_CPU = a.preflight, a.allow_cpu
    if PREFLIGHT:
        STAGES = ["preflight"]
    if a.results_repo:
        CFG["results_repo"] = a.results_repo
    if a.out_dir:
        CFG["out_dir"] = a.out_dir
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            CFG[k] = json.loads(v)
        except json.JSONDecodeError:
            CFG[k] = v


parse_cli()
if TINY:
    CFG.update(out_dir=CFG["out_dir"] if CFG["out_dir"] != "./eoh_out" else "./eoh_tiny_out", rmu_steps=20, relearn_seeds=[0], relearn_lrs=[1e-3],
               relearn_epochs=2, n_retain=32, min_forget=10, min_lang_acc=0.0, n_boot=200,
               rmu_mults=[4.0], probe_pca=8, eval_batch=8,
               rmu_overrides={"A1": {"rmu_steps": 25, "rmu_mults": [4.0]}})

# %% [markdown]
# ## Hugging Face login (required: Tiny Aya is a gated model)
# 1. Open https://huggingface.co/CohereLabs/tiny-aya-global while logged in and accept the licence
#    (do the same for `tiny-aya-fire` if you switch to it).
# 2. Create a token at https://huggingface.co/settings/tokens. "Read" is enough; choose "Write" only if you
#    set `results_repo`.
# 3. Give the notebook the token in ONE of these ways:
#    - **Colab secret (recommended):** key icon in the left bar > Add new secret > name `HF_TOKEN`, paste the
#      token, switch "Notebook access" on.
#    - **Paste it below** into `HF_TOKEN = ""`. Delete it again before you share the notebook.
#    - **Do nothing:** the cell will ask you to paste it into a hidden input box.
#
# The cell then checks that your account can actually open the model, so a licence problem shows up now
# instead of in the middle of a run.

# %%
HF_TOKEN = ""  # optional: paste your token here, e.g. "hf_abc123..." (never share the notebook with it filled in)


def hf_login() -> None:
    from huggingface_hub import hf_hub_download, login, whoami
    tok = HF_TOKEN.strip() or os.environ.get("HF_TOKEN", "").strip()
    if not tok and IN_COLAB:
        try:
            from google.colab import userdata
            tok = (userdata.get("HF_TOKEN") or "").strip()
        except Exception as e:
            print(f"[login] no Colab secret named HF_TOKEN ({type(e).__name__})")
    if not tok and (IN_COLAB or "ipykernel" in sys.modules or sys.stdin.isatty()):
        from getpass import getpass
        tok = getpass("Paste your Hugging Face token (starts with hf_): ").strip()
    if not tok:
        raise SystemExit("[login] no Hugging Face token found: set HF_TOKEN (see the instructions above)")
    os.environ["HF_TOKEN"] = tok
    login(token=tok, add_to_git_credential=False)
    print(f"[login] logged in as {whoami(token=tok)['name']}")
    try:
        hf_hub_download(CFG["model_id"], "config.json", token=tok)  # fails if the licence is not accepted
        print(f"[login] access to {CFG['model_id']} OK")
    except Exception as e:
        raise SystemExit(f"[login] your account cannot open {CFG['model_id']}: accept the licence on its page "
                         f"(https://huggingface.co/{CFG['model_id']}), then rerun this cell. ({type(e).__name__})")


if not TINY:
    hf_login()

# %%
OUT = Path(CFG["out_dir"])
for sub in ("data", "arms", "results", "plots"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEV == "cuda" else torch.float32
print(f"device={DEV} dtype={DTYPE} out={OUT} stages={STAGES} tiny={TINY}")
if DEV != "cuda" and not (TINY or ALLOW_CPU):
    raise SystemExit("No GPU is visible to PyTorch, so this would run for days on the CPU. Check `nvidia-smi`; on Colab "
                     "pick a GPU runtime. (Use --allow-cpu to run anyway.)")

# %% [markdown]
# ## Helpers: saving, Hub sync, seeding

# %%
def save_json(rel: str, obj) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(rel: str):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))


def exists(rel: str) -> bool:
    return (OUT / rel).exists()


def hub_pull() -> None:
    """Restore earlier stages' outputs from the results repo (lets HF Jobs resume)."""
    repo = CFG["results_repo"]
    if not repo:
        return
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(OUT))
        print(f"[hub] pulled {repo}")
    except Exception as e:
        print(f"[hub] nothing to pull yet ({type(e).__name__})")


def hub_push(msg: str) -> None:
    repo = CFG["results_repo"]
    if not repo:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    for attempt in (1, 2):
        try:
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)  # always private
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(OUT), commit_message=msg)
            print(f"[hub] pushed {msg}")
            return
        except Exception as e:  # never let an upload problem kill a long run: everything is already on disk
            print(f"[hub] push {attempt}/2 failed ({type(e).__name__}: {str(e)[:120]})")
            time.sleep(20)
    print(f"[hub] WARNING: not uploaded; results are still in {OUT}. The next successful push includes them.")


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


hub_pull()

# %% [markdown]
# ## Model loading
# The tiny mode builds a random 6-layer Llama with a character tokenizer so the whole pipeline can be
# checked on a CPU in about a minute. Its numbers mean nothing.

# %%
_MODEL = None
_TOK = None


def _tiny_model_and_tok():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    chars = [chr(c) for c in range(32, 127)] + [chr(c) for c in range(0x0900, 0x0980)] + ["\n"]
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for ch in chars:
        vocab.setdefault(ch, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant: {% endif %}")
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg), tok


def get_model():
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if TINY:
            _MODEL, _TOK = _tiny_model_and_tok()
        else:
            _TOK = AutoTokenizer.from_pretrained(CFG["model_id"])
            _MODEL = AutoModelForCausalLM.from_pretrained(CFG["model_id"], dtype=DTYPE)
        _MODEL.to(DEV).eval()
        for p in _MODEL.parameters():
            p.requires_grad_(False)
        if _TOK.pad_token is None:
            _TOK.pad_token = _TOK.eos_token
        _TOK.padding_side = "left"
    return _MODEL, _TOK


def get_layers(model) -> nn.ModuleList:
    n = model.config.num_hidden_layers
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n:
            return mod
    raise RuntimeError("could not find the decoder layer list")


def down_proj(model, i: int) -> nn.Module:
    return get_layers(model)[i].mlp.down_proj

# %% [markdown]
# ## Stage `data`
# Items are dicts `{id, subject, question, choices[4], answer}` per language, keyed by the same `id`
# across languages so we can compare them question by question.
# - English: `cais/mmlu` test split.
# - Hindi: `openai/MMMLU` (`HI_IN`), aligned to English by order within each subject; we check that the
#   answer keys agree before trusting the alignment.
# - Marathi: machine-translated from English with NLLB-200. Choices that are mostly numbers or formulas
#   are kept as they are. **Hand-check `data/mr_review.tsv`** before trusting Marathi numbers.

# %%
LETTERS = ["A", "B", "C", "D"]


def mmlu_items(subject: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject, split="test")
    return [{"id": f"{subject}-{i}", "subject": subject, "question": r["question"],
             "choices": list(r["choices"]), "answer": int(r["answer"])} for i, r in enumerate(ds)]


_MMMLU_CACHE: dict = {}


def mmmlu_items(config: str, subject: str) -> list[dict]:
    from datasets import load_dataset
    if config not in _MMMLU_CACHE:
        _MMMLU_CACHE[config] = load_dataset("openai/MMMLU", config, split="test")
    rows = [r for r in _MMMLU_CACHE[config] if r["Subject"] == subject]
    return [{"id": f"{subject}-{i}", "subject": subject, "question": r["Question"],
             "choices": [r["A"], r["B"], r["C"], r["D"]], "answer": "ABCD".index(r["Answer"].strip())}
            for i, r in enumerate(rows)]


SYMBOLIC = re.compile(r"^[\d\s\.,\-\+\^\(\)\[\]×x/=:%°±·*a-zA-Z]{0,14}$")


def is_symbolic(s: str) -> bool:
    """Short strings that are mostly numbers, units or formulas: keep untranslated."""
    s = s.strip()
    return bool(SYMBOLIC.match(s)) and sum(ch.isdigit() for ch in s) >= 1


def nllb_translate(texts: list[str], tgt: str, bs: int = 16) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["nllb_model"], src_lang="eng_Latn")
    m = AutoModelForSeq2SeqLM.from_pretrained(CFG["nllb_model"], dtype=DTYPE).to(DEV).eval()
    out = []
    bos = tok.convert_tokens_to_ids(tgt)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=400).to(DEV)
        with torch.no_grad():
            gen = m.generate(**enc, forced_bos_token_id=bos, max_new_tokens=400, num_beams=2)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"  translated {min(i + bs, len(texts))}/{len(texts)}", end="\r")
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def translate_items(items: list[dict], tgt: str) -> list[dict]:
    strings, slots = [], []
    for k, it in enumerate(items):
        strings.append(it["question"]); slots.append((k, "q"))
        for j, c in enumerate(it["choices"]):
            if not is_symbolic(c):
                strings.append(c); slots.append((k, j))
    trans = nllb_translate(strings, tgt)
    new = [dict(it, choices=list(it["choices"])) for it in items]
    for (k, slot), t in zip(slots, trans):
        if slot == "q":
            new[k]["question"] = t
        else:
            new[k]["choices"][slot] = t
    return new


def synthetic_items(subject: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [{"id": f"{subject}-{i}", "subject": subject,
             "question": f"{subject} q{i}: what is {rng.randint(1, 99)} + {rng.randint(1, 99)}?",
             "choices": [str(rng.randint(1, 200)) for _ in range(4)], "answer": rng.randint(0, 3)}
            for i in range(n)]


def stage_data():
    if exists("data/items.json"):
        print("[data] already done"); return
    rng = random.Random(CFG["seed"])
    subjects = [CFG["forget_subject"]] + CFG["ring1"] + CFG["ring2"] + CFG["ring3"]
    data = {lang: {} for lang in CFG["langs"]}
    if TINY:
        for s in subjects + ["general"]:
            items = synthetic_items(s, 40, hash(s) % 1000)
            for lang in CFG["langs"]:
                data[lang][s] = items
        save_json("data/items.json", data); return

    # English
    for s in subjects:
        data["en"][s] = mmlu_items(s)
    # ring sizes: cap per ring, spread over its subjects
    for ring in ("ring1", "ring2", "ring3"):
        per = max(1, CFG["n_ring"] // len(CFG[ring]))
        for s in CFG[ring]:
            keep = sorted(rng.sample(range(len(data["en"][s])), min(per, len(data["en"][s]))))
            data["en"][s] = [data["en"][s][i] for i in keep]
    # general: random MMLU subjects that are not science
    from datasets import load_dataset
    allq = load_dataset("cais/mmlu", "all", split="test")
    skip = set(subjects) | {"high_school_chemistry", "college_chemistry", "college_physics",
                            "high_school_physics", "high_school_biology", "college_biology"}
    idx = [i for i, r in enumerate(allq) if r["subject"] not in skip]
    pick = rng.sample(idx, CFG["n_general"])
    data["en"]["general"] = [{"id": f"general-{i}", "subject": allq[i]["subject"], "question": allq[i]["question"],
                              "choices": list(allq[i]["choices"]), "answer": int(allq[i]["answer"])} for i in pick]

    # Hindi from MMMLU, aligned by order within subject
    if "hi" in CFG["langs"]:
        align = {}
        for s in subjects:
            hi_all = mmmlu_items(CFG["hi_config"], s)
            en_full = mmlu_items(s)
            agree = np.mean([a["answer"] == b["answer"] for a, b in zip(en_full, hi_all)]) if hi_all else 0.0
            align[s] = float(agree)
            by_id = {it["id"]: it for it in hi_all}
            data["hi"][s] = [by_id[it["id"]] for it in data["en"][s] if it["id"] in by_id]
        # general: match by subject + position is not possible for the "all" sample; translate instead
        data["hi"]["general"] = translate_items(data["en"]["general"], "hin_Deva")
        save_json("data/hi_alignment.json", align)
        print("[data] Hindi answer-key agreement per subject:", align, "(below 0.95 = alignment suspect)")

    # Marathi by NLLB: forget subject, Ring 1 and general only (Rings 2-3 stay en/hi)
    if "mr" in CFG["langs"]:
        for s in [CFG["forget_subject"]] + CFG["ring1"] + ["general"]:
            print(f"[data] translating {s} to Marathi ({len(data['en'][s])} items)")
            data["mr"][s] = translate_items(data["en"][s], CFG["mr_code"])
        with open(OUT / "data/mr_review.tsv", "w", encoding="utf-8") as f:
            f.write("id\ten_question\tmr_question\n")
            for a, b in zip(data["en"][CFG["forget_subject"]], data["mr"][CFG["forget_subject"]]):
                f.write(f"{a['id']}\t{a['question']!r}\t{b['question']!r}\n")
    save_json("data/items.json", data)
    counts = {lang: {s: len(v) for s, v in d.items()} for lang, d in data.items()}
    print("[data] item counts:", json.dumps(counts, indent=1))


if "data" in STAGES:
    stage_data(); hub_push("data")

# %% [markdown]
# ## Multiple-choice scoring
# One forward pass per question; we read the next-token logits for A/B/C/D after the chat template.
# No LLM judge anywhere. `compliance` is how often the model's top token is one of the four letters: if it
# is low, set `answer_prefix` to `"Answer:"` and rerun the baseline.

# %%
def mc_prompt(tok, it: dict) -> str:
    body = "\n".join([it["question"]] + [f"{L}. {c}" for L, c in zip(LETTERS, it["choices"])])
    msg = [{"role": "user", "content": "Answer the following multiple choice question. "
                                       "Reply with only the letter (A, B, C or D).\n\n" + body}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return p + CFG["answer_prefix"]


_LETTER_IDS = None


def letter_ids(tok) -> list[list[int]]:
    global _LETTER_IDS
    if _LETTER_IDS is None:
        ids = []
        for L in LETTERS:
            cands = {tuple(tok.encode(v, add_special_tokens=False)) for v in (L, " " + L)}
            ids.append(sorted({c[0] for c in cands if len(c) == 1}) or [tok.encode(L, add_special_tokens=False)[-1]])
        _LETTER_IDS = ids
    return _LETTER_IDS


def batch_encode(tok, texts: list[str], max_len: int):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
              add_special_tokens=False)
    enc = {k: enc[k].to(DEV) for k in ("input_ids", "attention_mask")}
    enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return enc


@torch.no_grad()
def mc_eval(model, tok, items: list[dict]) -> dict:
    """Returns per-item correctness (list of 0/1), accuracy and format compliance."""
    lids = letter_ids(tok)
    union = set(i for ids in lids for i in ids)
    correct, comply = [], []
    tok.padding_side = "left"
    for i in range(0, len(items), CFG["eval_batch"]):
        chunk = items[i:i + CFG["eval_batch"]]
        enc = batch_encode(tok, [mc_prompt(tok, it) for it in chunk], CFG["max_len"])
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        scores = torch.stack([logits[:, ids].max(-1).values for ids in lids], -1)
        pred = scores.argmax(-1).tolist()
        top = logits.argmax(-1).tolist()
        for it, p, t in zip(chunk, pred, top):
            correct.append(int(p == it["answer"]))
            comply.append(int(t in union))
    acc = float(np.mean(correct)) if correct else float("nan")
    return {"correct": correct, "acc": acc, "compliance": float(np.mean(comply)) if comply else float("nan")}

# %% [markdown]
# ## Stage `baseline` (GO/NO-GO)
# Scores the base model on the forget subject in every language, then builds the splits:
# - **forget**: forget-subject questions the base model answers correctly in all languages (falls back to
#   en+hi if that leaves too few). Split into **A** (used by the relearning attack) and **B** (only scored).
# - **control**: forget-subject questions the base model gets wrong in every language ("never known"),
#   split the same way. The control is what turns relearning into *relearning excess*.

# %%
def stage_baseline():
    if exists("results/baseline.json"):
        print("[baseline] already done"); return load_json("results/baseline.json")
    model, tok = get_model()
    data = load_json("data/items.json")
    fs = CFG["forget_subject"]
    res = {}
    for lang in CFG["langs"]:
        r = mc_eval(model, tok, data[lang][fs])
        res[lang] = r
        print(f"[baseline] {fs} {lang}: acc={r['acc']:.3f} compliance={r['compliance']:.2f} n={len(r['correct'])}")
    ids = [it["id"] for it in data["en"][fs]]
    ok = {lang: dict(zip([it["id"] for it in data[lang][fs]], res[lang]["correct"])) for lang in CFG["langs"]}
    common = [i for i in ids if all(i in ok[l] for l in CFG["langs"])]
    rule_langs = list(CFG["langs"])
    forget = [i for i in common if all(ok[l][i] for l in rule_langs)]
    if len(forget) < CFG["min_forget"] and len(rule_langs) > 2:
        rule_langs = rule_langs[:2]
        forget = [i for i in common if all(ok[l][i] for l in rule_langs)]
    control = [i for i in common if not any(ok[l][i] for l in rule_langs)]
    rng = random.Random(CFG["seed"])
    rng.shuffle(forget); rng.shuffle(control)
    splits = {"forget_A": forget[: len(forget) // 2], "forget_B": forget[len(forget) // 2:],
              "control_A": control[: len(control) // 2], "control_B": control[len(control) // 2:],
              "rule_langs": rule_langs}
    go = len(forget) >= CFG["min_forget"] and all(res[l]["acc"] >= CFG["min_lang_acc"] for l in CFG["langs"])
    summary = {"acc": {l: res[l]["acc"] for l in CFG["langs"]},
               "compliance": {l: res[l]["compliance"] for l in CFG["langs"]},
               "n_forget": len(forget), "n_control": len(control), "rule_langs": rule_langs, "GO": bool(go)}
    save_json("data/splits.json", splits)
    save_json("results/baseline.json", summary)
    print("[baseline]", json.dumps(summary, indent=1))
    print("GO" if go else "NO-GO: see the fallbacks in the design doc (Fire variant, en+hi only, or knowledge injection)")
    return summary


if "baseline" in STAGES:
    stage_baseline(); hub_push("baseline")

# %% [markdown]
# ## Stage `layers`: pick the shared layer and the late layer
# For each layer, linear CKA between the hidden states of the same questions in two languages
# (mean-pooled over tokens). High CKA = the languages look alike there. The **shared layer** is the CKA peak
# inside the middle window; the **late layer** sits at `late_frac` of the depth.

# %%
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (X.T @ Y).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return float(num / den)


@torch.no_grad()
def pooled_hidden(model, tok, texts: list[str]) -> torch.Tensor:
    """[n_layers, n_texts, d] mean-pooled layer outputs (hidden_states[1:])."""
    outs = []
    for i in range(0, len(texts), CFG["eval_batch"]):
        enc = batch_encode(tok, texts[i:i + CFG["eval_batch"]], 256)
        hs = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states[1:]
        m = enc["attention_mask"].unsqueeze(-1).float()
        outs.append(torch.stack([(h.float() * m).sum(1) / m.sum(1) for h in hs]).cpu())
    return torch.cat(outs, 1)


def stage_layers():
    if exists("results/layers.json"):
        print("[layers] already done"); return load_json("results/layers.json")
    model, tok = get_model()
    data = load_json("data/items.json")
    fs = CFG["forget_subject"]
    ids_common = set.intersection(*[{it["id"] for it in data[l][fs]} for l in CFG["langs"]])
    texts = {l: [it["question"] for it in data[l][fs] if it["id"] in ids_common] for l in CFG["langs"]}
    H = {l: pooled_hidden(model, tok, texts[l]) for l in CFG["langs"]}
    L = H[CFG["langs"][0]].shape[0]
    pairs = [(a, b) for i, a in enumerate(CFG["langs"]) for b in CFG["langs"][i + 1:]]
    curves = {f"{a}-{b}": [linear_cka(H[a][k], H[b][k]) for k in range(L)] for a, b in pairs}
    mean = np.mean(list(curves.values()), 0)
    lo, hi = int(CFG["shared_range"][0] * L), int(CFG["shared_range"][1] * L)
    shared = int(lo + np.argmax(mean[lo:hi + 1]))
    late = min(L - 1, int(round(CFG["late_frac"] * (L - 1))))
    if late - shared < 3:
        late = min(L - 1, shared + 3)
    out = {"n_layers": L, "cka": curves, "cka_mean": mean.tolist(), "shared": max(2, shared), "late": late}
    save_json("results/layers.json", out)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.5))
        for k, v in curves.items():
            plt.plot(v, label=k)
        plt.axvline(out["shared"], ls="--", c="k"); plt.axvline(out["late"], ls=":", c="r")
        plt.xlabel("layer"); plt.ylabel("linear CKA"); plt.legend(); plt.title("Cross-language similarity per layer")
        plt.tight_layout(); plt.savefig(OUT / "plots/cka.png", dpi=150); plt.close()
    except Exception as e:
        print("[layers] plot skipped:", e)
    print(f"[layers] shared={out['shared']} late={out['late']} (of {L})")
    return out


if "layers" in STAGES:
    stage_layers(); hub_push("layers")

# %% [markdown]
# ## Stage `unlearn`: RMU per arm, at matched erasure
# RMU (Li et al., WMDP): on forget text, push the layer-ℓ activations toward a large random vector; on
# retain text, keep them where the frozen model had them. Only `down_proj` of layers ℓ-2, ℓ-1, ℓ train.
# We cache the frozen activations first, so only one model is ever in memory.
#
# **Matched erasure:** each arm tries steering strengths in `rmu_mults` (smallest first) and keeps the first
# that brings English forget-A accuracy to at most 25% + `erase_tol` while Ring-3 drops by at most
# `retain_tol`. Comparing arms that erased equally is what makes their relearning comparable. An arm that
# never reaches it is flagged `matched=false`.

# %%
class StopForward(Exception):
    pass


class F32Linear(nn.Module):
    """Trainable fp32 copy of a Linear inside an fp16/bf16 model."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.inner = lin.float()

    def forward(self, x):
        return self.inner(x.float()).to(x.dtype)


def layer_out(model, layer: int, enc: dict) -> torch.Tensor:
    store = {}

    def hook(_m, _i, out):
        store["h"] = out[0] if isinstance(out, tuple) else out
        raise StopForward

    h = get_layers(model)[layer].register_forward_hook(hook)
    try:
        model(**enc, use_cache=False)
    except StopForward:
        pass
    finally:
        h.remove()
    return store["h"]


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_tok = ((a.float() - b.float()) ** 2).mean(-1)
    m = mask.float()
    return (per_tok * m).sum() / m.sum().clamp(min=1)


_ORIG: dict = {}


def snapshot_orig(model, layers: list[int]) -> None:
    for i in layers:
        if i not in _ORIG:
            _ORIG[i] = down_proj(model, i).weight.detach().cpu().clone()


def restore_base(model) -> None:
    with torch.no_grad():
        for i, w in _ORIG.items():
            down_proj(model, i).weight.copy_(w.to(DEV))


def apply_delta(model, arm: str) -> None:
    """Load an arm's saved down_proj weights onto the base model."""
    restore_base(model)
    if arm == "A0":
        return
    from safetensors.torch import load_file
    sd = load_file(str(OUT / f"arms/{arm}/delta.safetensors"))
    snapshot_orig(model, [int(k.split(".")[1]) for k in sd])
    with torch.no_grad():
        for k, v in sd.items():
            down_proj(model, int(k.split(".")[1])).weight.copy_(v.to(DEV, down_proj(model, int(k.split(".")[1])).weight.dtype))


def forget_texts(items: list[dict]) -> list[str]:
    return [f"{it['question']}\nAnswer: {it['choices'][it['answer']]}" for it in items]


def retain_texts(n: int) -> list[str]:
    if TINY:
        return [f"retain sentence number {i} about history and law." for i in range(n)]
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    paras = [t.strip() for t in ds["text"] if len(t.strip()) > 300 and not t.strip().startswith("=")]
    return random.Random(CFG["seed"]).sample(paras, n)


RMU_KEYS = ("rmu_steps", "rmu_lr", "rmu_alpha", "rmu_mults")


def base_rmu() -> dict:
    return {k: CFG[k] for k in RMU_KEYS}


def rmu_cfg(arm: str) -> dict:
    """RMU settings for one arm: the defaults plus that arm's overrides."""
    rc = base_rmu()
    rc.update(CFG["rmu_overrides"].get(arm, {}))
    return rc


def canon(x) -> str:
    return json.dumps(x, sort_keys=True)


def arm_layers() -> dict:
    lay = load_json("results/layers.json")
    return {"A1": lay["late"], "A2": lay["shared"], **CFG["extra_arms"]}


def arm_sig(arm: str) -> str:
    """Identifies which version of an arm a result was computed on, so stale results get redone."""
    if arm == "A0":
        return "A0"
    m = load_json(f"arms/{arm}/meta.json")
    return canon({"layer": m["layer"], "cfg": m.get("rmu_cfg", base_rmu()), "mult": m["chosen"]["mult"]})


def rmu(model, tok, layer: int, f_texts: list[str], r_texts: list[str], mult: float, rc: dict | None = None) -> dict:
    seed_all(CFG["seed"])
    upd = [i for i in (layer - 2, layer - 1, layer) if i >= 0]
    snapshot_orig(model, upd)
    bs, ml = CFG["rmu_batch"], CFG["rmu_max_len"]
    tok.padding_side = "right"
    f_batches = [batch_encode(tok, f_texts[i:i + bs], ml) for i in range(0, len(f_texts), bs)]
    r_batches = [batch_encode(tok, r_texts[i:i + bs], ml) for i in range(0, len(r_texts), bs)]
    with torch.no_grad():
        frozen = [layer_out(model, layer, b).detach().cpu() for b in r_batches]
        norms = [(layer_out(model, layer, b).float().norm(dim=-1) * b["attention_mask"]).sum() / b["attention_mask"].sum()
                 for b in f_batches[:8]]
    norm_mean = float(torch.stack(norms).mean())
    d = frozen[0].shape[-1]
    u = torch.rand(d, generator=torch.Generator().manual_seed(CFG["seed"]))
    target = (mult * norm_mean * u / u.norm()).to(DEV)
    layers = get_layers(model)
    for i in upd:
        layers[i].mlp.down_proj = F32Linear(layers[i].mlp.down_proj)
        layers[i].mlp.down_proj.inner.weight.requires_grad_(True)
    params = [layers[i].mlp.down_proj.inner.weight for i in upd]
    rc = rc or base_rmu()
    opt = torch.optim.AdamW(params, lr=rc["rmu_lr"])
    log = []
    model.train()
    for step in range(rc["rmu_steps"]):
        fb = f_batches[step % len(f_batches)]
        k = step % len(r_batches)
        rb = r_batches[k]
        lf = masked_mse(layer_out(model, layer, fb), target.expand(1, 1, d), fb["attention_mask"])
        lr_ = masked_mse(layer_out(model, layer, rb), frozen[k].to(DEV), rb["attention_mask"])
        loss = lf + rc["rmu_alpha"] * lr_
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == rc["rmu_steps"] - 1:
            log.append({"step": step, "forget_loss": float(lf), "retain_loss": float(lr_)})
            print(f"  [rmu L{layer} x{mult}] step {step}: forget={float(lf):.3f} retain={float(lr_):.5f}")
    model.eval()
    with torch.no_grad():
        for i in upd:
            layers[i].mlp.down_proj = layers[i].mlp.down_proj.inner.to(DTYPE)
            layers[i].mlp.down_proj.weight.requires_grad_(False)
    tok.padding_side = "left"
    return {"layer": layer, "updated": upd, "mult": mult, "norm_mean": norm_mean, "log": log}


def items_by_ids(data, lang, ids, subject=None):
    pool = {it["id"]: it for s, v in data[lang].items() for it in v if subject is None or s == subject}
    return [pool[i] for i in ids if i in pool]


def ring_items(data, lang, ring):
    return [it for s in CFG[ring] for it in data[lang].get(s, [])]


def stage_unlearn():
    from safetensors.torch import save_file
    model, tok = get_model()
    data = load_json("data/items.json")
    splits = load_json("data/splits.json")
    fs = CFG["forget_subject"]
    forget_en = items_by_ids(data, "en", splits["forget_A"] + splits["forget_B"], fs)
    fA_en = items_by_ids(data, "en", splits["forget_A"], fs)
    r3_en = ring_items(data, "en", "ring3")
    base_r3 = mc_eval(model, tok, r3_en)["acc"]
    r_texts = retain_texts(CFG["n_retain"])
    arms = arm_layers()
    summary = {}
    for arm, layer in arms.items():
        rc = rmu_cfg(arm)
        if exists(f"arms/{arm}/meta.json"):
            old = load_json(f"arms/{arm}/meta.json")
            if old["layer"] == layer and canon(old.get("rmu_cfg", base_rmu())) == canon(rc):
                summary[arm] = old; print(f"[unlearn] {arm} already done"); continue
            print(f"[unlearn] {arm}: settings changed since the last run, redoing it with {rc}")
        tried, chosen = [], None
        t_arm = time.time()
        mo = CFG["match_overrides"].get(arm, {})
        erase_max = mo.get("erase_max", 0.25 + CFG["erase_tol"])
        drop_max = mo.get("drop_max", CFG["retain_tol"])
        for mult in rc["rmu_mults"]:
            restore_base(model)
            info = rmu(model, tok, layer, forget_texts(forget_en), r_texts, mult, rc)
            fa = mc_eval(model, tok, fA_en)["acc"]
            r3 = mc_eval(model, tok, r3_en)["acc"]
            rec = {"mult": mult, "forgetA_en": fa, "ring3_en": r3, "ring3_drop": base_r3 - r3}
            tried.append(rec)
            print(f"[unlearn] {arm} L{layer} x{mult}: forgetA_en={fa:.3f} ring3={r3:.3f} (base {base_r3:.3f}) "
                  f"[{(time.time() - t_arm) / 60:.0f} min]")
            if fa <= erase_max and base_r3 - r3 <= drop_max:
                chosen = (rec, info); break
        matched = chosen is not None
        if not matched:
            ok = [t for t in tried if t["ring3_drop"] <= 2 * drop_max] or tried
            best = min(ok, key=lambda t: t["forgetA_en"])
            restore_base(model)
            info = rmu(model, tok, layer, forget_texts(forget_en), r_texts, best["mult"], rc)
            chosen = (best, info)
        rec, info = chosen
        sd = {f"layers.{i}.mlp.down_proj.weight": down_proj(model, i).weight.detach().cpu().contiguous()
              for i in info["updated"]}
        (OUT / f"arms/{arm}").mkdir(parents=True, exist_ok=True)
        save_file(sd, str(OUT / f"arms/{arm}/delta.safetensors"))
        meta = {"arm": arm, "layer": layer, "matched": matched, "chosen": rec, "tried": tried, "rmu_cfg": rc,
                "match_rule": {"erase_max": erase_max, "drop_max": drop_max},
                "base_ring3_en": base_r3, "rmu": {k: v for k, v in info.items() if k != "log"}, "log": info["log"]}
        save_json(f"arms/{arm}/meta.json", meta)
        summary[arm] = meta
        restore_base(model)
        hub_push(f"unlearn {arm}")
    print("[unlearn]", {a: (m["layer"], m["matched"], m["chosen"]) for a, m in summary.items()})


if "unlearn" in STAGES:
    stage_unlearn()

# %% [markdown]
# ## Stage `eval`: every arm, every set, every language
# Sets: forget-A, forget-B, control-B, Ring 1, Ring 2, Ring 3, general. Marathi has the forget subject,
# Ring 1 and general only. Per-item correctness is saved so the report can bootstrap.

# %%
ARMS = ["A0", "A1", "A2"] + list(CFG["extra_arms"])


def eval_fresh(arm: str) -> bool:
    if not exists(f"results/eval_{arm}.json"):
        return False
    if exists(f"results/eval_{arm}.sig.json"):
        return load_json(f"results/eval_{arm}.sig.json")["sig"] == arm_sig(arm)
    # results from before signatures existed are valid only if the arm itself is also from back then
    return arm == "A0" or "rmu_cfg" not in load_json(f"arms/{arm}/meta.json")


def eval_sets(data, splits, lang):
    fs = CFG["forget_subject"]
    sets = {"forget_A": items_by_ids(data, lang, splits["forget_A"], fs),
            "forget_B": items_by_ids(data, lang, splits["forget_B"], fs),
            "control_B": items_by_ids(data, lang, splits["control_B"], fs),
            "ring1": ring_items(data, lang, "ring1"), "ring2": ring_items(data, lang, "ring2"),
            "ring3": ring_items(data, lang, "ring3"), "general": data[lang].get("general", [])}
    return {k: v for k, v in sets.items() if v}


def stage_eval():
    model, tok = get_model()
    data = load_json("data/items.json")
    splits = load_json("data/splits.json")
    for arm in ARMS:
        if eval_fresh(arm):
            print(f"[eval] {arm} already done"); continue
        apply_delta(model, arm)
        res = {}
        for lang in CFG["langs"]:
            for name, items in eval_sets(data, splits, lang).items():
                r = mc_eval(model, tok, items)
                res[f"{lang}/{name}"] = {"acc": r["acc"], "correct": r["correct"], "n": len(items)}
        save_json(f"results/eval_{arm}.json", res)
        save_json(f"results/eval_{arm}.sig.json", {"sig": arm_sig(arm)})
        print(f"[eval] {arm}:", {k: round(v["acc"], 3) for k, v in res.items()})
    restore_base(model)


if "eval" in STAGES:
    stage_eval(); hub_push("eval")

# %% [markdown]
# ## Stage `relearn`: does the knowledge come back, and faster than learning from scratch?
# For each arm, attack language and seed we train a small LoRA, score, then strip the LoRA off again.
# - **Attack 8 (targeted):** fine-tune on forget-A, score forget-B. Separately fine-tune on control-A and score
#   control-B. Relearning excess **RE = gain on forget-B − gain on control-B.** RE near 0: the knowledge was
#   gone. Large RE: it was still in the weights.
# - **Attack 9 (benign neighbour):** fine-tune on Ring-1 chemistry in Marathi; RE from forget-B vs control-B
#   gains in the same run.
# - We sweep `relearn_lrs` and keep the strongest attack per cell, so no arm looks robust because its attack
#   was badly tuned. forget-B is scored in **every** language, which is how cross-lingual relearning shows up.

# %%
def train_texts(tok, items: list[dict]) -> list[tuple[str, int]]:
    """(full text, prompt length in characters) with the answer as the assistant turn."""
    out = []
    for it in items:
        prompt = mc_prompt(tok, it)
        answer = f"{LETTERS[it['answer']]}. {it['choices'][it['answer']]}"
        out.append((prompt + answer + (tok.eos_token or ""), prompt))
    return out


def ensure_peft_importable() -> None:
    """Colab preinstalls torchao 0.10, and recent peft refuses to load LoRA next to torchao < 0.16.
    This experiment never uses torchao, so remove the old copy instead of fighting versions."""
    import importlib
    import importlib.metadata as md
    import subprocess
    from packaging.version import Version
    try:
        v = md.version("torchao")
    except md.PackageNotFoundError:
        return
    if Version(v) >= Version("0.16.0"):
        return
    print(f"[setup] removing torchao {v}: too old for peft, and this experiment does not use it")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    for m in [m for m in sys.modules if m == "torchao" or m.startswith("torchao.")]:
        del sys.modules[m]
    importlib.invalidate_caches()


ensure_peft_importable()


def answer_loss(pm, tok, chunk: list[tuple[str, str]]) -> torch.Tensor:
    """Mean cross-entropy over the answer tokens only.

    Left padding puts every row's answer at the end, so the model only has to compute the 262k-way vocabulary
    scores for the last few positions (logits_to_keep) instead of the whole sequence. Same loss as passing
    labels with the prompt masked, at a fraction of the memory."""
    enc = batch_encode(tok, [c[0] for c in chunk], CFG["max_len"])  # left-padded, with position_ids
    lens = enc["attention_mask"].sum(1).tolist()
    n_ans = [max(1, L - len(tok(prompt, add_special_tokens=False)["input_ids"])) for L, (_, prompt) in zip(lens, chunk)]
    K = min(max(n_ans) + 1, enc["input_ids"].shape[1])
    logits = pm(**enc, use_cache=False, logits_to_keep=K).logits[:, :-1, :].float()   # predicts the last K-1 tokens
    targets = enc["input_ids"][:, -(K - 1):]
    pos = torch.arange(K - 1, device=targets.device).unsqueeze(0)                    # 0 .. K-2
    mask = pos >= (K - 1 - torch.tensor(n_ans, device=targets.device).unsqueeze(1))  # last n_ans tokens of each row
    nll = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    return (nll * mask.reshape(-1)).sum() / mask.sum()


def lora_relearn(model, tok, items: list[dict], lr: float, seed: int):
    ensure_peft_importable()
    from peft import LoraConfig, get_peft_model
    seed_all(seed)
    names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in n})
    pm = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=2 * CFG["lora_r"], lora_dropout=0.0,
                                          target_modules=names, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    pairs = train_texts(tok, items)
    tok.padding_side = "left"
    pm.train()
    rng = random.Random(seed)
    step_i = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(CFG["relearn_epochs"]):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), CFG["relearn_batch"]):
            loss = answer_loss(pm, tok, pairs[i:i + CFG["relearn_batch"]]) / CFG["relearn_grad_accum"]
            loss.backward()
            step_i += 1
            if step_i % CFG["relearn_grad_accum"] == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    pm.eval()
    return pm


def strip_lora(pm):
    base = pm.unload()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def fold_sets(fold: int | None = None):
    """(train_forget, train_control, test_forget, test_control) split names for a relearning fold."""
    fold = fold or CFG["relearn_fold"]
    return ("forget_A", "control_A", "forget_B", "control_B") if fold == 1 else ("forget_B", "control_B", "forget_A", "control_A")


def relearn_dir(fold: int | None = None) -> str:
    fold = fold or CFG["relearn_fold"]
    return "results/relearn" if fold == 1 else f"results/relearn_fold{fold}"


def score_targets(model, tok, data, splits) -> dict:
    fs = CFG["forget_subject"]
    out = {}
    _, _, te_f, te_c = fold_sets()
    for lang in CFG["langs"]:
        for name in (te_f, te_c):
            items = items_by_ids(data, lang, splits[name], fs)
            if items:
                out[f"{lang}/{name}"] = mc_eval(model, tok, items)["correct"]
    return out


def stage_relearn():
    global _MODEL
    model, tok = get_model()
    data = load_json("data/items.json")
    splits = load_json("data/splits.json")
    fs = CFG["forget_subject"]
    tr_f, tr_c, te_f, te_c = fold_sets()
    print(f"[relearn] fold {CFG['relearn_fold']}: train on {tr_f}/{tr_c}, test on {te_f}/{te_c} -> {relearn_dir()}")
    jobs = []
    for arm in (CFG["relearn_arms"] or ARMS):
        if arm != "A0" and not exists(f"arms/{arm}/meta.json"):
            print(f"[relearn] skipping {arm}: not unlearned yet"); continue
        for lang in CFG["relearn_langs"]:
            jobs.append((arm, f"forget@{lang}", items_by_ids(data, lang, splits[tr_f], fs)))
            jobs.append((arm, f"control@{lang}", items_by_ids(data, lang, splits[tr_c], fs)))
        if CFG["benign_attack"] and "mr" in CFG["langs"]:
            jobs.append((arm, "benign_ring1@mr", ring_items(data, "mr", "ring1")))
        if CFG["benign_general_attack"] and "mr" in CFG["langs"]:
            jobs.append((arm, "benign_general@mr", data["mr"].get("general", [])))
    # most informative first, so a cut-off run still has the key comparison: A2 and A1 in English, then
    # Marathi, then the plain model (A0) and the benign attack
    def priority(job):
        arm, attack, _ = job
        tier = 2 if attack.startswith("benign") else (0 if attack.endswith("@en") else 1)
        tier += 1.5 if arm == "A0" else 0
        return (tier, {"A2": 0, "A1": 1}.get(arm, 2), not attack.startswith("forget"))
    jobs.sort(key=priority)
    rel_of = lambda arm, attack: f"{relearn_dir()}/{arm}__{attack.replace('@', '_')}.json"
    todo = [j for j in jobs if not (exists(rel_of(j[0], j[1])) and load_json(rel_of(j[0], j[1])).get("arm_sig") == arm_sig(j[0]))]
    print(f"[relearn] {len(jobs) - len(todo)} of {len(jobs)} jobs already done, {len(todo)} to run")
    t_start = time.time()
    for n_done, (arm, attack, items) in enumerate(todo):
        rel = rel_of(arm, attack)
        apply_delta(model, arm)
        before = score_targets(model, tok, data, splits)
        runs = []
        for seed in CFG["relearn_seeds"]:
            per_lr = []
            for lr in CFG["relearn_lrs"]:
                pm = lora_relearn(model, tok, items, lr, seed)
                after = score_targets(pm, tok, data, splits)
                model = strip_lora(pm); _MODEL = model
                tgt = te_c if attack.startswith("control") else te_f
                gain = np.mean([np.mean(after[k]) - np.mean(before[k]) for k in after if k.endswith(tgt)])
                per_lr.append({"lr": lr, "after": after, "mean_gain": float(gain)})
                print(f"[relearn] {arm} {attack} seed={seed} lr={lr}: mean gain {gain:+.3f}")
            runs.append({"seed": seed, **max(per_lr, key=lambda r: r["mean_gain"]), "all_lrs": [
                {"lr": r["lr"], "mean_gain": r["mean_gain"]} for r in per_lr]})
        save_json(rel, {"arm": arm, "arm_sig": arm_sig(arm), "attack": attack, "fold": CFG["relearn_fold"],
                        "test_sets": [te_f, te_c], "n_train": len(items),
                        "before": before, "runs": runs})
        el = (time.time() - t_start) / 60
        left = el / (n_done + 1) * (len(todo) - n_done - 1)
        print(f"[relearn] job {n_done + 1}/{len(todo)} done ({arm} {attack}): {el:.0f} min so far, about {left:.0f} min left")
        restore_base(model)
        hub_push(f"relearn fold{CFG['relearn_fold']} {arm} {attack}")


if "relearn" in STAGES:
    stage_relearn()

# %% [markdown]
# ## Stage `probe`: is the answer still readable inside?
# A linear probe (PCA + logistic regression) learns, from the **base** model's last-token activations, which
# option is correct. It is trained on items outside forget-B, then applied frozen to each arm's activations on
# forget-B. If an arm answers at chance but the probe still reads the right option at middle layers, the
# knowledge is hidden, not erased. Check the base model's own cross-validated probe accuracy first: if that is
# near chance, this readout tells you nothing and you should drop it.

# %%
@torch.no_grad()
def last_token_acts(model, tok, items) -> torch.Tensor:
    outs = []
    for i in range(0, len(items), CFG["eval_batch"]):
        enc = batch_encode(tok, [mc_prompt(tok, it) for it in items[i:i + CFG["eval_batch"]]], CFG["max_len"])
        hs = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states[1:]
        outs.append(torch.stack([h[:, -1, :].float() for h in hs]).cpu())
    return torch.cat(outs, 1)  # [L, n, d]


def stage_probe():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    model, tok = get_model()
    data = load_json("data/items.json")
    splits = load_json("data/splits.json")
    fs = CFG["forget_subject"]
    lang = "en"
    train_ids = splits["forget_A"] + splits["control_A"] + splits["control_B"]
    train = items_by_ids(data, lang, train_ids, fs) + ring_items(data, lang, "ring1")
    test = items_by_ids(data, lang, splits["forget_B"], fs)
    apply_delta(model, "A0")
    Xtr = last_token_acts(model, tok, train)
    ytr = np.array([it["answer"] for it in train])
    yte = np.array([it["answer"] for it in test])
    L = Xtr.shape[0]
    probes, base_cv = [], []
    for k in range(L):
        p = make_pipeline(PCA(n_components=min(CFG["probe_pca"], len(train) - 1)), LogisticRegression(max_iter=2000))
        base_cv.append(float(cross_val_score(p, Xtr[k].numpy(), ytr, cv=5).mean()))
        probes.append(p.fit(Xtr[k].numpy(), ytr))
    res = {"base_cv_acc": base_cv, "arms": {}}
    for arm in ARMS:
        apply_delta(model, arm)
        Xte = last_token_acts(model, tok, test)
        res["arms"][arm] = [float((probes[k].predict(Xte[k].numpy()) == yte).mean()) for k in range(L)]
    restore_base(model)
    save_json("results/probe.json", res)
    print("[probe] base CV acc per layer:", [round(x, 2) for x in base_cv])
    for arm, v in res["arms"].items():
        print(f"[probe] {arm} forget-B probe acc per layer:", [round(x, 2) for x in v])


if "probe" in STAGES:
    stage_probe(); hub_push("probe")

# %% [markdown]
# ## Stage `report`
# - `results/summary_accuracy.csv`: accuracy per arm × language × set.
# - `results/summary_RE.csv`: relearning excess per arm × attack × scored language, with a bootstrap 95% CI.
# - Plots in `plots/`.

# %%
def bootstrap_re(f_before, f_after, c_before, c_after, n_boot, rng):
    f_gain = np.array(f_after, float) - np.array(f_before, float)
    c_gain = np.array(c_after, float) - np.array(c_before, float)
    point = f_gain.mean() - c_gain.mean()
    bs = []
    for _ in range(n_boot):
        bs.append(rng.choice(f_gain, len(f_gain)).mean() - rng.choice(c_gain, len(c_gain)).mean())
    return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def stage_report():
    import pandas as pd
    rows = []
    for arm in ARMS:
        if exists(f"results/eval_{arm}.json"):
            for k, v in load_json(f"results/eval_{arm}.json").items():
                lang, name = k.split("/")
                rows.append({"arm": arm, "lang": lang, "set": name, "acc": v["acc"], "n": v["n"]})
    acc = pd.DataFrame(rows)
    if len(acc):
        acc.to_csv(OUT / "results/summary_accuracy.csv", index=False)
        print(acc.pivot_table(index=["set", "lang"], columns="arm", values="acc").round(3).to_string())

    rng = np.random.default_rng(CFG["seed"])
    re_rows = []
    for fold in (1, 2):
        rel_dir = OUT / relearn_dir(fold)
        if not rel_dir.exists():
            continue
        _, _, te_f, te_c = fold_sets(fold)
        get = lambda arm, att: load_json(f"{relearn_dir(fold)}/{arm}__{att}.json") if (rel_dir / f"{arm}__{att}.json").exists() else None
        for arm in ARMS:
            pairs = [(f"targeted@{lang}", get(arm, f"forget_{lang}"), get(arm, f"control_{lang}"))
                     for lang in ("en", "mr", "hi")]
            pairs += [(f"{b}@mr", get(arm, f"{b}_mr"), get(arm, f"{b}_mr")) for b in ("benign_ring1", "benign_general")]
            for attack, f, c in pairs:
                if not (f and c):
                    continue
                for scored in CFG["langs"]:
                    fk, ck = f"{scored}/{te_f}", f"{scored}/{te_c}"
                    if fk not in f["before"] or ck not in c["before"]:
                        continue
                    for rf, rc in zip(f["runs"], c["runs"]):
                        pt, lo, hi = bootstrap_re(f["before"][fk], rf["after"][fk], c["before"][ck], rc["after"][ck],
                                                  CFG["n_boot"], rng)
                        re_rows.append({"fold": fold, "arm": arm, "attack": attack, "scored": scored,
                                        "seed": rf["seed"], "RE": pt, "lo": lo, "hi": hi})
    re_df = pd.DataFrame(re_rows)
    if len(re_df):
        re_df.to_csv(OUT / "results/summary_RE.csv", index=False)
        for fold, sub in re_df.groupby("fold"):
            print(f"[report] relearning excess, fold {fold} (mean over seeds):")
            print(sub.groupby(["attack", "scored", "arm"])["RE"].mean().unstack().round(3).to_string())
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            g = re_df[re_df["fold"] == 1].groupby(["attack", "scored", "arm"])["RE"].mean().unstack()
            g.plot.bar(figsize=(9, 4)); plt.axhline(0, c="k", lw=0.8)
            plt.ylabel("relearning excess (higher = knowledge was still there)"); plt.tight_layout()
            plt.savefig(OUT / "plots/relearning_excess.png", dpi=150); plt.close()
        except Exception as e:
            print("[report] plot skipped:", e)
    if exists("results/baseline.json"):
        print("[report] baseline:", load_json("results/baseline.json"))
    for arm in ARMS[1:]:
        if exists(f"arms/{arm}/meta.json"):
            m = load_json(f"arms/{arm}/meta.json")
            print(f"[report] {arm}: layer {m['layer']}, matched erasure={m['matched']}, chosen={m['chosen']}")


if "report" in STAGES:
    stage_report(); hub_push("report")

# %% [markdown]
# ## Running on Hugging Face
# Every stage saves its outputs and skips work that is already done. With `results_repo` set, outputs are also
# pushed to a **private** HF dataset after each step and pulled back at the start, so a run can move between
# Colab, HF Jobs and a Space and carry on where it stopped.
#
# **Option 1: HF Jobs (recommended, billed per minute, stops by itself).**
# Easiest: open `eoh_hf_launcher.ipynb` in a CPU-only Colab. It uploads `MyDrive/eoh_out` to
# `ValueShift/eoh-results`, starts the Job and copies the results back. From a terminal instead:
# 1. `pip install -U huggingface_hub`, then `hf auth login` with a **Write** token.
# 2. Upload earlier results once: `hf upload ValueShift/eoh-results ./eoh_out . --repo-type dataset --private`
#    (the `.` keeps `data/`, `arms/`, `results/` at the top of the repo, which is where the script looks).
# 3. `hf jobs uv run --flavor l4x1 --namespace ValueShift --secrets HF_TOKEN --timeout 5h erased_or_hidden.py --stages all --results-repo ValueShift/eoh-results`
#    Drop `--namespace ValueShift` to bill your own account instead of the org (the org needs credit to pay).
# 4. `hf jobs logs <job_id> --namespace ValueShift --follow` to watch, `hf jobs cancel <job_id> --namespace ValueShift` to stop.
# An L4 is about $0.80/hour; what is left (A1 redo, relearn, probe, report) is roughly 3-4 hours.
#
# **Option 2: JupyterLab Space (this notebook, interactive).** Billed every minute it is switched on, used or not.
# 1. https://huggingface.co/new-space?template=SpacesExamples/jupyterlab : owner `ValueShift`, **Private**,
#    hardware `Nvidia L4` (or T4 medium), set the `JUPYTER_TOKEN` password, add a secret `HF_TOKEN`.
# 2. Open the Space, upload this notebook, set `"results_repo": "ValueShift/eoh-results"` in the config cell, run all.
#    Space storage is wiped on restart; the results repo is what keeps your work.
# 3. When it finishes: Space settings > **Pause** (or switch hardware to CPU basic), or the billing continues.


# %% [markdown]
# ## Samples: what does the model actually say?
# The scores above only compare the four letter options. Here the model writes its answer and a one-sentence
# explanation for held-out forget-B questions (never used in any training), under each condition: original,
# unlearned, unlearned + Marathi relearning attack, unlearned + benign Marathi fine-tune. The relearning
# adapters were not kept, so each is retrained here with the learning rate that run picked for seed 0.
# Output: `results/samples.md` (easy to read) and `results/samples.json`.

# %%
def sample_prompt(tok, it: dict) -> str:
    body = "\n".join([it["question"]] + [f"{L}. {c}" for L, c in zip(LETTERS, it["choices"])])
    msg = [{"role": "user", "content": "Answer the following multiple choice question. Give the letter of the correct "
                                       "option, then explain your answer in one sentence.\n\n" + body}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_answers(model, tok, items: list[dict]) -> list[str]:
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        enc = tok([sample_prompt(tok, it) for it in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEV)
        gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], do_sample=False,
                             max_new_tokens=CFG["sample_tokens"], pad_token_id=tok.pad_token_id)
        outs += [t.strip() for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs


def first_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def stage_samples():
    global _MODEL
    model, tok = get_model()
    data = load_json("data/items.json")
    splits = load_json("data/splits.json")
    fs = CFG["forget_subject"]
    qs = {lang: items_by_ids(data, lang, splits["forget_B"], fs)[:CFG["n_samples"]] for lang in ("en", "mr")}
    conds = [("A0", None)]
    for arm in ARMS[1:]:
        if exists(f"arms/{arm}/meta.json"):
            conds += [(arm, None), (arm, "forget@mr"), (arm, "benign_ring1@mr")]
    res = []
    for arm, attack in conds:
        apply_delta(model, arm)
        name = arm if attack is None else f"{arm} + {attack} relearning"
        pm = model
        if attack:
            rel = f"results/relearn/{arm}__{attack.replace('@', '_')}.json"
            lr = load_json(rel)["runs"][0]["lr"] if exists(rel) else CFG["relearn_lrs"][0]
            train = (items_by_ids(data, "mr", splits["forget_A"], fs) if attack.startswith("forget")
                     else ring_items(data, "mr", "ring1"))
            pm = lora_relearn(model, tok, train, lr, CFG["relearn_seeds"][0])
        for lang, items in qs.items():
            for it, out in zip(items, generate_answers(pm, tok, items)):
                res.append({"condition": name, "lang": lang, "id": it["id"], "question": it["question"],
                            "correct": f"{LETTERS[it['answer']]}. {it['choices'][it['answer']]}",
                            "output": out, "letter": first_letter(out),
                            "is_correct": first_letter(out) == LETTERS[it["answer"]]})
        n_ok = sum(r["is_correct"] for r in res if r["condition"] == name and r["lang"] == "en")
        print(f"[samples] {name}: {n_ok}/{len(qs['en'])} English answers start with the correct letter")
        if attack:
            model = strip_lora(pm); _MODEL = model
    restore_base(model)
    save_json("results/samples.json", res)
    lines = ["# Generated answers on held-out forget-B questions", "",
             "Greedy decoding. These questions were never in any unlearning-attack training set.", ""]
    for lang in ("en", "mr"):
        for it in qs[lang]:
            lines += [f"## [{lang}] {it['question'][:300]}", "",
                      f"**Correct:** {LETTERS[it['answer']]}. {it['choices'][it['answer']]}", ""]
            for r in res:
                if r["lang"] == lang and r["id"] == it["id"]:
                    mark = "correct" if r["is_correct"] else "wrong"
                    lines.append(f"- **{r['condition']}** ({mark}): {r['output'][:400]!r}")
            lines.append("")
    (OUT / "results/samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[samples] wrote {OUT / 'results/samples.md'}")


if "samples" in STAGES:
    stage_samples(); hub_push("samples")


# %% [markdown]
# ## Preflight (command line only: `--preflight`)
# Checks everything a long run needs in a few minutes, then exits without running any stage.

# %%
def run_preflight() -> None:
    import importlib.metadata as md
    print("[preflight] python", sys.version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda)
    for pkg in ("transformers", "peft", "datasets", "huggingface_hub"):
        print(f"[preflight] {pkg} {md.version(pkg)}")
    if DEV == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[preflight] GPU {torch.cuda.get_device_name(0)}: {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free")
    ensure_peft_importable()
    import peft  # noqa: F401  (the import that crashed on Colab)
    print("[preflight] peft imports OK")
    t = time.time()
    model, tok = get_model()
    print(f"[preflight] model loaded in {time.time() - t:.0f}s, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    if exists("data/items.json"):
        data = load_json("data/items.json")
        items = data["en"][CFG["forget_subject"]][:8]
        r = mc_eval(model, tok, items)
        print(f"[preflight] scored 8 questions: acc={r['acc']:.2f} format compliance={r['compliance']:.2f}")
        if DEV == "cuda":
            print(f"[preflight] peak GPU memory so far {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    found = {"data": exists("data/items.json"), "baseline": exists("results/baseline.json"),
             "layers": exists("results/layers.json")}
    for arm in ARMS:
        if arm != "A0":
            found[f"unlearn {arm}"] = exists(f"arms/{arm}/meta.json")
        found[f"eval {arm}"] = exists(f"results/eval_{arm}.json")
    n_rel = len(list((OUT / "results/relearn").glob("*.json"))) if (OUT / "results/relearn").exists() else 0
    print("[preflight] saved results found:", {k: ("yes" if v else "no") for k, v in found.items()}, f"| relearn files: {n_rel}")
    repo = CFG["results_repo"]
    if repo:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            assert api.repo_info(repo, repo_type="dataset").private, f"{repo} is PUBLIC: make it private first"
            api.upload_file(path_or_fileobj=b"preflight write check\n", path_in_repo="preflight_ok.txt",
                            repo_id=repo, repo_type="dataset", commit_message="preflight write check")
            print(f"[preflight] can write to private dataset {repo}")
        except Exception as e:
            raise SystemExit(f"[preflight] cannot write to {repo} ({type(e).__name__}: {str(e)[:200]}). Use a Write token "
                             f"from an account that is a member of the org, or run without --results-repo.")
    else:
        print("[preflight] no results repo: results stay only on this machine, copy them off before shutting it down")
    if not found["layers"]:
        print("[preflight] NOTE: no earlier results, so the run starts from scratch (adds about 1.5 h)")
    print("[preflight] all checks passed")


if "preflight" in STAGES:
    run_preflight()
