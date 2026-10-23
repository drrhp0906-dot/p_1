#!/usr/bin/env python3
"""Generate colab/02_tokenizer.ipynb (Task 3 — BPE tokenizer twins)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
C = []

C.append(nbf.v4.new_markdown_cell("""# 02 · BPE tokenizer — byte-level, from scratch (the Python twin)

Trains a **byte-level BPE** on the corpus TRAIN split and saves `merges.json`, then proves the
Python and JavaScript twins are **id-for-id identical** via a 200-case fixture file.

**Design (fixed contract — downstream tasks depend on it):**
- **Byte-level**: ids 0–255 are raw UTF-8 bytes (`id == byte`), so *any* string encodes, even
  with zero training data (byte fallback). Emoji / CJK / accents all just work.
- **Training corpus** = TRAIN split only, one text per entry: `question + " " + answer`,
  all joined with `"\\n"` into **one long byte sequence** (merges may cross the separators).
  Repeat 3838×: count adjacent pairs → pick the most frequent (tie → lexicographically
  smallest `(idA, idB)`) → append merge → replace every occurrence left-to-right.
  Merge *k* creates id `256 + k`.
- **Vocab arithmetic**: 256 bytes + **3838 merges** = ids `0..4093`, then specials
  `<bos>=4094`, `<eos>=4095` → **vocab_size = 4096**. (I.e. "vocab reaches 4094 before
  specials" — 4094 *merges* would collide with the special ids, so the merge count is 4094−256.)
- **Encode** = GPT-2-style rank merging: repeatedly merge ALL occurrences of the pair with
  the lowest merge rank, left-to-right, until no table pair remains. Never emits specials.
- **Decode** = ids → bytes (specials → nothing) → UTF-8 with `errors="replace"`.
  Guarantee: `decode(encode(s)) == s` for any valid UTF-8 string.

**Produces:** `corpus/merges.json` · `docs/tokenizer_fixtures.json` (+ `docs/tokenizer_data.js`,
a byte-identical snapshot so `docs/tokenizer_test.html` also works from `file://`, where
browsers block `fetch()`).

**Twins:** this notebook + `colab/tokenizer_py.py` ⇄ `docs/tokenizer.js`. Acceptance gate:
`node scripts/test_task3_tokenizer.js` → `PARITY 200/200 PASS`."""))

C.append(nbf.v4.new_markdown_cell("""## 1 · Load the corpus (train split feeds BPE)"""))

C.append(nbf.v4.new_code_cell("""# Data dir auto-detect, same convention as corpus_lib.py:
#   $P1_DATA_DIR  >  /content/drive/MyDrive/p_1/corpus (Colab, Drive mounted)
#   >  <repo_root>/corpus
import json, os
from pathlib import Path

if "google.colab" in str(getattr(__import__("sys"), "modules", {})) and not Path("/content/drive/MyDrive").is_dir():
    try:  # mount Drive on first use
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", e)

env = os.environ.get("P1_DATA_DIR", "").strip()
if env:
    CORPUS_DIR = Path(env)
elif Path("/content/drive/MyDrive").is_dir():
    CORPUS_DIR = Path("/content/drive/MyDrive/p_1/corpus")
else:
    CORPUS_DIR = Path("..") / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
print("corpus dir:", CORPUS_DIR.resolve())

entries = [json.loads(l) for l in open(CORPUS_DIR / "health_qa.jsonl", encoding="utf-8")]
train_texts = [e["question"] + " " + e["answer"] for e in entries if e["split"] == "train"]
val_texts   = [e["question"] + " " + e["answer"] for e in entries if e["split"] == "val"]
print(f"health_qa.jsonl: {len(entries)} entries -> {len(train_texts)} train / {len(val_texts)} val")
print("train corpus bytes:", sum(len(t.encode('utf-8')) for t in train_texts))"""))

C.append(nbf.v4.new_markdown_cell("""## 2 · Train BPE on the TRAIN split → `corpus/merges.json`

~2 MB of bytes, 3838 merges, pure stdlib. The trainer uses a linked-list + position-index +
lazy max-heap so it runs in **well under a minute** while staying *exact* (verified against
the naive O(passes×N) reference in `tokenizer_py.naive_train_bpe`)."""))

C.append(nbf.v4.new_code_cell("""import sys, time
from pathlib import Path

# import the Python twin (this notebook's folder in the repo, or the Colab clone)
for cand in ["", "colab", "/content/p_1/colab", "../colab", "."]:
    if Path(cand or ".").joinpath("tokenizer_py.py").is_file():
        sys.path.insert(0, str(Path(cand or ".").resolve()))
        break
import tokenizer_py as T

VOCAB_SIZE = 4096
N_MERGES = VOCAB_SIZE - 256 - 2          # 3838 merges -> ids 256..4093; 4094/4095 = specials
print(f"target: {N_MERGES} merges (vocab {256}+{N_MERGES}+{2} = {VOCAB_SIZE})")

t0 = time.time()
merges = T.train_bpe(train_texts, vocab_size=VOCAB_SIZE)
t_train = time.time() - t0
print(f"training time: {t_train:.1f}s ({len(train_texts)} texts, {N_MERGES} merges)")

MERGES_PATH = T.save_merges(merges, CORPUS_DIR / "merges.json", vocab_size=VOCAB_SIZE)
print("saved:", MERGES_PATH.resolve(), f"({MERGES_PATH.stat().st_size:,} bytes)")

tok = T.P1BPE(str(MERGES_PATH))          # load back from disk — the file is the source of truth
print(tok)
print("roundtrip of the file itself: format", json.load(open(MERGES_PATH))["format"],
      "| specials", tok.bos, tok.eos)"""))

C.append(nbf.v4.new_markdown_cell("""## 3 · Roundtrip test — 200 deterministic strings, must be lossless

Suite = **150** corpus texts (seeded sample) + **25** unicode torture strings (emoji, accents,
CJK, RTL, astral, combining marks — seeded) + **25** fixed edge cases (empty, spaces-only,
newlines, repeated chars, 1024-char word, NUL bytes, ZWJ family, …). Same seed → same 200
strings on every machine; the *same ordered list* feeds the fixtures in the next cell."""))

C.append(nbf.v4.new_code_cell("""cases = T.build_test_suite(train_texts + val_texts, seed=42)
print(f"suite: {len(cases)} cases = 150 corpus + 25 torture + 25 edge")
print("  corpus sample:", repr(cases[0][:70]))
print("  torture sample:", repr(cases[150]))
print("  edge samples:", repr(cases[175]), "|", repr(cases[192]), "|", repr(cases[196]))

bad = T.roundtrip_failures(tok, cases)
print(f"roundtrip: {len(cases) - len(bad)}/{len(cases)} LOSSLESS" + (f"  bad={bad}" if bad else ""))
assert not bad, "roundtrip failed — do not ship"
print("encode never emits specials:", max(max(tok.encode(c)) for c in cases if tok.encode(c)) < 256 + len(merges))"""))

C.append(nbf.v4.new_markdown_cell("""## 4 · Generate `docs/tokenizer_fixtures.json` (+ `tokenizer_data.js` snapshot)

The fixtures are the **bridge between the twins**: 200 cases with the exact ids the Python
twin computes. `docs/tokenizer_test.html` (browser) and `scripts/test_task3_tokenizer.js`
(Node) must reproduce them id-for-id. `tokenizer_data.js` embeds the same bytes for the
`file://` case where browsers block `fetch()` — regenerated together, so they cannot drift."""))

C.append(nbf.v4.new_code_cell("""# write next to the corpus (…/p_1/docs) and into a git clone if one exists
docs_dirs = [CORPUS_DIR.parent / "docs", Path("/content/p_1/docs")]
written = T.write_docs_artifacts(tok, cases, docs_dirs)
for p in written:
    print("wrote:", p.resolve(), f"({p.stat().st_size:,} bytes)")
print("also wrote tokenizer_data.js next to each fixtures file")
if not written:
    print("WARNING: no existing docs dir found — create one and re-run this cell")"""))

C.append(nbf.v4.new_markdown_cell("""## 5 · Stats — is this tokenizer any good?"""))

C.append(nbf.v4.new_code_cell("""val_bytes = sum(len(t.encode("utf-8")) for t in val_texts)
val_tokens = sum(len(tok.encode(t)) for t in val_texts)
print(f"vocab size          : {tok.vocab_size}")
print(f"merges              : {len(tok.merges)}  (ids 256..{256 + len(tok.merges) - 1})")
print(f"specials            : <bos>={tok.bos}  <eos>={tok.eos}")
print(f"VAL compression     : {val_bytes:,} bytes -> {val_tokens:,} tokens = {val_bytes / val_tokens:.3f} bytes/token")
print(f"VAL avg per Q&A     : {val_tokens / len(val_texts):.1f} tokens")

train_sample = train_texts[:500]
ts_b = sum(len(t.encode('utf-8')) for t in train_sample)
ts_t = sum(len(tok.encode(t)) for t in train_sample)
print(f"TRAIN(500) compress : {ts_b / ts_t:.3f} bytes/token (seen-in-training text compresses better)")

tb = tok._token_bytes
longest = sorted(range(256, 256 + len(tok.merges)), key=lambda i: (-len(tb[i]), i))[:5]
print("longest tokens (decoded):")
for i in longest:
    print(f"  id {i}  {len(tb[i]):2d} bytes  {tok.token_str(i)!r}")"""))

C.append(nbf.v4.new_markdown_cell("""## ✅ DONE WHEN

- [ ] `corpus/merges.json` exists: `format="p1-bpe-v1"`, `vocab_size=4096`, `special={bos:4094, eos:4095}`, **3838 merges**
- [ ] roundtrip cell printed **200/200 LOSSLESS**
- [ ] `docs/tokenizer_fixtures.json` + `docs/tokenizer_data.js` written (200 cases, deterministic seed 42)
- [ ] stats printed (val compression ≈ 4.4 bytes/token)
- [ ] back in the repo: `node scripts/test_task3_tokenizer.js` → **PARITY 200/200 PASS**
- [ ] `docs/tokenizer_test.html` opened in Chrome (from `http://` or `file://`) → three green tests

Then copy `merges.json` + fixtures (+ data.js) into the git repo if you ran this in Drive and
`git push` — the browser twin trains nothing, it only *loads* these files."""))

nb["cells"] = C
nbf.validate(nb)
out = "/home/z/my-project/download/p_1/colab/02_tokenizer.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "cells:", len(C))
