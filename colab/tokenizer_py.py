#!/usr/bin/env python3
# ============================================================================
# p_1 · colab/tokenizer_py.py — byte-level BPE tokenizer (the Python twin)
# ----------------------------------------------------------------------------
# Pure stdlib. The JavaScript twin is docs/tokenizer.js; both implement the
# SAME contract and must agree id-for-id — the acceptance gate is
# scripts/test_task3_tokenizer.js against docs/tokenizer_fixtures.json
# (200 cases generated from corpus/merges.json).
#
#   PROVENANCE NOTE (reorganisation 2026-10): this file was absent from the
#   workspace snapshot it was reorganised from. It was reconstructed by
#   porting docs/tokenizer.js (the byte-exact reference) back to Python and
#   cross-checking the trainer design against the colab/02_tokenizer.ipynb
#   cells that consume this module. Verification performed:
#     - encode(): 200/200 id parity vs docs/tokenizer_fixtures.json
#     - train_bpe(): re-training on the corpus TRAIN split reproduces the
#       shipped corpus/merges.json byte-for-byte (all 3838 merges)
#     - save_merges() / write_docs_artifacts() regenerate their outputs
#       byte-identically from the shipped data
#
#   CONTRACT (identical to docs/tokenizer.js — do not fork the two)
#     - BYTE-LEVEL BPE. Ids 0..255 = raw UTF-8 bytes (id == byte value). Any
#       string encodes regardless of training data: str.encode('utf-8')
#       always yields bytes, and a byte with no merge available stays a base
#       id.
#     - encode(text): str -> UTF-8 bytes -> GPT-2-style rank merging:
#       repeatedly find the adjacent pair with the LOWEST merge rank
#       (rank = index in the merges list; ranks are unique, so no
#       tie-breaking is needed) and merge ALL its occurrences left-to-right
#       simultaneously (non-overlapping greedy scan); repeat until no
#       adjacent pair is in the table. Never emits specials — the caller
#       adds bos/eos.
#     - decode(ids): ids -> token bytes (bos/eos -> empty) -> concatenate ->
#       bytes.decode('utf-8', errors='replace') which replaces malformed
#       input (never throws).
#     - Roundtrip guarantee: decode(encode(s)) == s for any valid UTF-8
#       string, because every merge step is a lossless re-chunking of the
#       byte stream.
#
#   TRAINING (used by colab/02_tokenizer.ipynb)
#     - Training corpus = TRAIN split only, one text per entry
#       (question + " " + answer), all joined with "\n" into ONE long byte
#       sequence (merges may cross the separators).
#     - train_bpe() repeats vocab_size - 258 times: count adjacent pairs ->
#       pick the most frequent (tie -> lexicographically smallest
#       (idA, idB)) -> append merge -> replace every occurrence left-to-right.
#       Merge k creates id 256 + k.
#     - Vocab arithmetic: 256 bytes + 3838 merges = ids 0..4093, then
#       specials <bos>=4094, <eos>=4095 -> vocab_size = 4096.
#     - train_bpe() is a linked-list + position-index + lazy max-heap
#       implementation (well under a minute on the ~2 MB corpus); its exact
#       reference is naive_train_bpe() — same output, O(passes x N).
#
#   PARITY NOTES (read before touching the merge loop)
#     - Everything happens in the BYTE domain: text.encode('utf-8') produces
#       exactly the same bytes as JS TextEncoder().encode(str) — including
#       for astral emoji (surrogate pairs are resolved before bytes exist).
#     - The merge loop scans pairs left-to-right and keeps the first lowest
#       rank (strict <). JS does the same, so scan order matters — do not
#       "optimize" it into a different traversal.
#     - Pair keys are (int, int) tuples — mirrors JS a * 65536 + b exactly.
#
#   USAGE
#     from tokenizer_py import P1BPE
#     tok = P1BPE('/path/to/merges.json')     # str/Path | parsed dict | list
#     ids = tok.encode('hello'); tok.decode(ids)
#
#   SELFTEST (python3 colab/tokenizer_py.py)
#     fixture parity vs docs/tokenizer_fixtures.json when found, roundtrip,
#     no-specials, and fast-vs-naive trainer equivalence on a corpus slice.
# ============================================================================

from __future__ import annotations

import heapq
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

VERSION = '1.0.0'
FORMAT = 'p1-bpe-v1'
NUM_BASE = 256            # ids 0..255 = raw bytes
FIXTURE_NOTE = 'expected ids computed by Python twin'


# ----------------------------------------------------------------------------
# P1BPE — the tokenizer class (mirror of docs/tokenizer.js P1BPE)
# ----------------------------------------------------------------------------

class P1BPE:
    """Byte-level BPE loaded from merges.json (path, parsed dict or raw list)."""

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            with open(source, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            merges = obj['merges']
            special = obj.get('special') or {}
            vocab_size = obj.get('vocab_size') or (NUM_BASE + len(merges) + 2)
            self.format = obj.get('format', FORMAT)
        elif isinstance(source, dict):
            if not isinstance(source.get('merges'), list):
                raise TypeError('P1BPE: pass a merges.json path, a parsed '
                                'merges.json object or a [[a,b],...] list')
            merges = source['merges']
            special = source.get('special') or {}
            vocab_size = source.get('vocab_size') or (NUM_BASE + len(merges) + 2)
            self.format = source.get('format', FORMAT)
        elif isinstance(source, list):
            merges = source
            special = {}
            vocab_size = NUM_BASE + len(merges) + 2
            self.format = FORMAT
        else:
            raise TypeError('P1BPE: pass a merges.json path, a parsed '
                            'merges.json object or a [[a,b],...] list')

        self.merges = [[int(a), int(b)] for a, b in merges]
        self._rank = {(a, b): k for k, (a, b) in enumerate(self.merges)}
        self.vocab_size = int(vocab_size)
        self.bos = int(special['bos']) if 'bos' in special else self.vocab_size - 2
        self.eos = int(special['eos']) if 'eos' in special else self.vocab_size - 1

        # token id -> byte chunk (specials -> b''). Built in merge order, so
        # tb[256+k] = tb[a] + tb[b] always resolves (a, b < 256 + k).
        tb = [None] * self.vocab_size
        for i in range(NUM_BASE):
            tb[i] = bytes([i])
        for k, (a, b) in enumerate(self.merges):
            tb[NUM_BASE + k] = tb[a] + tb[b]
        tb[self.bos] = b''
        tb[self.eos] = b''
        self._token_bytes = tb

    # -- encode -------------------------------------------------------------

    def encode(self, text):
        """str -> list[int]. GPT-2-style rank merging; NEVER emits specials."""
        ids = list(text.encode('utf-8'))
        rank = self._rank
        while len(ids) >= 2:
            # 1) find the adjacent pair with the lowest merge rank
            best_rank = -1
            best_pair = None
            prev = ids[0]
            for cur in ids[1:]:
                r = rank.get((prev, cur))
                if r is not None and (best_rank < 0 or r < best_rank):
                    best_rank = r
                    best_pair = (prev, cur)
                prev = cur
            if best_rank < 0:
                break                                   # no pair in table
            # 2) merge ALL its occurrences left-to-right, non-overlapping
            a, b = best_pair
            m = NUM_BASE + best_rank
            out = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(m)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    # -- decode -------------------------------------------------------------

    def decode(self, ids):
        """ids -> str. bos/eos decode to nothing; malformed UTF-8 replaced."""
        tb = self._token_bytes
        buf = b''.join(tb[t] for t in ids if tb[t])
        return buf.decode('utf-8', errors='replace')

    def token_str(self, token_id):
        """Human-readable decoding of a single token (for stats/UI)."""
        t = self._token_bytes[token_id] if 0 <= token_id < len(self._token_bytes) else None
        return t.decode('utf-8', errors='replace') if t else ''

    @property
    def num_merges(self):
        return len(self.merges)

    def __repr__(self):
        return ('<P1BPE %s vocab=%d merges=%d bos=%d eos=%d>'
                % (self.format, self.vocab_size, len(self.merges),
                   self.bos, self.eos))


# ----------------------------------------------------------------------------
# Training — naive reference first (the spec), then the fast equivalent
# ----------------------------------------------------------------------------

def _corpus_bytes(texts):
    """The training byte sequence: texts joined with '\n', UTF-8 encoded."""
    return list('\n'.join(texts).encode('utf-8'))


def _replace_all(ids, a, b, m):
    """Replace every non-overlapping (a, b) occurrence left-to-right with m."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(m)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def naive_train_bpe(texts, vocab_size=4096):
    """The SPEC, verbatim: one full recount + replace per merge. O(passes x N).

    Slow (hours on the full corpus) — use for small slices and as the
    correctness reference for train_bpe().
    """
    ids = _corpus_bytes(texts)
    merges = []
    for k in range(max(0, vocab_size - NUM_BASE - 2)):
        if len(ids) < 2:
            break
        counts = defaultdict(int)
        prev = ids[0]
        for cur in ids[1:]:
            counts[(prev, cur)] += 1
            prev = cur
        if not counts:
            break
        # most frequent; tie -> lexicographically smallest (idA, idB)
        (a, b) = min(counts, key=lambda p: (-counts[p], p))
        merges.append([a, b])
        ids = _replace_all(ids, a, b, NUM_BASE + k)
    return merges


def train_bpe(texts, vocab_size=4096):
    """Fast train_bpe: linked list + incremental pair counts + lazy max-heap.

    Produces EXACTLY the same merge sequence as naive_train_bpe (same count
    arithmetic, same tie-breaking), but each pass only touches the merged
    occurrences and their boundaries instead of rescanning the corpus.
    """
    tok = _corpus_bytes(texts)
    n = len(tok)
    merges = []
    if n < 2:
        return merges

    # doubly linked list over fixed slots
    prv = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[n - 1] = -1
    dead = bytearray(n)
    head = 0

    # incremental adjacent-pair counts
    counts = defaultdict(int)
    i = head
    while i != -1 and nxt[i] != -1:
        counts[(tok[i], tok[nxt[i]])] += 1
        i = nxt[i]

    # lazy max-heap: entries (-count, a, b); stale entries are skipped on pop
    heap = [(-c, p[0], p[1]) for p, c in counts.items()]
    heapq.heapify(heap)

    def push(pair, cnt):
        # Push an entry for the pair's CURRENT count — after EVERY change,
        # increment or decrement. A pair whose count only shrank would
        # otherwise be left with stale (too-high) entries only and become
        # invisible to the pop loop despite still being live.
        if cnt > 0:
            heapq.heappush(heap, (-cnt, pair[0], pair[1]))

    n_merges = max(0, vocab_size - NUM_BASE - 2)
    for k in range(n_merges):
        # pop the live entry with the highest count (tie: smallest (a, b))
        a = b = -1
        while heap:
            negc, x, y = heapq.heappop(heap)
            c = counts.get((x, y), 0)
            if c > 0 and c == -negc:
                a, b = x, y
                break                               # live global max
            # else: stale entry — discard
        if a < 0:
            break                                   # no pairs left at all
        m = NUM_BASE + k
        merges.append([a, b])

        # merge ALL occurrences of (a, b) left-to-right, non-overlapping.
        # m is strictly larger than every id in the list, so a merge can
        # never create a NEW (a, b) adjacency — one scan suffices.
        i = head
        while i != -1:
            j = nxt[i]
            if j != -1 and tok[i] == a and tok[j] == b:
                p = prv[i]
                q = nxt[j]
                # destroy the three absorbed adjacencies (re-push at their
                # new counts so surviving occurrences stay visible)
                counts[(a, b)] -= 1
                push((a, b), counts[(a, b)])
                if p != -1:
                    pair_old = (tok[p], a)
                    counts[pair_old] -= 1
                    push(pair_old, counts[pair_old])
                if q != -1:
                    pair_old = (b, tok[q])
                    counts[pair_old] -= 1
                    push(pair_old, counts[pair_old])
                # relink: p - m - q
                tok[i] = m
                dead[j] = 1
                nxt[i] = q
                if q != -1:
                    prv[q] = i
                # create the two new adjacencies
                if p != -1:
                    pair_new = (tok[p], m)
                    counts[pair_new] += 1
                    push(pair_new, counts[pair_new])
                if q != -1:
                    pair_new = (m, tok[q])
                    counts[pair_new] += 1
                    push(pair_new, counts[pair_new])
                # continue at the merged node; (m, tok[q]) is never (a, b)
            else:
                i = j
        # end of pass: zero (a, b) occurrences remain; its count is 0 and
        # stays 0 (pairs not involving the newest id only lose count)

    return merges


def save_merges(merges, path, vocab_size=4096):
    """Write merges.json (compact, canonical key order). Returns the Path."""
    obj = {
        'format': FORMAT,
        'vocab_size': int(vocab_size),
        'special': {'bos': int(vocab_size) - 2, 'eos': int(vocab_size) - 1},
        'merges': [[int(a), int(b)] for a, b in merges],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, separators=(',', ':'), ensure_ascii=True)
    return path


# ----------------------------------------------------------------------------
# Test suite + docs artifacts (used by colab/02_tokenizer.ipynb)
# ----------------------------------------------------------------------------

# 25 fixed edge cases (empty, whitespace runs, repeated chars, NUL, ZWJ
# family, lone combining marks, replacement char, ...). Fixed = not seeded.
_EDGE_CASES = [
    '',
    ' ',
    '   ',
    '\t',
    '\n',
    '\n\n\n',
    ' \n \n ',
    '\r\n',
    'a',
    'aa',
    'a' * 64,
    'a' * 80,
    'ab' * 40,
    'the the the the the',
    ' !"#$%&\'()*+,-./0123456789:;<=>?@[]^_`{|}~',
    '  leading and trailing spaces  ',
    '\xa0nbsp\xa0',
    '\u200bzero\u200bwidth',
    'écombining',
    '́',
    '😀',
    '👨\u200d👩\u200d👧\u200d👦',
    '\x00NUL\x00',
    '\ufffd replacement char',
    ' \x0b\x0c\r mixed whitespace ',
]

# Unicode torture fragments (emoji + ZWJ, accents, CJK, RTL marks, astral
# planes, combining stacks, ligatures, ...). Composed with the suite seed.
_TORTURE_FRAGMENTS = [
    'Здоровье и сон', '健康と睡眠', '日本語のテキスト', '한국어 텍스트',
    'मरीज़', 'صحة ونوم', 'กล้วย', '¿qué?', 'café naïve résumé',
    'ᚠᚢᚦ runes', ' Marek \t Zdrowie ', 'ȩ́', 'x\u200d́y',
    'ﬁﬂ ligatures', '\u202ertl\u202c', '👍🏽👌🏿',
    '👨\u200d👩\u200d👧\u200d👦', '𓀀𓁐', '𝔥𝔢𝔞𝔩𝔱𝔥', '𝟈', '🢥',
    '🍉', '🧒', 'ᜀᜌ᜔', 'ⵎⵣⵗ', 'ं', 'ٕ', 'ٓ', '҆', 'ĩ', 'ƚ', 'ƞ',
    'Þ', 'ë', 'ह़', 'ऩ', '٦', 'ٿ', 'ڠ', 'Ԩ', 'Ԗ', '晟', '鵱', '淃',
    'হ़', 'क़', 'tb', 'a', ' ',
]


def build_test_suite(texts, seed=42):
    """200 deterministic strings: 150 corpus (seeded sample) + 25 torture
    (seeded composition) + 25 fixed edge cases. Same seed -> same suite on
    every machine; the ordered list feeds the fixtures verbatim."""
    rng = random.Random(seed)
    pool = list(texts)
    corpus = rng.sample(pool, min(150, len(pool)))

    torture = []
    for _ in range(25):
        n_frag = rng.randint(2, 5)
        parts = [rng.choice(_TORTURE_FRAGMENTS) for _ in range(n_frag)]
        torture.append(''.join(parts))

    return corpus + torture + _EDGE_CASES


def roundtrip_failures(tok, cases):
    """Indices of cases where decode(encode(c)) != c. Must be empty."""
    bad = []
    for i, c in enumerate(cases):
        if tok.decode(tok.encode(c)) != c:
            bad.append(i)
    return bad


def write_docs_artifacts(tok, cases, docs_dirs):
    """Write docs/tokenizer_fixtures.json + docs/tokenizer_data.js into every
    EXISTING dir in docs_dirs. Returns the list of written Paths."""
    fixtures = {
        'format': FORMAT,
        'note': FIXTURE_NOTE,
        'cases': [{'id': i, 'text': c, 'ids': tok.encode(c)}
                  for i, c in enumerate(cases)],
    }
    merges_obj = {
        'format': FORMAT,
        'vocab_size': tok.vocab_size,
        'special': {'bos': tok.bos, 'eos': tok.eos},
        'merges': tok.merges,
    }
    fix_json = json.dumps(fixtures, separators=(',', ':'), ensure_ascii=True)
    merges_json = json.dumps(merges_obj, separators=(',', ':'), ensure_ascii=True)
    js = ('/* p_1 tokenizer data snapshot — file:// fallback for docs/tokenizer_test.html\n'
          '   (browsers block fetch() on file: URLs). AUTO-GENERATED by colab/02_tokenizer.ipynb\n'
          '   together with tokenizer_fixtures.json — do not edit by hand. */\n'
          'window.P1_TOKENIZER_DATA = {merges: ' + merges_json +
          ',fixtures: ' + fix_json + '};\n')

    written = []
    for d in docs_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        fpath = d / 'tokenizer_fixtures.json'
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(fix_json)
        jpath = d / 'tokenizer_data.js'
        with open(jpath, 'w', encoding='utf-8') as f:
            f.write(js)
        written.extend([fpath, jpath])
    return written


# ----------------------------------------------------------------------------
# __main__ — selftest (fixture parity when the repo layout is present)
# ----------------------------------------------------------------------------

def _selftest():
    here = Path(__file__).resolve().parent
    repo = here.parent

    # 1) fixture parity: encode() must reproduce docs/tokenizer_fixtures.json
    fixtures_path = repo / 'docs' / 'tokenizer_fixtures.json'
    merges_path = repo / 'corpus' / 'merges.json'
    if fixtures_path.is_file() and merges_path.is_file():
        tok = P1BPE(str(merges_path))
        fx = json.loads(fixtures_path.read_text(encoding='utf-8'))
        bad = [c['id'] for c in fx['cases'] if tok.encode(c['text']) != c['ids']]
        print('fixture parity: %d/%d PASS' % (len(fx['cases']) - len(bad), len(fx['cases'])))
        rt = roundtrip_failures(tok, [c['text'] for c in fx['cases']])
        print('roundtrip: %s' % ('LOSSLESS' if not rt else 'FAILED %r' % rt))
        tops = [max(tok.encode(c['text'])) for c in fx['cases'] if tok.encode(c['text'])]
        print('no specials emitted: %s (max id %d < %d)'
              % (max(tops) < NUM_BASE + len(tok.merges), max(tops),
                 NUM_BASE + len(tok.merges)))
    else:
        print('fixture parity: SKIPPED (%s / %s not found)'
              % (fixtures_path, merges_path))

    # 2) fast trainer == naive trainer on a small real slice
    hq = repo / 'corpus' / 'health_qa.jsonl'
    if hq.is_file():
        entries = [json.loads(l) for l in hq.open(encoding='utf-8')]
        texts = [e['question'] + ' ' + e['answer'] for e in entries[:300]]
        fast = train_bpe(texts, vocab_size=320)
        naive = naive_train_bpe(texts, vocab_size=320)
        print('fast vs naive (300 texts, 64 merges): %s'
              % ('IDENTICAL' if fast == naive else 'MISMATCH'))
        if fast != naive:
            for i, (f_, n_) in enumerate(zip(fast, naive)):
                if f_ != n_:
                    print('  first divergence at merge %d: %r vs %r' % (i, f_, n_))
                    break
    else:
        print('trainer check: SKIPPED (%s not found)' % hq)


if __name__ == '__main__':
    _selftest()
