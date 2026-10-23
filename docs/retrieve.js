/* ============================================================================
   p_1 · docs/retrieve.js — Task 8, handmade retrieval (zero libraries)
   ----------------------------------------------------------------------------
   Inverted-index retrieval over the health Q&A corpus, pure JavaScript.
   Runs in the browser (window.P1.retrieve) and in Node (module.exports).
   The Python twin is colab/retrieval_py.py — both implement the SAME
   algorithm contract; the acceptance gate is >95% top-3 agreement on the
   100-question eval (scripts/test_task8_retrieval.js).

   ALGORITHM CONTRACT (identical in colab/retrieval_py.py — do not deviate)

   Chunking    one chunk per Q&A pair: chunkId = entry.id,
               text = "Q: <question>\nA: <answer>", sources passed through.
               (A generic sliding-window chunker chunkDoc() — ~200 tokens,
               overlap 40 — also ships here for FUTURE long documents; the
               corpus path does NOT use it: short Q&A pairs must not be split,
               and one-chunk-per-pair keeps ids == citation ids.)

   Tokenize    lowercase; tokens = [a-z0-9]+  (JS: text.toLowerCase().match(/[a-z0-9]+/g)
               — Python twin: re.findall(r'[a-z0-9]+', text.lower())).
               NO stopword removal, NO stemming. Empty match list → [].

   Index       term → postings [(chunkIdx, tf)] built in chunk order, so each
               list is sorted by chunkIdx (deterministic: iterate corpus in
               file order). df per term = postings.length. N = chunk count.
               avgdl = mean token count over chunks (0-guarded to 1).

   TF-IDF      weight(t,d) = (1 + ln(tf)) × ln(N/df)          (doc AND query)
               score(q,d)  = dot(q,d) / (|q|·|d|)             (cosine)
               zero-guard: |q| == 0 or |d| == 0 → score 0.

   BM25        idf(t)      = ln((N - df + 0.5) / (df + 0.5) + 1)   (always > 0)
               score(q,d)  = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl(d)/avgdl))
               k1 = 1.5, b = 0.75. Query terms are the UNIQUE terms of the
               query in first-occurrence order (no query-TF multiplier —
               both twins sum each unique term once).

   Ranking     sort by (score DESC, chunkIdx ASC) — fully deterministic,
               including zero-score ties (they fall back to corpus order).
               search() returns the top-k as
               [{ chunkId, score, rank (1-based), sources, idx }].

   Float parity: both twins are IEEE-754 double and sum in the same order
   (query terms in first-occurrence order, postings in chunkIdx order, norms
   over each chunk's first-occurrence term map). Remaining libm noise can
   only flip exact near-ties — the gate tolerates that (>95%, observed 100%).

   API
     P1.retrieve.chunk(pairs)                 → chunks [{chunkId, text, sources}]
     P1.retrieve.chunkDoc(text, opts?)        → sliding-window chunks (~200 tok, overlap 40)
     P1.retrieve.tokenize(text)               → string[] (the shared tokenizer)
     P1.retrieve.buildIndex(items, opts?)     → index { search(query, {k, method}) }
       items: pairs [{id, question, answer, sources?}] OR chunks
       [{chunkId, text, sources?}] — pairs are auto-chunked (1 chunk/pair).
       index exposes: search(query, {k=5, method='bm25'|'tfidf'}), N, vocab,
       avgdl, chunks, dl, df, buildMs.
   ============================================================================ */

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.P1 = root.P1 || {};
  root.P1.retrieve = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const VERSION = '1.0.0';
  const BM25_K1 = 1.5;
  const BM25_B = 0.75;
  const WINDOW_TOKENS = 200;   // generic long-doc chunker (NOT used by the corpus path)
  const WINDOW_OVERLAP = 40;

  /* ------------------------------------------------------------------ tokenize */

  // THE shared tokenizer. Both twins: lowercase, [a-z0-9]+, no stopwords.
  function tokenize(text) {
    const m = String(text).toLowerCase().match(/[a-z0-9]+/g);
    return m ? m : [];
  }

  /* ------------------------------------------------------------------ chunking */

  // Corpus path: one chunk per Q&A pair. chunkId = entry id (citations stay stable).
  function chunk(pairs) {
    const out = [];
    for (let i = 0; i < pairs.length; i++) {
      const p = pairs[i];
      out.push({
        chunkId: p.id,
        text: 'Q: ' + p.question + '\nA: ' + p.answer,
        sources: p.sources || [],
        idx: out.length
      });
    }
    return out;
  }

  // Generic sliding-window chunker for FUTURE long documents: windows of
  // ~WINDOW_TOKENS whitespace words, stepping by (size - overlap). Text is
  // whitespace-normalized on rebuild. The corpus path does not use this —
  // documented here so future tasks inherit the convention.
  function chunkDoc(text, opts) {
    const o = opts || {};
    const size = o.sizeTokens || WINDOW_TOKENS;
    const overlap = (o.overlap === undefined) ? WINDOW_OVERLAP : o.overlap;
    const id = o.id || 'doc';
    const sources = o.sources || [];
    const step = Math.max(1, size - overlap);
    const words = String(text).split(/\s+/).filter(Boolean);
    const out = [];
    if (!words.length) return out;
    for (let start = 0; start < words.length; start += step) {
      const win = words.slice(start, start + size);
      out.push({
        chunkId: id + '#w' + String(out.length).padStart(4, '0'),
        text: win.join(' '),
        sources: sources,
        idx: out.length
      });
      if (start + size >= words.length) break;
    }
    return out;
  }

  /* ------------------------------------------------------------------ indexing */

  // Accepts pairs [{id, question, answer, sources?}] (auto 1-chunk-per-pair)
  // or ready chunks [{chunkId, text, sources?}]. Deterministic: file order in,
  // chunkIdx order out.
  function buildIndex(items, opts) {
    const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const o = opts || {};

    let chunks;
    if (items.length && items[0] && typeof items[0].question === 'string') {
      chunks = chunk(items);                    // corpus path: 1 chunk per pair
    } else {
      chunks = items.map((c, i) => ({
        chunkId: c.chunkId,
        text: c.text,
        sources: c.sources || [],
        idx: i
      }));
    }

    const N = chunks.length;
    const dl = new Array(N);
    const docTf = new Array(N);                 // per-chunk term→tf, insertion-ordered
    const postings = Object.create(null);       // term → [[chunkIdx, tf], ...] (chunkIdx-sorted)

    for (let i = 0; i < N; i++) {
      const toks = tokenize(chunks[i].text);
      dl[i] = toks.length;
      const tf = Object.create(null);
      for (let j = 0; j < toks.length; j++) {
        const t = toks[j];
        tf[t] = (tf[t] || 0) + 1;
      }
      docTf[i] = tf;
      for (const t in tf) {
        (postings[t] || (postings[t] = [])).push([i, tf[t]]);
      }
    }

    let total = 0;
    for (let i = 0; i < N; i++) total += dl[i];
    const avgdl = N > 0 ? total / N : 0;
    const avgdlGuard = avgdl > 0 ? avgdl : 1;

    let vocab = 0;
    for (const t in postings) vocab++;

    // TF-IDF doc norms, precomputed once df is final. Sum order = each chunk's
    // first-occurrence term order (matches the Python twin; keeps parity tight).
    const docNorm = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      const tf = docTf[i];
      for (const t in tf) {
        const w = (1 + Math.log(tf[t])) * Math.log(N / postings[t].length);
        s += w * w;
      }
      docNorm[i] = Math.sqrt(s);
    }
    if (o.keepDocTf !== true) {
      for (let i = 0; i < N; i++) docTf[i] = null;   // query path reads postings, not these
    }

    const t1 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const buildMs = t1 - t0;

    /* -------------------------------------------------------------- scoring */

    // BM25: sum over UNIQUE query terms (first-occurrence order), postings in
    // chunkIdx order. idf = ln((N - df + 0.5)/(df + 0.5) + 1) — the "+1" Lucene
    // variant, always positive.
    function scoreBm25(tokens, scores) {
      const seen = Object.create(null);
      for (let j = 0; j < tokens.length; j++) {
        const t = tokens[j];
        if (seen[t]) continue;
        seen[t] = 1;
        const list = postings[t];
        if (!list) continue;
        const df = list.length;
        const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
        for (let p = 0; p < list.length; p++) {
          const idx = list[p][0], tf = list[p][1];
          const denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl[idx] / avgdlGuard);
          scores[idx] += idf * (tf * (BM25_K1 + 1)) / denom;
        }
      }
    }

    // TF-IDF cosine: query weights use the SAME formula as doc weights.
    // Convention (both twins): query terms absent from the index (df = 0) are
    // SKIPPED — ln(N/0) is undefined and the term matches no document; it would
    // only rescale |q|, never change the ranking. (Also skipped when idf == 0,
    // i.e. the term is in every chunk — its weight is exactly 0.)
    function scoreTfidf(tokens, scores) {
      const qtf = Object.create(null);
      const qTerms = [];
      for (let j = 0; j < tokens.length; j++) {
        const t = tokens[j];
        if (!qtf[t]) { qtf[t] = 0; qTerms.push(t); }
        qtf[t]++;
      }
      if (!qTerms.length) return;               // empty query → all scores stay 0
      let qn2 = 0;
      const qwArr = [];                         // [term, weight] in first-occurrence order
      for (let j = 0; j < qTerms.length; j++) {
        const t = qTerms[j];
        const list = postings[t];
        if (!list) continue;                    // df = 0 → skipped (see convention above)
        const w = (1 + Math.log(qtf[t])) * Math.log(N / list.length);
        qwArr.push([t, w]);
        qn2 += w * w;
      }
      const qnorm = Math.sqrt(qn2);
      if (qnorm === 0) return;                  // zero-guard: |q| == 0 → all 0
      for (let j = 0; j < qwArr.length; j++) {
        const t = qwArr[j][0], qw = qwArr[j][1];
        const list = postings[t];
        const idf = Math.log(N / list.length);
        if (idf === 0) continue;                // df == N → weight 0 everywhere
        for (let p = 0; p < list.length; p++) {
          const idx = list[p][0], tf = list[p][1];
          scores[idx] += qw * (1 + Math.log(tf)) * idf;
        }
      }
      // cosine normalization: dot / (|q|·|d|), zero-guarded
      for (let i = 0; i < N; i++) {
        if (scores[i] !== 0 && docNorm[i] > 0) scores[i] /= qnorm * docNorm[i];
      }
    }

    /* -------------------------------------------------------------- search */

    function search(query, optsQ) {
      const q = optsQ || {};
      const k = q.k === undefined ? 5 : q.k;
      const method = q.method === undefined ? 'bm25' : q.method;
      const tokens = tokenize(query);

      const scores = new Float64Array(N);
      if (method === 'tfidf') scoreTfidf(tokens, scores);
      else scoreBm25(tokens, scores);           // default (and only) fallback: bm25

      // Deterministic ranking: (score DESC, chunkIdx ASC). Zero scores stay in
      // corpus order, so top-k is stable even for empty/garbage queries.
      const order = new Array(N);
      for (let i = 0; i < N; i++) order[i] = i;
      order.sort(function (a, b) {
        if (scores[b] !== scores[a]) return scores[b] - scores[a];
        return a - b;
      });

      const out = [];
      const kk = Math.max(0, Math.min(k, N));
      for (let r = 0; r < kk; r++) {
        const idx = order[r];
        out.push({
          chunkId: chunks[idx].chunkId,
          score: scores[idx],
          rank: r + 1,
          sources: chunks[idx].sources,
          idx: idx
        });
      }
      return out;
    }

    return {
      search: search,
      N: N,
      vocab: vocab,
      avgdl: avgdl,
      chunks: chunks,
      dl: dl,
      df: (function () { const d = Object.create(null); for (const t in postings) d[t] = postings[t].length; return d; })(),
      postingsCount: vocab,                     // alias, same as vocab
      buildMs: buildMs,
      k1: BM25_K1,
      b: BM25_B,
      version: VERSION
    };
  }

  return {
    VERSION: VERSION,
    BM25_K1: BM25_K1,
    BM25_B: BM25_B,
    WINDOW_TOKENS: WINDOW_TOKENS,
    WINDOW_OVERLAP: WINDOW_OVERLAP,
    tokenize: tokenize,
    chunk: chunk,
    chunkDoc: chunkDoc,
    buildIndex: buildIndex
  };
});
