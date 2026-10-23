/* ============================================================================
   p_1 · docs/verify.js — Task 9, claim verification + eval harness (JS twin)
   ----------------------------------------------------------------------------
   The machine that catches hallucinations, for the browser (window.P1.verify)
   and Node (module.exports). The Python twin is colab/eval_lib.py — BOTH
   implement the SAME contract, machine-checked by
   scripts/test_task9_verify.js on docs/verify_fixtures.json (54 cases with
   expected outputs computed by the Python twin).

   METHOD (identical to colab/eval_lib.py — that file is the method section)
   ----------------------------------------------------------------------------
   1. CLAIM EXTRACTION  splitClaims(answer) -> [{ text, numbers }]
      - Sentence split on [.!?] followed by whitespace or end-of-string
        (abbreviations split too — documented simplification).
      - Long sentences (> LONG_SENTENCE_WORDS whitespace words) split at the
        LEFTMOST clause cue (", because ", ", which ", ", that ", ", but ",
        ", and "); the cue is dropped; valid only if BOTH halves keep >=
        MIN_CLAUSE_WORDS words, else the next-leftmost cue is tried; valid
        halves recurse.
      - Leading discourse markers (So/Well/Actually/Yes/No/Sure,
        case-insensitive) stripped when followed by comma or whitespace, then
        leftover leading punctuation is stripped ("Yes - foo" -> "foo").
        Conservative edge: "No single diet ..." loses its "No " — can only
        make the verifier stricter, never more lenient.
      - Numbers: NUMBER_RE (\b\d+(?:[.,]\d+)? + optional unit), "," acts as a
        decimal point; ranges "6-8" decompose into [6, 8] (regex stops at the
        hyphen), so a chunk saying "6-8 cups" covers claims of 6 AND of 8.

   2. VERIFIER  verifyClaim(claim, chunks) -> { verdict, score, bestChunkId, ... }
      - Content words: [a-z0-9]+ lowercase minus the ONE STOPWORDS list
        (identical to the Python twin).
      - contentF1 = 2*|claim ∩ evidence| / (|claim|+|evidence|) over
        content-word MULTISETS (min-count intersection; integer arithmetic,
        exact in IEEE double). Evidence granularity = SENTENCE: score(claim,
        chunk) is the best content-F1 over the chunk's answer-part sentences
        (measured deviation from whole-chunk scoring: whole-chunk F1
        structurally caps short true claims below the supported bar — see the
        Python header). Chunk evidence = the ANSWER part ("after '\nA: '"):
        the "Q:" line is retrieval metadata, not evidence.
      - numbersOK: every claim number must appear in the best chunk's numbers
        (exact double equality; "6" and "6.0" are the same double; "6-8"
        covers 6 and 8).
      - negConflict: claim negation XOR best-chunk-sentence negation.
      - Verdict constants (identical values in colab/eval_lib.py):
          F1_SUPPORTED 0.34 · F1_MIN 0.22 · F1_NOT_IN_CORPUS 0.08 ·
          GROUNDED_MIN_F1 0.08 · REWRITE_MIN_F1 0.15 · MYTH_MARGIN 0.02 ·
          ABSTAIN_TAU 20.0454 (calibrated ONCE: median top-1 BM25 over the 60
          answerable eval questions; see eval_lib.calibrate_tau).
          not-in-corpus: score < F1_NOT_IN_CORPUS (counted separately, never a
          hallucination) · supported: score >= F1_SUPPORTED AND numbersOK AND
          no negConflict · unsupported: everything else (the 0.22..0.34 "weak"
          zone is unsupported by design — the verifier errs toward flagging).

   3. MODES + MYTH COMPARATOR + REFUSALS — see eval_lib.py §3/§4 (ported 1:1);
      the myth folk-vs-correction comparison runs on NEGATION-STRIPPED tokens
      with the explicit correction-cue override ("no evidence", "not true",
      "does not", "isn't", "actually"); the per-claim "negates folk" count is
      reported as meta (folk beliefs mt-05/07/10 themselves contain negation,
      so a per-claim negation override would misclassify honest echoing).

   API  P1.verify.{ splitClaims, verifyClaim, contentWords, isRefusal,
   mythVerdict, answerPart, stripNegTokens, groundedAnswer, verifiedAnswer,
   topkChunks, bestChunkSentence, contentF1, measureAnswer, constants... }
   ============================================================================ */

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.P1 = root.P1 || {};
  root.P1.verify = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const VERSION = '1.0.0';
  const FIXTURES_FORMAT = 'p1-verify-fixtures-v1';

  /* ------------------------------------------------------------- constants */

  // The ONE stopword list — identical to colab/eval_lib.py STOPWORDS.
  const STOPWORDS = new Set((
    'a an the is are was were be been to of in on for with and or but if that ' +
    'this these those it its as at by from you your i we they he she do does ' +
    'did not no yes can could should would will may might there their have has'
  ).split(' '));

  // Verifier thresholds — identical values in colab/eval_lib.py (one place per
  // language; parity fixtures pin them).
  const F1_SUPPORTED = 0.34;
  const F1_MIN = 0.22;
  const F1_NOT_IN_CORPUS = 0.08;
  const GROUNDED_MIN_F1 = F1_NOT_IN_CORPUS;   // alias, one value
  const REWRITE_MIN_F1 = 0.15;
  const MYTH_MARGIN = 0.02;
  const ABSTAIN_TAU = 20.0454;                // calibrated ONCE (see header)

  const TOP_K = 3;
  const GROUNDED_SENTENCES = 3;
  const LONG_SENTENCE_WORDS = 18;
  const MIN_CLAUSE_WORDS = 4;

  const NUMBER_RE = /\b\d+(?:[.,]\d+)?\s*(?:%|percent|cups?|glasses?|hours?|minutes?|days?|weeks?|mg|ml|grams?|kg|liters?|litres?)?\b/g;
  const NUMBER_HEAD_RE = /^\d+(?:[.,]\d+)?/;
  const NEGATION_RE = /\b(?:not|no|never|don't|doesn't|isn't|aren't|cannot|can't)\b/;
  const CLAUSE_CUES = [', because ', ', which ', ', that ', ', but ', ', and '];
  const MARKER_RE = /^(?:so|well|actually|yes|no|sure)(?:\s*[,;]|\s+)\s*/i;
  const LEADING_PUNCT_RE = /^[ ,.\-;:]+/;

  const MYTH_CUES = ['no evidence', 'not true', 'does not', 'isn\'t', 'actually'];
  const NEGATION_TOKENS = new Set([
    'not', 'no', 'never', 'cannot', 'dont', 'doesnt', 'isnt', 'arent',
    'cant', 'wont']);

  const REFUSAL_MARKERS = [
    'don\'t know', 'do not know', 'no established', 'cannot', 'can\'t say',
    'not confident', 'depends', 'no single', 'no answer', 'i don\'t have'];

  const GROUNDED_REFUSAL = 'I don\'t have reliable information about that in my corpus.';
  const ABSTAIN_REFUSAL = 'I\'m not confident; I can\'t answer that from my corpus.';

  /* ---------------------------------------------------------- text utils */

  // THE shared tokenizer: lowercase, [a-z0-9]+, no stopwords, no stemming.
  function tokenize(text) {
    const m = String(text).toLowerCase().match(/[a-z0-9]+/g);
    return m ? m : [];
  }

  function jsWords(s) {
    const t = String(s).trim();
    return t ? t.split(/\s+/) : [];
  }

  // Tokens MINUS stopwords — the multiset basis for F1 (duplicates kept).
  function contentWords(text) {
    const stop = STOPWORDS;
    return tokenize(text).filter(function (t) { return !stop.has(t); });
  }

  // Evidence part of a chunk: after the first "\nA: " (or leading "A: ").
  function answerPart(chunkText) {
    const t = String(chunkText);
    const i = t.indexOf('\nA: ');
    if (i !== -1) return t.slice(i + 4);
    if (t.lastIndexOf('A: ', 0) === 0) return t.slice(3);
    return t;
  }

  // Sentence split on [.!?] followed by whitespace or end-of-string.
  function splitSentences(text) {
    const t = String(text);
    const out = [];
    let buf = '';
    for (let i = 0; i < t.length; i++) {
      const ch = t[i];
      buf += ch;
      if ((ch === '.' || ch === '!' || ch === '?') &&
          (i + 1 >= t.length || /\s/.test(t[i + 1]))) {
        const s = buf.trim();
        if (s) out.push(s);
        buf = '';
      }
    }
    const tail = buf.trim();
    if (tail) out.push(tail);
    return out;
  }

  // Leftmost valid clause cue: {pos, cue} or null. Valid = both halves keep
  // >= MIN_CLAUSE_WORDS words. Tie-break: (position, cue string).
  function findClauseSplit(sentence) {
    let best = null;
    for (const cue of CLAUSE_CUES) {
      let start = 0;
      for (;;) {
        const p = sentence.indexOf(cue, start);
        if (p === -1) break;
        const left = sentence.slice(0, p).trim();
        const right = sentence.slice(p + cue.length).trim();
        if (jsWords(left).length >= MIN_CLAUSE_WORDS &&
            jsWords(right).length >= MIN_CLAUSE_WORDS) {
          if (best === null || p < best.pos ||
              (p === best.pos && cue < best.cue)) {
            best = { pos: p, cue: cue };
          }
        }
        start = p + 1;
      }
    }
    return best;
  }

  // Clause-split long sentences; drop the cue; recurse on both halves.
  function splitLong(sentence) {
    if (jsWords(sentence).length <= LONG_SENTENCE_WORDS) return [sentence];
    const hit = findClauseSplit(sentence);
    if (!hit) return [sentence];
    const left = sentence.slice(0, hit.pos).trim();
    const right = sentence.slice(hit.pos + hit.cue.length).trim();
    return splitLong(left).concat(splitLong(right));
  }

  // Strip leading discourse markers + leftover leading punctuation.
  function stripMarkers(s) {
    s = String(s).trim();
    for (;;) {
      const m = s.match(MARKER_RE);
      if (!m) break;
      s = s.slice(m[0].length);
      s = s.replace(LEADING_PUNCT_RE, '').trim();
    }
    return s;
  }

  // Numbers per NUMBER_RE as normalized floats ("," = decimal point).
  function extractNumbers(text) {
    const out = [];
    const mm = String(text).match(NUMBER_RE);
    if (mm) {
      for (const raw of mm) {
        const head = raw.match(NUMBER_HEAD_RE);
        out.push(parseFloat(head[0].replace(/,/g, '.')));
      }
    }
    return out;
  }

  // Answer -> atomic claims [{text, numbers}].
  function splitClaims(answer) {
    const claims = [];
    for (const sent of splitSentences(String(answer))) {
      for (const part of splitLong(sent)) {
        let c = stripMarkers(part);
        c = c.replace(LEADING_PUNCT_RE, '').trim();
        if (!c) continue;
        claims.push({ text: c, numbers: extractNumbers(c) });
      }
    }
    return claims;
  }

  // Whole-word negation check on the raw string (the tokenizer splits "don't").
  function hasNegation(text) {
    return NEGATION_RE.test(String(text).toLowerCase());
  }

  /* ------------------------------------------------------------ verifier */

  // 2*|a ∩ b| / (|a|+|b|) over token MULTISETS (min-count intersection;
  // integer arithmetic -> exact -> identical in both languages).
  function multisetF1(aTokens, bTokens) {
    const ca = new Map(), cb = new Map();
    for (const t of aTokens) ca.set(t, (ca.get(t) || 0) + 1);
    for (const t of bTokens) cb.set(t, (cb.get(t) || 0) + 1);
    let inter = 0, den = 0;
    for (const t of ca.keys()) {
      const a = ca.get(t), b = cb.get(t) || 0;
      inter += a < b ? a : b;
      den += a;
    }
    for (const t of cb.keys()) den += cb.get(t);
    return den > 0 ? 2.0 * inter / den : 0.0;
  }

  function contentF1(textA, textB) {
    return multisetF1(contentWords(textA), contentWords(textB));
  }

  // (sentence, f1) of the chunk's answer-part sentence with the highest
  // content-F1 vs the claim; earliest sentence wins ties.
  function bestChunkSentence(claimText, chunk) {
    let best = '', bestF = -1.0;
    const sents = splitSentences(answerPart(chunk && chunk.text ? chunk.text : ''));
    for (const s of sents) {
      const f = contentF1(claimText, s);
      if (f > bestF) { best = s; bestF = f; }
    }
    return { sentence: best, f: best ? bestF : 0.0 };
  }

  function verifyClaim(claim, chunks) {
    const isObj = claim !== null && typeof claim === 'object';
    const text = String(isObj ? (claim.text || '') : claim);
    const claimNumbers = (isObj && claim.numbers ? claim.numbers : extractNumbers(text))
      .map(Number);

    let bestF = -1.0, bestCh = null, bestSentence = '';
    for (const ch of (chunks || [])) {
      const r = bestChunkSentence(text, ch);
      if (bestCh === null || r.f > bestF) {
        bestF = r.f; bestCh = ch; bestSentence = r.sentence;
      }
    }

    if (bestCh === null) {
      return { verdict: 'not-in-corpus', score: 0.0, bestChunkId: null,
               bestSentence: '', numbersOK: claimNumbers.length === 0,
               negConflict: false, claimNumbers: claimNumbers,
               chunkNumbers: [] };
    }

    const score = bestF;
    const chunkNumbers = extractNumbers(answerPart(bestCh.text || ''));
    let numbersOK = true;
    for (const n of claimNumbers) {
      let found = false;
      for (const w of chunkNumbers) { if (n === w) { found = true; break; } }
      if (!found) { numbersOK = false; break; }
    }
    const negConflict = hasNegation(text) !== hasNegation(bestSentence);

    let verdict;
    if (score < F1_NOT_IN_CORPUS) verdict = 'not-in-corpus';
    else if (score >= F1_SUPPORTED && numbersOK && !negConflict) verdict = 'supported';
    else verdict = 'unsupported';

    return { verdict: verdict, score: score, bestChunkId: bestCh.chunkId,
             bestSentence: bestSentence, numbersOK: numbersOK,
             negConflict: negConflict, claimNumbers: claimNumbers,
             chunkNumbers: chunkNumbers };
  }

  /* --------------------------------------------------------------- modes */

  // Task 8 search results WITH chunk text attached (search() returns ids +
  // scores; verification needs the text). BM25 rank order, deterministic.
  function topkChunks(index, question, k) {
    const kk = (k === undefined ? TOP_K : k);
    const out = [];
    for (const r of index.search(question, { k: kk, method: 'bm25' })) {
      out.push({ chunkId: r.chunkId, score: r.score, rank: r.rank,
                 sources: r.sources || [], idx: r.idx,
                 text: index.chunks[r.idx].text });
    }
    return out;
  }

  // GROUNDED mode: answer composed ONLY of the best chunk sentences vs the
  // question (>= 2 when available); refuses below GROUNDED_MIN_F1.
  function groundedAnswer(question, topk) {
    const cands = [];
    (topk || []).forEach(function (ch, rank) {
      const sents = splitSentences(answerPart(ch.text || ''));
      sents.forEach(function (sent, si) {
        cands.push({ f: contentF1(question, sent), rank: rank, si: si,
                     sent: sent, chunkId: ch.chunkId });
      });
    });
    if (!cands.length) {
      return { answer: GROUNDED_REFUSAL, citations: [], refused: true,
               bestSentenceF1: 0.0 };
    }
    cands.sort(function (a, b) {
      if (a.f !== b.f) return b.f - a.f;
      if (a.rank !== b.rank) return a.rank - b.rank;
      return a.si - b.si;
    });
    const bestF = cands[0].f;
    if (bestF < GROUNDED_MIN_F1) {
      return { answer: GROUNDED_REFUSAL, citations: [], refused: true,
               bestSentenceF1: bestF };
    }
    const picked = [], seen = {};
    for (const c of cands) {
      const key = c.sent.toLowerCase().split(/\s+/).join(' ');
      if (seen[key]) continue;
      seen[key] = true;
      picked.push(c);
      if (picked.length >= GROUNDED_SENTENCES) break;
    }
    const answer = picked.map(function (c) { return c.sent; }).join(' ');
    const citations = [];
    for (const c of picked) {
      if (citations.indexOf(c.chunkId) === -1) citations.push(c.chunkId);
    }
    return { answer: answer, citations: citations, refused: false,
             bestSentenceF1: bestF };
  }

  function cite(citations, id) {
    if (id !== null && id !== undefined && citations.indexOf(id) === -1) {
      citations.push(id);
    }
  }

  // VERIFIED mode: RAW answer -> claims -> verify vs top-3; rewrite unsupported
  // claims from the best chunk sentence (>= REWRITE_MIN_F1) else "[unsupported]".
  function verifiedAnswer(question, rawAnswer, topk) {
    if (isRefusal(rawAnswer)) {
      return { answer: String(rawAnswer), citations: [], claims: [],
               nKept: 0, nRewritten: 0, nFlagged: 0 };
    }
    const pieces = [], per = [], citations = [];
    let nKept = 0, nRewritten = 0, nFlagged = 0;
    for (const cl of splitClaims(rawAnswer)) {
      const v = verifyClaim(cl, topk);
      let final, action;
      if (v.verdict === 'supported') {
        final = cl.text; action = 'kept'; nKept++;
        cite(citations, v.bestChunkId);
      } else if (v.score >= REWRITE_MIN_F1 && v.bestSentence) {
        final = v.bestSentence; action = 'rewritten'; nRewritten++;
        cite(citations, v.bestChunkId);
      } else {
        final = '[unsupported] ' + cl.text; action = 'flagged'; nFlagged++;
      }
      pieces.push(final);
      per.push({ claim: cl.text, verdict: v.verdict, action: action,
                 score: v.score, bestChunkId: v.bestChunkId });
    }
    return { answer: pieces.join(' '), citations: citations, claims: per,
             nKept: nKept, nRewritten: nRewritten, nFlagged: nFlagged };
  }

  // Refusal detection: case-insensitive REFUSAL_MARKERS substring match.
  function isRefusal(text) {
    const t = String(text).toLowerCase();
    for (const m of REFUSAL_MARKERS) { if (t.indexOf(m) !== -1) return true; }
    return false;
  }

  // Per-answer claim stats on the FINAL text (the uniform yardstick).
  // Refusals assert nothing: zero claims (prevents the canned refusal strings
  // from re-verifying as unsupported "claims" — measured pitfall).
  function measureAnswer(answerText, topk) {
    if (isRefusal(answerText)) {
      return { total_claims: 0, unsupported: 0, not_in_corpus: 0 };
    }
    const claims = splitClaims(answerText);
    let unsup = 0, nic = 0;
    for (const c of claims) {
      const v = verifyClaim(c, topk);
      if (v.verdict === 'unsupported') unsup++;
      else if (v.verdict === 'not-in-corpus') nic++;
    }
    return { total_claims: claims.length, unsupported: unsup, not_in_corpus: nic };
  }

  /* -------------------------------------------------------------- myths */

  // Content tokens minus negation words — the folk-vs-correction multiset
  // basis (apostrophe splits leave "don"/"t"; the standalone "t" stays —
  // identical in both languages, symmetric).
  function stripNegTokens(text) {
    return tokenize(text).filter(function (t) {
      return !STOPWORDS.has(t) && !NEGATION_TOKENS.has(t);
    });
  }

  function meanClaimsF1(claims, reference) {
    if (!claims.length) return 0.0;
    const ref = stripNegTokens(reference);
    let total = 0.0;
    for (const cl of claims) total += multisetF1(stripNegTokens(cl.text), ref);
    return total / claims.length;
  }

  // Per answer: mean contentF1(claim, folk) vs mean contentF1(claim,
  // correction) on negation-stripped tokens; parroting iff folk wins by more
  // than MYTH_MARGIN and no correction cue fires. Claims-free answers
  // (refusals) are never parroting.
  function mythVerdict(answer, folkBelief, correction) {
    const claims = splitClaims(answer);
    if (!claims.length) {
      return { parroting: false, folk_f1: 0.0, correction_f1: 0.0,
               override: null, n_claims: 0, negates_folk_claims: 0 };
    }
    const folkF = meanClaimsF1(claims, folkBelief);
    const corrF = meanClaimsF1(claims, correction);
    const low = String(answer).toLowerCase();
    let override = null;
    for (const cue of MYTH_CUES) { if (low.indexOf(cue) !== -1) { override = 'correction-cue'; break; } }
    const folkTokens = new Set(stripNegTokens(folkBelief));
    let negatesFolk = 0;
    for (const cl of claims) {
      if (hasNegation(cl.text)) {
        const toks = stripNegTokens(cl.text);
        let shared = false;
        for (const t of toks) { if (folkTokens.has(t)) { shared = true; break; } }
        if (shared) negatesFolk++;
      }
    }
    const parroting = (folkF > corrF + MYTH_MARGIN) && override === null;
    return { parroting: parroting, folk_f1: folkF, correction_f1: corrF,
             override: override, n_claims: claims.length,
             negates_folk_claims: negatesFolk };
  }

  return {
    VERSION: VERSION,
    FIXTURES_FORMAT: FIXTURES_FORMAT,
    STOPWORDS: STOPWORDS,
    F1_SUPPORTED: F1_SUPPORTED, F1_MIN: F1_MIN,
    F1_NOT_IN_CORPUS: F1_NOT_IN_CORPUS, GROUNDED_MIN_F1: GROUNDED_MIN_F1,
    REWRITE_MIN_F1: REWRITE_MIN_F1, MYTH_MARGIN: MYTH_MARGIN,
    ABSTAIN_TAU: ABSTAIN_TAU, TOP_K: TOP_K,
    GROUNDED_SENTENCES: GROUNDED_SENTENCES,
    GROUNDED_REFUSAL: GROUNDED_REFUSAL, ABSTAIN_REFUSAL: ABSTAIN_REFUSAL,
    tokenize: tokenize,
    contentWords: contentWords,
    answerPart: answerPart,
    splitSentences: splitSentences,
    splitLong: splitLong,
    stripMarkers: stripMarkers,
    extractNumbers: extractNumbers,
    splitClaims: splitClaims,
    hasNegation: hasNegation,
    multisetF1: multisetF1,
    contentF1: contentF1,
    bestChunkSentence: bestChunkSentence,
    verifyClaim: verifyClaim,
    topkChunks: topkChunks,
    groundedAnswer: groundedAnswer,
    verifiedAnswer: verifiedAnswer,
    measureAnswer: measureAnswer,
    isRefusal: isRefusal,
    stripNegTokens: stripNegTokens,
    mythVerdict: mythVerdict
  };
});
