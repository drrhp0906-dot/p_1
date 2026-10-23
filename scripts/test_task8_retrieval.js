#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/test_task8_retrieval.js — Task 8 acceptance (Node)
   ----------------------------------------------------------------------------
   Gate: >95% top-3 agreement between the JS twin (docs/retrieve.js) and the
   Python twin (colab/retrieval_py.py --eval -> results/retrieval_top3.json)
   on the 100-question eval (60 answerable + 20 mythtraps + 20 unanswerables),
   for BOTH methods (BM25 + TF-IDF). Expect 100% — same formulas, IEEE-754
   doubles both sides; float noise could only flip exact near-ties.

   Also prints:
     - index stats + timings (JS side; Python timings are printed by the CLI)
     - per-method agreement % + up to 5 mismatches with scores
     - sanity: own-id MRR@3 / hit@1 / hit@3 on the answerable set (the 60 eval
       entries appended as chunks — their gold answers are rewrites of corpus
       facts and the corpus carries no seed provenance, so "own id" needs the
       entry itself in the index; near-duplicate corpus chunks make it a real
       test). Reported honestly, not part of the gate.
     - self-test: 10 sampled exact corpus questions must retrieve their own
       fact at rank 1. NOTE every corpus question text has ~3-6 sibling chunks
       (Task 2 expansion: one pair per phrasing x answer variant), so a sibling
       sharing the identical question text counts as a hit; strict own-id also
       reported.

   Run:  node scripts/test_task8_retrieval.js
   (If results/retrieval_top3.json is missing it runs
    `python3 colab/retrieval_py.py --eval` first — pass --no-py to skip.)
   ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const R = require(path.join(ROOT, 'docs', 'retrieve.js'));

const NO_PY = process.argv.includes('--no-py');
const K = 3;
const GATE = 0.95;
const METHODS = ['bm25', 'tfidf'];

/* ------------------------------------------------------------------ helpers */
function loadJsonl(p) {
  return fs.readFileSync(p, 'utf8').split('\n').filter(l => l.trim().length)
    .map(l => JSON.parse(l));
}
const fmtPct = (x) => (100 * x).toFixed(1) + '%';

/* --------------------------------------------------------------------- data */
const corpus = loadJsonl(path.join(ROOT, 'corpus', 'health_qa.jsonl'));
const evalSets = [
  ['answerable', loadJsonl(path.join(ROOT, 'corpus', 'eval_answerable.jsonl'))],
  ['mythtraps', loadJsonl(path.join(ROOT, 'corpus', 'eval_mythtraps.jsonl'))],
  ['unanswerable', loadJsonl(path.join(ROOT, 'corpus', 'eval_unanswerable.jsonl'))],
];
const questions = [];
for (const [, rows] of evalSets) {
  for (const r of rows) questions.push({ id: r.id, question: r.question });
}
if (questions.length !== 100) {
  console.error('FATAL: expected exactly 100 eval questions, got ' + questions.length);
  process.exit(1);
}

/* --------------------------------------------------------- gate JSON (py) */
const resultsPath = path.join(ROOT, 'results', 'retrieval_top3.json');
if (!fs.existsSync(resultsPath)) {
  if (NO_PY) {
    console.error('FATAL: ' + resultsPath + ' missing. Run: python3 colab/retrieval_py.py --eval');
    process.exit(1);
  }
  console.log('results/retrieval_top3.json missing — generating via the Python twin…');
  execFileSync('python3', [path.join(ROOT, 'colab', 'retrieval_py.py'), '--eval'], { stdio: 'inherit' });
  console.log('');
}
const py = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
const pyByKey = new Map();                    // "method|id" -> {top3, scores}
for (const q of py.questions) pyByKey.set(q.method + '|' + q.id, q);
if (py.format !== 'p1-retrieval-eval-v1') {
  console.error('FATAL: unexpected format ' + py.format);
  process.exit(1);
}

/* -------------------------------------------------------------- JS index */
console.log('p_1 · Task 8 retrieval acceptance — JS twin vs Python twin');
console.log('corpus: ' + corpus.length + ' pairs · eval: 100 questions · k=' + K +
            ' · gate: >' + (GATE * 100) + '% top-3 agreement per method');

const t0 = performance.now();
const index = R.buildIndex(corpus);
const buildMs = performance.now() - t0;
console.log('JS index : N=' + index.N + '  vocab=' + index.vocab + '  avgdl=' +
            index.avgdl.toFixed(3) + '  build=' + buildMs.toFixed(1) + ' ms' +
            '  (python meta: N=' + py.meta.n_chunks + ', vocab=' + py.meta.vocab +
            ', avgdl=' + py.meta.avgdl + ')');

const t1 = performance.now();
const jsTop3 = new Map();                     // "method|id" -> [{chunkId, score}...]
for (const method of METHODS) {
  for (const q of questions) {
    jsTop3.set(method + '|' + q.id,
      index.search(q.question, { k: K, method }).map(r => ({ chunkId: r.chunkId, score: r.score })));
  }
}
const perQueryMs = (performance.now() - t1) / (questions.length * METHODS.length);

/* ------------------------------------------------------- agreement (gate) */
let overallPass = true;
for (const method of METHODS) {
  let listMatch = 0, slotMatch = 0, slots = 0;
  const mismatches = [];
  for (const q of questions) {
    const key = method + '|' + q.id;
    const a = jsTop3.get(key), b = pyByKey.get(key);
    if (!b) { console.error('FATAL: python JSON missing ' + key); process.exit(1); }
    let same = true;
    for (let s = 0; s < K; s++) {
      slots++;
      const ok = a[s] && b.top3[s] === a[s].chunkId;
      if (ok) slotMatch++; else same = false;
    }
    if (same) listMatch++;
    else if (mismatches.length < 5) {
      mismatches.push({ id: q.id, js: a.map(r => r.chunkId), py: b.top3,
        jsScores: a.map(r => +r.score.toFixed(6)), pyScores: b.scores.map(v => +v.toFixed(6)) });
    }
  }
  const agree = listMatch / questions.length;
  const pass = agree > GATE;
  if (!pass) overallPass = false;
  console.log('');
  console.log('[' + method + '] top-3 full-list agreement: ' + listMatch + '/' + questions.length +
              ' = ' + fmtPct(agree) + '  · per-slot ' + slotMatch + '/' + slots +
              ' = ' + fmtPct(slotMatch / slots) + '  ' + (pass ? 'PASS' : 'FAIL'));
  for (const m of mismatches) {
    console.log('  mismatch ' + m.id + '\n    js: ' + m.js.join(', ') + '  ' + JSON.stringify(m.jsScores) +
                '\n    py: ' + m.py.join(', ') + '  ' + JSON.stringify(m.pyScores));
  }
}

/* ------------------------------------------------- sanity: own-id metric */
console.log('');
console.log('sanity — own-id retrieval on the answerable set (60 eval entries appended as chunks):');
for (const method of METHODS) {
  const idx2 = R.buildIndex(corpus.concat(evalSets[0][1].map(e => ({
    id: e.id, question: e.question, answer: e.answer, sources: e.sources || [] }))));
  let rr = 0, h1 = 0, h3 = 0;
  for (const e of evalSets[0][1]) {
    const top3 = idx2.search(e.question, { k: 3, method });
    const hit = top3.find(r => r.chunkId === e.id);
    if (hit) { rr += 1 / hit.rank; h3++; if (hit.rank === 1) h1++; }
  }
  const n = evalSets[0][1].length;
  console.log('  ' + method.padEnd(5) + ' MRR@3 ' + (rr / n).toFixed(3) +
              ' · hit@1 ' + fmtPct(h1 / n) + ' · hit@3 ' + fmtPct(h3 / n) +
              '  (own gold chunk vs ' + index.N + ' near-dup corpus chunks — honest signal, not the gate)');
}

/* --------------------------------------------- sanity: exact-question test */
console.log('');
console.log('self-test — exact corpus question must retrieve its own fact at rank 1 (10 sampled):');
{
  const N = corpus.length;
  let hits = 0, strict = 0;
  const fails = [];
  for (let i = 0; i < 10; i++) {
    const ci = Math.round(i * (N - 1) / 9);
    const entry = corpus[ci];
    const top = index.search(entry.question, { k: 1, method: 'bm25' })[0];
    if (top.chunkId === entry.id) { hits++; strict++; }
    else if (corpus[top.idx].question === entry.question) hits++;   // sibling: same question text
    else fails.push(entry.id + '->' + top.chunkId);
  }
  console.log('  hits ' + hits + '/10 (strict own-id ' + strict + '/10; siblings share the question text)' +
              (fails.length ? '  FAILS: ' + fails.join(', ') : ''));
  if (hits < 10) overallPass = false;
}

/* ---------------------------------------------------------------- verdict */
console.log('');
console.log('timings: JS index build ' + buildMs.toFixed(1) + ' ms · JS search ' +
            perQueryMs.toFixed(2) + ' ms/query avg (' + questions.length * METHODS.length + ' searches)' +
            ' — python side printed by retrieval_py.py --eval');
console.log('');
if (overallPass) {
  console.log('PASS — top-3 agreement > ' + (GATE * 100) + '% on both methods; the twins agree. Tick the box.');
  process.exit(0);
} else {
  console.log('FAIL — agreement gate not met. Do not tick the box; paste this log into the session.');
  process.exit(1);
}
