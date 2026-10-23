#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/gen_corpus_mini.js — Task 6 data prep
   ----------------------------------------------------------------------------
   Reads corpus/health_qa.jsonl (Task 2 output) and emits docs/corpus_mini.js:
   the FIRST 500 train-split entries in FILE ORDER (deterministic — no
   shuffling, no RNG), each flattened to one string "Q: <question>\nA: <answer>".

   Output shape (consumed by docs/train.js + docs/train_demo.html):

     window.P1_CORPUS_MINI = { n: 500, pairs: [ "Q: ...\nA: ...", ... ] };

   The file is a plain <script> (no fetch) so the demo page works from
   file:// — the one deviation from a literal `window.P1_CORPUS_MINI = ...`
   is a tiny guard so Node can also require() it for the Task 6 smoke test
   (in a browser `window` IS the global object, so behavior is identical).

   Run:  node scripts/gen_corpus_mini.js
   ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SRC = path.join(ROOT, 'corpus', 'health_qa.jsonl');
const OUT = path.join(ROOT, 'docs', 'corpus_mini.js');
const N = 500;

const lines = fs.readFileSync(SRC, 'utf8').split('\n').filter(l => l.trim().length);
const pairs = [];
let trainSeen = 0;
for (const line of lines) {
  const rec = JSON.parse(line);
  if (rec.split !== 'train') continue;
  trainSeen++;
  if (pairs.length < N) {
    pairs.push('Q: ' + rec.question + '\nA: ' + rec.answer);
  }
}
if (pairs.length < N) {
  throw new Error('only ' + trainSeen + ' train rows — need ' + N);
}

// JSON.stringify produces a valid JS string literal for these payloads.
// (U+2028/2029 would be line terminators in JS but not in JSON — escaped here.)
const lit = (s) => JSON.stringify(s)
  .replace(/\u2028/g, '\\u2028')
  .replace(/\u2029/g, '\\u2029');

const body = pairs.map(lit).join(',\n  ');
const date = new Date().toISOString().slice(0, 10);
const js =
`/* ============================================================================
   p_1 · docs/corpus_mini.js — GENERATED FILE, do not edit by hand.
   The first ${N} train-split Q&A pairs of corpus/health_qa.jsonl (file order),
   each flattened to "Q: <question>\\nA: <answer>". Char-level training food
   for docs/train.js / docs/train_demo.html (Task 6). Plain <script> include,
   no fetch — safe from file://.
   ============================================================================ */
'use strict';
(typeof window !== 'undefined' ? window : globalThis).P1_CORPUS_MINI = {
  n: ${N},
  pairs: [
  ${body}
  ]
};
// source: corpus/health_qa.jsonl (train split, first ${N} in file order) · generated ${date} by scripts/gen_corpus_mini.js
`;

fs.writeFileSync(OUT, js);
console.log('gen_corpus_mini: wrote ' + OUT + ' (' + (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB)');
console.log('train rows in source: ' + trainSeen + ' · emitted pairs: ' + pairs.length);
console.log('--- first 2 pairs ---');
console.log(pairs[0]);
console.log('~~~');
console.log(pairs[1]);
