#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/gen_corpus_docs.js — Task 8 data prep
   ----------------------------------------------------------------------------
   Reads corpus/health_qa.jsonl (Task 2 output) and emits docs/data/corpus_docs.js:

     (window|globalThis).P1_CORPUS_DOCS = { n: 4278, pairs: [ {id, question, answer,
     sources, split}, ... ] };

   ALL 4278 entries in FILE ORDER (deterministic — no shuffling, no RNG), full
   records. This is the file:// fallback for docs/retrieve_test.html: browsers
   block fetch() on file: URLs, so the demo page tries
   fetch('../corpus/health_qa.jsonl') first and falls back to this <script>
   snapshot (same pattern as tokenizer_data.js / model_fixtures_data.js /
   corpus_mini.js). Regenerate after any corpus rebuild:

     node scripts/gen_corpus_docs.js
   ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SRC = path.join(ROOT, 'corpus', 'health_qa.jsonl');
const OUT_DIR = path.join(ROOT, 'docs', 'data');
const OUT = path.join(OUT_DIR, 'corpus_docs.js');

const lines = fs.readFileSync(SRC, 'utf8').split('\n').filter(l => l.trim().length);
const pairs = lines.map(l => {
  const r = JSON.parse(l);
  // keep the record schema exactly as authored; page only needs these fields
  return { id: r.id, question: r.question, answer: r.answer, sources: r.sources, split: r.split };
});
if (pairs.length < 4000) {
  throw new Error('expected ~4278 corpus rows, got ' + pairs.length + ' — wrong source file?');
}

// JSON.stringify produces a valid JS string literal for these payloads.
// (U+2028/2029 would be line terminators in JS but not in JSON — escaped here.)
const lit = (s) => JSON.stringify(s)
  .replace(/\u2028/g, '\\u2028')
  .replace(/\u2029/g, '\\u2029');

const body = pairs.map(p => '  {' +
  '"id":' + lit(p.id) + ',' +
  '"question":' + lit(p.question) + ',' +
  '"answer":' + lit(p.answer) + ',' +
  '"sources":' + JSON.stringify(p.sources).replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029') + ',' +
  '"split":' + lit(p.split) + '}').join(',\n');
const date = new Date().toISOString().slice(0, 10);
const js =
`/* ============================================================================
   p_1 · docs/data/corpus_docs.js — GENERATED FILE, do not edit by hand.
   Full health Q&A corpus (${pairs.length} entries, file order) as a plain <script>
   snapshot for docs/retrieve_test.html — the file:// fallback when fetch()
   is blocked. Regenerate: node scripts/gen_corpus_docs.js
   ============================================================================ */
'use strict';
(typeof window !== 'undefined' ? window : globalThis).P1_CORPUS_DOCS = {
  n: ${pairs.length},
  pairs: [
${body}
  ]
};
// source: corpus/health_qa.jsonl (all splits, file order) · generated ${date} by scripts/gen_corpus_docs.js
`;

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(OUT, js);
console.log('gen_corpus_docs: wrote ' + OUT + ' (' + (fs.statSync(OUT).size / 1024 / 1024).toFixed(2) + ' MB)');
console.log('entries: ' + pairs.length);
console.log('--- first pair ---');
console.log(JSON.stringify(pairs[0]).slice(0, 220) + '…');
