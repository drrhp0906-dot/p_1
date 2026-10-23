#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/test_task6_train.js — Node mirror of docs/train_demo.html
   ----------------------------------------------------------------------------
   The page's training loop, headless: build vocab from corpus_mini, train
   Config A for 300 steps (lr 1e-3, B=8, ctx=128 to keep this quick — the
   PAGE uses the full ctx=256, same code path, different window size),
   print the loss curve, then sample 2 generations at temp 0.8.

   Asserts: losses finite · last-50 avg < first-50 avg (falling curve) ·
   generations non-empty + ASCII-ish. Prints "TASK6 SMOKE PASS" + stats.
   Run: node scripts/test_task6_train.js
   ============================================================================ */
'use strict';
require('../docs/corpus_mini.js');                       // -> globalThis.P1_CORPUS_MINI
const P1T = require('../docs/train.js');                 // -> { buildVocab, Trainer, AdamW }
const P1 = require('../docs/tensor.js');                 // for VERSION reporting

const CORPUS = globalThis.P1_CORPUS_MINI;
if (!CORPUS || !Array.isArray(CORPUS.pairs)) {
  console.error('FAIL: P1_CORPUS_MINI missing — run node scripts/gen_corpus_mini.js first');
  process.exit(1);
}

console.log('tensor.js v' + P1.VERSION + ' · train.js v' + P1T.VERSION);
console.log('corpus_mini: n=' + CORPUS.n + ' pairs=' + CORPUS.pairs.length);

/* ---- vocab (same call the page makes) ---- */
const text = CORPUS.pairs.join('\n\n') + '\n\n';
const vocab = P1T.buildVocab(text);
const V = vocab.itos.length;
console.log('vocab size: ' + V + ' chars · ln(vocab) baseline ≈ ' + Math.log(V).toFixed(3));

/* ---- trainer: test uses ctx=128 for speed; the page uses ctx=256 ---- */
const trainer = new P1T.Trainer({
  pairs: CORPUS.pairs,
  lr: 1e-3, ctx: 128, batchSize: 8, seed: 42, warmupSteps: 200, clipNorm: 1.0
});
console.log('model params: ' + trainer.numParams +
  ' · config: ' + JSON.stringify(trainer.model.config));

const STEPS = parseInt(process.env.P1_SMOKE_STEPS || "300", 10);
const losses = [];
let tFirst = 0, tAll = 0;
const t0 = Date.now();
for (let i = 1; i <= STEPS; i++) {
  const r = trainer.step();
  losses.push(r.loss);
  tAll += r.stepMs;
  if (i === 1) tFirst = r.stepMs;
  if (i % 50 === 0 || i === 1) {
    console.log('  step ' + String(i).padStart(4) + ' · loss ' + r.loss.toFixed(4) +
      ' · lr ' + r.lr.toExponential(2) + ' · tok/s ' + Math.round(r.tokensPerSec));
  }
}
const wall = (Date.now() - t0) / 1000;

/* ---- curve checks ---- */
const avg = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const first50 = avg(losses.slice(0, 50));
const last50 = avg(losses.slice(-50));
const minLoss = Math.min.apply(null, losses);
const allFinite = losses.every(Number.isFinite);

console.log('--- curve ---');
console.log('first-50 avg : ' + first50.toFixed(4));
console.log('last-50  avg : ' + last50.toFixed(4));
console.log('min loss     : ' + minLoss.toFixed(4) + ' · drop ' + (first50 - last50).toFixed(4));

/* ---- benchmark the PAGE shapes (B=8, ctx=256) for the rAF budget ---- */
const pageTrainer = new P1T.Trainer({
  pairs: CORPUS.pairs, lr: 1e-3, ctx: 256, batchSize: 8, seed: 42
});
let tPage = Infinity, tpsPage = 0;
for (let i = 0; i < 6; i++) { const r = pageTrainer.step(); if (i >= 1) tPage = Math.min(tPage, r.stepMs); tpsPage = r.tokensPerSec; }
console.log('page-shape step (B=8, ctx=256): ~' + tPage.toFixed(1) + ' ms/step · ' +
  Math.round(tpsPage) + ' tok/s');

/* ---- generation ---- */
console.log('--- generation (temp 0.8) ---');
const g1 = trainer.sample('Q:', 220, 0.8);
const g2 = trainer.sample('Q: How much water', 220, 0.8);
console.log('[1] prompt "Q:"');
console.log(g1.replace(/^\n+/, ''));
console.log('[2] prompt "Q: How much water"');
console.log(g2.replace(/^\n+/, ''));

/* ---- asserts ---- */
const fails = [];
if (!allFinite) fails.push('non-finite loss in curve');
if (!(last50 < first50)) fails.push('loss did not fall (last-50 >= first-50)');
for (const [name, g] of [['gen1', g1], ['gen2', g2]]) {
  if (!g || g.trim().length < 10) fails.push(name + ' empty/too short');
  // ASCII-ish: every decoded char must be in the (ASCII) vocab
  for (const ch of g) {
    if (!vocab.stoi.has(ch)) { fails.push(name + ' has out-of-vocab char ' + JSON.stringify(ch)); break; }
    if (ch.charCodeAt(0) > 126) { fails.push(name + ' non-ASCII char ' + JSON.stringify(ch)); break; }
  }
}
if (minLoss > 3.0) {
  console.log('NOTE: min loss ' + minLoss.toFixed(3) + ' not below 3.0 yet in 300 steps ' +
    '(curve is falling; the page keeps going — spec target "toward < 3.0")');
}

if (fails.length) {
  console.error('TASK6 SMOKE FAIL:\n  - ' + fails.join('\n  - '));
  process.exit(1);
}
console.log('--- stats ---');
console.log('300 steps in ' + wall.toFixed(1) + 's wall · ' + (tAll / STEPS).toFixed(1) +
  ' ms/step avg (ctx=128) · ' + trainer.tokensTotal + ' tokens processed');
console.log('TASK6 SMOKE PASS');
