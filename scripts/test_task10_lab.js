#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/test_task10_lab.js — Node smoke for the web lab (Task 10)
   ----------------------------------------------------------------------------
   docs/app.js is browser-first, but its PURE logic is exported (window.P1_LAB
   / module.exports). This script exercises those functions in Node with the
   REAL libs (verify.js / retrieve.js) and the REAL snapshots
   (corpus_docs.js / eval_mythtraps.js / eval_unanswerables.js / metrics.js).
   The full UI check (tabs / streaming / chips / screenshots) is the browser
   validation in the Task 10 worklog — this is the no-browser safety net.

   Run from repo root:  node scripts/test_task10_lab.js
   ============================================================================ */
'use strict';
const path = require('path');
const DOCS = path.join(__dirname, '..', 'docs');

const P1 = require(path.join(DOCS, 'tensor.js'));
P1.verify = require(path.join(DOCS, 'verify.js'));
P1.retrieve = require(path.join(DOCS, 'retrieve.js'));
const LAB = require(path.join(DOCS, 'app.js'));

// <script> snapshots attach to globalThis in Node
require(path.join(DOCS, 'data', 'corpus_docs.js'));
require(path.join(DOCS, 'data', 'eval_mythtraps.js'));
require(path.join(DOCS, 'data', 'eval_unanswerables.js'));
require(path.join(DOCS, 'results', 'metrics.js'));

let pass = 0, fail = 0;
function ok(cond, msg) {
  if (cond) { pass++; console.log('  PASS ' + msg); }
  else { fail++; console.log('  FAIL ' + msg); }
}

console.log('== Task 10 lab: pure-function smoke ==');

/* 1 · prompt + stop conditions ------------------------------------------ */
ok(LAB.buildPrompt('How do I treat a minor burn?') ===
   'Q: How do I treat a minor burn?\nA:', 'buildPrompt contract Q: <q>\\nA:');
ok(LAB.hitsNewlineQ('run cool water\nQ: next') === true, 'hitsNewlineQ sees a new question');
ok(LAB.hitsNewlineQ('run cool water for 10 minutes') === false, 'hitsNewlineQ ignores plain text');
ok(LAB.truncate('abcdefgh', 5) === 'abcd…', 'truncate cuts to n with ellipsis');
ok(LAB.truncate('abc', 5) === 'abc', 'truncate keeps short strings');
ok(LAB.fmtInt(4278) === '4,278', 'fmtInt thousands separator');
ok(LAB.fmtPct(0.135) === '13.5%' && LAB.fmtPct(null) === '–', 'fmtPct');

/* 2 · claim chips against the real index --------------------------------- */
console.log('building the real retrieval index over ' +
  globalThis.P1_CORPUS_DOCS.pairs.length + ' pairs…');
const idx = P1.retrieve.buildIndex(globalThis.P1_CORPUS_DOCS.pairs);
const q = 'How do I treat a minor burn?';
const topk = P1.verify.topkChunks(idx, q, 3);
ok(topk.length === 3 && topk[0].score >= topk[1].score, 'top-3 retrieval works, ranked');

const grounded = P1.verify.groundedAnswer(q, topk);
const gchips = LAB.claimChips(grounded.answer, topk, P1.verify);
ok(grounded.refused === false && grounded.citations.length >= 1,
   'groundedAnswer composes a cited extractive answer');
ok(gchips.claims.length >= 1 &&
   gchips.claims.every((c) => c.verdict === 'supported'),
   'extractive sentences verify as supported (' +
   gchips.claims.map((c) => c.verdict).join(',') + ')');

const nonsense = LAB.claimChips(
  'Cover the burn with butter and ice. Then peel the skin off after 3 hours.', topk, P1.verify);
ok(nonsense.claims.length >= 2 &&
   nonsense.claims.every((c) => c.verdict !== 'supported'),
   'invented advice is flagged unsupported/not-in-corpus');

const refu = LAB.claimChips(P1.verify.GROUNDED_REFUSAL, topk, P1.verify);
ok(refu.refused === true && refu.claims.length === 0, 'refusal string short-circuits chips');

/* 3 · summary aggregation ------------------------------------------------ */
const fake = [
  { item: { kind: 'myth' }, mythRaw: { parroting: true }, mythGrd: { parroting: false } },
  { item: { kind: 'myth' }, mythRaw: { parroting: false }, mythGrd: { parroting: true } },
  { item: { kind: 'myth' }, error: 'x' },
  { item: { kind: 'unanswerable' }, refusedRaw: false, refusedGrd: true },
  { item: { kind: 'unanswerable' }, refusedRaw: false, refusedGrd: true }
];
const s = LAB.aggregateSummary(fake);
ok(s.myths === 2 && s.parrotRaw === 1 && s.parrotGrd === 1, 'aggregateSummary myth counters');
ok(s.unans === 2 && s.refusedGrd === 2 && s.refusedRaw === 0, 'aggregateSummary refusal counters');
ok(s.errors === 1, 'aggregateSummary counts error cards');

/* 4 · metrics table shaping ---------------------------------------------- */
const rows = LAB.metricsRows(globalThis.P1_METRICS);
ok(rows.length === 3, 'metricsRows: 3 eval sets (got ' + rows.length + ')');
ok(rows.every((r) => r.cells.length === 4), 'metricsRows: 4 modes per set');
const rawAns = rows[0].cells[0];
ok(isFinite(rawAns.hallucination_rate) && isFinite(rawAns.refusal_rate),
   'metricsRows carries hallucination/refusal numbers');
const mythRow = rows.find((r) => r.set === 'mythtraps');
ok(mythRow && mythRow.cells.every((c) => typeof c.parroting_rate === 'number'),
   'metricsRows: mythtraps row carries parroting_rate');
ok(globalThis.P1_METRICS.stub === true, 'snapshot metrics are flagged stub:true');
ok(globalThis.P1_TRAPS.length === 20 && globalThis.P1_UNANSWERABLES.length === 20,
   'snapshots carry 20 traps + 20 unanswerables');

/* 5 · trap data shape ----------------------------------------------------- */
ok(globalThis.P1_TRAPS.every((t) => t.id && t.question && t.folk_belief && t.correction),
   'all traps have id/question/folk_belief/correction');
ok(globalThis.P1_UNANSWERABLES.every((u) => u.id && u.question),
   'all unanswerables have id/question');

console.log('\n== LAB SMOKE: ' + pass + ' PASS / ' + fail + ' FAIL ==');
process.exit(fail ? 1 : 0);
