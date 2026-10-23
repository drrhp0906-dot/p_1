#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/gen_task10_data.js — Task 10 data prep
   ----------------------------------------------------------------------------
   Emits the file:// <script> snapshots the web lab (docs/index.html) needs,
   in the SAME auto-generated-snapshot pattern the repo already uses
   (tokenizer_data.js / model_fixtures_data.js / corpus_docs.js):

     docs/data/eval_mythtraps.js     -> (window|globalThis).P1_TRAPS
                                        from corpus/eval_mythtraps.jsonl (20)
     docs/data/eval_unanswerables.js -> (window|globalThis).P1_UNANSWERABLES
                                        from corpus/eval_unanswerable.jsonl (20)
     docs/results/metrics.js         -> (window|globalThis).P1_METRICS
                                        from results/metrics_stub.json

   Browsers block fetch() on file:// URLs, so the lab loads these snapshots
   first and uses fetch ONLY as the "real data later" upgrade path
   (e.g. results/metrics.json from colab/08_eval.ipynb — same schema, same
   p1-metrics-v1 format; the page prefers fetch when it succeeds).

   Regenerate after any eval-set or metrics rebuild:
     node scripts/gen_task10_data.js
   ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

function readJsonl(p) {
  return fs.readFileSync(p, 'utf8').split('\n').filter(l => l.trim().length)
    .map(l => JSON.parse(l));
}

function wrap(header, name, value) {
  return header + '\n\'use strict\';\n' +
    '(typeof window !== \'undefined\' ? window : globalThis).' + name + ' =\n' +
    JSON.stringify(value, null, 1) + ';\n';
}

function main() {
  const traps = readJsonl(path.join(ROOT, 'corpus', 'eval_mythtraps.jsonl'));
  const unans = readJsonl(path.join(ROOT, 'corpus', 'eval_unanswerable.jsonl'));
  const metrics = JSON.parse(
    fs.readFileSync(path.join(ROOT, 'results', 'metrics_stub.json'), 'utf8'));

  if (traps.length !== 20) throw new Error('expected 20 myth traps, got ' + traps.length);
  if (unans.length !== 20) throw new Error('expected 20 unanswerables, got ' + unans.length);
  for (const t of traps) {
    if (!t.id || !t.question || !t.folk_belief || !t.correction) {
      throw new Error('myth trap missing id/question/folk_belief/correction: ' + t.id);
    }
  }
  for (const u of unans) {
    if (!u.id || !u.question || !u.why_unanswerable || !u.best_response) {
      throw new Error('unanswerable missing id/question/why_unanswerable/best_response: ' + u.id);
    }
  }
  if (metrics.format !== 'p1-metrics-v1') {
    throw new Error('metrics format ' + metrics.format + ' != p1-metrics-v1');
  }

  const outDirData = path.join(ROOT, 'docs', 'data');
  const outDirRes = path.join(ROOT, 'docs', 'results');
  fs.mkdirSync(outDirData, { recursive: true });
  fs.mkdirSync(outDirRes, { recursive: true });

  const hTraps =
    '/* ============================================================================\n' +
    '   p_1 · docs/data/eval_mythtraps.js — GENERATED FILE, do not edit by hand.\n' +
    '   The 20 myth-trap eval questions (id, question, folk_belief, correction,\n' +
    '   sources) as a plain <script> snapshot for docs/index.html (TRAP LAB) —\n' +
    '   the file:// fallback when fetch() is blocked.\n' +
    '   Regenerate: node scripts/gen_task10_data.js\n' +
    '   ============================================================================ */';
  const hUnans =
    '/* ============================================================================\n' +
    '   p_1 · docs/data/eval_unanswerables.js — GENERATED FILE, do not edit by hand.\n' +
    '   The 20 unanswerable eval questions (id, question, why_unanswerable,\n' +
    '   best_response) as a plain <script> snapshot for docs/index.html\n' +
    '   (TRAP LAB) — the file:// fallback when fetch() is blocked.\n' +
    '   Regenerate: node scripts/gen_task10_data.js\n' +
    '   ============================================================================ */';
  const hMetrics =
    '/* ============================================================================\n' +
    '   p_1 · docs/results/metrics.js — GENERATED FILE, do not edit by hand.\n' +
    '   Snapshot of results/metrics_stub.json (p1-metrics-v1, stub:true — the\n' +
    '   demo stub answerer) for docs/index.html (ABOUT tab), file://-friendly.\n' +
    '   A REAL metrics.json (colab/08_eval.ipynb, same schema) is preferred by\n' +
    '   the page via fetch(\'results/metrics.json\') whenever HTTP is available.\n' +
    '   Regenerate: node scripts/gen_task10_data.js\n' +
    '   ============================================================================ */';

  fs.writeFileSync(path.join(outDirData, 'eval_mythtraps.js'),
    wrap(hTraps, 'P1_TRAPS', traps));
  fs.writeFileSync(path.join(outDirData, 'eval_unanswerables.js'),
    wrap(hUnans, 'P1_UNANSWERABLES', unans));
  fs.writeFileSync(path.join(outDirRes, 'metrics.js'),
    wrap(hMetrics, 'P1_METRICS', metrics));

  console.log('gen_task10_data: wrote ' + traps.length + ' traps -> docs/data/eval_mythtraps.js');
  console.log('gen_task10_data: wrote ' + unans.length + ' unanswerables -> docs/data/eval_unanswerables.js');
  console.log('gen_task10_data: wrote metrics (stub:' + !!metrics.stub + ') -> docs/results/metrics.js');
}

main();
