#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/test_task7_bridge.js — Task 7 acceptance (Node)
   --------------------------------------------------------------------------------
   The full weights-bridge flow, end to end, on the REAL 150-step checkpoint:

     a. EXPORT  : child_process -> python3 colab/05_export.py --ckpt
                  scripts/t5_runA_uninterrupted --out download/p_1/docs/weights
                  (+ --parity-fixtures / --snapshot: the Python twin generates
                  the 10-prompt fixture so this test needs no notebook).
     b. LOAD    : docs/weights.js Node path (loadWebModelFile) parses
                  docs/weights/web_model.bin; manifest.json cross-checked and
                  the byte-layout formula re-derived independently.
     c. REBUILD : docs/model.js Transformer from the config inside the bin,
                  weights copied in BY NAME via model.setParams (both
                  languages gained the helper in this task).
     d. PROMPTS : the SAME 10 fixed prompts (ea-01..ea-10) built here from
                  corpus/merges.json (docs/tokenizer.js) + corpus/eval_answerable
                  exactly like colab/05_export.build_prompts (bos + encode,
                  left-truncate to last ctx, left-pad id 0 to ctx) — and
                  asserted id-for-id against the fixture.
     e. PARITY  : per prompt — max |logit diff| over the first 64 final-
                  position logits (< 1e-2, expect ~1e-6), argmax equal,
                  greedy-20 ids equal (20 re-forwards each).

   Run:  node scripts/test_task7_bridge.js     (from the project root, or
        from download/p_1/scripts — the script finds the roots itself)
   Exit 0 = all green.  ~1 min (the JS twin does 210 real forwards).
   ============================================================================ */
'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

/* ------------------------------------------------------------------ roots */
function findAbove(start, predicate) {
  // walk up from `start`; at each level test the dir itself AND
  // <dir>/download/p_1 (the repo mirror layout used in this workspace).
  let d = path.resolve(start);
  for (let i = 0; i < 5; i++) {
    const cands = [d, path.join(d, 'download', 'p_1')];
    for (const c of cands) {
      try { if (predicate(c)) return c; } catch (e) { /* keep walking */ }
    }
    const up = path.dirname(d);
    if (up === d) break;
    d = up;
  }
  return null;
}
// project root: holds the real Task 5 checkpoints (scripts/t5_runA_uninterrupted)
const PROJ = findAbove(__dirname, (d) =>
  fs.existsSync(path.join(d, 'scripts', 't5_runA_uninterrupted', 'latest.txt')));
// repo mirror: holds colab/ + corpus/ + docs/
const REPO = findAbove(__dirname, (d) =>
  fs.existsSync(path.join(d, 'corpus', 'merges.json')) &&
  fs.existsSync(path.join(d, 'colab', '05_export.py')));
if (!PROJ || !REPO) {
  console.error('cannot locate project/repo roots from ' + __dirname);
  process.exit(2);
}
const CKPT_DIR = path.join(PROJ, 'scripts', 't5_runA_uninterrupted');
const DOCS = path.join(REPO, 'docs');
const BIN = path.join(DOCS, 'weights', 'web_model.bin');
const MANIFEST = path.join(DOCS, 'weights', 'manifest.json');
const FIXTURES = path.join(REPO, 'results', 'parity_outputs.json');
const IDS = ['ea-01', 'ea-02', 'ea-03', 'ea-04', 'ea-05',
             'ea-06', 'ea-07', 'ea-08', 'ea-09', 'ea-10'];

const P1 = require(path.join(DOCS, 'tensor.js'));
const M = require(path.join(DOCS, 'model.js'));
const W = require(path.join(DOCS, 'weights.js'));
const P1BPE = require(path.join(DOCS, 'tokenizer.js'));

let failures = 0;
const ok = (cond, msg) => {
  console.log((cond ? '  PASS ' : '  FAIL ') + msg);
  if (!cond) failures++;
  return cond;
};

/* ------------------------------------------------------- a. python export */
function runExport() {
  console.log('\nTEST A — python3 colab/05_export.py (checkpoint -> browser weights + fixtures)');
  const cmd = path.join(REPO, 'colab', '05_export.py');
  const args = ['--ckpt', CKPT_DIR,
                '--out', path.join(DOCS, 'weights'),
                '--parity-fixtures', FIXTURES,
                '--snapshot', path.join(DOCS, 'parity_outputs_data.js')];
  console.log('  $ python3 ' + path.relative(PROJ, cmd) + ' ' +
              args.map(a => a.startsWith('/') ? path.relative(PROJ, a) : a).join(' '));
  const out = execFileSync('python3', [cmd].concat(args),
                           { cwd: PROJ, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  console.log(out.trim().split('\n').map(l => '  | ' + l).join('\n'));
  ok(fs.existsSync(BIN), 'web_model.bin written');
  ok(fs.existsSync(MANIFEST), 'manifest.json written');
  ok(fs.existsSync(FIXTURES), 'parity_outputs.json (Python twin fixture) written');
  return out;
}

/* --------------------------------------------- b. load bin + manifest math */
function loadAndCheckManifest() {
  console.log('\nTEST B — parse web_model.bin (weights.js Node path) + manifest sanity');
  const parsed = W.loadWebModelFile(BIN);
  ok(parsed.config !== null && typeof parsed.config === 'object',
     'config JSON parsed: ' + JSON.stringify(parsed.config));
  ok(parsed.meta.step === 150, 'meta.step = ' + parsed.meta.step);
  ok(parsed.meta.tokenizer_version === 'p1-bpe-v1:3838',
     'meta.tokenizer_version = ' + parsed.meta.tokenizer_version);
  ok(parsed.tensors.size === M.paramNames(parsed.config).length,
     'tensor count ' + parsed.tensors.size + ' == canonical paramNames() length (wte stored ONCE)');

  const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
  let threw = null;
  try { W.checkManifest(parsed, manifest); } catch (e) { threw = e; }
  ok(!threw, 'weights.js checkManifest: ' + (threw ? threw.message : 'OK'));

  // byte-layout formula re-derived independently of the exporter:
  //   4 + 4 + 4 + len(config) + SUM(4 + len(name) + 4 + 4*ndim + 4*prod(shape))
  let expected = parsed.headerBytes;                    // 4+4+4+len(config)
  let payload = 0, contiguous = true, prevDataEnd = -1;
  const names = [];
  for (let i = 0; i < manifest.tensors.length; i++) {
    const m = manifest.tensors[i];
    const nameLen = Buffer.byteLength(m.name, 'utf8');
    let n = 1;
    for (const d of m.shape) n *= d;
    expected += 4 + nameLen + 4 + 4 * m.shape.length + 4 * n;
    payload += 4 * n;
    // each data block starts after the previous data block + its OWN record
    // prefix (nameLen+name+ndim+dims); block 0 starts after the header
    const prefix = 4 + nameLen + 4 + 4 * m.shape.length;
    const want = (i === 0 ? parsed.headerBytes : prevDataEnd) + prefix;
    if (m.offset !== want) contiguous = false;
    prevDataEnd = m.offset + m.nbytes;
    names.push(m.name);
  }
  ok(contiguous, 'tensor data blocks contiguous + offsets monotonic');
  ok(expected === manifest.total_bytes,
     'formula total = ' + expected + ' == manifest.total_bytes ' + manifest.total_bytes);
  const actualSize = fs.statSync(BIN).size;
  ok(actualSize === manifest.total_bytes && actualSize === parsed.byteLength,
     'file size ' + actualSize + ' == manifest.total_bytes == parsed byteLength');
  ok(manifest.num_params === payload / 4,
     'num_params ' + manifest.num_params + ' == payload/4 ' + (payload / 4) +
     ' (file = payload ' + payload + ' + header ' + parsed.headerBytes + ' + records ' +
     (manifest.total_bytes - payload - parsed.headerBytes) + ')');
  ok(JSON.stringify(names) === JSON.stringify(M.paramNames(parsed.config)),
     'manifest order == canonical paramNames() order');
  console.log('  web_model.bin: ' + actualSize.toLocaleString() + ' bytes · ' +
              manifest.tensors.length + ' tensors · ' + manifest.num_params.toLocaleString() +
              ' params · header ' + parsed.headerBytes + ' B');
  return { parsed: parsed, manifest: manifest };
}

/* --------------------------------------------------------- c. rebuild model */
function rebuildModel(parsed, manifest) {
  console.log('\nTEST C — rebuild Transformer from the bin config + setParams by name');
  const model = new M.Transformer(parsed.config, 0);   // seed irrelevant: overwritten
  const named = {};
  for (const [name, t] of parsed.tensors) named[name] = t.f32;
  model.setParams(named);
  ok(model.numParams() === manifest.num_params,
     'numParams ' + model.numParams() + ' == manifest.num_params');
  let bad = 0;
  for (const [name, t] of parsed.tensors) {
    const p = model.P[name];
    for (let i = 0; i < t.f32.length; i++) {
      if (p.data[i] !== t.f32[i]) { bad++; break; }
    }
    if (JSON.stringify(p.shape) !== JSON.stringify(t.shape)) bad++;
  }
  ok(bad === 0, 'every param holds the parsed f32 values + shapes (28/28 exact)');
  // negative control: setParams must reject a bogus/missing name
  let rejected = false;
  try { model.setParams({ wte: parsed.tensors.get('wte').f32 }); } catch (e) { rejected = true; }
  ok(rejected, 'setParams rejects an incomplete name set (missing 27 params)');
  return model;
}

/* ------------------------------------------------- d. the SAME 10 prompts */
function buildPrompts(tok, entries, ctx) {
  // EXACT mirror of colab/05_export.py build_prompts
  const byId = new Map(entries.map(e => [e.id, e]));
  const out = [];
  for (const pid of IDS) {
    const e = byId.get(pid);
    if (!e) throw new Error('eval_answerable.jsonl missing ' + pid);
    let seq = [tok.bos].concat(tok.encode(e.question));
    if (seq.length > ctx) seq = seq.slice(seq.length - ctx);   // left-truncate
    while (seq.length < ctx) seq.unshift(0);                   // left-pad id 0
    out.push({ id: pid, text: e.question, ids: seq });
  }
  return out;
}

function loadFixtures() {
  console.log('\nTEST D — rebuild the 10 fixed prompts with docs/tokenizer.js (BPE)');
  const merges = JSON.parse(fs.readFileSync(path.join(REPO, 'corpus', 'merges.json'), 'utf8'));
  const tok = new P1BPE(merges);
  const entries = fs.readFileSync(path.join(REPO, 'corpus', 'eval_answerable.jsonl'), 'utf8')
    .split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
  const fx = JSON.parse(fs.readFileSync(FIXTURES, 'utf8'));
  ok(fx.format === 'p1-parity-v1', 'fixture format ' + fx.format);
  const prompts = buildPrompts(tok, entries, fx.config.ctx);
  let idOk = 0;
  for (let i = 0; i < prompts.length; i++) {
    const f = fx.prompts[i];
    if (f.id === prompts[i].id &&
        f.ids.length === prompts[i].ids.length &&
        f.ids.every((v, k) => v === prompts[i].ids[k])) idOk++;
  }
  ok(idOk === IDS.length,
     'node-BPE ids == python-fixture ids for ' + idOk + '/' + IDS.length +
     ' prompts (bos + encode + left-truncate + left-pad, ctx=' + fx.config.ctx + ')');
  return { fx: fx, prompts: prompts };
}

/* ------------------------------------------------------------ e. parity */
function parity(fx, prompts, model) {
  console.log('\nTEST E — parity: JS forward vs Python fixture (bar < 1e-2 per prompt)');
  const V = model.config.vocab_size;
  let worst = 0, worstId = '', argmaxBad = 0, greedyBad = 0;
  for (let i = 0; i < fx.prompts.length; i++) {
    const p = fx.prompts[i];
    const t0 = Date.now();
    const logits = model.forward(prompts[i].ids, 1);      // [T, V], T = ctx
    const off = (prompts[i].ids.length - 1) * V;          // final position
    let maxDiff = 0;
    for (let j = 0; j < p.logits_first.length; j++) {
      const d = Math.abs(logits.data[off + j] - p.logits_first[j]);
      if (d > maxDiff) maxDiff = d;
    }
    // argmax — first-max scan (strict >), mirrors np.argmax
    let best = 0, bv = -Infinity;
    for (let j = 0; j < V; j++) {
      const v = logits.data[off + j];
      if (v > bv) { bv = v; best = j; }
    }
    const argmaxOk = best === p.argmax_top5[0][0];
    if (!argmaxOk) argmaxBad++;
    // greedy 20 — re-forward the windowed sequence each step
    const ctx = model.config.ctx;
    const seq = prompts[i].ids.slice(), g = [];
    for (let s = 0; s < 20; s++) {
      const win = seq.length > ctx ? seq.slice(seq.length - ctx) : seq;
      const lo = model.forward(win, 1);
      const o = (win.length - 1) * V;
      let b2 = 0, v2 = -Infinity;
      for (let j = 0; j < V; j++) {
        const v = lo.data[o + j];
        if (v > v2) { v2 = v; b2 = j; }
      }
      seq.push(b2); g.push(b2);
    }
    const greedyOk = g.length === p.greedy_next20.length &&
                     g.every((v, k) => v === p.greedy_next20[k]);
    if (!greedyOk) greedyBad++;
    if (maxDiff > worst) { worst = maxDiff; worstId = p.id; }
    console.log('  ' + p.id + ' max|Δlogit| ' + maxDiff.toExponential(3) +
                ' · argmax ' + (argmaxOk ? '=' : '≠') + ' · greedy20 ' +
                (greedyOk ? '=' : '≠') + ' · ' + (Date.now() - t0) + ' ms');
  }
  const allOk = worst < 1e-2 && argmaxBad === 0 && greedyBad === 0;
  console.log('\n  BRIDGE PARITY max diff = ' + worst.toExponential(6) +
              ' (worst: ' + worstId + ') · ' + fx.prompts.length + '/' +
              fx.prompts.length + (allOk ? ' PASS' : ' FAIL') +
              ' · argmax mismatches ' + argmaxBad + ' · greedy mismatches ' + greedyBad);
  ok(allOk, 'BRIDGE PARITY max diff = ' + worst.toExponential(6) + ' · 10/10 PASS');
  return allOk;
}

/* ------------------------------------------------------------------ main */
(function main() {
  console.log('p_1 · test_task7_bridge.js — Task 7 acceptance (engine v' + P1.VERSION +
              ', model v' + M.VERSION + ', weights v' + W.VERSION + ')');
  console.log('  project root : ' + PROJ);
  console.log('  repo root    : ' + REPO);
  runExport();
  const lm = loadAndCheckManifest();
  const model = rebuildModel(lm.parsed, lm.manifest);
  const pf = loadFixtures();
  parity(pf.fx, pf.prompts, model);
  console.log('\n==== ' + (failures === 0 ? 'ALL TESTS PASSED' : failures + ' FAILURES PRESENT') + ' ====');
  process.exit(failures ? 1 : 0);
})();
