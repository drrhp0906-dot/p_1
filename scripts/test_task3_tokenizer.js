#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/test_task3_tokenizer.js — Task 3 acceptance gate (Node mirror)
   ----------------------------------------------------------------------------
   Loads corpus/merges.json + docs/tokenizer_fixtures.json from disk and runs
   the JavaScript twin (docs/tokenizer.js) against the Python twin's answers:

     Test 1  PARITY     : JS encode(text) === fixture ids, all 200 cases
     Test 2  ROUNDTRIP  : decode(encode(text)) === text, all 200 cases
     Test 3  TORTURE    : roundtrip on 20 extra nasty unicode strings

   Prints "PARITY 200/200 PASS" on success; failing case details otherwise.
   Exit code 0 = pass, 1 = fail. This is the gate to tick Task 3's box.
   ============================================================================ */
'use strict';

const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const P1BPE = require(path.join(ROOT, 'docs', 'tokenizer.js'));

/* ---- 20 extra torture strings (valid Unicode only — no lone surrogates,
        which are not encodable UTF-8) ------------------------------------ */
const EXTRA_TORTURE = [
  '👨‍👩‍👧‍👦 👍🏽 👌🏿 — ZWJ + skin tones',
  '🇺🇳🇯🇵🇰🇪 regional indicator pairs',
  '健康に関する質問：睡眠は何時間？',
  'В здоровье важно sleep и питание.',
  'صحة ونوم وتغذية سليمة',
  'e\u0301\u0327 combining acute + cedilla',
  'ÅÄÖ åäö ß ÆØœ œuf à côté',
  'Ω≈ç√∫˜µ≤≥÷ math-ish symbols',
  '𓀀𓁐𓃭 Egyptian hieroglyphs (astral)',
  '𝔥𝔢𝔞𝔩𝔱𝔥 𝐦𝐞𝐭𝐫𝐢𝐜𝐬 double-struck',
  'café\nnaïve\nline\nbreaks',
  '  spaces\ttabs\nnewlines\r\nand more  ',
  'ththththth ththth the the the thethe',
  'a',
  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  'question? answer! It is 100% true — see §4.2 (c) 2024.',
  'ᚠᚢᚦᚨᚱᚲ runes ⵎⵣⵗ tifinagh ᜀᜌ᜔ baybayin',
  'กล้วยทุกวัน Thai combining marks मरीज़ Devanagari',
  '‮overridden bidi‬ normal again',
  '\u0000NUL\u0000 inside \uFFFD replaced',
];

/* ---- helpers ------------------------------------------------------------ */
let failures = 0;
function fail(msg) { failures++; console.log('  ✗ ' + msg); }
function sameArr(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/* ---- main --------------------------------------------------------------- */
const t0 = Date.now();
const mergesObj = JSON.parse(fs.readFileSync(path.join(ROOT, 'corpus', 'merges.json'), 'utf8'));
const fixtures = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'tokenizer_fixtures.json'), 'utf8'));
const tok = new P1BPE(mergesObj);

console.log('p_1 · task 3 acceptance gate — BPE tokenizer twins');
console.log(`  merges.json : format=${mergesObj.format} vocab_size=${tok.vocabSize} merges=${tok.numMerges} bos=${tok.bos} eos=${tok.eos}`);
console.log(`  fixtures    : ${fixtures.format} · ${fixtures.cases.length} cases · "${fixtures.note}"`);

/* contract constants */
if (tok.vocabSize !== 4096) fail(`vocabSize ${tok.vocabSize} !== 4096`);
if (tok.bos !== 4094) fail(`bos ${tok.bos} !== 4094`);
if (tok.eos !== 4095) fail(`eos ${tok.eos} !== 4095`);
if (tok.numMerges !== 3838) fail(`numMerges ${tok.numMerges} !== 3838 (ids 256..4093)`);

/* TEST 1 — parity against Python-generated fixtures */
let parity = 0;
for (const c of fixtures.cases) {
  const got = tok.encode(c.text);
  if (sameArr(got, c.ids)) { parity++; continue; }
  let d = 0;
  while (d < got.length && d < c.ids.length && got[d] === c.ids[d]) d++;
  fail(`case ${c.id}: len ${got.length} vs ${c.ids.length}, first diff at ${d} ` +
       `(got ${JSON.stringify(got.slice(Math.max(0, d - 2), d + 3))} ` +
       `want ${JSON.stringify(c.ids.slice(Math.max(0, d - 2), d + 3))}) ` +
       `text=${JSON.stringify(c.text.slice(0, 60))}`);
}
const parityLine = `PARITY ${parity}/${fixtures.cases.length} ` + (parity === fixtures.cases.length ? 'PASS' : 'FAIL');
console.log('  ' + parityLine);

/* TEST 2 — roundtrip on all fixture cases */
let rt = 0;
for (const c of fixtures.cases) {
  const back = tok.decode(tok.encode(c.text));
  if (back === c.text) rt++;
  else fail(`roundtrip case ${c.id} MISMATCH: ${JSON.stringify(c.text.slice(0, 50))} -> ${JSON.stringify(back.slice(0, 50))}`);
}
console.log(`  ROUNDTRIP ${rt}/${fixtures.cases.length} ` + (rt === fixtures.cases.length ? 'PASS' : 'FAIL'));

/* specials hygiene: encode never emits specials, and specials decode away */
let hyg = true;
for (const c of fixtures.cases) {
  for (const id of tok.encode(c.text)) {
    if (id >= 256 + tok.numMerges) { hyg = false; fail(`case ${c.id}: emitted id ${id} >= ${256 + tok.numMerges}`); break; }
  }
}
const withSpecials = [tok.bos, ...tok.encode('hello'), tok.eos];
if (tok.decode(withSpecials) !== 'hello') { hyg = false; fail('specials do not decode to empty'); }
console.log('  SPECIALS HYGIENE ' + (hyg ? 'PASS' : 'FAIL') + ` (bos ${tok.bos} / eos ${tok.eos} never emitted; decode to nothing)`);

/* TEST 3 — extra torture roundtrip */
let xt = 0;
for (const s of EXTRA_TORTURE) {
  const back = tok.decode(tok.encode(s));
  if (back === s) xt++;
  else fail(`torture MISMATCH: ${JSON.stringify(s.slice(0, 50))} -> ${JSON.stringify(back.slice(0, 50))}`);
}
console.log(`  TORTURE ROUNDTRIP ${xt}/${EXTRA_TORTURE.length} ` + (xt === EXTRA_TORTURE.length ? 'PASS' : 'FAIL'));

/* a tiny encode demo for the log */
const demo = tok.encode('How much sleep do adults need?');
console.log(`  demo: 'How much sleep do adults need?' -> ${demo.length} ids: ${demo.slice(0, 12).join(',')}${demo.length > 12 ? ',…' : ''}`);

const dt = Date.now() - t0;
const okAll = parity === fixtures.cases.length && rt === fixtures.cases.length && xt === EXTRA_TORTURE.length && hyg && failures === 0;
console.log(okAll
  ? `\n✓ ALL GATES GREEN in ${dt} ms — ${parityLine} · tokenizer twins agree. Tick the box.`
  : `\n✗ FAILURES PRESENT (${failures}) — paste this log into the session. Do not tick the box.`);
process.exit(okAll ? 0 : 1);
