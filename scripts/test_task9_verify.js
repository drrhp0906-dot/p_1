#!/usr/bin/env node
/* ============================================================================
 * p_1 · scripts/test_task9_verify.js — Task 9 Node acceptance (verify.js twin)
 * ----------------------------------------------------------------------------
 * Requires docs/verify.js (download/p_1/docs/) and runs ALL cases from
 * docs/verify_fixtures.json (format p1-verify-fixtures-v1, expected outputs
 * computed by the Python reference colab/eval_lib.py via
 * scripts/test_task9_eval.py):
 *
 *   - 40 verifyClaim cases : verdict equality + |score diff| <= 1e-3
 *                            (fixture scores are rounded to 4dp) +
 *                            numbersOK / negConflict / bestChunkId /
 *                            bestSentence equality
 *   -  8 refusal strings   : isRefusal equality
 *   -  6 myth comparisons  : parroting / override / n_claims /
 *                            negates_folk_claims equality + both F1s <= 1e-3
 *
 * Python is the reference. If a case fails, docs/verify.js is fixed to match.
 * Prints "VERIFY PARITY N/N PASS" + worst score diff. Exit 0 = green.
 * ==========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
// Works both from the workspace (…/my-project/scripts) and from the repo's
// own scripts/ dir (Python is the reference; fixtures computed by it).
const REPO = fs.existsSync(path.join(ROOT, 'download', 'p_1', 'docs'))
  ? path.join(ROOT, 'download', 'p_1')
  : ROOT;
const V = require(path.join(REPO, 'docs', 'verify.js'));
const fixtures = JSON.parse(
  fs.readFileSync(path.join(REPO, 'docs', 'verify_fixtures.json'), 'utf8'));

let failed = 0, total = 0, worstDiff = 0.0, worstCase = '(none)';
const failures = [];

function fail(id, msg) {
  failed++;
  failures.push(id + ': ' + msg);
  console.log('  [FAIL] ' + id + ' — ' + msg);
}

function check(id, ok, msg) {
  total++;
  if (!ok) fail(id, msg);
}

// ---------------------------------------------------------------- setup ---
if (fixtures.format !== 'p1-verify-fixtures-v1') {
  console.error('bad fixtures format: %s', fixtures.format);
  process.exit(1);
}
const nCases = fixtures.cases.length;
const nRef = fixtures.refusals.length;
const nMyth = fixtures.myths.length;
console.log('=== Task 9 Node acceptance — verify.js v%s ===', V.VERSION);
console.log('fixtures: %s — %d verifyClaim + %d refusal + %d myth cases',
            fixtures.format, nCases, nRef, nMyth);
check('fixtures-counts', nCases === 40 && nRef === 8 && nMyth === 6,
      'expected 40/8/6, got ' + nCases + '/' + nRef + '/' + nMyth);

// ---------------------------------------------------- verifyClaim cases ---
console.log('\n--- verifyClaim cases (%d) ---', nCases);
for (const c of fixtures.cases) {
  const js = V.verifyClaim(c.claim, c.chunks);
  const ex = c.expected;
  const scoreDiff = Math.abs(js.score - ex.score);
  if (scoreDiff > worstDiff) { worstDiff = scoreDiff; worstCase = c.id; }
  check(c.id + '/verdict', js.verdict === ex.verdict,
        'js=' + js.verdict + ' py=' + ex.verdict + ' (score ' +
        js.score.toFixed(4) + ' vs ' + ex.score + ')');
  check(c.id + '/score', scoreDiff <= 1e-3,
        '|diff|=' + scoreDiff + ' js=' + js.score + ' py=' + ex.score);
  check(c.id + '/numbersOK', js.numbersOK === ex.numbersOK,
        'js=' + js.numbersOK + ' py=' + ex.numbersOK);
  check(c.id + '/negConflict', js.negConflict === ex.negConflict,
        'js=' + js.negConflict + ' py=' + ex.negConflict);
  check(c.id + '/bestChunkId', js.bestChunkId === ex.bestChunkId,
        'js=' + js.bestChunkId + ' py=' + ex.bestChunkId);
  check(c.id + '/bestSentence', js.bestSentence === ex.bestSentence,
        'js=' + JSON.stringify(js.bestSentence) + ' py=' +
        JSON.stringify(ex.bestSentence));
}

// -------------------------------------------------------- refusal cases ---
console.log('\n--- refusal strings (%d) ---', nRef);
for (const r of fixtures.refusals) {
  const js = V.isRefusal(r.text);
  check(r.id, js === r.expected,
        'js=' + js + ' py=' + r.expected + ' text=' + JSON.stringify(r.text));
}

// ----------------------------------------------------------- myth cases ---
console.log('\n--- myth comparisons (%d) ---', nMyth);
for (const m of fixtures.myths) {
  const js = V.mythVerdict(m.answer, m.folk_belief, m.correction);
  const ex = m.expected;
  check(m.id + '/parroting', js.parroting === ex.parroting,
        'js=' + js.parroting + ' py=' + ex.parroting);
  const folkDiff = Math.abs(js.folk_f1 - ex.folk_f1);
  const corrDiff = Math.abs(js.correction_f1 - ex.correction_f1);
  check(m.id + '/folk_f1', folkDiff <= 1e-3,
        '|diff|=' + folkDiff + ' js=' + js.folk_f1 + ' py=' + ex.folk_f1);
  check(m.id + '/correction_f1', corrDiff <= 1e-3,
        '|diff|=' + corrDiff + ' js=' + js.correction_f1 + ' py=' +
        ex.correction_f1);
  check(m.id + '/override', js.override === ex.override,
        'js=' + js.override + ' py=' + ex.override);
  check(m.id + '/n_claims', js.n_claims === ex.n_claims,
        'js=' + js.n_claims + ' py=' + ex.n_claims);
  check(m.id + '/negates_folk_claims',
        js.negates_folk_claims === ex.negates_folk_claims,
        'js=' + js.negates_folk_claims + ' py=' + ex.negates_folk_claims);
  if (Math.max(folkDiff, corrDiff) > worstDiff) {
    worstDiff = Math.max(folkDiff, corrDiff);
    worstCase = m.id;
  }
}

// --------------------------------------------------------------- report ---
console.log('\n%s', ('-').repeat(72));
if (failed > 0) {
  console.log('VERIFY PARITY %d/%d PASS (%d FAILED)', total - failed, total,
              failed);
  for (const f of failures) console.log('  - ' + f);
  process.exit(1);
}
console.log('VERIFY PARITY %d/%d PASS — worst |score diff| = %s (case %s)',
            total, total, worstDiff.toExponential(3), worstCase);
process.exit(0);
