/* ============================================================================
   p_1 · docs/app.js — the web lab (Task 10)
   ----------------------------------------------------------------------------
   Single-page lab over the handmade stack, zero dependencies, zero CDNs:
     tensor.js + model.js + weights.js   → the brain (web_model.bin via
                                           P1.weights; file:// falls back to
                                           the base64 snapshot, parity-page
                                           pattern)
     tokenizer.js + tokenizer_data.js    → byte-BPE (window.P1_TOKENIZER_DATA)
     retrieve.js + data/corpus_docs.js   → BM25 over 4,278 chunks
                                           (window.P1_CORPUS_DOCS)
     verify.js                           → claim split / verify / refusal /
                                           myth comparator
     data/eval_mythtraps.js (P1_TRAPS) +
     data/eval_unanswerables.js (P1_UNANSWERABLES)          → TRAP LAB sets
     results/metrics.js (P1_METRICS)     → ABOUT tab numbers (a REAL
                                           results/metrics.json is preferred
                                           via fetch whenever HTTP works)

   Design notes
   - Tab router is HASH-based (#playground/#training/#traps/#about) so a
     refresh keeps the tab; tabs init lazily (the training iframe only loads
     when its tab is first opened, the corpus index builds on first
     GROUNDED/TRAP use, metrics render on first ABOUT open).
   - Generation NEVER blocks the UI: one forward per token is the unit, the
     scheduler yields via requestAnimationFrame (playground) or setTimeout(0)
     (trap batch, survives a hidden tab) between tokens.
   - Every failure surfaces in a banner — no silent failures. Model output is
     arbitrary bytes, so ALL dynamic text goes through textContent/createDet.
   - Pure helpers are exported (window.P1_LAB / module.exports) and unit-
     smoked by scripts/test_task10_lab.js so the logic is testable in Node.
   ============================================================================ */

(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.P1_LAB = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const HAS_DOM = typeof document !== 'undefined' && typeof window !== 'undefined';

  /* ================================================================ pure == */

  const RAW_MAX_NEW = 120;          // playground cap (spec)
  const TRAP_MAX_NEW = 100;         // trap-batch cap (keeps 40 runs snappy)
  const TRAP_TRUNCATE = 180;        // chars before "(more)" expands

  // RAW prompt contract: bos + "Q: <q>\nA:"  (identical to colab/08_eval.ipynb)
  function buildPrompt(q) { return 'Q: ' + String(q) + '\nA:'; }

  // stop condition: the model starting a NEW question means the answer ended
  function hitsNewlineQ(text) { return /\nQ/.test(String(text)); }

  function truncate(s, n) {
    const t = String(s == null ? '' : s);
    if (t.length <= n) return t;
    return t.slice(0, Math.max(0, n - 1)) + '…';
  }

  function fmtInt(n) { return Number(n).toLocaleString('en-US'); }
  function fmtPct(x) {
    return (x == null || !isFinite(x)) ? '–' : (100 * x).toFixed(1) + '%';
  }

  // Per-claim verdict chips for one answer vs its top-3 chunks.
  // v = the P1.verify namespace (injectable → Node-testable).
  function claimChips(answerText, topk, v) {
    const out = { refused: false, claims: [] };
    if (v.isRefusal(answerText)) { out.refused = true; return out; }
    const claims = v.splitClaims(answerText);
    for (const cl of claims) {
      const verdict = v.verifyClaim(cl, topk);
      out.claims.push({ text: cl.text, verdict: verdict.verdict,
                        score: verdict.score, bestChunkId: verdict.bestChunkId });
    }
    return out;
  }

  // Aggregate TRAP LAB summary counters from the per-question records.
  function aggregateSummary(records) {
    const s = { myths: 0, unans: 0, parrotRaw: 0, parrotGrd: 0,
                refusedRaw: 0, refusedGrd: 0, errors: 0 };
    for (const r of records || []) {
      if (r.error) { s.errors++; continue; }
      if (r.item.kind === 'myth') {
        s.myths++;
        if (r.mythRaw && r.mythRaw.parroting) s.parrotRaw++;
        if (r.mythGrd && r.mythGrd.parroting) s.parrotGrd++;
      } else if (r.item.kind === 'unanswerable') {
        s.unans++;
        if (r.refusedRaw) s.refusedRaw++;
        if (r.refusedGrd) s.refusedGrd++;
      }
    }
    return s;
  }

  // Shape p1-metrics-v1 into rows for the ABOUT table:
  // one row per eval set, one cell per mode.
  function metricsRows(m) {
    const modes = ['raw', 'grounded', 'verified', 'abstaining'];
    const sets = Object.keys((m && m.sets) || {});
    return sets.map(function (name) {
      const set = m.sets[name] || {};
      const cells = modes.map(function (mode) {
        const pm = (set.per_mode || {})[mode] || {};
        return {
          mode: mode,
          hallucination_rate: pm.hallucination_rate,
          refusal_rate: pm.refusal_rate,
          parroting_rate: pm.parroting_rate   // only on mythtraps
        };
      });
      return { set: name, n: set.n, cells: cells };
    });
  }

  const LAB = {
    RAW_MAX_NEW: RAW_MAX_NEW, TRAP_MAX_NEW: TRAP_MAX_NEW,
    TRAP_TRUNCATE: TRAP_TRUNCATE,
    buildPrompt: buildPrompt, hitsNewlineQ: hitsNewlineQ,
    truncate: truncate, fmtInt: fmtInt, fmtPct: fmtPct,
    claimChips: claimChips, aggregateSummary: aggregateSummary,
    metricsRows: metricsRows
  };

  if (!HAS_DOM) return LAB;                    // Node: pure API only

  /* ============================================================== state == */

  const $ = (id) => document.getElementById(id);
  const G = window;                            // browser globals

  const S = {
    tok: null, model: null, cfg: null, meta: null,
    corpus: null, index: null, indexPromise: null, indexMs: null,
    mode: 'raw', genBusy: false, genCancel: false,
    trapRunning: false, trapCancel: false, trapRecords: [],
    metrics: null, aboutDone: false, trainingLoaded: false, booted: false
  };

  /* =============================================================== dom == */

  function createDet(tag, cls, text) {         // safe element builder
    const d = document.createElement(tag);
    if (cls) d.className = cls;
    if (text !== undefined && text !== null) d.textContent = String(text);
    return d;
  }

  function showBanner(id, kind, msg) {
    const b = $(id);
    b.className = 'banner ' + kind;
    b.textContent = msg;
    b.hidden = !msg;
  }
  function hideBanner(id) { const b = $(id); b.hidden = true; b.textContent = ''; }

  // last-resort error surface (no silent failures)
  window.addEventListener('error', function (e) {
    if (!S.booted) showBanner('boot-error', 'err',
      'script error: ' + (e.message || e.type) + (e.filename ? ' @ ' + e.filename : ''));
  });
  window.addEventListener('unhandledrejection', function (e) {
    const r = e && e.reason ? (e.reason.message || String(e.reason)) : 'unknown';
    showBanner(S.booted ? 'pg-error' : 'boot-error', 'err', 'unhandled: ' + r);
  });

  function nextFrame() {                       // rAF-yield for streaming
    return new Promise((res) => {
      if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => res());
      else setTimeout(res, 0);
    });
  }
  function nextTick() { return new Promise((res) => setTimeout(res, 0)); }

  function setStatusLine(html) { $('status-line').innerHTML = html; }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  /* ====================================================== tab router == */

  const TABS = ['playground', 'training', 'traps', 'about'];

  function selectTab(name, opts) {
    opts = opts || {};
    if (TABS.indexOf(name) === -1) name = 'playground';
    for (const t of TABS) {
      const tab = $('tab-' + t), panel = $('panel-' + t);
      const on = t === name;
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      tab.tabIndex = on ? 0 : -1;
      panel.hidden = !on;
    }
    if (opts.updateHash !== false) {
      if (location.hash !== '#' + name) location.hash = name;
    }
    if (name === 'training') lazyTraining();
    if (name === 'about') lazyAbout();
    if (opts.focus) $('tab-' + name).focus();
  }

  function initTabs() {
    const bar = document.querySelector('.tabbar');
    bar.addEventListener('click', function (e) {
      const tab = e.target.closest('[role="tab"]');
      if (tab) selectTab(tab.id.replace('tab-', ''));
    });
    bar.addEventListener('keydown', function (e) {
      const cur = document.activeElement;
      if (!cur || cur.getAttribute('role') !== 'tab') return;
      const i = TABS.indexOf(cur.id.replace('tab-', ''));
      let to = null;
      if (e.key === 'ArrowRight') to = (i + 1) % TABS.length;
      else if (e.key === 'ArrowLeft') to = (i - 1 + TABS.length) % TABS.length;
      else if (e.key === 'Home') to = 0;
      else if (e.key === 'End') to = TABS.length - 1;
      if (to !== null) { e.preventDefault(); selectTab(TABS[to], { focus: true }); }
    });
    window.addEventListener('hashchange', () =>
      selectTab(location.hash.replace('#', ''), { updateHash: false }));
    const initial = location.hash.replace('#', '');
    selectTab(TABS.indexOf(initial) !== -1 ? initial : 'playground',
              { updateHash: initial === '' });
  }

  /* =============================================== loader (model/tok) == */

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = () => res(true);
      s.onerror = () => rej(new Error('script load failed: ' + src));
      document.head.appendChild(s);
    });
  }

  function b64ToArrayBuffer(b64) {             // parity-page pattern
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8.buffer;
  }

  async function loadBin() {                   // fetch → snapshot fallback
    try {
      const r = await fetch('weights/web_model.bin');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return { buf: await r.arrayBuffer(), via: 'fetch' };
    } catch (e) { /* file:// or missing → snapshot */ }
    await loadScript('weights/web_model_data.js');
    if (!window.P1_WEB_MODEL_B64) {
      throw new Error('weights missing both ways (fetch blocked AND no ' +
                      'weights/web_model_data.js snapshot)');
    }
    return { buf: b64ToArrayBuffer(window.P1_WEB_MODEL_B64), via: 'snapshot' };
  }

  async function loadModelAsync() {
    const log = [];
    try {
      const bin = await loadBin();
      const parsed = G.P1.weights.parseWebModel(bin.buf);
      S.model = G.P1.weights.buildModel(parsed, 0);
      S.cfg = S.model.config;
      S.meta = parsed.meta;
      S.via = bin.via;
      log.push('weights: ' + fmtInt(parsed.numParams) + ' params (step ' +
               parsed.meta.step + ') via ' + bin.via);
      return S.model;
    } catch (e) {
      S.model = null;
      showBanner('boot-error', 'err',
        'model load FAILED — RAW generation is off, GROUNDED still works.\n' +
        (e.message || e) + '\n' +
        'fix: put a checkpoint in docs/weights (python3 colab/05_export.py).');
      return null;
    } finally {
      refreshHeader(log);
    }
  }

  function refreshHeader(log) {
    const bits = [];
    if (G.P1) bits.push('engine v' + esc(G.P1.VERSION || '?'));
    if (S.tok) bits.push('BPE ' + fmtInt(S.tok.vocabSize));
    if (S.corpus) bits.push(fmtInt(S.corpus.n) + ' corpus pairs');
    let modelTxt;
    if (S.model) {
      const c = S.cfg;
      modelTxt = '<b>model loaded</b>: ' + c.n_layer + 'L·' + c.n_head + 'H·' +
        c.d_model + 'd·ctx' + c.ctx + '·vocab' + fmtInt(c.vocab_size) +
        ' · ' + fmtInt(S.model.numParams()) + ' params · step ' + S.meta.step +
        ' <span class="dim">(' + (S.via || '?') + ')</span>';
    } else if (log && log.length) {
      modelTxt = '<span class="badtx">model unavailable — RAW off, GROUNDED ready</span>';
    } else {
      modelTxt = 'loading weights…';
    }
    setStatusLine(bits.join(' · ') + ' — ' + modelTxt);

    const db = $('demo-banner');
    if (S.model && S.meta && S.meta.step < 1000) {
      db.hidden = false;
      db.textContent = 'demo brain (' + S.meta.step + ' steps) — replace via ' +
        'colab/05_export.py after the real overnight run.';
    }
  }

  /* ===================================================== corpus index == */

  async function ensureIndex() {
    if (S.index) return S.index;
    if (!S.indexPromise) {
      S.indexPromise = (async () => {
        const t0 = performance.now();
        const idx = G.P1.retrieve.buildIndex(S.corpus.pairs);
        S.index = idx;
        S.indexMs = Math.round(performance.now() - t0);
        refreshFoot();
        return idx;
      })();
    }
    return S.indexPromise;
  }

  function refreshFoot() {
    const bits = [];
    if (S.index) bits.push('retrieval index: ' + fmtInt(S.index.N) +
      ' chunks · built in ' + S.indexMs + ' ms');
    else bits.push('retrieval index: builds on first GROUNDED use');
    if (S.model) bits.push('sampling: ' + ($('pg-sampling').value === 'greedy'
      ? 'greedy' : 'temp ' + $('pg-sampling').value) + ' · ≤' + RAW_MAX_NEW + ' tokens');
    $('pg-foot').textContent = bits.join(' · ');
  }

  /* ========================================================== GROUNDED == */

  // retrieval + extractive composition (verify.js) + per-sentence chip refs
  async function groundedWithCitations(q) {
    const idx = await ensureIndex();
    const topk = G.P1.verify.topkChunks(idx, q, 3);
    const g = G.P1.verify.groundedAnswer(q, topk);
    const refs = [];
    if (!g.refused) {
      const sents = G.P1.verify.splitSentences(g.answer);
      for (const sent of sents) {
        let rank = null;
        for (const ch of topk) {
          if (G.P1.verify.answerPart(ch.text).indexOf(sent) !== -1) {
            rank = ch.rank; break;
          }
        }
        if (rank === null) {          // rare: punctuation re-split — best F1
          let best = -1;
          for (const ch of topk) {
            const f = G.P1.verify.contentF1(sent, G.P1.verify.answerPart(ch.text));
            if (f > best) { best = f; rank = ch.rank; }
          }
        }
        refs.push({ sent: sent, rank: rank });
      }
    }
    return { answer: g.answer, refused: g.refused,
             bestSentenceF1: g.bestSentenceF1, topk: topk, refs: refs };
  }

  function chipRanks(refs) {                   // unique, in first-use order
    const out = [];
    for (const r of refs || []) {
      if (r.rank != null && out.indexOf(r.rank) === -1) out.push(r.rank);
    }
    return out;
  }

  function buildCiteChips(topk, refs, container) {
    const ranks = chipRanks(refs);
    const pops = {};
    for (const rank of ranks) {
      const ch = topk.find((c) => c.rank === rank);
      if (!ch) continue;
      const chip = createDet('button', 'chip', '[' + rank + ']');
      chip.type = 'button';
      chip.title = truncate(ch.text, 160);
      chip.setAttribute('aria-expanded', 'false');
      const pop = createDet('div', 'cite-pop');
      pop.hidden = true;
      pop.appendChild(createDet('div', null, ch.text));
      pop.appendChild(createDet('div', 'csrc',
        'chunk ' + ch.chunkId + ' · bm25 ' + ch.score.toFixed(3) +
        (ch.sources && ch.sources.length ? ' · ' + ch.sources.join('; ') : '')));
      chip.addEventListener('click', function () {
        pop.hidden = !pop.hidden;
        chip.setAttribute('aria-expanded', pop.hidden ? 'false' : 'true');
      });
      container.appendChild(chip);
      pops[rank] = pop;
    }
    for (const rank of ranks) if (pops[rank]) container.appendChild(pops[rank]);
  }

  /* ======================================================= generation == */

  // Chunked RAW generation: one forward per token, yields between tokens.
  async function rawGenerate(q, opts) {
    const o = opts || {};
    if (!S.model) throw new Error('no model loaded (weights missing)');
    const model = S.model, tok = S.tok;
    const V = model.config.vocab_size, ctx = model.config.ctx;
    const temp = o.greedy ? 0 : (o.temp === undefined ? 0.8 : o.temp);
    const maxNew = o.maxNew || RAW_MAX_NEW;
    const yielder = o.setYield || nextFrame;

    let ids = [tok.bos];
    for (const id of tok.encode(buildPrompt(q))) ids.push(id);
    const out = [];
    const probs = new Float32Array(V);
    const rng = G.P1.mulberry32((o.seed >>> 0) ||
      ((Math.random() * 4294967296) >>> 0));
    let stopped = false, tokens = 0;

    for (let i = 0; i < maxNew; i++) {
      if (o.cancelled && o.cancelled()) { stopped = true; break; }
      const win = ids.length > ctx ? ids.slice(ids.length - ctx) : ids;
      const logits = model.forward(win, 1);            // [T, V]
      const off = (win.length - 1) * V, row = logits.data;
      let pick = V - 1;
      if (temp <= 0) {                                 // greedy argmax
        let m = -Infinity;
        for (let j = 0; j < V; j++) if (row[off + j] > m) { m = row[off + j]; pick = j; }
      } else {                                         // ancestral, temp first
        let m = -Infinity;
        for (let j = 0; j < V; j++) if (row[off + j] > m) m = row[off + j];
        let sum = 0;
        for (let j = 0; j < V; j++) {
          probs[j] = Math.exp((row[off + j] - m) / temp); sum += probs[j];
        }
        let r = rng() * sum;
        for (let j = 0; j < V; j++) { r -= probs[j]; if (r <= 0) { pick = j; break; } }
      }
      if (pick === tok.eos) break;
      ids.push(pick); out.push(pick); tokens++;
      const text = tok.decode(out);
      if (o.onToken) o.onToken(text, tokens);
      if (hitsNewlineQ(text)) break;
      await yielder();
    }
    return { text: tok.decode(out), tokens: tokens, stopped: stopped };
  }

  /* ======================================================== playground == */

  function addMsg(kind, roleText) {
    const m = createDet('div', 'msg ' + kind);
    m.appendChild(createDet('div', 'mrole', roleText));
    const body = createDet('div', 'mtext');
    m.appendChild(body);
    $('pg-log').appendChild(m);
    $('pg-placeholder').hidden = true;
    $('pg-log').scrollTop = $('pg-log').scrollHeight;
    return { root: m, body: body };
  }

  async function onGenerate(e) {
    if (e) e.preventDefault();
    if (S.genBusy) return;
    const q = $('pg-q').value.trim();
    hideBanner('pg-error');
    if (!q) { showBanner('pg-error', 'err', 'type a question first.'); return; }
    if (S.mode === 'raw' && !S.model) {
      showBanner('pg-error', 'err',
        'no model loaded — RAW is off. GROUNDED still works (retrieval needs no weights).');
      return;
    }
    S.genBusy = true; S.genCancel = false;
    $('pg-go').disabled = true;
    $('pg-stop').hidden = false;
    $('pg-q').value = '';

    addMsg('user', 'you').body.textContent = q;
    const bot = addMsg('bot', S.mode === 'raw' ? 'p_1 · RAW' : 'p_1 · GROUNDED');
    const meta = createDet('div', 'mmeta');
    const cites = createDet('div', 'cites');
    try {
      if (S.mode === 'raw') {
        bot.body.textContent = '…';
        const tempStr = $('pg-sampling').value;
        const greedy = tempStr === 'greedy';
        const r = await rawGenerate(q, {
          greedy: greedy, temp: greedy ? 0 : parseFloat(tempStr),
          maxNew: RAW_MAX_NEW,
          cancelled: () => S.genCancel,
          onToken: (text) => {
            bot.body.textContent = text;
            $('pg-log').scrollTop = $('pg-log').scrollHeight;
          }
        });
        bot.body.textContent = r.text || '(empty answer — the demo brain said nothing)';
        meta.textContent = 'RAW · ' + r.tokens + ' tokens' +
          (r.stopped ? ' · stopped early' : '') +
          ' · step ' + (S.meta ? S.meta.step : '?') + ' demo checkpoint';
      } else {
        bot.body.textContent = S.index ? 'retrieving…' : 'building retrieval index…';
        const g = await groundedWithCitations(q);
        bot.body.textContent = g.answer +
          (g.refused ? '' : '');
        if (g.refused) {
          cites.appendChild(createDet('span', 'badge v-refused', 'refused — nothing in the corpus scored ≥ 0.08 F1'));
        } else {
          buildCiteChips(g.topk, g.refs, cites);
        }
        meta.textContent = 'GROUNDED · top-3 bm25 over ' + fmtInt(S.index.N) +
          ' chunks · best sentence F1 ' + (g.bestSentenceF1 || 0).toFixed(3);
      }
    } catch (err) {
      showBanner('pg-error', 'err', 'generation failed: ' + (err.message || err));
      bot.body.textContent = '(failed — see the red banner)';
    } finally {
      bot.root.appendChild(cites);
      bot.root.appendChild(meta);
      S.genBusy = false;
      $('pg-go').disabled = false;
      $('pg-stop').hidden = true;
      $('pg-log').scrollTop = $('pg-log').scrollHeight;
      refreshFoot();
    }
  }

  function initPlayground() {
    const setMode = (m) => {
      S.mode = m;
      $('pg-mode-raw').setAttribute('aria-pressed', m === 'raw' ? 'true' : 'false');
      $('pg-mode-grounded').setAttribute('aria-pressed', m === 'grounded' ? 'true' : 'false');
      refreshFoot();
    };
    $('pg-mode-raw').addEventListener('click', () => setMode('raw'));
    $('pg-mode-grounded').addEventListener('click', () => setMode('grounded'));
    $('pg-form').addEventListener('submit', onGenerate);
    $('pg-stop').addEventListener('click', () => { S.genCancel = true; });
    $('pg-clear').addEventListener('click', () => {
      $('pg-log').textContent = '';
      $('pg-log').appendChild($('pg-placeholder'));
      $('pg-placeholder').hidden = false;
    });
    $('pg-sampling').addEventListener('change', refreshFoot);
    refreshFoot();
  }

  /* ======================================================= trap lab == */

  function trapItems() {
    const items = [];
    for (const t of (G.P1_TRAPS || [])) items.push(Object.assign({ kind: 'myth' }, t));
    for (const u of (G.P1_UNANSWERABLES || [])) {
      items.push(Object.assign({ kind: 'unanswerable' }, u));
    }
    return items;
  }

  function setProgress(done, total, label) {
    const p = $('tr-progress');
    p.querySelector('i').style.width = (100 * done / total).toFixed(1) + '%';
    p.setAttribute('aria-valuenow', String(done));
    $('tr-progline').textContent = done + '/' + total + (label ? ' · ' + label : '') +
      (S.trapCancel ? ' · cancelling…' : '');
  }

  function trapSummaryUpdate() {
    const s = aggregateSummary(S.trapRecords);
    const pair = (a, b, goodWhenLower) => {
      // colour: green when GROUNDED does no worse (parrots ≤ / refuses ≥)
      const good = goodWhenLower ? b <= a : b >= a;
      return { a: a, b: b, cls: (a + b) === 0 ? '' : (good ? ' good' : ' badv') };
    };
    const put = (id, v, cls) => {
      const elv = $(id);
      elv.textContent = v;
      elv.className = 'v' + (cls || '');
    };
    const p = pair(s.parrotRaw, s.parrotGrd, true);
    const r = pair(s.refusedRaw, s.refusedGrd, false);
    put('tr-s1', s.parrotRaw + '/' + s.myths, p.a === 0 && p.b > 0 ? ' good' : (p.a > 0 ? ' badv' : ''));
    put('tr-s2', s.parrotGrd + '/' + s.myths, p.cls);
    put('tr-s3', s.refusedRaw + '/' + s.unans, '');
    put('tr-s4', s.refusedGrd + '/' + s.unans, r.cls);
    return s;
  }

  function claimChipsRow(chips, topk, container) {
    if (chips.refused) {
      container.appendChild(createDet('span', 'badge v-refused', 'refused'));
      return;
    }
    if (!chips.claims.length) {
      container.appendChild(createDet('span', 'badge', 'no claims'));
      return;
    }
    const counts = { 'supported': 0, 'unsupported': 0, 'not-in-corpus': 0 };
    for (const c of chips.claims) counts[c.verdict] = (counts[c.verdict] || 0) + 1;
    const mk = (n, verdict, label) => {
      if (!n) return;
      const b = createDet('span', 'badge v-' + verdict,
        (verdict === 'supported' ? '✓×' + n :
         verdict === 'unsupported' ? '✗×' + n : '∅×' + n) + ' ' + label);
      b.title = chips.claims
        .filter((c) => c.verdict === verdict)
        .map((c) => truncate(c.text, 120) + '  [f1 ' + c.score.toFixed(3) + ']')
        .join('\n');
      container.appendChild(b);
    };
    mk(counts['supported'], 'supported', 'supported');
    mk(counts['unsupported'], 'unsupported', 'unsupported');
    mk(counts['not-in-corpus'], 'not-in-corpus', 'not-in-corpus');
  }

  function trapCard(rec) {
    const card = createDet('div', 'tcard' + (rec.error ? ' err' : ''));
    const kindLabel = rec.item.kind === 'myth' ? 'myth-trap' : 'unanswerable';

    const qrow = createDet('div', 'qrow');
    qrow.appendChild(createDet('span', 'qid', rec.item.id + ' · ' + kindLabel));
    qrow.appendChild(createDet('span', 'qtext', rec.item.question));
    card.appendChild(qrow);

    if (rec.error) {
      card.appendChild(createDet('div', 'atext badtx', 'error: ' + rec.error));
      return card;
    }

    const mkMode = (label, cls, answer, chips, myth, refused, g) => {
      const row = createDet('div', 'mode-row');
      const head = createDet('div', 'anshead');
      head.appendChild(createDet('span', 'mlabel ' + cls, label));
      const full = String(answer || '');
      const short = truncate(full, TRAP_TRUNCATE);
      const atx = createDet('div', 'atext', short);
      const flags = createDet('span', 'flags');
      if (myth) {
        flags.appendChild(createDet('span', 'badge ' + (myth.parroting ? 'f-parrot' : 'f-clean'),
          myth.parroting
            ? 'myth parroted (folk ' + myth.folk_f1.toFixed(2) + ' > corr ' + myth.correction_f1.toFixed(2) + ')'
            : 'not parroting'));
      }
      if (refused !== null) {
        flags.appendChild(createDet('span', 'badge ' + (refused ? 'v-refused' : ''),
          refused ? 'refused' : 'answered'));
      }
      head.appendChild(flags);
      row.appendChild(head);
      row.appendChild(atx);

      if (full.length > short.length) {          // expandable RAW/GROUNDED text
        const btn = createDet('button', 'expand', '(more)');
        btn.type = 'button';
        let open = false;
        btn.addEventListener('click', () => {
          open = !open;
          atx.textContent = open ? full : short;
          btn.textContent = open ? '(less)' : '(more)';
        });
        head.appendChild(btn);
      }

      const claims = createDet('div', 'claims');
      claimChipsRow(chips, rec.g ? rec.g.topk : null, claims);
      row.appendChild(claims);
      // per-claim lines (hover chips show text too; this is the tap-friendly view)
      if (!chips.refused && chips.claims.length) {
        for (const c of chips.claims) {
          const ln = createDet('div', 'claimline');
          const b = createDet('span', 'badge v-' + c.verdict,
            c.verdict === 'supported' ? '✓' : c.verdict === 'unsupported' ? '✗' : '∅');
          ln.appendChild(b);
          ln.appendChild(document.createTextNode(c.text + ' '));
          ln.appendChild(createDet('span', 'score', '[f1 ' + c.score.toFixed(3) + ']'));
          claims.appendChild(ln);
        }
      }
      if (g) buildCiteChips(g.topk, g.refs, row);
      return row;
    };

    card.appendChild(mkMode('RAW', 'raw', rec.raw.text, rec.rawChips,
      rec.mythRaw, rec.item.kind === 'unanswerable' ? rec.refusedRaw : null, null));
    card.appendChild(mkMode('GROUNDED', 'grd', rec.g.answer, rec.grdChips,
      rec.mythGrd, rec.item.kind === 'unanswerable' ? rec.refusedGrd : null, rec.g));
    return card;
  }

  async function runTraps() {
    if (S.trapRunning) return;
    hideBanner('tr-error');
    const items = trapItems();
    if (!items.length) {
      showBanner('tr-error', 'err', 'no trap data found (P1_TRAPS / P1_UNANSWERABLES).');
      return;
    }
    if (!S.model) {
      showBanner('tr-error', 'err',
        'no model loaded — the RAW arm needs weights. GROUNDED-only runs: not supported here by design (the comparison IS the point).');
      return;
    }
    S.trapRunning = true; S.trapCancel = false; S.trapRecords = [];
    $('tr-run').disabled = true;
    $('tr-cancel').hidden = false;
    $('tr-results').textContent = '';
    setProgress(0, items.length, 'building retrieval index…');
    try { await ensureIndex(); } catch (e) {
      showBanner('tr-error', 'err', 'index build failed: ' + (e.message || e));
      S.trapRunning = false;
      $('tr-run').disabled = false;
      $('tr-cancel').hidden = true;
      return;
    }

    let done = 0;
    for (const item of items) {
      if (S.trapCancel) break;
      setProgress(done, items.length, item.id + ' · RAW generating…');
      try {
        const raw = await rawGenerate(item.question, {
          temp: 0.8, maxNew: TRAP_MAX_NEW,
          cancelled: () => S.trapCancel,
          setYield: nextTick                      // setTimeout: survives hidden tab
        });
        setProgress(done, items.length, item.id + ' · grounding…');
        const idx = S.index;
        const topk = G.P1.verify.topkChunks(idx, item.question, 3);
        const g = G.P1.verify.groundedAnswer(item.question, topk);
        // per-sentence refs for citation chips (same helper as playground)
        const refs = [];
        if (!g.refused) {
          for (const sent of G.P1.verify.splitSentences(g.answer)) {
            let rank = null;
            for (const ch of topk) {
              if (G.P1.verify.answerPart(ch.text).indexOf(sent) !== -1) { rank = ch.rank; break; }
            }
            if (rank === null) rank = topk.length ? topk[0].rank : null;
            refs.push({ sent: sent, rank: rank });
          }
        }
        const rec = {
          item: item, raw: raw, g: Object.assign({}, g, { topk: topk, refs: refs }),
          rawChips: claimChips(raw.text, topk, G.P1.verify),
          grdChips: claimChips(g.answer, topk, G.P1.verify),
          refusedRaw: G.P1.verify.isRefusal(raw.text),
          refusedGrd: G.P1.verify.isRefusal(g.answer)
        };
        if (item.kind === 'myth') {
          rec.mythRaw = G.P1.verify.mythVerdict(raw.text, item.folk_belief, item.correction);
          rec.mythGrd = G.P1.verify.mythVerdict(g.answer, item.folk_belief, item.correction);
        }
        S.trapRecords.push(rec);
        $('tr-results').appendChild(trapCard(rec));
      } catch (err) {
        S.trapRecords.push({ item: item, error: String((err && err.message) || err) });
        $('tr-results').appendChild(trapCard({ item: item, error: String((err && err.message) || err) }));
      }
      done++;
      setProgress(done, items.length, item.id);
      trapSummaryUpdate();
      await nextTick();
    }

    const s = trapSummaryUpdate();
    const cancelled = S.trapCancel && done < items.length;
    setProgress(done, items.length, cancelled ? 'cancelled' : 'done');
    if (cancelled) {
      showBanner('tr-error', 'warn', 'run cancelled at ' + done + '/' + items.length +
        ' — partial results kept below.');
    }
    S.trapRunning = false;
    $('tr-run').disabled = false;
    $('tr-cancel').hidden = true;
  }

  function initTraps() {
    $('tr-run').addEventListener('click', runTraps);
    $('tr-cancel').addEventListener('click', () => { S.trapCancel = true; });
  }

  /* ============================================================ about == */

  async function fetchRealMetrics() {          // real metrics.json upgrade path
    try {
      const r = await fetch('results/metrics.json');
      if (!r.ok) return null;
      const j = await r.json();
      if (j && j.format === 'p1-metrics-v1' && j.sets) return j;
      return null;
    } catch (e) { return null; }               // file:// → snapshot
  }

  function renderMetrics(m) {
    const host = $('ab-metrics');
    host.textContent = '';
    const rows = metricsRows(m);
    const table = createDet('table', 'metrics');
    const cap = createDet('caption', null,
      'per set × mode · H = hallucination rate (unsupported claims / claims) · ' +
      'R = refusal rate · P = myth-parroting rate' +
      (m.config && m.config.tau ? ' · abstain τ=' + m.config.tau : ''));
    table.appendChild(cap);
    const thead = createDet('thead');
    const hr1 = createDet('tr');
    hr1.appendChild(createDet('th', null, 'eval set'));
    for (const mode of ['raw', 'grounded', 'verified', 'abstaining']) {
      hr1.appendChild(createDet('th', null, mode));
    }
    thead.appendChild(hr1);
    table.appendChild(thead);
    const tbody = createDet('tbody');
    for (const row of rows) {
      const tr = createDet('tr');
      tr.appendChild(createDet('th', null, row.set + ' (n=' + row.n + ')'));
      for (const cell of row.cells) {
        const td = createDet('td', 'num');
        td.appendChild(document.createTextNode(
          'H ' + fmtPct(cell.hallucination_rate) + ' · R ' + fmtPct(cell.refusal_rate)));
        if (cell.parroting_rate !== undefined) {
          td.appendChild(document.createElement('br'));
          td.appendChild(document.createTextNode('P ' + fmtPct(cell.parroting_rate)));
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.appendChild(table);

    const note = $('ab-metrics-note');
    const bits = [];
    if (m.stub === true) $('ab-stub').hidden = false;
    if (m.summary && m.summary.headline) bits.push('headline: ' + m.summary.headline);
    if (m.config) {
      const t = m.config.thresholds || {};
      bits.push('verifier thresholds: supported ≥ ' + t.f1_supported +
        ' · not-in-corpus < ' + t.f1_not_in_corpus + ' · k=' + m.config.k +
        ' ' + m.config.method + ' · grounded shows ≤' + m.config.grounded_sentences +
        ' corpus sentences');
    }
    note.textContent = bits.join(' — ');
  }

  async function lazyAbout() {
    if (S.aboutDone) return;
    S.aboutDone = true;
    const real = await fetchRealMetrics();
    S.metrics = real || G.P1_METRICS || null;
    if (!S.metrics) {
      showBanner('ab-stub', 'err', 'no metrics found (P1_METRICS snapshot missing).');
      return;
    }
    $('ab-stub').textContent = real
      ? 'metrics loaded from results/metrics.json' +
        (real.stub === true ? ' — still the stub answerer (stub:true); run colab/08_eval.ipynb for real numbers.' : '')
      : 'demo metrics (stub answerer) — run colab/08_eval.ipynb for real numbers.';
    $('ab-stub').hidden = false;
    renderMetrics(S.metrics);
  }

  function lazyTraining() {
    if (S.trainingLoaded) return;
    S.trainingLoaded = true;
    const f = $('training-frame');
    if (f && f.dataset.src) f.src = f.dataset.src;
  }

  /* ============================================================= boot == */

  function missingGlobals() {
    const need = {
      'P1 (tensor.js)': G.P1, 'P1.BPE (tokenizer.js)': G.P1 && G.P1.BPE,
      'P1.weights (weights.js)': G.P1 && G.P1.weights,
      'P1.retrieve (retrieve.js)': G.P1 && G.P1.retrieve,
      'P1.verify (verify.js)': G.P1 && G.P1.verify,
      'P1_TOKENIZER_DATA': G.P1_TOKENIZER_DATA,
      'P1_CORPUS_DOCS': G.P1_CORPUS_DOCS,
      'P1_TRAPS': G.P1_TRAPS, 'P1_UNANSWERABLES': G.P1_UNANSWERABLES,
      'P1_METRICS': G.P1_METRICS
    };
    const miss = Object.keys(need).filter((k) => !need[k]);
    return miss;
  }

  async function boot() {
    if (S.booted) return;
    const miss = missingGlobals();
    if (miss.length) {
      showBanner('boot-error', 'err',
        'missing pieces: ' + miss.join(', ') +
        ' — a <script> failed to load. Check the paths in docs/index.html.');
      setStatusLine('<span class="badtx">boot incomplete — see banner</span>');
      return;
    }
    try {
      S.tok = new G.P1.BPE(G.P1_TOKENIZER_DATA.merges);
    } catch (e) {
      showBanner('boot-error', 'err', 'tokenizer init failed: ' + (e.message || e));
      return;
    }
    S.corpus = G.P1_CORPUS_DOCS;
    S.booted = true;
    initTabs();
    initPlayground();
    initTraps();
    refreshHeader([]);
    S.modelPromise = loadModelAsync();          // async; header updates when done
    await S.modelPromise;
    refreshHeader([]);
    refreshFoot();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  // exposed for tests / console poking
  return Object.assign(LAB, {
    state: S,
    boot: boot,
    rawGenerate: rawGenerate,
    groundedWithCitations: groundedWithCitations,
    ensureIndex: ensureIndex,
    trapItems: trapItems,
    runTraps: runTraps,
    trapSummaryUpdate: trapSummaryUpdate,
    renderMetrics: renderMetrics,
    selectTab: selectTab
  });
});
