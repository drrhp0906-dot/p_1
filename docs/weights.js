/* ============================================================================
   p_1 · docs/weights.js — Task 7 weights bridge: load the Colab brain
   ----------------------------------------------------------------------------
   Pure JavaScript, zero dependencies. Parses the little-endian binary written
   by colab/05_export.py (web_model.bin, format "p1wm"), rebuilds a
   docs/model.js Transformer and copies the weights in BY NAME. The browser
   sibling of the NumPy forward in colab/06_parity.ipynb.

   BINARY LAYOUT (all integers uint32 LE, floats float32 LE, tensors in the
   canonical paramNames() order; weight-tied head = wte stored ONCE):
     magic "P1WM" | uint32 version=1 | uint32 configJsonLen | config JSON
     per tensor: uint32 nameLen | name UTF-8 | uint32 ndim | ndim x uint32
                 dims (row-major) | raw float32 data
   The config JSON carries the full model config dict PLUS tokenizer_version
   and step trained.

   API (window.P1.weights / module.exports):
     parseWebModel(arrayBuffer [, manifest]) -> {config, tensors: Map(name ->
       {shape, f32: Float32Array, offset, nbytes}), numParams, headerBytes,
       byteLength, meta:{step, tokenizer_version}}
       — parses the bin DIRECTLY (manifest.json is for humans/debug; when one
       is passed it is cross-checked: names, shapes, offsets, sizes, totals).
     buildModel(parsed [, seed]) -> Transformer with the parsed weights
       copied in via model.setParams (names + sizes + shapes asserted).
     loadModel(url [, opts]) : Promise<{model, config, tensors, meta,
       manifest}> — fetch(url) -> parse -> build. Tries the sibling
       manifest.json for a cross-check unless opts.checkManifest is false.
     loadWebModelFile(path [, opts]) — Node: fs.readFileSync -> parseWebModel.

   UMD: window.P1.weights (needs tensor.js + model.js loaded first) /
   module.exports (requires ./tensor.js + ./model.js).
   ============================================================================ */

(function (root, factory) {
  let P1 = root.P1;
  if (typeof module === 'object' && module.exports) {
    if (!P1) P1 = require('./tensor.js');
    const MODEL = require('./model.js');
    let fs = null;
    try { fs = require('fs'); } catch (e) { /* browser build */ }
    module.exports = factory(P1, MODEL, fs);
  } else {
    if (!P1 || !P1.Transformer || !P1.MODEL) {
      throw new Error('weights.js needs tensor.js + model.js loaded first (window.P1)');
    }
    root.P1.weights = factory(P1, root.P1.MODEL, null);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (P1, MODEL, fs) {
  'use strict';

  const VERSION = '1.0.0';
  const MAGIC = [0x50, 0x31, 0x57, 0x4D];   // "P1WM"
  const FORMAT_VERSION = 1;
  const REQUIRED_KEYS = ['n_layer', 'n_head', 'd_model', 'd_ff', 'ctx',
                         'vocab_size'];

  function decodeUtf8(u8) {
    if (typeof TextDecoder !== 'undefined') {
      return new TextDecoder('utf-8').decode(u8);
    }
    let out = '';
    for (let i = 0; i < u8.length; i++) out += String.fromCharCode(u8[i]);
    return decodeURIComponent(escape(out));
  }

  /* ------------------------------------------------------------------ parse */

  function parseWebModel(buffer, manifest) {
    const dv = (buffer instanceof DataView)
      ? buffer
      : (ArrayBuffer.isView(buffer)
          ? new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
          : new DataView(buffer));
    if (dv.byteLength < 16) {
      throw new Error('web_model: too small (' + dv.byteLength + ' bytes)');
    }
    for (let i = 0; i < 4; i++) {
      if (dv.getUint8(i) !== MAGIC[i]) {
        throw new Error('web_model: bad magic — not a "P1WM" file');
      }
    }
    const version = dv.getUint32(4, true);
    if (version !== FORMAT_VERSION) {
      throw new Error('web_model: unsupported version ' + version +
                      ' (want ' + FORMAT_VERSION + ')');
    }
    const cfgLen = dv.getUint32(8, true);
    if (12 + cfgLen > dv.byteLength) {
      throw new Error('web_model: config JSON (' + cfgLen +
                      ' bytes) overruns the file');
    }
    const config = JSON.parse(decodeUtf8(
      new Uint8Array(dv.buffer, dv.byteOffset + 12, cfgLen)));
    for (const k of REQUIRED_KEYS) {
      if (typeof config[k] !== 'number') {
        throw new Error('web_model: config JSON missing key ' + k);
      }
    }

    const tensors = new Map();
    let off = 12 + cfgLen;
    let numParams = 0;
    while (off < dv.byteLength) {
      if (off + 4 > dv.byteLength) {
        throw new Error('web_model: truncated tensor record at ' + off);
      }
      const nameLen = dv.getUint32(off, true); off += 4;
      if (off + nameLen > dv.byteLength) {
        throw new Error('web_model: truncated name at ' + off);
      }
      const name = decodeUtf8(new Uint8Array(dv.buffer, dv.byteOffset + off,
                                             nameLen));
      off += nameLen;
      if (off + 4 > dv.byteLength) {
        throw new Error('web_model: truncated ndim for ' + name);
      }
      const ndim = dv.getUint32(off, true); off += 4;
      const shape = [];
      let n = 1;
      for (let i = 0; i < ndim; i++) {
        if (off + 4 > dv.byteLength) {
          throw new Error('web_model: truncated dims for ' + name);
        }
        const dim = dv.getUint32(off, true); off += 4;
        shape.push(dim);
        n *= dim;
      }
      const nbytes = n * 4;
      if (off + nbytes > dv.byteLength) {
        throw new Error('web_model: truncated data for ' + name +
                        ' (want ' + nbytes + ' bytes at ' + off + ')');
      }
      // Explicit little-endian decode per contract (DataView, platform-safe)
      const f32 = new Float32Array(n);
      for (let i = 0; i < n; i++) f32[i] = dv.getFloat32(off + i * 4, true);
      tensors.set(name, { shape: shape, f32: f32, offset: off, nbytes: nbytes });
      off += nbytes;
      numParams += n;
    }
    if (off !== dv.byteLength) {
      throw new Error('web_model: ' + (dv.byteLength - off) +
                      ' trailing bytes after the last tensor');
    }

    const parsed = {
      config: config,
      tensors: tensors,
      numParams: numParams,
      headerBytes: 12 + cfgLen,
      byteLength: dv.byteLength,
      meta: { step: config.step, tokenizer_version: config.tokenizer_version }
    };
    if (manifest) checkManifest(parsed, manifest);
    return parsed;
  }

  /* ------------------------------------------------- manifest cross-check */

  function checkManifest(parsed, manifest) {
    if (!manifest || manifest.format !== 'p1wm' || manifest.version !== 1) {
      throw new Error('manifest: not a p1wm v1 manifest');
    }
    const mt = manifest.tensors || [];
    if (mt.length !== parsed.tensors.size) {
      throw new Error('manifest: ' + mt.length + ' tensors vs bin ' +
                      parsed.tensors.size);
    }
    let prev = -1, sum = 0;
    for (let i = 0; i < mt.length; i++) {
      const m = mt[i];
      const t = parsed.tensors.get(m.name);
      if (!t) throw new Error('manifest: unknown tensor ' + m.name);
      if (JSON.stringify(m.shape) !== JSON.stringify(t.shape)) {
        throw new Error('manifest: ' + m.name + ' shape mismatch');
      }
      if (m.offset !== t.offset || m.nbytes !== t.nbytes) {
        throw new Error('manifest: ' + m.name + ' offset/size mismatch ' +
                        '(manifest ' + m.offset + '/' + m.nbytes +
                        ', bin ' + t.offset + '/' + t.nbytes + ')');
      }
      if (m.offset <= prev) {
        throw new Error('manifest: offsets not strictly monotonic at ' +
                        m.name);
      }
      prev = m.offset;
      sum += t.f32.length;
    }
    if (manifest.num_params !== sum) {
      throw new Error('manifest: num_params ' + manifest.num_params +
                      ' vs bin ' + sum);
    }
    if (manifest.total_bytes !== parsed.byteLength) {
      throw new Error('manifest: total_bytes ' + manifest.total_bytes +
                      ' vs bin ' + parsed.byteLength);
    }
    return true;
  }

  /* ------------------------------------------------------------------ build */

  function buildModel(parsed, seed) {
    const model = new MODEL.Transformer(parsed.config, seed || 0);
    const named = {};
    for (const [name, t] of parsed.tensors) named[name] = t.f32;
    model.setParams(named);          // asserts name set + element counts
    for (const [name, t] of parsed.tensors) {
      const p = model.P[name];
      if (JSON.stringify(p.shape) !== JSON.stringify(t.shape)) {
        throw new Error(name + ': shape ' + JSON.stringify(t.shape) +
                        ' vs model ' + JSON.stringify(p.shape));
      }
    }
    return model;
  }

  /* ------------------------------------------------------------------ load */

  async function tryFetchManifest(url) {
    try {
      const i = url.lastIndexOf('/');
      const murl = (i >= 0 ? url.slice(0, i) : '.') + '/manifest.json';
      const r = await fetch(murl);
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;                    // manifest is optional by design
    }
  }

  async function loadModel(url, opts) {
    opts = opts || {};
    const r = await fetch(url);
    if (!r.ok) throw new Error('weights.loadModel: HTTP ' + r.status +
                               ' for ' + url);
    const buf = await r.arrayBuffer();
    let manifest = opts.manifest || null;
    if (!manifest && opts.checkManifest !== false) {
      manifest = await tryFetchManifest(url);
    }
    const parsed = parseWebModel(buf, manifest || undefined);
    const model = buildModel(parsed, opts.seed);
    return { model: model, config: parsed.config, tensors: parsed.tensors,
             meta: parsed.meta, manifest: manifest, parsed: parsed };
  }

  function loadWebModelFile(path, opts) {
    if (!fs) throw new Error('loadWebModelFile: fs unavailable (browser? use loadModel)');
    const buf = fs.readFileSync(path);
    return parseWebModel(buf, opts && opts.manifest);
  }

  return {
    VERSION: VERSION,
    MAGIC_BYTES: MAGIC,
    FORMAT_VERSION: FORMAT_VERSION,
    parseWebModel: parseWebModel,
    checkManifest: checkManifest,
    buildModel: buildModel,
    loadModel: loadModel,
    loadWebModelFile: loadWebModelFile
  };
});
