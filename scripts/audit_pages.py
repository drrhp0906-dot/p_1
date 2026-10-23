#!/usr/bin/env python3
# ============================================================================
# p_1 · scripts/audit_pages.py — GitHub Pages deploy audit (Task 12)
#
# Checks the docs/ folder that GitHub Pages serves and PRINTS a report:
#   (a) every src=/href=/data-src= in docs/**/*.html and every fetch('...')
#       string literal in docs/**/*.js resolves to an existing file
#       (relative paths resolved against the containing file; http(s):,
#       data:, #, mailto:, javascript: are ignored)
#   (b) grep audit: no absolute "/paths, no http:// and no protocol-relative
#       //cdn references inside src=/href=/fetch( contexts (Pages project
#       sites are served from /<repo>/ — absolute paths break; insecure
#       http:// is blocked on Pages; everything must be same-origin relative)
#   (c) weights presence: docs/weights/web_model.bin + manifest.json
#   (d) every *.js under docs/ parses as JavaScript (node --check)
#   (e) docs/index.html loads all of its <script src> / <link href> /
#       iframe data-src files (listed one by one)
#   (f) total docs/ size vs the 1 GB Pages limit
#
# Exit code 0 = no ERRORs (warnings allowed), 1 = at least one ERROR.
#
# Path-dual: auto-detects the repo root by walking up from this file's
# directory until a docs/index.html is found (works from the repo's own
# scripts/ copy AND from the workspace /home/z/my-project/scripts/ copy).
# Override: python3 audit_pages.py /path/to/p_1
# ============================================================================
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys

PAGES_LIMIT_BYTES = 1 << 30  # 1 GiB — GitHub Pages hard limit

ERRORS, WARNS, INFOS = [], [], []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNS.append(msg)


def info(msg):
    INFOS.append(msg)


def find_repo_root():
    """Walk up from this script until a dir containing docs/index.html."""
    override = sys.argv[1] if len(sys.argv) > 1 else None
    if override:
        root = os.path.abspath(override)
        if not os.path.isfile(os.path.join(root, "docs", "index.html")):
            sys.exit(f"audit_pages: {root} has no docs/index.html")
        return root
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    d = here
    for _ in range(6):                       # repo copy: scripts/ sits in the repo
        candidates.append(d)
        d = os.path.dirname(d)
    candidates.append(os.path.join(here, os.pardir, "download", "p_1"))  # workspace copy
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(os.path.join(c, "docs", "index.html")):
            return c
    sys.exit("audit_pages: could not locate repo root (docs/index.html not found)")


ROOT = find_repo_root()
DOCS = os.path.join(ROOT, "docs")

# ---------------------------------------------------------------- collectors
ATTR_RE = re.compile(r"""\b(?:src|href|data-src)\s*=\s*["']([^"']+)["']""", re.I)
FETCH_RE = re.compile(r"""fetch\(\s*["']([^"']+)["']""")

IGNORE_PREFIXES = ("http://", "https://", "data:", "#", "mailto:", "javascript:")


def html_files():
    out = []
    for dirpath, _dirnames, filenames in os.walk(DOCS):
        for f in sorted(filenames):
            if f.endswith(".html"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def js_files():
    out = []
    for dirpath, _dirnames, filenames in os.walk(DOCS):
        for f in sorted(filenames):
            if f.endswith(".js"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def resolve_ref(ref, container, js_document_base=False):
    """Resolve a reference. Returns (status, resolved_path_or_None).
    status: 'ok' | 'escape' | 'absolute' | 'ignore'
    js_document_base: fetch() inside a browser JS file resolves against the
    DOCUMENT base (docs/ — where index.html lives), not the script's own URL.
    For this repo docs/ is flat (no js in subdirs that fetch), so docs-root
    resolution is tried first, file-dir second as fallback."""
    if ref.startswith(IGNORE_PREFIXES):
        return "ignore", None
    if ref.startswith("//"):
        return "absolute", None  # protocol-relative — flagged separately
    if ref.startswith("/"):
        return "absolute", None
    file_rel = posixpath.normpath(posixpath.join(
        os.path.relpath(os.path.dirname(container), ROOT).replace(os.sep, "/"),
        ref))
    # candidate 1: document base (docs/) — where browser fetch() resolves;
    # candidate 2: file-relative (covers non-flat layouts / <script src>)
    cands = [posixpath.normpath(posixpath.join("docs", ref))]
    if cands[0] != file_rel:
        cands.append(file_rel)
    for c in cands:
        if c.startswith("docs/"):
            return "ok", os.path.join(ROOT, c)      # under docs/ — must exist
    return "escape", os.path.normpath(os.path.join(ROOT, file_rel))


# --------------------------------------------------- (a) reference resolution
print("=" * 74)
print("(a) reference resolution — src/href/data-src in *.html, fetch() in *.js")
print("=" * 74)
n_refs = 0
for path in html_files() + js_files():
    is_html = path.endswith(".html")
    text = open(path, encoding="utf-8", errors="replace").read()
    refs = ATTR_RE.findall(text) if is_html else []
    refs += FETCH_RE.findall(text)
    for ref in sorted(set(refs)):
        n_refs += 1
        status, target = resolve_ref(ref, path, js_document_base=not is_html)
        if status == "ignore":
            continue
        if status == "absolute":
            err(f"{rel(path)}: absolute path reference {ref!r} — "
                f"Pages serves the site under /<repo>/, use a relative path")
            continue
        if status == "escape":
            if target and os.path.isfile(target):
                warn(f"{rel(path)}: {ref!r} escapes docs/ -> {rel(target)} "
                     f"(exists in repo; deliberate file:// upgrade path — "
                     f"404s on Pages and the in-page snapshot fallback engages)")
            else:
                err(f"{rel(path)}: {ref!r} escapes docs/ AND is missing "
                    f"(resolved {target})")
            continue
        if target and os.path.isfile(target):
            print(f"  ok   {rel(path):42s} -> {ref}")
        else:
            err(f"{rel(path)}: reference {ref!r} does not resolve "
                f"(looked for {rel(target) if target else ref})")
print(f"  ... {n_refs} references scanned\n")

# ------------------------------------------------------- (b) CDN / abs audit
print("=" * 74)
print("(b) no absolute /paths, no http:// , no protocol-relative // in code")
print("=" * 74)
BAD_PATTERNS = [
    (re.compile(r"""\b(?:src|href)\s*=\s*["']/(?!/)""", re.I), "absolute /path"),
    (re.compile(r"""\b(?:src|href)\s*=\s*["']//""", re.I), "protocol-relative //"),
    (re.compile(r"""\b(?:src|href)\s*=\s*["']http://""", re.I), "insecure http://"),
    (re.compile(r"""fetch\(\s*["']http://"""), "insecure fetch('http://"),
]
n_bad = 0
for path in html_files() + js_files():
    for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        for rx, label in BAD_PATTERNS:
            if rx.search(line):
                n_bad += 1
                err(f"{rel(path)}:{i}: {label}: {line.strip()[:90]}")
if n_bad == 0:
    print("  clean — no absolute, insecure or CDN references in code contexts")
print()

# --------------------------------------------------------- (c) weights files
print("=" * 74)
print("(c) weights presence — docs/weights/web_model.bin + manifest.json")
print("=" * 74)
binp = os.path.join(DOCS, "weights", "web_model.bin")
manp = os.path.join(DOCS, "weights", "manifest.json")
if os.path.isfile(binp) and os.path.getsize(binp) > 0:
    print(f"  ok   web_model.bin  ({os.path.getsize(binp):,} bytes)")
else:
    err("docs/weights/web_model.bin missing or empty — the playground brain "
        "cannot load on Pages")
if os.path.isfile(manp):
    try:
        man = json.load(open(manp, encoding="utf-8"))
        keys = {k: man[k] for k in ("num_params", "total_bytes", "step",
                                    "tokenizer_version") if k in man}
        print(f"  ok   manifest.json (valid JSON: {keys})")
    except Exception as e:
        err(f"docs/weights/manifest.json does not parse: {e}")
else:
    err("docs/weights/manifest.json missing")
print()

# ----------------------------------------------------- (d) node --check *.js
print("=" * 74)
print("(d) every *.js under docs/ parses as JavaScript (node --check)")
print("=" * 74)
node = shutil.which("node")
if node is None:
    warn("node not found — skipping JS syntax check")
else:
    files = js_files()
    n_pass = 0
    for path in files:
        r = subprocess.run([node, "--check", path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            n_pass += 1
        else:
            err(f"{rel(path)}: node --check FAILED:\n"
                f"{r.stderr.decode('utf-8', 'replace').strip()[:400]}")
    print(f"  {n_pass}/{len(files)} files parse cleanly")
print()

# ----------------------------------------------------- (e) index.html assets
print("=" * 74)
print("(e) docs/index.html script/link/iframe files, one by one")
print("=" * 74)
idx = os.path.join(DOCS, "index.html")
for m in re.finditer(
        r"""<(script[^>]*\bsrc|link[^>]*\bhref|iframe[^>]*\bdata-src)\s*=\s*["']([^"']+)["']""",
        open(idx, encoding="utf-8").read(), re.I):
    tag, ref = m.group(1).split()[0].lower(), m.group(2)
    status, target = resolve_ref(ref, idx)
    if status == "ignore":
        continue
    ok = status == "ok" and target and os.path.isfile(target)
    if ok:
        print(f"  ok   <{tag}> {ref}")
    else:
        err(f"index.html <{tag}> {ref!r} does not resolve under docs/")
print()

# ------------------------------------------------------------- (f) docs size
print("=" * 74)
print("(f) total docs/ size vs 1 GiB Pages limit")
print("=" * 74)
total = 0
per_dir = {}
for dirpath, _dirnames, filenames in os.walk(DOCS):
    for f in filenames:
        s = os.path.getsize(os.path.join(dirpath, f))
        total += s
        d = os.path.relpath(dirpath, DOCS)
        per_dir[d] = per_dir.get(d, 0) + s
print(f"  docs/ total: {total:,} bytes ({total / (1 << 20):.1f} MiB) "
      f"= {100.0 * total / PAGES_LIMIT_BYTES:.3f}% of the 1 GiB Pages limit")
for d, s in sorted(per_dir.items(), key=lambda kv: -kv[1])[:8]:
    print(f"    {('.' if d == '.' else d):24s} {s / (1 << 20):8.2f} MiB")
if total >= PAGES_LIMIT_BYTES:
    err("docs/ exceeds the 1 GiB Pages limit")
print()

# ------------------------------------------------------------------- summary
print("=" * 74)
print(f"AUDIT SUMMARY — {len(ERRORS)} error(s), {len(WARNS)} warning(s)")
print("=" * 74)
for w in WARNS:
    print(f"  WARN  {w}")
for e in ERRORS:
    print(f"  ERROR {e}")
if not ERRORS:
    print("  RESULT: CLEAN — docs/ is ready to ship to GitHub Pages")
    sys.exit(0)
print("  RESULT: FAIL — fix the errors above and re-run")
sys.exit(1)
