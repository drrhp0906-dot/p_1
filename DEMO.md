# p_1 — the 3-minute live demo

Open **https://YOURUSERNAME.github.io/p_1/** (replace `YOURUSERNAME`). Every beat below is timed; total ≈ 3:00. Talk while the page loads — everything is static files, it boots in a second or two.

---

**0:00 — ABOUT: what this is (30 s).** Open the **4·ABOUT** tab first.
Say: *"This is a language model raised from raw arithmetic — no PyTorch, no APIs, no pretrained anything. The autograd engine, the tokenizer, the transformer, the optimizer, the retrieval, the verifier: all written by hand, twice, in JavaScript and Python, and pinned to each other by parity tests. You're talking to the weights in your browser right now, not to a server."*
Point at the **metrics table**: "And at the bottom — the experiment's numbers. I'll come back to those."

**0:30 — LIVE TRAINING: raise the baby while you talk (60 s).** Switch to **2·LIVE TRAINING**, press **start** inside the frame.
Say, over the falling loss: *"While I talk, a char-level transformer — built on the same handmade engine — is training live in this tab. That's the real loss curve, computed in your browser. Watch the samples turn from noise into question-and-answer shapes."* (Let it run; do not stop it — the falling numbers ARE the demo.)
If asked about speed: "The Colab big sibling — 4.3M params — trains overnight on a free GPU and is what answers your questions in the next tab."

**1:30 — PLAYGROUND: the grounded answer (30 s).** Switch to **1·PLAYGROUND**, select **GROUNDED**, type **"How do I treat a minor burn?"**, Generate.
Point at: the extractive answer ("cool or lukewarm running water for at least 20 minutes…"), the **[n] citation chips** (tap one — it scrolls to the retrieved corpus chunk), and the retrieval footer ("top-3 bm25 over 4,278 chunks").
Say: *"GROUNDED answers only from sentences retrieved out of the corpus it was raised on — every claim carries its receipt."*

**2:00 — the same question, RAW (20 s).** Toggle **RAW** (temp 0.8), same question, Generate.
Say: *"Same brain, network alone — it invents. With a 4-million-parameter model the inventions are obvious. At scale they're not; that's the problem this lab measures."* (If the demo brain is still the stub — see the note below — the RAW output is babble; say exactly that: "an under-trained brain babbles, which is RAW honesty of a kind.")

**2:20 — TRAP LAB: a myth, RAW vs GROUNDED (30 s).** Switch to **3·TRAP LAB**, press **run the traps** (40 questions — with the stub brain ~a minute; start it right after you finish the 2:00 beat if it's slow).
While the progress bar fills, say: *"Twenty folk beliefs the corpus actively corrects, twenty questions no honest corpus can settle — each run twice, every answer claim-split and verified against the evidence."*
When it lands, pick one myth row: show the claim chips — **✓ supported** (green) / **✗ unsupported** (red) / **∅ not-in-corpus** — and the **myth parroted** vs **clean** flags, RAW vs GROUNDED side by side.

**2:50 — ABOUT: the numbers + the takeaway (10 s).** Back to **4·ABOUT**, point at the metrics table and read it live (stub-run example: hallucination rate 13.5 % RAW → 1.7 % GROUNDED; refusal on unanswerables 0 % → 95 %):
*"Grounding and verification measurably reduce hallucination — on a model small enough to watch think. Full numbers in the repo."*
Close: *"Everything you just saw — engine to eval — is handmade and in the README."*

---

### If the demo brain is still the 150-step stub

The shipped `docs/weights/web_model.bin` may be the acceptance-run checkpoint (step 150 of the tiny config): PLAYGROUND GROUNDED works fully (retrieval is what carries GROUNDED), but RAW output is babble and TRAP LAB counters stay near zero — the pages banner-flag it. Hot-swap in the real Config B brain:

1. Open `colab/04_train.ipynb` in Colab, run the overnight training (or resume — checkpoints live in Drive `MyDrive/p_1/checkpoints/` and a dead session resumes bit-exact).
2. In `colab/06_parity.ipynb`, set the repo/ckpt cells and run top-to-bottom: it re-derives the parity fixtures, runs `colab/05_export.py` as a subprocess and writes the new `web_model.bin` + `manifest.json` + `web_model_data.js` straight into `docs/weights/`, then `results/parity_outputs.json` → `docs/parity_outputs_data.js`.
3. `git pull && git add -A && git commit -m "real Config B brain" && git push` — Pages redeploys in ~1 min.
4. Confirm: `docs/parity_test.html` shows the green PASS banner with the new checkpoint's step, and PLAYGROUND RAW answers in (garbled but word-shaped) English.

### Demo-day checklist

- [ ] Pages URL loads on the phone (data or wifi) — this is the DONE WHEN of the whole project.
- [ ] PLAYGROUND answers GROUNDED with citation chips.
- [ ] LIVE TRAINING starts and the loss visibly falls.
- [ ] TRAP LAB completes a full 40-question run.
- [ ] ABOUT metrics table renders (and the "demo metrics (stub)" banner is understood — or gone because you ran 08_eval).
