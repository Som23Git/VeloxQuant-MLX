# Venue Recommendation Memo — VeloxQuant-MLX

**Paper profile:** strong, reproducible *systems/engineering* contribution (a Metal-accelerated KV-cache quantization suite for Apple Silicon) with **limited algorithmic novelty** (7 of 9 methods are re-implementations of published work) and **no downstream-task evaluation** (quality = cosine similarity + a coherence proxy). One verified standout systems result (13×/98% Metal quantize kernel) and one genuinely fresh algorithm (SpectralQuant).

> **Deadlines shift every year — verify all dates on each venue's CFP before relying on them.** Dates below are typical cycles, not commitments.

## Ranked options

| Rank | Venue | Fit | Novelty bar for *this* paper | Page limit (typical) | Cycle (verify!) | Clears bar? |
|---|---|---|---|---|---|---|
| **1** | **MLSys — Artifact / Systems track**, or an MLSys-affiliated workshop (e.g. on-device / efficient inference) | **Excellent.** MLSys explicitly values systems contributions, kernels, and reproducibility over algorithmic novelty. The Metal kernel + unified suite + 12-model sweep is exactly its lane. | Systems novelty accepted; algorithmic novelty not required. The "re-implementation" framing is fine *if* the systems delta (Metal kernel, unified API, unified-memory eval) is the headline. | 10–12 pp | Abstracts ~Oct, papers ~Nov; workshops spring | **Yes — main track is a stretch without downstream eval; the workshop is a clean fit now.** |
| **2** | **EuroMLSys (EuroSys workshop)** | **Excellent first target.** Designed for systems-y ML work, on-device/edge inference, early but solid results. | Low–moderate; welcomes engineering + measurement papers. | 6–8 pp | Submissions ~Jan–Feb for spring | **Yes — most realistic first acceptance.** |
| **3** | **NeurIPS/ICML Efficient-ML / ENLSP / on-device workshops** | **Very good.** KV-cache compression is squarely on-topic; workshops accept re-implementation + systems + negative results. | Low; non-archival, fast feedback, good for visibility. | 4–6 pp (ext. abstract) | Summer (NeurIPS) / spring (ICML) | **Yes.** |
| **4** | **JMLR MLOSS / SoftwareX / JOSS** (software-artifact venues) | **Good, orthogonal.** The library *is* the contribution; these reward open-source tooling with tests + docs. | N/A — judged on software quality, not novelty. The 314-test suite, docs site, and PyPI package qualify. | short | rolling | **Yes — do this in parallel; it's nearly free.** |
| **5** | **EMNLP/ACL — Demo track** (or Efficient Methods) | **Moderate.** Demo track suits a plug-in-3-lines tool; main Efficient-Methods track would demand task accuracy (perplexity/LongBench). | Demo: low. Main: high — needs downstream eval you don't have. | Demo 6 pp | EMNLP spring/summer; ACL winter | **Demo: yes. Main: not yet.** |
| **6** | **arXiv preprint** (baseline for all of the above) | **Always.** Establishes the record, dated, citable; pairs with the blog/landing page. | None. | n/a | now | **Yes — do immediately.** |

## Top recommendation
**Primary: submit to EuroMLSys (or an MLSys-affiliated on-device/efficient-inference workshop) as the first archival target, and file the library to JMLR MLOSS/JOSS in parallel.** This pair matches the profile exactly — a measured, reproducible Apple-Silicon systems artifact — and does not require the downstream-accuracy results you currently lack. Post the arXiv preprint first.

**Fallback / aspirational: MLSys main track**, but only after closing the evaluation gap (below). The Metal kernel result is main-track-worthy; the missing task accuracy and the non-functional RaBitQ search are what a reviewer would reject on today.

## Is a workshop/demo the right first target vs. a main conference? — Yes.
Given (a) re-implementation-heavy novelty and (b) no downstream eval, a **workshop or demo is the realistic first acceptance**, with a main-track systems submission as a v2 once the gaps close. Trying the main track first risks a desk-or-review reject that you'd have to live down.

## Concrete submission plan
1. **Now:** post `paper.tex` to arXiv (cs.LG / cs.DC). Fix the README badges first (231/243/314 tests; scope "13×" to quantize; reconcile SpectralQuant 5.33×/5.95×) so the artifact matches the paper.
2. **Parallel:** submit the library to **JOSS** (fastest) or JMLR MLOSS — near-free given the existing tests/docs.
3. **First archival paper:** **EuroMLSys** or an **MLSys/NeurIPS efficient-inference workshop** with the current 8–10 pp draft. Lead with the Metal kernel + unified-memory framing.
4. **Before any main-track (MLSys) attempt — close the gap:**
   - Add a **WikiText-2 perplexity sweep** across bit-widths/methods (you already have a single Falcon3 fp16 PPL=20.16 hook).
   - Add **LongBench or needle-in-a-haystack** for at least the 3–4 headline configs.
   - Either **fix RaBitQ search recall** or **drop the search story** and present RaBitQ purely as memory compression (recommended — it's faster to cut than to fix).
   - Broaden throughput beyond a single M4 (one more M-series tier) and to longer contexts.
5. **v2 → MLSys main track** once 4 is done; cite the workshop/preprint.
