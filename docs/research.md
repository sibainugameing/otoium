# Research log

**Status: NOT PERFORMED.** The environment had no network, so the Moresampler (hungrierr/Moresampler) and
OpenMoresampler (LingYi0110/OpenMoresampler) repositories, changelogs and any model files were not read.
Nothing below is confirmed fact; do not treat it as research.

## To verify (checklist)
- [ ] Whether official Moresampler ships/loads a learned alignment model; where it is stored; its format and how it is loaded.
- [ ] How phoneme boundaries are obtained and how they map to offset/consonant/cutoff/preutterance/overlap.
- [ ] `index.csv` exact column layout and how it maps to phoneme sequences (this repo assumes `file,alias`).
- [ ] CV / VCV / CVVC handling and known bugs/limits.
- [ ] Do not infer "no model" from small binary size; inspect source, side files and runtime loads.
- [ ] Official vs unofficial (OpenMoresampler) behaviour recorded separately.

## Alignment technology candidates (to evaluate, none tried)
Learned forced alignment (e.g. wav2vec2/CTC-based, Montreal-Forced-Aligner-style HMM-GMM), Japanese singing-voice
aligners, hybrid learned + acoustic refinement. For each: Japanese support, phone-level output, CPU-only inference,
macOS ARM + Ubuntu, offline use, model size, code/model licenses, evaluation method. Recognition accuracy ≠ boundary accuracy.

## Recalled (unverified) UTAU semantics used by this code
See `docs/oto-parameters.md`.
