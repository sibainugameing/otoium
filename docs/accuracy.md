# Accuracy

**No measurements exist.** No ground-truth data, no Moresampler output, no compiled binary.
Any number you may see elsewhere about this project is not from this repository.

## Protocol to follow once data is available
1. Data: hand-verified UTAU voicebanks and/or licensed phone-aligned corpora; split by recording session/speaker; record licences and ambiguity of labels.
2. Run `otoium auto --json report.json`, then `otoium eval generated/oto.ini reference/oto.ini --large-ms 20`.
3. Report per parameter and per absolute position (offset, preutterance, consonant end, used end): MAE, median, p90, max, large-error %.
4. Report alias missing/extra rates and auto-completion rate.
5. Join `report.json` with errors to measure **error by confidence class** (not implemented yet) and calibrate `confidence.rs` thresholds. Until then `calibrated=false`.
6. Break down by CV/VCV/CVVC, phoneme class, recording condition (not implemented yet).
7. Compare with Moresampler on identical inputs, recording versions/settings; do not treat its output as ground truth.
8. Iterate: classify errors → fix → regression → before/after table.
