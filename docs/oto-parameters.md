# oto.ini parameters

**Semantics below are recalled from common UTAU conventions and were NOT verified against a primary source or tested
in UTAU/Moresampler here.** Verify before trusting output.

Format: `file.wav=alias,offset,consonant,cutoff,preutterance,overlap` (ms; fractional allowed).

| Param | Meaning (assumed) | Reference time | Computation in `params::compute` |
|---|---|---|---|
| offset | head skipped | file start | `max(0, first_start − preroll)` (preroll 10 ms) |
| consonant | unstretched region | from offset | `anchor + min(15 ms, 0.25·anchor_dur) − offset` (VC: end of consonant), clamped to used length |
| cutoff | end of used region | negative: from offset (`−(end−offset)`); positive: from file end | `end = min(dur, last_end + 5 ms)`; negative by default (`--positive-cutoff` to switch) |
| preutterance | point aligned to note start | from offset | `anchor_start − offset`; anchor = last vowel/ん segment (CV, VCV); consonant start for VC |
| overlap | crossfade start | from offset | CV: first fine frame reaching 50 % of first segment level, ≤ anchor; VCV/VC: consonant start − min(30 ms, 0.5·prev vowel) |

The anchor/overlap/consonant-tail rules are **design choices**, parameterised in `Policy`, not spec-derived. Exceptions:
overlap clamped to [0, preutter]; `sanitize` clamps everything into a self-consistent range, and results are re-validated.
Unit conversion helpers: `ms_to_samples`, `samples_to_ms`. Tests: `params::tests`. Engine compatibility: unknown.
