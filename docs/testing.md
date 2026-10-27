# Testing

**Status: written, never executed.** Run `cargo test` after `cargo build` succeeds. Report real results here.

| Module | What is tested |
|---|---|
| wav | 16-bit parse, garbage rejection, truncated data chunk |
| dsp | FFT vs naive DFT, FFT round trip, voicing harmonic vs noise |
| kana | youon, katakana, foreign combos, sokuon, hatsuon, chouon, romaji, tokens |
| format | alias kinds, format detection |
| sjis | kana round trip, known bytes, kanji rejected |
| oto | number format, round trip incl. CRLF, Shift_JIS with undecodable line preserved, empty/invalid fields |
| align | synthetic sh+a boundaries within 15–20 ms (thresholds set a priori), error cases, heuristic ordering |
| refine | order/contiguity kept, shift bounded |
| params | hand-computed CV / VCV / positive-cutoff cases, sanitize, unit conversion |
| validate | good params pass; offset/end/silent/NaN errors detected |
| confidence | high/low/fallback behaviour |
| store | create + idempotent re-run, manual-edit protection, foreign lines + dry run + backup, Shift_JIS output |
| pipeline | end-to-end synthetic file; silent/garbage fail cleanly |
| batch / eval / report / sources | candidate priority, stats, compare, JSON escaping, index.csv |

Not covered: real recordings, interrupted writes (power loss), cancellation, parallel race behaviour, macOS ARM/Ubuntu builds.
