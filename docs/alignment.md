# Alignment and boundary estimation

## Engine: `AcousticAligner` (rule-based DP)
Per frame (5 ms): normalised energy, voicing (FFT autocorrelation), HF-energy ratio (>4 kHz), relative spectral flux.
States are expanded from the phoneme list: plosive → closure+burst, affricate → burst+frication, others 1 state;
each state has a minimum duration; leading/trailing silence states. Viterbi finds the monotonic assignment minimising
squared distance to per-class prototype vectors. Prototypes and weights are **hand-set design values, not fitted to data.**
Two weightings (`primary`, `alternate`) give a weak second opinion; their boundary disagreement feeds confidence.

## Refinement
Each boundary moves ±15 ms on a 1 ms track to maximise evidence chosen by class pair (lead/trail energy slope,
closure→burst rise, unvoiced→vowel voicing onset & HF drop, voiced→vowel flux/energy, vowel→consonant, vowel→vowel flux).
Closure-adjacent boundaries inside silence are left fixed (no acoustic evidence).

## Known limits (by construction)
* No language knowledge beyond phoneme class → cannot distinguish e.g. か/た by acoustics; relies on given text.
* Devoiced vowels, creaky voice, breathy recordings, noisy rooms, whispered or very long/short samples will degrade it.
* Two engines share the same features, so agreement overstates independence.
* Not a substitute for a trained forced aligner; plug one in via `Aligner` when a model/runtime is available.
