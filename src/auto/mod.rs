/// Automatic boundary estimation for UTAU voicebank files.
///
/// # Implementation status
///
/// Phase 1 (implemented):
///   - Silence/onset detection via RMS energy thresholding
///   - Simple heuristic parameter assignment from onset/offset
///
/// Phase 2 (planned, not yet implemented):
///   - Phoneme label parsing (hiragana/katakana/romaji)
///   - MFCC feature extraction
///   - HMM-based phoneme alignment (Viterbi decoding)
///   - Per-phoneme-class boundary refinement
///
/// Note: Phase 2 requires pre-trained HMM models for Japanese phonemes.
/// Training data, initialization procedures, and model licensing must be
/// determined before implementation. This module documents known unknowns
/// explicitly rather than shipping placeholder "high-precision" results.

use crate::audio::WavData;
use crate::oto::{EntryStatus, OtoEntry};

/// Confidence level of an estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Confidence {
    /// Simple energy-based heuristic only. High uncertainty.
    Low,
    /// Energy + spectral features. Moderate uncertainty.
    Medium,
    /// Full HMM alignment. Lower uncertainty (but not validated).
    High,
}

/// Result of automatic boundary estimation for one WAV file.
#[derive(Debug, Clone)]
pub struct EstimationResult {
    pub wav_name: String,
    pub entries: Vec<OtoEntry>,
    pub confidence: Confidence,
    pub warnings: Vec<String>,
}

/// Estimate oto.ini parameters for a single WAV using simple silence detection.
///
/// This is Phase 1 only: onset and offset are detected by RMS energy,
/// and the consonant/preutterance/overlap are assigned by fixed ratios
/// of the voiced region. This is NOT phoneme alignment.
///
/// # Accuracy
/// Not validated against a ground-truth dataset. Do not represent as high-precision.
pub fn estimate_simple(
    wav: &WavData,
    wav_name: &str,
    alias_hint: Option<&str>,
) -> EstimationResult {
    const SILENCE_THRESHOLD: f32 = 0.02; // RMS threshold for onset/offset
    const WINDOW_MS: f64 = 10.0;

    let onset_ms = wav
        .find_onset_ms(SILENCE_THRESHOLD, WINDOW_MS)
        .unwrap_or(0.0);
    let offset_ms = wav
        .find_offset_ms(SILENCE_THRESHOLD, WINDOW_MS)
        .unwrap_or(wav.duration_ms);

    let voiced_dur = (offset_ms - onset_ms).max(0.0);
    let mut warnings = Vec::new();

    if voiced_dur < 20.0 {
        warnings.push(format!(
            "Very short voiced region ({:.0}ms): file may be silent or threshold too high",
            voiced_dur
        ));
    }

    // Heuristic parameter assignment (these ratios are illustrative, not validated):
    //   overlap      = 10% of voiced duration
    //   preutterance = 30% of voiced duration  (cursor point)
    //   consonant    = 50% of voiced duration  (end of consonant burst)
    //   cutoff       = relative to end of WAV
    let overlap = (voiced_dur * 0.10).min(50.0).max(5.0);
    let preutterance = (voiced_dur * 0.30).max(overlap + 5.0);
    let consonant = voiced_dur * 0.50;
    let cutoff = OtoEntry::abs_end_to_cutoff(offset_ms, wav.duration_ms);

    let stem = alias_hint.unwrap_or_else(|| {
        wav_name
            .strip_suffix(".wav")
            .or_else(|| wav_name.strip_suffix(".WAV"))
            .unwrap_or(wav_name)
    });

    // Strip leading underscore that UTAU voicebanks conventionally use
    let alias = stem.trim_start_matches('_').to_string();

    let entry = OtoEntry {
        wav_file: wav_name.to_string(),
        alias,
        offset: onset_ms,
        consonant,
        cutoff,
        preutterance,
        overlap,
        source_line: None,
        status: EntryStatus::Estimated {
            confidence: match voiced_dur {
                d if d > 50.0 => 0.5,
                _ => 0.3,
            },
        },
    };

    EstimationResult {
        wav_name: wav_name.to_string(),
        entries: vec![entry],
        confidence: Confidence::Low,
        warnings,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planned: phoneme label parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a Japanese voicebank filename into phoneme labels.
///
/// # Status: PLANNED – not implemented.
///
/// UTAU voicebanks use conventions like:
///   _あ.wav  → phoneme /a/
///   _かあ.wav → phonemes /k a/  (VCV: preceding vowel + CV)
///   _- か.wav → CV with leading silence
///
/// Parsing rules depend on the voicebank format (CV, VCV, CVVC, etc.)
/// and cannot be assumed without knowing the format. This function
/// currently returns None for all inputs.
pub fn parse_phoneme_label(_filename_stem: &str) -> Option<Vec<String>> {
    // TODO: implement hiragana/katakana/romaji parser
    None
}
