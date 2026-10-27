//! Boundaries -> oto.ini five parameters. Kept separate from boundary estimation on purpose.
//!
//! Semantics used (see docs/oto-parameters.md; recalled from UTAU conventions, NOT verified against a
//! primary source in this environment):
//!   offset    : ms skipped from the start of the WAV.
//!   consonant : ms from `offset` that are played unstretched (fixed region).
//!   cutoff    : end of the used region. Negative => -(length from offset); positive => ms from WAV end.
//!   preutter  : ms from `offset` to the point aligned with the note start.
//!   overlap   : ms from `offset` where the crossfade with the previous note begins.
use crate::align::Alignment;
use crate::dsp::FeatureTrack;
use crate::error::{Error, Result};
use crate::format::AliasKind;
use crate::kana::is_nucleus;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OtoParams {
    pub offset: f64,
    pub consonant: f64,
    pub cutoff: f64,
    pub preutter: f64,
    pub overlap: f64,
}

impl OtoParams {
    /// Absolute end (ms from file start) of the used region.
    pub fn used_end(&self, duration_ms: f64) -> f64 {
        if self.cutoff < 0.0 { self.offset - self.cutoff } else { duration_ms - self.cutoff }
    }
}

#[derive(Debug, Clone)]
pub struct Policy {
    /// Margin kept before the first voiced/consonant onset.
    pub preroll_ms: f64,
    /// Margin kept after the end of the last phoneme.
    pub tailroom_ms: f64,
    /// Fixed region extends this far past the anchor, capped by `consonant_tail_frac` of the anchor segment.
    pub consonant_tail_ms: f64,
    pub consonant_tail_frac: f64,
    pub negative_cutoff: bool,
    /// Fraction of the first segment's level that defines the end of the onset ramp (overlap for plain aliases).
    pub onset_level: f64,
    /// VCV/VC overlap lies this far (cap) / this fraction of the previous vowel before the consonant onset.
    pub vc_overlap_back_ms: f64,
    pub vc_overlap_frac: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            preroll_ms: 10.0,
            tailroom_ms: 5.0,
            consonant_tail_ms: 15.0,
            consonant_tail_frac: 0.25,
            negative_cutoff: true,
            onset_level: 0.5,
            vc_overlap_back_ms: 30.0,
            vc_overlap_frac: 0.5,
        }
    }
}

pub fn ms_to_samples(ms: f64, sr: u32) -> f64 {
    ms * sr as f64 / 1000.0
}
pub fn samples_to_ms(s: f64, sr: u32) -> f64 {
    s * 1000.0 / sr as f64
}

pub fn compute(al: &Alignment, kind: AliasKind, fine: &FeatureTrack, duration_ms: f64, pol: &Policy) -> Result<OtoParams> {
    let segs = &al.segments;
    if segs.is_empty() {
        return Err(Error::Align("no segments".into()));
    }
    let first_start = segs[0].start_ms;
    let last_end = segs[segs.len() - 1].end_ms;
    let offset = (first_start - pol.preroll_ms).max(0.0);

    // Anchor segment: the one whose start is aligned to the note start.
    let anchor_idx = match kind {
        AliasKind::VcLike if segs.len() >= 2 => 1,
        _ => segs
            .iter()
            .rposition(|s| is_nucleus(&s.phoneme))
            .unwrap_or(segs.len() - 1),
    };
    let anchor = &segs[anchor_idx];
    let anchor_t = anchor.start_ms;
    let preutter = (anchor_t - offset).max(0.0);

    let anchor_dur = (anchor.end_ms - anchor.start_ms).max(0.0);
    let fixed_end = if kind == AliasKind::VcLike && segs.len() >= 2 {
        segs[1].end_ms
    } else {
        anchor_t + pol.consonant_tail_ms.min(pol.consonant_tail_frac * anchor_dur)
    };
    let end_used = duration_ms.min(last_end + pol.tailroom_ms);
    let len = (end_used - offset).max(0.0);
    let consonant = (fixed_end - offset).clamp(0.0, len);
    let cutoff = if pol.negative_cutoff { -len } else { duration_ms - end_used };

    let overlap_abs = if segs.len() >= 2 && matches!(kind, AliasKind::VcvLike | AliasKind::VcLike) {
        let prev = &segs[0];
        let prev_dur = (prev.end_ms - prev.start_ms).max(0.0);
        segs[1].start_ms - pol.vc_overlap_back_ms.min(pol.vc_overlap_frac * prev_dur)
    } else {
        // Onset ramp end: first fine frame reaching `onset_level` of the first segment's mean level.
        let s0 = &segs[0];
        let seg_end = s0.end_ms.min(anchor_t.max(s0.start_ms + 1.0));
        let mean_db = fine.mean_over(s0.start_ms, seg_end, |f| f.energy_db);
        let thr = fine.floor_db + pol.onset_level * (mean_db - fine.floor_db);
        let i0 = fine.index_at(s0.start_ms);
        let i1 = fine.index_at(seg_end);
        let mut t = anchor_t;
        for i in i0..=i1 {
            if fine.frames[i].energy_db >= thr {
                t = fine.frames[i].t_ms;
                break;
            }
        }
        t.min(anchor_t)
    };
    let overlap = (overlap_abs - offset).clamp(0.0, preutter);

    Ok(OtoParams { offset, consonant, cutoff, preutter, overlap })
}

/// Force parameters into a self-consistent range (used only for best-effort output; re-validated after).
pub fn sanitize(p: &OtoParams, duration_ms: f64) -> OtoParams {
    let min_len = 30.0f64.min(duration_ms);
    let offset = p.offset.clamp(0.0, (duration_ms - min_len).max(0.0));
    let end = p.used_end(duration_ms).clamp(offset + min_len, duration_ms);
    let len = end - offset;
    let cutoff = if p.cutoff < 0.0 { -len } else { duration_ms - end };
    let preutter = p.preutter.clamp(0.0, len);
    OtoParams {
        offset,
        consonant: p.consonant.clamp(0.0, len),
        cutoff,
        preutter,
        overlap: p.overlap.clamp(0.0, preutter),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::Segment;
    use crate::dsp::{extract, ExtractCfg};
    use crate::kana::PhClass;

    fn seg(p: &str, c: PhClass, a: f64, b: f64) -> Segment {
        Segment { phoneme: p.into(), class: c, start_ms: a, end_ms: b, sub_ms: None }
    }

    fn track() -> FeatureTrack {
        let x = crate::testutil::sh_a(44100);
        extract(&x, 44100, ExtractCfg::fine())
    }

    #[test]
    fn cv_math() {
        let al = Alignment {
            segments: vec![seg("sh", PhClass::Fricative, 100.0, 220.0), seg("a", PhClass::Vowel, 220.0, 470.0)],
            cost_per_frame: 0.0,
            aligner: "t".into(),
            duration_ms: 570.0,
        };
        let p = compute(&al, AliasKind::Plain, &track(), 570.0, &Policy::default()).unwrap();
        assert!((p.offset - 90.0).abs() < 1e-9);
        assert!((p.preutter - 130.0).abs() < 1e-9); // 220 - 90
        // tail = min(15, 0.25*250)=15 -> fixed_end 235 -> consonant 145
        assert!((p.consonant - 145.0).abs() < 1e-9);
        // used_end = min(570, 470+5)=475 -> len 385 -> cutoff -385
        assert!((p.cutoff + 385.0).abs() < 1e-9);
        assert!(p.overlap >= 0.0 && p.overlap <= p.preutter);
        assert!((p.used_end(570.0) - 475.0).abs() < 1e-9);
    }

    #[test]
    fn positive_cutoff_mode() {
        let al = Alignment {
            segments: vec![seg("a", PhClass::Vowel, 50.0, 300.0)],
            cost_per_frame: 0.0,
            aligner: "t".into(),
            duration_ms: 400.0,
        };
        let pol = Policy { negative_cutoff: false, ..Policy::default() };
        let p = compute(&al, AliasKind::Plain, &track(), 400.0, &pol).unwrap();
        assert!((p.cutoff - 95.0).abs() < 1e-9); // 400 - 305
        assert!((p.used_end(400.0) - 305.0).abs() < 1e-9);
    }

    #[test]
    fn vcv_overlap_inside_previous_vowel() {
        let al = Alignment {
            segments: vec![
                seg("a", PhClass::Vowel, 20.0, 120.0),
                seg("k", PhClass::Plosive, 120.0, 180.0),
                seg("a", PhClass::Vowel, 180.0, 400.0),
            ],
            cost_per_frame: 0.0,
            aligner: "t".into(),
            duration_ms: 450.0,
        };
        let p = compute(&al, AliasKind::VcvLike, &track(), 450.0, &Policy::default()).unwrap();
        assert!((p.offset - 10.0).abs() < 1e-9);
        assert!((p.preutter - 170.0).abs() < 1e-9);
        // overlap_abs = 120 - min(30, 0.5*100)=90 -> 80 from offset
        assert!((p.overlap - 80.0).abs() < 1e-9);
    }

    #[test]
    fn sanitize_fixes_out_of_range() {
        let bad = OtoParams { offset: -5.0, consonant: 9999.0, cutoff: -9999.0, preutter: 5000.0, overlap: 7000.0 };
        let s = sanitize(&bad, 300.0);
        assert!(s.offset >= 0.0 && s.used_end(300.0) <= 300.0 + 1e-9);
        assert!(s.overlap <= s.preutter && s.preutter <= -s.cutoff);
    }

    #[test]
    fn unit_conversion_roundtrip() {
        assert!((samples_to_ms(ms_to_samples(123.456, 44100), 44100) - 123.456).abs() < 1e-9);
    }
}
