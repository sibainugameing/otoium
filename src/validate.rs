//! Self-validation of alignments and parameters.
use crate::align::Alignment;
use crate::dsp::FeatureTrack;
use crate::kana::{is_voiced, PhClass};
use crate::params::OtoParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct Issue {
    pub severity: Severity,
    pub code: &'static str,
    pub msg: String,
}

fn issue(sev: Severity, code: &'static str, msg: String) -> Issue {
    Issue { severity: sev, code, msg }
}

pub fn has_errors(v: &[Issue]) -> bool {
    v.iter().any(|i| i.severity == Severity::Error)
}
pub fn count_errors(v: &[Issue]) -> usize {
    v.iter().filter(|i| i.severity == Severity::Error).count()
}

pub fn validate_alignment(al: &Alignment, track: &FeatureTrack, duration_ms: f64) -> Vec<Issue> {
    use Severity::*;
    let mut v = Vec::new();
    let mut prev_end = -1e9;
    for s in &al.segments {
        if !(s.start_ms.is_finite() && s.end_ms.is_finite()) {
            v.push(issue(Error, "SEG_NONFINITE", format!("{}: non-finite boundary", s.phoneme)));
            continue;
        }
        if s.start_ms < -0.5 || s.end_ms > duration_ms + 0.5 {
            v.push(issue(Error, "SEG_OUT_OF_RANGE", format!("{}: [{:.1},{:.1}] outside [0,{:.1}]", s.phoneme, s.start_ms, s.end_ms, duration_ms)));
        }
        if s.end_ms - s.start_ms < 1.0 {
            v.push(issue(Error, "SEG_NONPOSITIVE", format!("{}: duration {:.2} ms", s.phoneme, s.end_ms - s.start_ms)));
        }
        if s.start_ms < prev_end - 0.5 {
            v.push(issue(Error, "SEG_ORDER", format!("{}: starts before previous segment ends", s.phoneme)));
        }
        prev_end = s.end_ms;
        let dur = s.end_ms - s.start_ms;
        match s.class {
            PhClass::Vowel => {
                if dur < 15.0 {
                    v.push(issue(Warning, "VOWEL_SHORT", format!("{}: {:.1} ms", s.phoneme, dur)));
                }
                let vo = track.mean_over(s.start_ms, s.end_ms, |f| f.voicing);
                if vo < 0.3 {
                    v.push(issue(Warning, "ACOUSTIC_MISMATCH", format!("vowel {} mean voicing {:.2}", s.phoneme, vo)));
                }
            }
            PhClass::Fricative if !is_voiced(&s.phoneme) => {
                let hf = track.mean_over(s.start_ms, s.end_ms, |f| f.hf_ratio);
                if hf < 0.2 {
                    v.push(issue(Warning, "ACOUSTIC_MISMATCH", format!("fricative {} mean HF ratio {:.2}", s.phoneme, hf)));
                }
            }
            _ => {}
        }
        if s.class != PhClass::Vowel && dur > 400.0 {
            v.push(issue(Warning, "CONSONANT_LONG", format!("{}: {:.0} ms", s.phoneme, dur)));
        }
    }
    v
}

pub fn validate_params(p: &OtoParams, dur: f64, track: &FeatureTrack) -> Vec<Issue> {
    use Severity::*;
    let mut v = Vec::new();
    let all = [p.offset, p.consonant, p.cutoff, p.preutter, p.overlap];
    if !all.iter().all(|x| x.is_finite()) {
        v.push(issue(Error, "PARAM_NONFINITE", "non-finite parameter".into()));
        return v;
    }
    if p.offset < 0.0 || p.offset >= dur {
        v.push(issue(Error, "OFFSET_RANGE", format!("offset {:.1} not in [0,{:.1})", p.offset, dur)));
        return v;
    }
    let end = p.used_end(dur);
    let len = end - p.offset;
    if end > dur + 0.5 {
        v.push(issue(Error, "END_BEYOND_FILE", format!("used end {:.1} > duration {:.1}", end, dur)));
    }
    if len < 30.0 {
        v.push(issue(Error, "USED_TOO_SHORT", format!("used length {:.1} ms", len)));
        return v;
    }
    if p.consonant < 0.0 || p.consonant > len + 0.5 {
        v.push(issue(Error, "CONSONANT_RANGE", format!("consonant {:.1} outside used length {:.1}", p.consonant, len)));
    }
    if p.preutter < 0.0 || p.preutter > len + 0.5 {
        v.push(issue(Error, "PREUTTER_RANGE", format!("preutter {:.1} outside used length {:.1}", p.preutter, len)));
    }
    if p.overlap < 0.0 {
        v.push(issue(Warning, "NEG_OVERLAP", format!("overlap {:.1}", p.overlap)));
    }
    if p.overlap > p.preutter + 0.5 {
        v.push(issue(Warning, "OVERLAP_GT_PREUTTER", format!("overlap {:.1} > preutter {:.1}", p.overlap, p.preutter)));
    }
    if p.consonant + 0.5 < p.preutter {
        v.push(issue(Warning, "CONSONANT_LT_PREUTTER", format!("consonant {:.1} < preutter {:.1}", p.consonant, p.preutter)));
    }
    if p.preutter > 400.0 {
        v.push(issue(Warning, "PREUTTER_LONG", format!("preutter {:.0} ms", p.preutter)));
    }
    if p.consonant > 0.8 * len {
        v.push(issue(Warning, "CONSONANT_DOMINATES", format!("consonant {:.0} of {:.0} ms", p.consonant, len)));
    }
    let lvl = track.mean_over(p.offset + p.preutter.min(len), end, |f| track.norm_energy(f.energy_db));
    if lvl < 0.15 {
        v.push(issue(Error, "SILENT_REGION", format!("mean normalised level {:.2} after preutterance", lvl)));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{extract, ExtractCfg};

    fn track() -> FeatureTrack {
        extract(&crate::testutil::sh_a(44100), 44100, ExtractCfg::coarse())
    }

    #[test]
    fn good_params_pass() {
        let p = OtoParams { offset: 90.0, consonant: 145.0, cutoff: -385.0, preutter: 130.0, overlap: 40.0 };
        let v = validate_params(&p, 570.0, &track());
        assert!(!has_errors(&v), "{:?}", v);
    }

    #[test]
    fn detects_each_gross_error() {
        let t = track();
        let bad_off = OtoParams { offset: -1.0, consonant: 0.0, cutoff: -100.0, preutter: 10.0, overlap: 0.0 };
        assert!(validate_params(&bad_off, 570.0, &t).iter().any(|i| i.code == "OFFSET_RANGE"));
        let beyond = OtoParams { offset: 100.0, consonant: 10.0, cutoff: -900.0, preutter: 10.0, overlap: 0.0 };
        assert!(validate_params(&beyond, 570.0, &t).iter().any(|i| i.code == "END_BEYOND_FILE"));
        let silent = OtoParams { offset: 0.0, consonant: 10.0, cutoff: -80.0, preutter: 5.0, overlap: 0.0 };
        assert!(validate_params(&silent, 570.0, &t).iter().any(|i| i.code == "SILENT_REGION"));
        let nan = OtoParams { offset: f64::NAN, consonant: 0.0, cutoff: 0.0, preutter: 0.0, overlap: 0.0 };
        assert!(has_errors(&validate_params(&nan, 570.0, &t)));
    }
}
