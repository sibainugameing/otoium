//! Confidence from measurable evidence. Component thresholds are UNCALIBRATED: they have not been
//! compared against measured boundary error (no ground-truth data was available). `calibrated` is false
//! until docs/accuracy.md contains a measured score-vs-error relationship.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfClass {
    High,
    Medium,
    Low,
    Failed,
}

impl ConfClass {
    pub fn label(&self) -> &'static str {
        match self {
            ConfClass::High => "high",
            ConfClass::Medium => "medium",
            ConfClass::Low => "low",
            ConfClass::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfInputs {
    /// Max boundary disagreement (ms) between two independent engines, if both succeeded.
    pub disagreement_ms: Option<f64>,
    pub cost_per_frame: f64,
    pub snr_db: f64,
    pub clipped_ratio: f64,
    pub warnings: usize,
    pub used_fallback_engine: bool,
    pub sanitized: bool,
}

#[derive(Debug, Clone)]
pub struct Confidence {
    pub class: ConfClass,
    pub score: f64,
    pub calibrated: bool,
    pub components: Vec<(&'static str, f64)>,
}

fn ramp(x: f64, bad: f64, good: f64) -> f64 {
    // 1.0 at/below `good`, 0.0 at/above `bad` when bad > good; mirrored otherwise.
    ((x - bad) / (good - bad)).clamp(0.0, 1.0)
}

pub fn failed() -> Confidence {
    Confidence { class: ConfClass::Failed, score: 0.0, calibrated: false, components: vec![] }
}

pub fn assess(i: &ConfInputs) -> Confidence {
    let agree = match i.disagreement_ms {
        Some(d) => ramp(d, 30.0, 3.0),
        None => 0.3,
    };
    let fit = ramp(i.cost_per_frame, 0.35, 0.08);
    let snr = ramp(i.snr_db, 10.0, 30.0);
    let clip = ramp(i.clipped_ratio, 0.02, 0.0);
    let mut score = 0.45 * agree + 0.30 * fit + 0.15 * snr + 0.10 * clip;
    score -= 0.03 * i.warnings as f64;
    if i.sanitized {
        score -= 0.2;
    }
    let score = score.clamp(0.0, 1.0);
    let mut class = if score >= 0.75 {
        ConfClass::High
    } else if score >= 0.5 {
        ConfClass::Medium
    } else {
        ConfClass::Low
    };
    if i.used_fallback_engine && class == ConfClass::High {
        class = ConfClass::Medium;
    }
    Confidence {
        class,
        score,
        calibrated: false,
        components: vec![("agreement", agree), ("acoustic_fit", fit), ("snr", snr), ("clipping", clip)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ConfInputs {
        ConfInputs {
            disagreement_ms: Some(2.0),
            cost_per_frame: 0.05,
            snr_db: 40.0,
            clipped_ratio: 0.0,
            warnings: 0,
            used_fallback_engine: false,
            sanitized: false,
        }
    }

    #[test]
    fn good_evidence_is_high() {
        assert_eq!(assess(&base()).class, ConfClass::High);
    }

    #[test]
    fn disagreement_lowers_confidence() {
        let mut b = base();
        b.disagreement_ms = Some(60.0);
        assert!(assess(&b).score < assess(&base()).score);
        assert_ne!(assess(&b).class, ConfClass::High);
    }

    #[test]
    fn fallback_caps_at_medium_and_flag_uncalibrated() {
        let mut b = base();
        b.used_fallback_engine = true;
        let c = assess(&b);
        assert_eq!(c.class, ConfClass::Medium);
        assert!(!c.calibrated);
    }
}
