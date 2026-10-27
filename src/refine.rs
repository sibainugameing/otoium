//! Class-pair-specific boundary refinement on a 1 ms feature track.
//! Each boundary is moved within +/- window to the position maximising an evidence score chosen by the
//! phoneme classes on either side (not a single energy threshold).
use crate::align::{Alignment, Segment};
use crate::dsp::FeatureTrack;
use crate::kana::{is_voiced, PhClass};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Slot {
    Start(usize),
    Sub(usize),
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ev {
    Lead,
    Trail,
    ClosureBurst,
    UnvoicedToVowel,
    VoicedToVowel,
    VowelToCons,
    VowelToVowel,
    Other,
    Fixed,
}

pub fn slots(al: &Alignment) -> Vec<(Slot, f64)> {
    let mut v = Vec::new();
    for (i, s) in al.segments.iter().enumerate() {
        v.push((Slot::Start(i), s.start_ms));
        if let Some(sub) = s.sub_ms {
            v.push((Slot::Sub(i), sub));
        }
    }
    v.push((Slot::End, al.end_ms()));
    v
}

pub fn apply(al: &mut Alignment, pos: &[(Slot, f64)]) {
    let n = al.segments.len();
    for (slot, t) in pos {
        match *slot {
            Slot::Start(i) => {
                al.segments[i].start_ms = *t;
                if i > 0 {
                    // segments are contiguous
                }
            }
            Slot::Sub(i) => al.segments[i].sub_ms = Some(*t),
            Slot::End => {}
        }
    }
    // contiguity: end of segment i == start of segment i+1; final end from Slot::End
    for i in 0..n {
        let end = if i + 1 < n {
            al.segments[i + 1].start_ms
        } else {
            pos.iter().find(|(s, _)| *s == Slot::End).map(|(_, t)| *t).unwrap_or(al.segments[i].end_ms)
        };
        al.segments[i].end_ms = end;
    }
}

fn evidence_kind(slot: Slot, segs: &[Segment]) -> Ev {
    match slot {
        Slot::Sub(i) => {
            if segs[i].class == PhClass::Plosive { Ev::ClosureBurst } else { Ev::Other }
        }
        Slot::End => Ev::Trail,
        Slot::Start(0) => Ev::Lead,
        Slot::Start(i) => {
            let l = &segs[i - 1];
            let r = &segs[i];
            if l.class == PhClass::Closure {
                return Ev::Fixed;
            }
            let rv = r.class == PhClass::Vowel;
            let lv = l.class == PhClass::Vowel;
            if rv && lv {
                Ev::VowelToVowel
            } else if rv && !is_voiced(&l.phoneme) {
                Ev::UnvoicedToVowel
            } else if rv {
                Ev::VoicedToVowel
            } else if lv {
                Ev::VowelToCons
            } else {
                Ev::Other
            }
        }
    }
}

fn smooth(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| {
            let a = i.saturating_sub(1);
            let b = (i + 1).min(n - 1);
            v[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
        })
        .collect()
}

pub struct RefineStats {
    pub shifts_ms: Vec<f64>,
    pub max_shift_ms: f64,
}

pub fn refine(al: &mut Alignment, fine: &FeatureTrack, window_ms: f64) -> RefineStats {
    let n = fine.frames.len();
    if n < 8 || al.segments.is_empty() {
        return RefineStats { shifts_ms: vec![], max_shift_ms: 0.0 };
    }
    let e: Vec<f64> = smooth(&fine.frames.iter().map(|f| fine.norm_energy(f.energy_db)).collect::<Vec<_>>());
    let h: Vec<f64> = smooth(&fine.frames.iter().map(|f| f.hf_ratio).collect::<Vec<_>>());
    let v: Vec<f64> = smooth(&fine.frames.iter().map(|f| f.voicing).collect::<Vec<_>>());
    let fl: Vec<f64> = smooth(&fine.frames.iter().map(|f| (f.flux / fine.flux_ref).min(1.5)).collect::<Vec<_>>());
    let hop = fine.hop_ms;
    let d = ((3.0 / hop).round() as isize).max(1);
    let at = |a: &Vec<f64>, i: isize| -> f64 { a[i.clamp(0, n as isize - 1) as usize] };
    let min_gap = 2.0;

    let orig = slots(al);
    let kinds: Vec<Ev> = orig.iter().map(|(s, _)| evidence_kind(*s, &al.segments)).collect();
    let mut out: Vec<(Slot, f64)> = Vec::with_capacity(orig.len());
    let mut shifts = Vec::new();
    for (idx, (slot, p0)) in orig.iter().enumerate() {
        let kind = kinds[idx];
        let lo_bound = out.last().map(|(_, t)| t + min_gap).unwrap_or(0.0);
        let hi_bound = orig.get(idx + 1).map(|(_, t)| t - min_gap).unwrap_or(al.duration_ms);
        if kind == Ev::Fixed {
            let t = p0.clamp(lo_bound.min(*p0), hi_bound.max(*p0));
            out.push((*slot, t));
            shifts.push(0.0);
            continue;
        }
        let lo = (p0 - window_ms).max(lo_bound);
        let hi = (p0 + window_ms).min(hi_bound);
        if lo > hi {
            out.push((*slot, *p0));
            shifts.push(0.0);
            continue;
        }
        let ja = (lo / hop).round() as isize;
        let jb = (hi / hop).round() as isize;
        let mut best_t = *p0;
        let mut best_s = f64::NEG_INFINITY;
        for j in ja..=jb {
            let de = at(&e, j + d) - at(&e, j - d);
            let dh = at(&h, j + d) - at(&h, j - d);
            let dv = at(&v, j + d) - at(&v, j - d);
            let f = at(&fl, j);
            let s = match kind {
                Ev::Lead => de,
                Ev::Trail => -de,
                Ev::ClosureBurst => de + 0.5 * dh,
                Ev::UnvoicedToVowel => dv - dh + 0.5 * de,
                Ev::VoicedToVowel => 0.5 * f + de,
                Ev::VowelToCons => -de + 0.5 * dh,
                Ev::VowelToVowel | Ev::Other => f,
                Ev::Fixed => 0.0,
            };
            let t = j as f64 * hop;
            let s = s - 0.05 * (t - p0).abs() / window_ms.max(1.0);
            if s > best_s {
                best_s = s;
                best_t = t;
            }
        }
        let t = best_t.clamp(lo.min(hi), hi);
        shifts.push(t - p0);
        out.push((*slot, t));
    }
    let max_shift = shifts.iter().fold(0.0f64, |m, s| m.max(s.abs()));
    apply(al, &out);
    RefineStats { shifts_ms: shifts, max_shift_ms: max_shift }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::{Aligner, AcousticAligner};
    use crate::dsp::{extract, ExtractCfg};

    #[test]
    fn refine_keeps_order_and_moves_toward_truth() {
        let sr = 44100;
        let x = crate::testutil::sh_a(sr);
        let dur = x.len() as f64 * 1000.0 / sr as f64;
        let coarse = extract(&x, sr, ExtractCfg::coarse());
        let fine = extract(&x, sr, ExtractCfg::fine());
        let mut al = AcousticAligner::primary().align(&coarse, &["sh".into(), "a".into()], dur).unwrap();
        let before = (al.segments[1].start_ms - 220.0).abs();
        let st = refine(&mut al, &fine, 15.0);
        for w in al.segments.windows(2) {
            assert!(w[0].start_ms < w[0].end_ms && (w[0].end_ms - w[1].start_ms).abs() < 1e-9);
        }
        let after = (al.segments[1].start_ms - 220.0).abs();
        assert!(after <= before + 5.0, "before {before} after {after}");
        assert!(st.max_shift_ms <= 15.0 + 1e-9);
    }
}
