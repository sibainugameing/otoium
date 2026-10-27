//! Phoneme alignment. Engines are swappable via the `Aligner` trait.
//!
//! `AcousticAligner` is a *rule-based acoustic DP aligner* (no learned model). It is NOT a substitute for a
//! trained forced aligner; see docs/alignment.md for its limits.
use crate::dsp::{FeatureTrack, Frame};
use crate::error::{Error, Result};
use crate::kana::{classify, is_voiced, PhClass};

#[derive(Debug, Clone)]
pub struct Segment {
    pub phoneme: String,
    pub class: PhClass,
    pub start_ms: f64,
    pub end_ms: f64,
    /// Internal boundary: closure->burst for plosives, burst->frication for affricates.
    pub sub_ms: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Alignment {
    pub segments: Vec<Segment>,
    /// Mean DP emission cost per frame (lower = better acoustic fit). 1.0 for non-DP engines.
    pub cost_per_frame: f64,
    pub aligner: String,
    pub duration_ms: f64,
}

impl Alignment {
    pub fn start_ms(&self) -> f64 {
        self.segments.first().map(|s| s.start_ms).unwrap_or(0.0)
    }
    pub fn end_ms(&self) -> f64 {
        self.segments.last().map(|s| s.end_ms).unwrap_or(0.0)
    }
}

pub trait Aligner: Send + Sync {
    fn name(&self) -> &str;
    fn align(&self, track: &FeatureTrack, phones: &[String], duration_ms: f64) -> Result<Alignment>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum K {
    Sil,
    Vowel,
    Nasal,
    Liquid,
    Glide,
    Fric,
    Closure,
    Burst,
}

/// Prototype feature vectors [norm_energy, voicing, hf_ratio, rel_flux].
/// These are hand-set design values, NOT fitted to data (see docs/alignment.md).
fn proto(k: K, voiced: bool) -> [f64; 4] {
    match k {
        K::Sil => [0.0, 0.15, 0.3, 0.1],
        K::Vowel => [0.85, 0.9, 0.08, 0.15],
        K::Nasal => [0.45, 0.85, 0.04, 0.2],
        K::Liquid => [0.6, 0.85, 0.06, 0.3],
        K::Glide => [0.6, 0.85, 0.06, 0.35],
        K::Fric => {
            if voiced { [0.5, 0.55, 0.5, 0.3] } else { [0.45, 0.15, 0.7, 0.3] }
        }
        K::Closure => {
            if voiced { [0.12, 0.6, 0.1, 0.1] } else { [0.04, 0.1, 0.25, 0.1] }
        }
        K::Burst => {
            if voiced { [0.5, 0.45, 0.45, 1.0] } else { [0.45, 0.2, 0.6, 1.0] }
        }
    }
}

/// Per-kind multipliers on the feature weights. Silence and closures are judged mainly by energy:
/// room noise is broadband, so its HF ratio / flux look like a fricative and must not drive the decision.
fn kind_weight(k: K) -> [f64; 4] {
    match k {
        K::Sil => [3.0, 0.3, 0.0, 0.0],
        K::Closure => [2.0, 1.0, 0.3, 0.2],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

fn min_ms(k: K) -> f64 {
    match k {
        K::Sil => 1.0,
        K::Vowel => 25.0,
        K::Nasal => 20.0,
        K::Liquid | K::Glide => 15.0,
        K::Fric => 25.0,
        K::Closure => 10.0,
        K::Burst => 5.0,
    }
}

struct Elem {
    kind: K,
    voiced: bool,
    phone: usize,
    sub: u8,
    self_loop: bool,
}

const NONE: usize = usize::MAX;

fn groups(sym: &str) -> Result<Vec<(K, bool)>> {
    let cls = classify(sym).ok_or_else(|| Error::Align(format!("unknown phoneme '{sym}'")))?;
    let v = is_voiced(sym);
    Ok(match cls {
        PhClass::Vowel => vec![(K::Vowel, true)],
        PhClass::Nasal => vec![(K::Nasal, true)],
        PhClass::Liquid => vec![(K::Liquid, true)],
        PhClass::Glide => vec![(K::Glide, true)],
        PhClass::Fricative => vec![(K::Fric, v)],
        PhClass::Closure => vec![(K::Closure, false)],
        PhClass::Plosive => vec![(K::Closure, v), (K::Burst, v)],
        PhClass::Affricate => vec![(K::Burst, v), (K::Fric, v)],
    })
}

fn build_elems(phones: &[String], hop_ms: f64, min1: bool) -> Result<Vec<Elem>> {
    let frames = |k: K| -> usize {
        if min1 { 1 } else { ((min_ms(k) / hop_ms).ceil() as usize).max(1) }
    };
    let mut v = vec![Elem { kind: K::Sil, voiced: false, phone: NONE, sub: 0, self_loop: true }];
    for (pi, sym) in phones.iter().enumerate() {
        for (gi, (k, voiced)) in groups(sym)?.into_iter().enumerate() {
            let n = frames(k);
            for j in 0..n {
                v.push(Elem { kind: k, voiced, phone: pi, sub: gi as u8, self_loop: j + 1 == n });
            }
        }
    }
    v.push(Elem { kind: K::Sil, voiced: false, phone: NONE, sub: 0, self_loop: true });
    Ok(v)
}

fn fvec(track: &FeatureTrack, f: &Frame) -> [f64; 4] {
    [
        track.norm_energy(f.energy_db).min(1.2),
        f.voicing,
        f.hf_ratio,
        (f.flux / track.flux_ref).min(1.5),
    ]
}

pub struct AcousticAligner {
    name: &'static str,
    w: [f64; 4],
}

impl AcousticAligner {
    pub fn primary() -> Self {
        Self { name: "acoustic-dp/primary", w: [1.0, 1.0, 0.8, 0.5] }
    }
    /// Different feature weighting, used as an independent second opinion.
    pub fn alternate() -> Self {
        Self { name: "acoustic-dp/alternate", w: [1.3, 0.6, 1.2, 0.3] }
    }
}

impl Aligner for AcousticAligner {
    fn name(&self) -> &str {
        self.name
    }

    fn align(&self, track: &FeatureTrack, phones: &[String], duration_ms: f64) -> Result<Alignment> {
        if phones.is_empty() {
            return Err(Error::Align("empty phoneme list".into()));
        }
        let t = track.frames.len();
        if t < 3 {
            return Err(Error::Align("audio too short".into()));
        }
        let hop = track.hop_ms;
        let mut elems = build_elems(phones, hop, false)?;
        if elems.len() > t {
            elems = build_elems(phones, hop, true)?;
            if elems.len() > t {
                return Err(Error::Align("more phoneme states than frames".into()));
            }
        }
        let n = elems.len();
        let feats: Vec<[f64; 4]> = track.frames.iter().map(|f| fvec(track, f)).collect();
        let w = self.w;
        let emit = |ti: usize, e: &Elem| -> f64 {
            let p = proto(e.kind, e.voiced);
            let kw = kind_weight(e.kind);
            let f = &feats[ti];
            let mut s = 0.0;
            for k in 0..4 {
                let d = f[k] - p[k];
                s += w[k] * kw[k] * d * d;
            }
            s
        };
        const INF: f64 = 1e18;
        let mut prev = vec![INF; n];
        let mut cur = vec![INF; n];
        let mut back = vec![0u8; t * n];
        prev[0] = emit(0, &elems[0]);
        for ti in 1..t {
            for i in 0..n {
                let stay = if elems[i].self_loop { prev[i] } else { INF };
                let adv = if i > 0 { prev[i - 1] } else { INF };
                let (best, bk) = if adv < stay { (adv, 1u8) } else { (stay, 0u8) };
                if best >= INF {
                    cur[i] = INF;
                } else {
                    cur[i] = best + emit(ti, &elems[i]);
                    back[ti * n + i] = bk;
                }
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        let total = prev[n - 1];
        if total >= INF {
            return Err(Error::Align("no valid alignment path".into()));
        }
        let mut path = vec![0usize; t];
        let mut i = n - 1;
        for ti in (1..t).rev() {
            path[ti] = i;
            if back[ti * n + i] == 1 {
                i -= 1;
            }
        }
        path[0] = i;

        let mut first = vec![NONE; phones.len()];
        let mut last = vec![0usize; phones.len()];
        let mut sub_first = vec![NONE; phones.len()];
        for (ti, &ei) in path.iter().enumerate() {
            let e = &elems[ei];
            if e.phone == NONE {
                continue;
            }
            if first[e.phone] == NONE {
                first[e.phone] = ti;
            }
            last[e.phone] = ti;
            if e.sub == 1 && sub_first[e.phone] == NONE {
                sub_first[e.phone] = ti;
            }
        }
        let mut segments = Vec::with_capacity(phones.len());
        for (pi, sym) in phones.iter().enumerate() {
            if first[pi] == NONE {
                return Err(Error::Align(format!("phoneme '{sym}' received no frames")));
            }
            segments.push(Segment {
                phoneme: sym.clone(),
                class: classify(sym).unwrap(),
                start_ms: (first[pi] as f64 * hop - hop / 2.0).max(0.0),
                end_ms: (last[pi] as f64 * hop + hop / 2.0).min(duration_ms),
                sub_ms: if sub_first[pi] != NONE { Some((sub_first[pi] as f64 * hop - hop / 2.0).max(0.0)) } else { None },
            });
        }
        Ok(Alignment { segments, cost_per_frame: total / t as f64, aligner: self.name.into(), duration_ms })
    }
}

/// Last-resort engine: detect the active region by energy, then split by nominal class durations.
pub struct HeuristicAligner;

fn nominal_weight(c: PhClass) -> (f64, Option<f64>) {
    // (total weight, closure/first-part fraction for two-part phonemes)
    match c {
        PhClass::Vowel => (150.0, None),
        PhClass::Nasal => (70.0, None),
        PhClass::Liquid | PhClass::Glide => (50.0, None),
        PhClass::Fricative => (100.0, None),
        PhClass::Closure => (50.0, None),
        PhClass::Plosive => (70.0, Some(0.6)),
        PhClass::Affricate => (90.0, Some(0.25)),
    }
}

impl Aligner for HeuristicAligner {
    fn name(&self) -> &str {
        "heuristic-proportional"
    }

    fn align(&self, track: &FeatureTrack, phones: &[String], duration_ms: f64) -> Result<Alignment> {
        if phones.is_empty() {
            return Err(Error::Align("empty phoneme list".into()));
        }
        let active: Vec<usize> = track
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| track.norm_energy(f.energy_db) >= 0.2)
            .map(|(i, _)| i)
            .collect();
        let (a, b) = match (active.first(), active.last()) {
            (Some(&a), Some(&b)) if b > a => (a, b),
            _ => return Err(Error::Align("no active region found".into())),
        };
        let start = (a as f64 * track.hop_ms).max(0.0);
        let end = (b as f64 * track.hop_ms + track.hop_ms).min(duration_ms);
        let mut ws = Vec::new();
        let mut tot = 0.0;
        for p in phones {
            let c = classify(p).ok_or_else(|| Error::Align(format!("unknown phoneme '{p}'")))?;
            let (w, f) = nominal_weight(c);
            ws.push((c, w, f));
            tot += w;
        }
        let mut t = start;
        let mut segments = Vec::new();
        for (p, (c, w, f)) in phones.iter().zip(ws) {
            let len = (end - start) * w / tot;
            segments.push(Segment {
                phoneme: p.clone(),
                class: c,
                start_ms: t,
                end_ms: t + len,
                sub_ms: f.map(|fr| t + len * fr),
            });
            t += len;
        }
        Ok(Alignment { segments, cost_per_frame: 1.0, aligner: self.name().into(), duration_ms })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{extract, ExtractCfg};

    #[test]
    fn dp_aligns_synthetic_sh_a() {
        let sr = 44100;
        let x = crate::testutil::sh_a(sr);
        let dur = x.len() as f64 * 1000.0 / sr as f64;
        let track = extract(&x, sr, ExtractCfg::coarse());
        let al = AcousticAligner::primary().align(&track, &["sh".into(), "a".into()], dur).unwrap();
        // truth: 100, 220, 470 ms. Thresholds chosen a priori, not tuned.
        assert!((al.segments[0].start_ms - 100.0).abs() < 15.0, "{:?}", al.segments);
        assert!((al.segments[1].start_ms - 220.0).abs() < 15.0, "{:?}", al.segments);
        assert!((al.segments[1].end_ms - 470.0).abs() < 20.0, "{:?}", al.segments);
    }

    #[test]
    fn rejects_unknown_phoneme_and_empty() {
        let x = crate::testutil::sh_a(16000);
        let track = extract(&x, 16000, ExtractCfg::coarse());
        assert!(AcousticAligner::primary().align(&track, &["qq".into()], 500.0).is_err());
        assert!(AcousticAligner::primary().align(&track, &[], 500.0).is_err());
    }

    #[test]
    fn heuristic_orders_segments() {
        let sr = 44100;
        let x = crate::testutil::sh_a(sr);
        let dur = x.len() as f64 * 1000.0 / sr as f64;
        let track = extract(&x, sr, ExtractCfg::coarse());
        let al = HeuristicAligner.align(&track, &["k".into(), "a".into()], dur).unwrap();
        assert!(al.segments[0].end_ms <= al.segments[1].start_ms + 1e-9);
        assert!(al.segments[0].sub_ms.unwrap() > al.segments[0].start_ms);
    }
}
