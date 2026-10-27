//! Accuracy tooling: compare a generated oto.ini against a reference, and sanity-check an oto directory.
//! Implemented: per-parameter and absolute-position errors (MAE, median, p90, large-error fraction),
//! alias coverage. NOT implemented: error-by-confidence (needs the JSON report joined by key),
//! per-phoneme-class and per-recording-condition breakdowns, Moresampler comparison runner.
use crate::oto::{OtoEntry, OtoFile};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub n: usize,
    pub mae: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
    pub large_fraction: f64,
}

pub fn stats(errors: &[f64], large_threshold_ms: f64) -> Stats {
    if errors.is_empty() {
        return Stats::default();
    }
    let mut v: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let pick = |p: f64| v[((n - 1) as f64 * p).round() as usize];
    Stats {
        n,
        mae: v.iter().sum::<f64>() / n as f64,
        median: pick(0.5),
        p90: pick(0.9),
        max: v[n - 1],
        large_fraction: v.iter().filter(|e| **e > large_threshold_ms).count() as f64 / n as f64,
    }
}

pub struct EvalReport {
    pub matched: usize,
    pub missing: Vec<(String, String)>,
    pub extra: Vec<(String, String)>,
    pub params: Vec<(&'static str, Stats)>,
    pub positions: Vec<(&'static str, Stats)>,
}

/// Absolute positions (ms from file start) implied by an entry; `dur` is needed for positive cutoff.
fn positions(e: &OtoEntry, dur: Option<f64>) -> (f64, f64, f64, Option<f64>) {
    let end = if e.cutoff < 0.0 { Some(e.offset - e.cutoff) } else { dur.map(|d| d - e.cutoff) };
    (e.offset, e.offset + e.preutter, e.offset + e.consonant, end)
}

pub fn compare(generated: &OtoFile, reference: &OtoFile, large_ms: f64) -> EvalReport {
    let g: HashMap<(String, String), &OtoEntry> = generated.entries().map(|e| (e.key(), e)).collect();
    let r: HashMap<(String, String), &OtoEntry> = reference.entries().map(|e| (e.key(), e)).collect();
    let mut missing: Vec<_> = r.keys().filter(|k| !g.contains_key(*k)).cloned().collect();
    let mut extra: Vec<_> = g.keys().filter(|k| !r.contains_key(*k)).cloned().collect();
    missing.sort();
    extra.sort();
    let (mut eo, mut ec, mut ecut, mut ep, mut eov) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut p_off, mut p_pre, mut p_con, mut p_end) = (vec![], vec![], vec![], vec![]);
    let mut matched = 0;
    for (k, ge) in &g {
        if let Some(re) = r.get(k) {
            matched += 1;
            eo.push(ge.offset - re.offset);
            ec.push(ge.consonant - re.consonant);
            ecut.push(ge.cutoff - re.cutoff);
            ep.push(ge.preutter - re.preutter);
            eov.push(ge.overlap - re.overlap);
            let (a, b, c, d) = positions(ge, None);
            let (a2, b2, c2, d2) = positions(re, None);
            p_off.push(a - a2);
            p_pre.push(b - b2);
            p_con.push(c - c2);
            if let (Some(x), Some(y)) = (d, d2) {
                p_end.push(x - y);
            }
        }
    }
    EvalReport {
        matched,
        missing,
        extra,
        params: vec![
            ("offset", stats(&eo, large_ms)),
            ("consonant", stats(&ec, large_ms)),
            ("cutoff", stats(&ecut, large_ms)),
            ("preutter", stats(&ep, large_ms)),
            ("overlap", stats(&eov, large_ms)),
        ],
        positions: vec![
            ("offset_pos", stats(&p_off, large_ms)),
            ("preutter_pos", stats(&p_pre, large_ms)),
            ("consonant_end_pos", stats(&p_con, large_ms)),
            ("used_end_pos", stats(&p_end, large_ms)),
        ],
    }
}

impl EvalReport {
    pub fn render(&self, ref_count: usize) -> String {
        let mut s = format!(
            "matched {} | missing {} ({:.1}% of reference) | extra {}\n",
            self.matched,
            self.missing.len(),
            if ref_count > 0 { 100.0 * self.missing.len() as f64 / ref_count as f64 } else { 0.0 },
            self.extra.len()
        );
        s.push_str("name               n      MAE   median      p90      max  large%\n");
        for (name, st) in self.params.iter().chain(self.positions.iter()) {
            s.push_str(&format!(
                "{:<16} {:>4} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>6.1}\n",
                name, st.n, st.mae, st.median, st.p90, st.max, st.large_fraction * 100.0
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(a: &str, off: f64, pre: f64) -> OtoEntry {
        OtoEntry { file: format!("{a}.wav"), alias: a.into(), offset: off, consonant: 100.0, cutoff: -300.0, preutter: pre, overlap: 20.0 }
    }

    fn file(es: &[OtoEntry]) -> OtoFile {
        let mut f = OtoFile::new(crate::oto::Encoding::Utf8);
        for x in es {
            f.lines.push(crate::oto::Line::Entry(x.clone()));
        }
        f
    }

    #[test]
    fn stats_basic() {
        let s = stats(&[1.0, -2.0, 3.0, 4.0, 100.0], 20.0);
        assert_eq!(s.n, 5);
        assert!((s.mae - 22.0).abs() < 1e-9);
        assert!((s.median - 3.0).abs() < 1e-9);
        assert!((s.large_fraction - 0.2).abs() < 1e-9);
    }

    #[test]
    fn compare_counts_missing_extra_and_errors() {
        let generated = file(&[e("ka", 10.0, 80.0), e("zz", 1.0, 1.0)]);
        let rf = file(&[e("ka", 14.0, 90.0), e("ki", 5.0, 50.0)]);
        let r = compare(&generated, &rf, 20.0);
        assert_eq!(r.matched, 1);
        assert_eq!(r.missing.len(), 1);
        assert_eq!(r.extra.len(), 1);
        let off = &r.params[0].1;
        assert!((off.mae - 4.0).abs() < 1e-9);
        // preutter absolute position error: (10+80) - (14+90) = -14
        assert!((r.positions[1].1.mae - 14.0).abs() < 1e-9);
    }
}
