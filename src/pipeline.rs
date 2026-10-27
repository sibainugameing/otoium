//! Per-file automatic pipeline: decode -> features -> pronunciation choice -> align (2 engines) ->
//! refine -> parameters -> validate -> (bounded) re-estimate -> confidence.
use crate::align::{AcousticAligner, Aligner, Alignment, HeuristicAligner};
use crate::confidence::{self, ConfClass, ConfInputs, Confidence};
use crate::dsp::{extract, ExtractCfg, FeatureTrack};
use crate::error::{Error, Result};
use crate::format::{BankFormat, ParsedAlias};
use crate::oto::OtoEntry;
use crate::params::{compute, sanitize, OtoParams, Policy};
use crate::refine::refine;
use crate::store::SaveOptions;
use crate::validate::{count_errors, has_errors, validate_alignment, validate_params, Issue, Severity};
use crate::wav::read_wav;
use std::path::Path;

pub struct Config {
    pub policy: Policy,
    pub jobs: usize,
    pub recursive: bool,
    /// Hard cap on estimation attempts per file (primary, alternate, heuristic).
    pub max_attempts: usize,
    pub refine_window_ms: f64,
    pub format_hint: Option<BankFormat>,
    pub store: SaveOptions,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            policy: Policy::default(),
            jobs: 1,
            recursive: false,
            max_attempts: 3,
            refine_window_ms: 15.0,
            format_hint: None,
            store: SaveOptions::default(),
        }
    }
}

/// Swappable alignment engines.
pub struct Engine {
    pub primary: Box<dyn Aligner>,
    pub alternate: Box<dyn Aligner>,
    pub fallback: Box<dyn Aligner>,
}

impl Engine {
    pub fn acoustic() -> Self {
        Self {
            primary: Box::new(AcousticAligner::primary()),
            alternate: Box::new(AcousticAligner::alternate()),
            fallback: Box::new(HeuristicAligner),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    ExistingOto,
    IndexCsv,
    Filename,
}

impl Source {
    pub fn label(&self) -> &'static str {
        match self {
            Source::ExistingOto => "existing oto.ini",
            Source::IndexCsv => "index.csv",
            Source::Filename => "filename",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PronCandidate {
    pub alias: String,
    pub parsed: ParsedAlias,
    pub source: Source,
}

#[derive(Debug, Clone)]
pub struct FileResult {
    pub file: String,
    pub entry: Option<OtoEntry>,
    pub confidence: Confidence,
    pub strategy: String,
    pub issues: Vec<Issue>,
    pub notes: Vec<String>,
    /// Final boundaries as (label, ms) for the report.
    pub boundaries: Vec<(String, f64)>,
    pub failure: Option<String>,
    pub attempts: usize,
}

impl FileResult {
    pub fn failed(file: &str, why: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            entry: None,
            confidence: confidence::failed(),
            strategy: "none".into(),
            issues: vec![],
            notes: vec![],
            boundaries: vec![],
            failure: Some(why.into()),
            attempts: 0,
        }
    }
}

struct Attempt {
    name: String,
    al: Alignment,
    params: OtoParams,
    issues: Vec<Issue>,
}

#[allow(clippy::too_many_arguments)]
fn run_attempt(
    name: &str,
    aligner: &dyn Aligner,
    coarse: &FeatureTrack,
    fine: &FeatureTrack,
    phones: &[String],
    cand: &PronCandidate,
    dur: f64,
    cfg: &Config,
) -> Result<Attempt> {
    let mut al = aligner.align(coarse, phones, dur)?;
    refine(&mut al, fine, cfg.refine_window_ms);
    let params = compute(&al, cand.parsed.kind, fine, dur, &cfg.policy)?;
    let mut issues = validate_alignment(&al, coarse, dur);
    issues.extend(validate_params(&params, dur, coarse));
    Ok(Attempt { name: name.into(), al, params, issues })
}

fn max_boundary_diff(a: &Alignment, b: &Alignment) -> Option<f64> {
    if a.segments.len() != b.segments.len() {
        return None;
    }
    let mut m = 0.0f64;
    for (x, y) in a.segments.iter().zip(b.segments.iter()) {
        m = m.max((x.start_ms - y.start_ms).abs());
    }
    m = m.max((a.end_ms() - b.end_ms()).abs());
    Some(m)
}

pub fn process_file(path: &Path, file_name: &str, cands: &[PronCandidate], cfg: &Config, engine: &Engine) -> FileResult {
    match process_inner(path, file_name, cands, cfg, engine) {
        Ok(r) => r,
        Err(e) => FileResult::failed(file_name, e.to_string()),
    }
}

fn process_inner(path: &Path, file_name: &str, cands: &[PronCandidate], cfg: &Config, engine: &Engine) -> Result<FileResult> {
    if cands.is_empty() {
        return Err(Error::Pron("no usable pronunciation candidate".into()));
    }
    let wav = read_wav(path)?;
    let dur = wav.duration_ms();
    if dur < 50.0 {
        return Err(Error::Wav(format!("too short ({dur:.0} ms)")));
    }
    if wav.peak() < 0.001 {
        return Err(Error::Wav("signal is silent (peak below -60 dBFS)".into()));
    }
    let coarse = extract(&wav.samples, wav.sample_rate, ExtractCfg::coarse());
    let fine = extract(&wav.samples, wav.sample_rate, ExtractCfg::fine());
    let mut notes: Vec<String> = Vec::new();
    if coarse.degenerate {
        notes.push(format!("low dynamic range ({:.1} dB): relative thresholds unreliable", coarse.snr_db()));
    }
    if wav.clipped_ratio > 0.001 {
        notes.push(format!("clipping: {:.2}% of samples", wav.clipped_ratio * 100.0));
    }

    // Pronunciation choice: lowest acoustic cost + small source-priority penalty.
    let mut chosen = 0usize;
    if cands.len() > 1 {
        let mut best = f64::INFINITY;
        for (rank, c) in cands.iter().enumerate() {
            let cost = match engine.primary.align(&coarse, &c.parsed.phonemes, dur) {
                Ok(a) => a.cost_per_frame + 0.02 * rank as f64,
                Err(_) => f64::INFINITY,
            };
            if cost < best {
                best = cost;
                chosen = rank;
            }
        }
        if chosen != 0 {
            notes.push(format!(
                "pronunciation conflict: chose '{}' ({}) over '{}' ({}) by acoustic fit",
                cands[chosen].alias, cands[chosen].source.label(), cands[0].alias, cands[0].source.label()
            ));
        }
    }
    let cand = &cands[chosen];
    let phones = &cand.parsed.phonemes;

    // Bounded attempts.
    let mut attempts: Vec<Attempt> = Vec::new();
    let mut errs: Vec<String> = Vec::new();
    let max = cfg.max_attempts.clamp(1, 3);
    let plan: [(&str, &dyn Aligner); 3] = [
        ("primary", engine.primary.as_ref()),
        ("alternate", engine.alternate.as_ref()),
        ("fallback", engine.fallback.as_ref()),
    ];
    for (i, (name, al)) in plan.iter().enumerate() {
        if i >= max {
            break;
        }
        // The fallback engine only runs when no earlier attempt is error-free.
        if i == 2 && attempts.iter().any(|a| !has_errors(&a.issues)) {
            break;
        }
        match run_attempt(name, *al, &coarse, &fine, phones, cand, dur, cfg) {
            Ok(a) => attempts.push(a),
            Err(e) => errs.push(format!("{name}: {e}")),
        }
    }
    if attempts.is_empty() {
        return Err(Error::Align(format!("all engines failed: {}", errs.join("; "))));
    }
    let n_attempts = attempts.len() + errs.len();
    for e in &errs {
        notes.push(format!("engine failure: {e}"));
    }

    // Selection: first error-free attempt in priority order; otherwise fewest errors.
    let sel = attempts
        .iter()
        .position(|a| !has_errors(&a.issues))
        .unwrap_or_else(|| {
            attempts
                .iter()
                .enumerate()
                .min_by_key(|(_, a)| count_errors(&a.issues))
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
    let disagreement = {
        let p = attempts.iter().find(|a| a.name == "primary");
        let q = attempts.iter().find(|a| a.name == "alternate");
        match (p, q) {
            (Some(p), Some(q)) => max_boundary_diff(&p.al, &q.al),
            _ => None,
        }
    };
    let chosen_attempt = attempts.swap_remove(sel);
    let mut params = chosen_attempt.params;
    let mut issues = chosen_attempt.issues;
    let mut sanitized = false;
    if has_errors(&issues) {
        let s = sanitize(&params, dur);
        let mut again = validate_params(&s, dur, &coarse);
        again.extend(validate_alignment(&chosen_attempt.al, &coarse, dur).into_iter().filter(|i| i.severity == Severity::Warning));
        if has_errors(&again) {
            let msgs: Vec<String> = issues.iter().filter(|i| i.severity == Severity::Error).map(|i| format!("{}: {}", i.code, i.msg)).collect();
            return Err(Error::Align(format!("no valid parameters after retries: {}", msgs.join("; "))));
        }
        notes.push("parameters sanitized into valid range after failed validation".into());
        params = s;
        issues = again;
        sanitized = true;
    }

    let conf = confidence::assess(&ConfInputs {
        disagreement_ms: disagreement,
        cost_per_frame: chosen_attempt.al.cost_per_frame,
        snr_db: coarse.snr_db(),
        clipped_ratio: wav.clipped_ratio,
        warnings: issues.iter().filter(|i| i.severity == Severity::Warning).count(),
        used_fallback_engine: chosen_attempt.name == "fallback",
        sanitized,
    });
    if conf.class == ConfClass::Low {
        notes.push("low confidence: best estimate saved; review recommended".into());
    }

    let mut boundaries = Vec::new();
    for s in &chosen_attempt.al.segments {
        boundaries.push((format!("{} start", s.phoneme), s.start_ms));
        if let Some(sub) = s.sub_ms {
            boundaries.push((format!("{} sub", s.phoneme), sub));
        }
    }
    boundaries.push(("end".into(), chosen_attempt.al.end_ms()));

    Ok(FileResult {
        file: file_name.into(),
        entry: Some(OtoEntry {
            file: file_name.into(),
            alias: cand.alias.clone(),
            offset: params.offset,
            consonant: params.consonant,
            cutoff: params.cutoff,
            preutter: params.preutter,
            overlap: params.overlap,
        }),
        confidence: conf,
        strategy: chosen_attempt.name,
        issues,
        notes,
        boundaries,
        failure: None,
        attempts: n_attempts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::parse_alias;

    fn write_wav(path: &Path, x: &[f32], sr: u32) {
        let pcm: Vec<i16> = x.iter().map(|v| (v.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
        let dl = (pcm.len() * 2) as u32;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + dl).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&sr.to_le_bytes());
        b.extend_from_slice(&(sr * 2).to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&dl.to_le_bytes());
        for s in pcm {
            b.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn end_to_end_synthetic_sh_a() {
        let d = std::env::temp_dir().join(format!("otoium_pipe_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("sha.wav");
        write_wav(&p, &crate::testutil::sh_a(44100), 44100);
        let cand = PronCandidate { alias: "sha".into(), parsed: parse_alias("sha").unwrap(), source: Source::Filename };
        let r = process_file(&p, "sha.wav", &[cand], &Config::default(), &Engine::acoustic());
        assert!(r.failure.is_none(), "{:?}", r.failure);
        let e = r.entry.unwrap();
        // truth: consonant onset 100 ms, vowel onset 220 ms -> preutter ~ 130 with 10 ms preroll.
        assert!((e.offset - 90.0).abs() < 25.0, "{e:?}");
        assert!((e.preutter - 130.0).abs() < 25.0, "{e:?}");
        assert!(e.cutoff < 0.0);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn silent_and_garbage_files_fail_cleanly() {
        let d = std::env::temp_dir().join(format!("otoium_pipe2_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("a.wav");
        write_wav(&p, &vec![0.0f32; 20000], 44100);
        let cand = PronCandidate { alias: "a".into(), parsed: parse_alias("a").unwrap(), source: Source::Filename };
        let r = process_file(&p, "a.wav", &[cand.clone()], &Config::default(), &Engine::acoustic());
        assert!(r.entry.is_none() && r.failure.is_some());
        let g = d.join("g.wav");
        std::fs::write(&g, b"garbage").unwrap();
        let r = process_file(&g, "g.wav", &[cand], &Config::default(), &Engine::acoustic());
        assert!(r.entry.is_none());
        std::fs::remove_dir_all(&d).ok();
    }
}
