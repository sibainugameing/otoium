//! Whole-folder processing: scan, build pronunciation candidates, process files (optionally in parallel),
//! tolerate per-file failures and panics, honour cancellation, then save through `store`.
use crate::confidence::ConfClass;
use crate::error::Result;
use crate::format::{detect_format, expected_phoneme_keys, parse_alias, ParsedAlias};
use crate::oto::{Line, OtoEntry, OtoFile};
use crate::pipeline::{process_file, Config, Engine, FileResult, PronCandidate, Source};
use crate::report::{BatchReport, DirReport};
use crate::sources::read_index_csv;
use crate::store;
use crate::validate::Severity;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub struct Progress {
    pub dir: String,
    pub current: String,
    pub stage: &'static str,
    pub done: usize,
    pub total: usize,
    pub ok: usize,
    pub warn: usize,
    pub fail: usize,
    pub elapsed_secs: f64,
}

fn list_wavs(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.to_lowercase().ends_with(".wav"))
            .collect(),
        Err(_) => vec![],
    };
    v.sort();
    v
}

fn collect_dirs(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) {
    if !list_wavs(dir).is_empty() {
        out.push(dir.to_path_buf());
    }
    if recursive {
        let mut subs: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect(),
            Err(_) => vec![],
        };
        subs.sort();
        for s in subs {
            collect_dirs(&s, true, out);
        }
    }
}

fn stem(name: &str) -> String {
    match name.rfind('.') {
        Some(i) => name[..i].to_string(),
        None => name.to_string(),
    }
}

struct Shared {
    done: AtomicUsize,
    ok: AtomicUsize,
    warn: AtomicUsize,
    fail: AtomicUsize,
    total: usize,
    started: Instant,
}

pub fn run_auto(
    root: &Path,
    cfg: &Config,
    engine: &Engine,
    cancel: &AtomicBool,
    on_progress: &(dyn Fn(&Progress) + Sync),
) -> Result<BatchReport> {
    let mut dirs = Vec::new();
    collect_dirs(root, cfg.recursive, &mut dirs);
    let total: usize = dirs.iter().map(|d| list_wavs(d).len()).sum();
    let shared = Shared {
        done: AtomicUsize::new(0),
        ok: AtomicUsize::new(0),
        warn: AtomicUsize::new(0),
        fail: AtomicUsize::new(0),
        total,
        started: Instant::now(),
    };
    let mut report = BatchReport::default();
    for d in &dirs {
        if cancel.load(Ordering::Relaxed) {
            report.cancelled = true;
            break;
        }
        match process_dir(d, cfg, engine, cancel, on_progress, &shared) {
            Ok(r) => report.dirs.push(r),
            Err(e) => report.dirs.push(DirReport {
                dir: d.display().to_string(),
                notes: vec![format!("directory failed: {e}")],
                ..Default::default()
            }),
        }
    }
    if cancel.load(Ordering::Relaxed) {
        report.cancelled = true;
    }
    report.elapsed_secs = shared.started.elapsed().as_secs_f64();
    Ok(report)
}

fn build_candidates(
    name: &str,
    existing: &HashMap<String, Vec<OtoEntry>>,
    index: &HashMap<String, Vec<String>>,
    notes: &mut Vec<String>,
) -> Vec<PronCandidate> {
    let lname = name.to_lowercase();
    let mut raw: Vec<(String, Source)> = Vec::new();
    if let Some(es) = existing.get(&lname) {
        for e in es {
            let a = if e.alias.trim().is_empty() { stem(name) } else { e.alias.clone() };
            raw.push((a, Source::ExistingOto));
        }
    }
    if let Some(al) = index.get(&lname) {
        for a in al {
            raw.push((a.clone(), Source::IndexCsv));
        }
    }
    raw.push((stem(name), Source::Filename));
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (alias, source) in raw {
        if !seen.insert(alias.clone()) {
            continue;
        }
        match parse_alias(&alias) {
            Ok(parsed) => out.push(PronCandidate { alias, parsed, source }),
            Err(e) => notes.push(format!("{name}: candidate '{alias}' ({}) rejected: {e}", source.label())),
        }
    }
    out
}

fn process_dir(
    dir: &Path,
    cfg: &Config,
    engine: &Engine,
    cancel: &AtomicBool,
    on_progress: &(dyn Fn(&Progress) + Sync),
    shared: &Shared,
) -> Result<DirReport> {
    let wavs = list_wavs(dir);
    let mut rep = DirReport { dir: dir.display().to_string(), ..Default::default() };

    // existing oto.ini grouped by lower-case file name
    let mut existing: HashMap<String, Vec<OtoEntry>> = HashMap::new();
    if let Ok(b) = std::fs::read(dir.join("oto.ini")) {
        let f = OtoFile::parse(&b);
        for l in &f.lines {
            match l {
                Line::Entry(e) => existing.entry(e.file.to_lowercase()).or_default().push(e.clone()),
                Line::Bytes(_) => {}
                Line::Text(_) => {}
            }
        }
    }
    let index = read_index_csv(&dir.join("index.csv"));

    // candidates per file
    let mut cand_notes = Vec::new();
    let mut work: Vec<(String, Vec<PronCandidate>, Option<String>)> = Vec::new();
    for w in &wavs {
        let multi = existing.get(&w.to_lowercase()).map(|v| v.len() > 1).unwrap_or(false);
        if multi {
            work.push((w.clone(), vec![], Some("MULTI_ALIAS_UNSUPPORTED: several aliases share this WAV; existing entries left untouched (not implemented)".into())));
            continue;
        }
        let c = build_candidates(w, &existing, &index, &mut cand_notes);
        work.push((w.clone(), c, None));
    }
    rep.notes.extend(cand_notes);

    // format detection over top-priority candidate per file
    let top: Vec<ParsedAlias> = work.iter().filter_map(|(_, c, _)| c.first().map(|x| x.parsed.clone())).collect();
    rep.formats = detect_format(&top);
    let fmt = cfg.format_hint.unwrap_or_else(|| rep.formats.first().map(|h| h.format).unwrap_or(crate::format::BankFormat::Cv));
    let present: HashSet<String> = top.iter().map(|p| p.phonemes.join(" ")).collect();
    rep.missing_expected = expected_phoneme_keys(fmt).into_iter().filter(|k| !present.contains(k)).collect();

    // process
    let n = work.len();
    let results: Mutex<Vec<Option<FileResult>>> = Mutex::new((0..n).map(|_| None).collect());
    let next = AtomicUsize::new(0);
    let jobs = cfg.jobs.max(1).min(n.max(1));
    let dir_s = dir.display().to_string();
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let (name, cands, pre_fail) = &work[i];
                let emit = |stage: &'static str| {
                    on_progress(&Progress {
                        dir: dir_s.clone(),
                        current: name.clone(),
                        stage,
                        done: shared.done.load(Ordering::Relaxed),
                        total: shared.total,
                        ok: shared.ok.load(Ordering::Relaxed),
                        warn: shared.warn.load(Ordering::Relaxed),
                        fail: shared.fail.load(Ordering::Relaxed),
                        elapsed_secs: shared.started.elapsed().as_secs_f64(),
                    })
                };
                emit("analysing");
                let r = match pre_fail {
                    Some(why) => FileResult::failed(name, why.clone()),
                    None => catch_unwind(AssertUnwindSafe(|| process_file(&dir.join(name), name, cands, cfg, engine)))
                        .unwrap_or_else(|_| FileResult::failed(name, "internal panic while processing this file")),
                };
                match r.confidence.class {
                    ConfClass::Failed => shared.fail.fetch_add(1, Ordering::Relaxed),
                    ConfClass::Low => shared.warn.fetch_add(1, Ordering::Relaxed),
                    _ => {
                        if r.issues.iter().any(|i| i.severity == Severity::Warning) {
                            shared.warn.fetch_add(1, Ordering::Relaxed)
                        } else {
                            shared.ok.fetch_add(1, Ordering::Relaxed)
                        }
                    }
                };
                shared.done.fetch_add(1, Ordering::Relaxed);
                results.lock().unwrap()[i] = Some(r);
                emit("done");
            });
        }
    });
    let results: Vec<FileResult> = results.into_inner().unwrap().into_iter().flatten().collect();

    // duplicate (file, alias) check before saving; keep all distinct keys
    let mut entries: Vec<OtoEntry> = Vec::new();
    let mut keys = HashSet::new();
    for r in &results {
        if let Some(e) = &r.entry {
            if keys.insert(e.key()) {
                entries.push(e.clone());
            }
        }
    }
    // empty-alias guard
    entries.retain(|e| !e.alias.trim().is_empty());

    if !entries.is_empty() {
        match store::save(dir, &entries, &cfg.store) {
            Ok(s) => rep.save = Some(s),
            Err(e) => rep.notes.push(format!("SAVE FAILED, existing oto.ini preserved: {e}")),
        }
    }
    rep.results = results;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_prefer_existing_then_index_then_filename() {
        let mut existing = HashMap::new();
        existing.insert(
            "ka.wav".to_string(),
            vec![OtoEntry { file: "ka.wav".into(), alias: "か".into(), offset: 0.0, consonant: 0.0, cutoff: 0.0, preutter: 0.0, overlap: 0.0 }],
        );
        let mut index = HashMap::new();
        index.insert("ka.wav".to_string(), vec!["ka".to_string()]);
        let mut notes = vec![];
        let c = build_candidates("ka.wav", &existing, &index, &mut notes);
        assert_eq!(c.len(), 2); // "ka" from index and filename deduplicated
        assert_eq!(c[0].source, Source::ExistingOto);
        assert_eq!(c[1].source, Source::IndexCsv);
    }

    #[test]
    fn unparsable_stem_is_noted_not_fatal() {
        let mut notes = vec![];
        let c = build_candidates("qqq.wav", &HashMap::new(), &HashMap::new(), &mut notes);
        assert!(c.is_empty() && !notes.is_empty());
    }
}
