//! Safe oto.ini persistence: backup, temp write, verify, atomic replace, reload verify, rollback.
//! WAV files are never opened for writing anywhere in this crate.
use crate::error::{Error, Result};
use crate::oto::{Encoding, Line, OtoEntry, OtoFile};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    /// Add entries whose (file, alias) key is absent; never modify existing entries.
    Append,
    /// Update existing entries only if they are unchanged since otoium generated them (manual edits are
    /// protected unless `overwrite_manual`); add absent ones.
    Update,
    /// Overwrite entries with matching keys regardless of origin; add absent ones. Other entries are kept.
    Replace,
}

#[derive(Debug, Clone)]
pub struct SaveOptions {
    pub mode: MergeMode,
    pub dry_run: bool,
    pub overwrite_manual: bool,
    pub default_encoding: Encoding,
    /// If a Shift_JIS file cannot encode a new alias, convert the whole file to UTF-8 instead of failing.
    pub allow_utf8_fallback: bool,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            mode: MergeMode::Update,
            dry_run: false,
            overwrite_manual: false,
            default_encoding: Encoding::ShiftJis,
            allow_utf8_fallback: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    Added,
    Updated,
    Unchanged,
    ProtectedKept,
}

#[derive(Debug, Clone)]
pub struct DiffItem {
    pub kind: DiffKind,
    pub key: (String, String),
    pub old: Option<String>,
    pub new: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SaveReport {
    pub diff: Vec<DiffItem>,
    pub backup: Option<PathBuf>,
    pub wrote: bool,
    pub notes: Vec<String>,
}

impl SaveReport {
    pub fn count(&self, k: DiffKind) -> usize {
        self.diff.iter().filter(|d| d.kind == k).count()
    }
}

const STATE_FILE: &str = ".otoium-state.tsv";

fn load_state(dir: &Path) -> HashMap<(String, String), String> {
    let mut m = HashMap::new();
    if let Ok(t) = std::fs::read_to_string(dir.join(STATE_FILE)) {
        for l in t.lines() {
            let mut it = l.splitn(3, '\t');
            if let (Some(f), Some(a), Some(line)) = (it.next(), it.next(), it.next()) {
                m.insert((f.to_string(), a.to_string()), line.to_string());
            }
        }
    }
    m
}

fn save_state(dir: &Path, st: &HashMap<(String, String), String>) -> Result<()> {
    let mut keys: Vec<_> = st.keys().collect();
    keys.sort();
    let mut s = String::new();
    for k in keys {
        if k.0.contains(['\t', '\n']) || k.1.contains(['\t', '\n']) {
            continue;
        }
        s.push_str(&format!("{}\t{}\t{}\n", k.0, k.1, st[k]));
    }
    let tmp = dir.join(format!("{STATE_FILE}.tmp"));
    std::fs::write(&tmp, s)?;
    std::fs::rename(&tmp, dir.join(STATE_FILE))?;
    Ok(())
}

fn next_backup_path(dir: &Path) -> PathBuf {
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut n = 0;
    loop {
        let p = dir.join(format!("oto.ini.bak-{ts}-{n}"));
        if !p.exists() {
            return p;
        }
        n += 1;
    }
}

fn verify_file(path: &Path, expected: &OtoFile) -> Result<()> {
    let got = OtoFile::parse(&std::fs::read(path)?);
    let a: Vec<&OtoEntry> = expected.entries().collect();
    let b: Vec<&OtoEntry> = got.entries().collect();
    if a.len() != b.len() || expected.lines.len() != got.lines.len() {
        return Err(Error::Store(format!(
            "verification failed: expected {} entries/{} lines, got {}/{}",
            a.len(),
            expected.lines.len(),
            b.len(),
            got.lines.len()
        )));
    }
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.same_values(y) {
            return Err(Error::Store(format!("verification mismatch for {}={}", x.file, x.alias)));
        }
    }
    Ok(())
}

pub fn save(dir: &Path, new_entries: &[OtoEntry], opts: &SaveOptions) -> Result<SaveReport> {
    let path = dir.join("oto.ini");
    let existing_bytes = std::fs::read(&path).ok();
    let mut file = match &existing_bytes {
        Some(b) => OtoFile::parse(b),
        None => OtoFile::new(opts.default_encoding),
    };
    let state = load_state(dir);
    let mut report = SaveReport::default();
    let mut changed = false;
    let mut generated: Vec<(String, String, String)> = Vec::new();

    for e in new_entries {
        let key = e.key();
        let pos = file.lines.iter().position(|l| matches!(l, Line::Entry(x) if x.key() == key));
        let new_line = e.to_line();
        match pos {
            None => {
                file.lines.push(Line::Entry(e.clone()));
                report.diff.push(DiffItem { kind: DiffKind::Added, key: key.clone(), old: None, new: Some(new_line.clone()) });
                generated.push((key.0, key.1, new_line));
                changed = true;
            }
            Some(i) => {
                let old = match &file.lines[i] {
                    Line::Entry(x) => x.clone(),
                    _ => unreachable!(),
                };
                if old.same_values(e) {
                    report.diff.push(DiffItem { kind: DiffKind::Unchanged, key, old: Some(old.to_line()), new: None });
                    continue;
                }
                let was_generated_unchanged = state.get(&key).map(|s| *s == old.to_line()).unwrap_or(false);
                let allow = match opts.mode {
                    MergeMode::Append => false,
                    MergeMode::Update => was_generated_unchanged || opts.overwrite_manual,
                    MergeMode::Replace => true,
                };
                if allow {
                    report.diff.push(DiffItem { kind: DiffKind::Updated, key: key.clone(), old: Some(old.to_line()), new: Some(new_line.clone()) });
                    file.lines[i] = Line::Entry(e.clone());
                    generated.push((key.0, key.1, new_line));
                    changed = true;
                } else {
                    report.diff.push(DiffItem { kind: DiffKind::ProtectedKept, key, old: Some(old.to_line()), new: Some(new_line) });
                }
            }
        }
    }

    if !changed {
        report.notes.push("no changes; oto.ini left untouched".into());
        return Ok(report);
    }
    if opts.dry_run {
        report.notes.push("dry run: nothing written".into());
        return Ok(report);
    }

    if file.encoding == Encoding::Ascii && !file.is_ascii() {
        file.encoding = opts.default_encoding;
    }
    file.final_newline = true;
    let bytes = match file.to_bytes() {
        Ok(b) => b,
        Err(e) if file.encoding == Encoding::ShiftJis && opts.allow_utf8_fallback => {
            report.notes.push(format!("Shift_JIS encoding failed ({e}); file converted to UTF-8"));
            file.encoding = Encoding::Utf8;
            file.to_bytes()?
        }
        Err(e) => return Err(e),
    };

    // 1. backup
    if existing_bytes.is_some() {
        let b = next_backup_path(dir);
        std::fs::copy(&path, &b)?;
        report.backup = Some(b);
    }
    // 2. temp write + verify
    let tmp = dir.join("oto.ini.otoium-tmp");
    std::fs::write(&tmp, &bytes)?;
    if let Err(e) = verify_file(&tmp, &file) {
        std::fs::remove_file(&tmp).ok();
        return Err(e);
    }
    // 3. atomic replace
    std::fs::rename(&tmp, &path)?;
    // 4. reload verify, roll back on failure
    if let Err(e) = verify_file(&path, &file) {
        if let Some(b) = &report.backup {
            std::fs::copy(b, &path).ok();
        }
        return Err(Error::Store(format!("post-save verification failed, restored backup: {e}")));
    }
    report.wrote = true;

    let mut st = state;
    for (f, a, l) in generated {
        st.insert((f, a), l);
    }
    if let Err(e) = save_state(dir, &st) {
        report.notes.push(format!("could not write state file: {e}"));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("otoium_store_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn entry(a: &str, off: f64) -> OtoEntry {
        OtoEntry { file: format!("{a}.wav"), alias: a.into(), offset: off, consonant: 100.0, cutoff: -300.0, preutter: 80.0, overlap: 20.0 }
    }

    #[test]
    fn create_then_idempotent_rerun() {
        let d = tmpdir("create");
        let opts = SaveOptions::default();
        let r1 = save(&d, &[entry("ka", 10.0), entry("ki", 12.0)], &opts).unwrap();
        assert!(r1.wrote && r1.count(DiffKind::Added) == 2 && r1.backup.is_none());
        let before = std::fs::read(d.join("oto.ini")).unwrap();
        let r2 = save(&d, &[entry("ka", 10.0), entry("ki", 12.0)], &opts).unwrap();
        assert!(!r2.wrote && r2.count(DiffKind::Unchanged) == 2);
        assert_eq!(std::fs::read(d.join("oto.ini")).unwrap(), before);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn manual_edit_is_protected_in_update_mode() {
        let d = tmpdir("manual");
        save(&d, &[entry("ka", 10.0)], &SaveOptions::default()).unwrap();
        // simulate a manual edit
        let p = d.join("oto.ini");
        let txt = std::fs::read_to_string(&p).unwrap().replace("ka,10,", "ka,99,");
        std::fs::write(&p, txt).unwrap();
        let r = save(&d, &[entry("ka", 11.0)], &SaveOptions::default()).unwrap();
        assert_eq!(r.count(DiffKind::ProtectedKept), 1);
        assert!(std::fs::read_to_string(&p).unwrap().contains("ka,99,"));
        let opts = SaveOptions { overwrite_manual: true, ..SaveOptions::default() };
        let r = save(&d, &[entry("ka", 11.0)], &opts).unwrap();
        assert_eq!(r.count(DiffKind::Updated), 1);
        assert!(r.backup.is_some());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn untouched_foreign_lines_and_dry_run() {
        let d = tmpdir("foreign");
        std::fs::write(d.join("oto.ini"), "# my note\r\nzz.wav=zz,1,2,3,4,5\r\n").unwrap();
        let opts = SaveOptions { dry_run: true, ..SaveOptions::default() };
        let r = save(&d, &[entry("ka", 10.0)], &opts).unwrap();
        assert!(!r.wrote);
        assert_eq!(std::fs::read_to_string(d.join("oto.ini")).unwrap(), "# my note\r\nzz.wav=zz,1,2,3,4,5\r\n");
        let r = save(&d, &[entry("ka", 10.0)], &SaveOptions::default()).unwrap();
        assert!(r.wrote && r.backup.is_some());
        let out = std::fs::read_to_string(d.join("oto.ini")).unwrap();
        assert!(out.starts_with("# my note\r\nzz.wav=zz,1,2,3,4,5\r\n") && out.contains("ka.wav=ka,10,"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn kana_alias_written_as_shift_jis_for_new_file() {
        let d = tmpdir("sjis");
        let e = OtoEntry { file: "か.wav".into(), alias: "か".into(), offset: 1.0, consonant: 2.0, cutoff: -3.0, preutter: 4.0, overlap: 5.0 };
        save(&d, &[e], &SaveOptions::default()).unwrap();
        let b = std::fs::read(d.join("oto.ini")).unwrap();
        assert_eq!(&b[0..2], &[0x82, 0xA9]); // か in Shift_JIS
        std::fs::remove_dir_all(&d).ok();
    }
}
