//! Text and JSON reports. JSON is hand-written (no serde dependency).
use crate::confidence::ConfClass;
use crate::format::FormatHypothesis;
use crate::pipeline::FileResult;
use crate::store::{DiffKind, SaveReport};
use crate::validate::Severity;

#[derive(Default)]
pub struct DirReport {
    pub dir: String,
    pub results: Vec<FileResult>,
    pub formats: Vec<FormatHypothesis>,
    pub missing_expected: Vec<String>,
    pub save: Option<SaveReport>,
    pub notes: Vec<String>,
}

#[derive(Default)]
pub struct BatchReport {
    pub dirs: Vec<DirReport>,
    pub cancelled: bool,
    pub elapsed_secs: f64,
}

pub fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn count_class(rs: &[FileResult], c: ConfClass) -> usize {
    rs.iter().filter(|r| r.confidence.class == c).count()
}

impl BatchReport {
    pub fn totals(&self) -> (usize, usize, usize, usize, usize) {
        let mut t = (0, 0, 0, 0, 0);
        for d in &self.dirs {
            t.0 += d.results.len();
            t.1 += count_class(&d.results, ConfClass::High);
            t.2 += count_class(&d.results, ConfClass::Medium);
            t.3 += count_class(&d.results, ConfClass::Low);
            t.4 += count_class(&d.results, ConfClass::Failed);
        }
        t
    }

    pub fn render_text(&self) -> String {
        let mut s = String::new();
        let (n, h, m, l, f) = self.totals();
        s.push_str(&format!(
            "otoium report: {n} files | high {h} | medium {m} | low {l} | failed {f} | {:.1}s{}\n",
            self.elapsed_secs,
            if self.cancelled { " | CANCELLED (partial)" } else { "" }
        ));
        s.push_str("NOTE: confidence classes are uncalibrated (no measured error data); see docs/accuracy.md\n");
        for d in &self.dirs {
            s.push_str(&format!("\n== {} ==\n", d.dir));
            if let Some(top) = d.formats.first() {
                s.push_str(&format!("format guess: {:?} (score {:.2})", top.format, top.score));
                if d.formats.len() > 1 {
                    s.push_str(&format!("; runner-up {:?} ({:.2})", d.formats[1].format, d.formats[1].score));
                }
                s.push('\n');
            }
            if !d.missing_expected.is_empty() {
                s.push_str(&format!("missing expected aliases: {} (e.g. {})\n", d.missing_expected.len(), d.missing_expected.iter().take(5).cloned().collect::<Vec<_>>().join(", ")));
            }
            for n in &d.notes {
                s.push_str(&format!("note: {n}\n"));
            }
            if let Some(sr) = &d.save {
                s.push_str(&format!(
                    "save: added {} | updated {} | unchanged {} | protected {} | wrote {}{}\n",
                    sr.count(DiffKind::Added), sr.count(DiffKind::Updated), sr.count(DiffKind::Unchanged),
                    sr.count(DiffKind::ProtectedKept), sr.wrote,
                    sr.backup.as_ref().map(|b| format!(" | backup {}", b.display())).unwrap_or_default()
                ));
                for it in &sr.diff {
                    match it.kind {
                        DiffKind::Added => s.push_str(&format!("  + {}\n", it.new.clone().unwrap_or_default())),
                        DiffKind::Updated => s.push_str(&format!("  ~ {}\n      was {}\n", it.new.clone().unwrap_or_default(), it.old.clone().unwrap_or_default())),
                        DiffKind::ProtectedKept => s.push_str(&format!("  ! kept {}\n      proposed {}\n", it.old.clone().unwrap_or_default(), it.new.clone().unwrap_or_default())),
                        DiffKind::Unchanged => {}
                    }
                }
                for n in &sr.notes {
                    s.push_str(&format!("  note: {n}\n"));
                }
            }
            for r in &d.results {
                s.push_str(&format!("[{}] {} score={:.2} engine={} attempts={}\n", r.confidence.class.label(), r.file, r.confidence.score, r.strategy, r.attempts));
                if let Some(fl) = &r.failure {
                    s.push_str(&format!("    FAILED: {fl}\n"));
                }
                for i in &r.issues {
                    let tag = if i.severity == Severity::Error { "error" } else { "warn" };
                    s.push_str(&format!("    {tag} {}: {}\n", i.code, i.msg));
                }
                for n in &r.notes {
                    s.push_str(&format!("    note: {n}\n"));
                }
            }
        }
        s
    }

    pub fn render_json(&self) -> String {
        let mut s = String::from("{\n  \"calibrated_confidence\": false,\n  \"dirs\": [\n");
        for (di, d) in self.dirs.iter().enumerate() {
            s.push_str(&format!("    {{\"dir\": \"{}\", \"files\": [\n", json_escape(&d.dir)));
            for (ri, r) in d.results.iter().enumerate() {
                s.push_str(&format!(
                    "      {{\"file\": \"{}\", \"confidence\": \"{}\", \"score\": {:.4}, \"engine\": \"{}\", \"attempts\": {}",
                    json_escape(&r.file), r.confidence.class.label(), r.confidence.score, json_escape(&r.strategy), r.attempts
                ));
                if let Some(e) = &r.entry {
                    s.push_str(&format!(
                        ", \"alias\": \"{}\", \"oto\": {{\"offset\": {:.3}, \"consonant\": {:.3}, \"cutoff\": {:.3}, \"preutter\": {:.3}, \"overlap\": {:.3}}}",
                        json_escape(&e.alias), e.offset, e.consonant, e.cutoff, e.preutter, e.overlap
                    ));
                }
                if let Some(f) = &r.failure {
                    s.push_str(&format!(", \"failure\": \"{}\"", json_escape(f)));
                }
                let issues: Vec<String> = r.issues.iter().map(|i| format!("\"{}\"", json_escape(&format!("{}: {}", i.code, i.msg)))).collect();
                s.push_str(&format!(", \"issues\": [{}]", issues.join(", ")));
                let notes: Vec<String> = r.notes.iter().map(|n| format!("\"{}\"", json_escape(n))).collect();
                s.push_str(&format!(", \"notes\": [{}]}}{}\n", notes.join(", "), if ri + 1 < d.results.len() { "," } else { "" }));
            }
            s.push_str(&format!("    ]}}{}\n", if di + 1 < self.dirs.len() { "," } else { "" }));
        }
        s.push_str("  ]\n}\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes() {
        assert_eq!(json_escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
        assert_eq!(json_escape("\u{1}"), "\\u0001");
    }

    #[test]
    fn empty_report_renders() {
        let r = BatchReport::default();
        assert!(r.render_text().contains("0 files"));
        assert!(r.render_json().contains("\"dirs\""));
    }
}
