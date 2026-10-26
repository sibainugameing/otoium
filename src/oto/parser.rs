use anyhow::{bail, Context, Result};
use encoding_rs::{SHIFT_JIS, UTF_8};
use std::fs;
use std::io::Write;
use std::path::Path;

use super::{OtoEncoding, OtoEntry, OtoIni};

/// Load an oto.ini file, auto-detecting UTF-8 vs Shift-JIS encoding.
pub fn load(path: &Path) -> Result<OtoIni> {
    let raw = fs::read(path)
        .with_context(|| format!("Failed to read oto.ini: {}", path.display()))?;

    // Detect encoding: try UTF-8 first (strict), fall back to Shift-JIS
    let (text, encoding) = decode_bytes(&raw);

    let mut ini = OtoIni::new(path.to_path_buf());
    ini.encoding = encoding;

    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match parse_line(line, line_no + 1) {
            Ok(entry) => {
                ini.entries.push(entry);
            }
            Err(e) => {
                log::warn!("oto.ini line {}: {}", line_no + 1, e);
                // Push a placeholder entry with error status so the UI can show it
                let placeholder = OtoEntry {
                    wav_file: format!("(parse error line {})", line_no + 1),
                    alias: String::new(),
                    offset: 0.0,
                    consonant: 0.0,
                    cutoff: 0.0,
                    preutterance: 0.0,
                    overlap: 0.0,
                    source_line: Some(line_no + 1),
                    status: super::EntryStatus::Error(e.to_string()),
                };
                ini.entries.push(placeholder);
            }
        }
    }

    ini.rebuild_index();
    Ok(ini)
}

/// Decode raw bytes as UTF-8 (strict) or Shift-JIS fallback.
fn decode_bytes(raw: &[u8]) -> (String, OtoEncoding) {
    // Check for UTF-8 BOM
    let (bytes, has_bom) = if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        (&raw[3..], true)
    } else {
        (raw, false)
    };

    // Try strict UTF-8 decoding
    if let Ok(s) = std::str::from_utf8(bytes) {
        return (s.to_string(), OtoEncoding::Utf8);
    }

    // Fall back to Shift-JIS
    let (cow, _enc, had_errors) = SHIFT_JIS.decode(bytes);
    if had_errors {
        log::warn!("Shift-JIS decoding had replacement characters; file may be corrupted");
    }
    (cow.into_owned(), OtoEncoding::ShiftJis)
}

/// Parse one non-empty line of oto.ini.
///
/// Format: wavfile.wav=alias,offset,consonant,cutoff,preutterance,overlap
///
/// The alias may be empty (resulting in an alias equal to the wav stem).
fn parse_line(line: &str, line_no: usize) -> Result<OtoEntry> {
    // Split at first '='
    let eq_pos = line
        .find('=')
        .with_context(|| format!("line {}: no '=' separator", line_no))?;

    let wav_file = line[..eq_pos].trim().to_string();
    if wav_file.is_empty() {
        bail!("line {}: empty wav filename", line_no);
    }

    let rest = &line[eq_pos + 1..];

    // Split rest by comma into at most 7 parts:
    // alias, offset, consonant, cutoff, preutterance, overlap
    let parts: Vec<&str> = rest.splitn(7, ',').collect();

    if parts.len() < 6 {
        bail!(
            "line {}: expected 6 fields after '=', got {}",
            line_no,
            parts.len()
        );
    }

    let alias = parts[0].trim().to_string();
    let offset = parse_f64(parts[1], "offset", line_no)?;
    let consonant = parse_f64(parts[2], "consonant", line_no)?;
    let cutoff = parse_f64(parts[3], "cutoff", line_no)?;
    let preutterance = parse_f64(parts[4], "preutterance", line_no)?;
    let overlap = parse_f64(parts[5].trim_end(), "overlap", line_no)?;

    Ok(OtoEntry {
        wav_file,
        alias,
        offset,
        consonant,
        cutoff,
        preutterance,
        overlap,
        source_line: Some(line_no),
        status: super::EntryStatus::Unchecked,
    })
}

fn parse_f64(s: &str, field: &str, line_no: usize) -> Result<f64> {
    let s = s.trim();
    s.parse::<f64>()
        .with_context(|| format!("line {}: invalid {} value {:?}", line_no, field, s))
}

/// Save OtoIni to a file.
/// If `backup` is true and the target file exists, copy it to `<path>.bak` first.
pub fn save(ini: &OtoIni, path: &Path, backup: bool) -> Result<()> {
    // Create backup if requested and file exists
    if backup && path.exists() {
        let bak = path.with_extension(format!(
            "{}.bak",
            path.extension().and_then(|e| e.to_str()).unwrap_or("ini")
        ));
        fs::copy(path, &bak)
            .with_context(|| format!("Failed to create backup at {}", bak.display()))?;
        log::info!("Backup created at {}", bak.display());
    }

    // Build text content
    let mut text = String::new();
    for entry in &ini.entries {
        // Skip error placeholder entries
        if matches!(entry.status, super::EntryStatus::Error(_)) {
            if entry.wav_file.starts_with("(parse error") {
                continue;
            }
        }
        text.push_str(&entry.to_oto_line());
        text.push('\n');
    }

    // Encode to target encoding
    let bytes: Vec<u8> = match ini.encoding {
        OtoEncoding::Utf8 => text.into_bytes(),
        OtoEncoding::ShiftJis => {
            let (encoded, _enc, had_errors) = SHIFT_JIS.encode(&text);
            if had_errors {
                log::warn!("Some characters could not be encoded as Shift-JIS and were replaced");
            }
            encoded.into_owned()
        }
    };

    // Write atomically: write to temp file then rename
    let tmp_path = path.with_extension("ini.tmp");
    {
        let mut f = fs::File::create(&tmp_path)
            .with_context(|| format!("Cannot create temp file: {}", tmp_path.display()))?;
        f.write_all(&bytes)
            .with_context(|| format!("Failed to write: {}", tmp_path.display()))?;
        f.flush()?;
    }

    fs::rename(&tmp_path, path)
        .with_context(|| format!("Failed to finalize save to {}", path.display()))?;

    log::info!("Saved oto.ini to {}", path.display());
    Ok(())
}

// ───────────────────────────── tests ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_temp_oto(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_parse_basic_line() {
        let entry = parse_line("_あ.wav=あ,10,70,80,-20,95,10", 1).unwrap();
        // Note: The spec line has 7 fields: alias,offset,consonant,cutoff,preutterance,overlap
        // Actually UTAU format is:  alias,offset,consonant,cutoff,preutterance,overlap  (6 fields)
        // But let's test with the correct 6-field format:
        let entry = parse_line("_あ.wav=あ,10.0,80.0,-20.0,70.0,10.0", 1).unwrap();
        assert_eq!(entry.wav_file, "_あ.wav");
        assert_eq!(entry.alias, "あ");
        assert!((entry.offset - 10.0).abs() < 1e-6);
        assert!((entry.consonant - 80.0).abs() < 1e-6);
        assert!((entry.cutoff - (-20.0)).abs() < 1e-6);
        assert!((entry.preutterance - 70.0).abs() < 1e-6);
        assert!((entry.overlap - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_cutoff_conversion() {
        let mut entry = OtoEntry::new_empty("a.wav".into(), "a".into());
        entry.cutoff = -50.0;
        let wav_duration = 1000.0;
        let abs_end = entry.cutoff_to_abs_end(wav_duration);
        assert!((abs_end - 950.0).abs() < 1e-6);

        entry.cutoff = 800.0;
        let abs_end = entry.cutoff_to_abs_end(wav_duration);
        assert!((abs_end - 800.0).abs() < 1e-6);

        entry.cutoff = 0.0;
        let abs_end = entry.cutoff_to_abs_end(wav_duration);
        assert!((abs_end - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip() {
        let content = "_あ.wav=あ,10.000,80.000,-20.000,70.000,10.000\n\
                       _い.wav=い,5.000,60.000,-15.000,55.000,8.000\n";
        let tmp = make_temp_oto(content);
        let ini = load(tmp.path()).unwrap();
        assert_eq!(ini.entries.len(), 2);
        assert_eq!(ini.entries[0].alias, "あ");
        assert_eq!(ini.entries[1].alias, "い");

        // Modify and save
        let mut ini2 = ini.clone();
        ini2.entries[0].offset = 15.0;
        ini2.encoding = OtoEncoding::Utf8;
        let out = NamedTempFile::new().unwrap();
        save(&ini2, out.path(), false).unwrap();

        let reloaded = load(out.path()).unwrap();
        assert!((reloaded.entries[0].offset - 15.0).abs() < 1e-3);
    }

    #[test]
    fn test_parse_error_entry() {
        // Missing '=' should produce error but not panic
        let content = "this_is_not_valid\n_あ.wav=あ,10.000,80.000,-20.000,70.000,10.000\n";
        let tmp = make_temp_oto(content);
        let ini = load(tmp.path()).unwrap();
        // Should have 2 entries: 1 error + 1 valid
        assert_eq!(ini.entries.len(), 2);
        assert!(matches!(ini.entries[0].status, super::super::EntryStatus::Error(_)));
        assert_eq!(ini.entries[1].alias, "あ");
    }
}
