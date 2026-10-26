pub mod history;

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Recursively (depth=1) find all WAV files in `dir`, sorted by filename.
/// Files with names not representable in UTF-8 are silently skipped.
pub fn scan_wav_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_ascii_lowercase() == "wav" {
                    result.push(path);
                }
            }
        }
    }
    // Sort case-insensitively by filename
    result.sort_by(|a, b| {
        let na = a.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
        let nb = b.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
        na.cmp(&nb)
    });
    Ok(result)
}
