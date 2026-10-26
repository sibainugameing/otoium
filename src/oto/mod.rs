pub mod parser;
pub mod validator;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One entry in oto.ini: the phoneme configuration for a single alias within a WAV file.
///
/// oto.ini line format:
///   wavfile.wav=alias,offset,consonant,cutoff,preutterance,overlap
///
/// Parameters (all in milliseconds):
///   offset      - start of the usable region from the beginning of the WAV
///   consonant   - end of the consonant portion (relative to offset)
///   cutoff      - end of the usable region.
///                 POSITIVE value: ms from the beginning of the WAV (absolute).
///                 NEGATIVE value: ms from the END of the WAV (relative to end).
///                 This is the most frequently misunderstood parameter.
///   preutterance - the "pre-utterance" point relative to offset (cursor position)
///   overlap     - the "overlap" length, starts at (preutterance - overlap)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OtoEntry {
    /// Source WAV filename (basename only, e.g. "_あ.wav")
    pub wav_file: String,
    /// Alias for this entry (e.g. "あ", "- あ", "あ い")
    pub alias: String,
    /// Offset from start of WAV (ms). Always >= 0.
    pub offset: f64,
    /// End of consonant portion relative to offset (ms). Always >= 0.
    pub consonant: f64,
    /// End of usable region.
    /// Positive: ms from start of WAV (absolute position).
    /// Negative: ms before end of WAV (e.g. -50 means 50ms before EOF).
    pub cutoff: f64,
    /// Pre-utterance position relative to offset (ms).
    pub preutterance: f64,
    /// Overlap length (ms). Region starts at preutterance - overlap from offset.
    pub overlap: f64,
    /// Which line in the source file this came from (for error reporting).
    #[serde(skip)]
    pub source_line: Option<usize>,
    /// Modification status for UI display.
    pub status: EntryStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum EntryStatus {
    #[default]
    Unchecked,
    /// Values were set by auto-estimation
    Estimated { confidence: f32 },
    /// User has reviewed / manually edited
    Confirmed,
    /// Validation found issues
    HasWarnings(Vec<String>),
    /// Entry has errors (missing WAV, parse error, etc.)
    Error(String),
}

impl OtoEntry {
    /// Create a new entry with all-zero parameters.
    pub fn new_empty(wav_file: String, alias: String) -> Self {
        Self {
            wav_file,
            alias,
            offset: 0.0,
            consonant: 0.0,
            cutoff: 0.0,
            preutterance: 0.0,
            overlap: 0.0,
            source_line: None,
            status: EntryStatus::Unchecked,
        }
    }

    /// Convert this entry to oto.ini line format.
    /// Note: cutoff is stored internally as the UTAU convention value
    /// (positive = ms from start, negative = ms before end).
    pub fn to_oto_line(&self) -> String {
        format!(
            "{}={},{:.3},{:.3},{:.3},{:.3},{:.3}",
            self.wav_file,
            self.alias,
            self.offset,
            self.consonant,
            self.cutoff,
            self.preutterance,
            self.overlap
        )
    }

    /// Compute the absolute end position of the usable region in ms,
    /// given the total duration of the WAV file.
    ///
    /// UTAU cutoff convention:
    ///   cutoff > 0: absolute ms position from start of WAV
    ///   cutoff < 0: ms before end of WAV  (end_ms = wav_duration + cutoff)
    ///   cutoff = 0: interpreted as "end of WAV"
    pub fn cutoff_to_abs_end(&self, wav_duration_ms: f64) -> f64 {
        if self.cutoff < 0.0 {
            wav_duration_ms + self.cutoff
        } else if self.cutoff == 0.0 {
            wav_duration_ms
        } else {
            self.cutoff
        }
    }

    /// Convert an absolute end position (ms from start) back to the cutoff value,
    /// choosing the sign based on which representation is more natural.
    /// Convention: use negative (relative to end) when the position is near the end
    /// of the file; use positive (absolute) only when explicitly needed.
    pub fn abs_end_to_cutoff(abs_end_ms: f64, wav_duration_ms: f64) -> f64 {
        // Use negative representation (UTAU's typical style)
        abs_end_ms - wav_duration_ms // will be <= 0 for positions before EOF
    }
}

/// The complete contents of one oto.ini file.
/// A single oto.ini file covers one directory of WAV files.
#[derive(Debug, Default, Clone)]
pub struct OtoIni {
    /// Absolute path to the oto.ini file
    pub path: PathBuf,
    /// All entries, in order of appearance
    pub entries: Vec<OtoEntry>,
    /// Index from wav_file -> list of entry indices (for quick lookup)
    pub wav_index: HashMap<String, Vec<usize>>,
    /// Whether entries have been modified since last save
    pub dirty: bool,
    /// Source encoding detected on load
    pub encoding: OtoEncoding,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum OtoEncoding {
    Utf8,
    #[default]
    ShiftJis,
}

impl OtoIni {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            entries: Vec::new(),
            wav_index: HashMap::new(),
            dirty: false,
            encoding: OtoEncoding::ShiftJis,
        }
    }

    /// Load from disk, auto-detecting encoding.
    pub fn load(path: &Path) -> Result<Self> {
        parser::load(path)
    }

    /// Save back to the same path. Creates a backup first.
    pub fn save(&self) -> Result<()> {
        parser::save(self, &self.path, true)
    }

    /// Save to a specific path.
    pub fn save_to(&self, path: &Path, backup: bool) -> Result<()> {
        parser::save(self, path, backup)
    }

    /// Rebuild the wav_index from current entries.
    pub fn rebuild_index(&mut self) {
        self.wav_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.wav_index
                .entry(entry.wav_file.clone())
                .or_default()
                .push(i);
        }
    }

    /// Get all entries for a given wav filename.
    pub fn entries_for_wav(&self, wav_name: &str) -> Vec<&OtoEntry> {
        match self.wav_index.get(wav_name) {
            Some(indices) => indices.iter().map(|&i| &self.entries[i]).collect(),
            None => Vec::new(),
        }
    }

    /// Update (replace) an entry at a given index.
    pub fn update_entry(&mut self, index: usize, entry: OtoEntry) {
        if index < self.entries.len() {
            self.entries[index] = entry;
            self.dirty = true;
            self.rebuild_index();
        }
    }

    /// Check for duplicate aliases.
    pub fn find_duplicate_aliases(&self) -> Vec<(&OtoEntry, &OtoEntry)> {
        let mut seen: HashMap<&str, &OtoEntry> = HashMap::new();
        let mut dups = Vec::new();
        for entry in &self.entries {
            if let Some(prev) = seen.get(entry.alias.as_str()) {
                dups.push((*prev, entry));
            } else {
                seen.insert(&entry.alias, entry);
            }
        }
        dups
    }
}
