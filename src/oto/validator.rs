use super::{EntryStatus, OtoEntry};

/// Result of validating one OtoEntry.
#[derive(Debug, Default)]
pub struct ValidationResult {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.warnings.is_empty() && self.errors.is_empty()
    }
}

/// Validate a single OtoEntry against the WAV file duration (in ms).
pub fn validate_entry(entry: &OtoEntry, wav_duration_ms: f64) -> ValidationResult {
    let mut r = ValidationResult::default();

    // --- Basic range checks ---
    if entry.offset < 0.0 {
        r.errors.push(format!("offset ({:.1}) must be >= 0", entry.offset));
    }
    if entry.consonant < 0.0 {
        r.errors.push(format!("consonant ({:.1}) must be >= 0", entry.consonant));
    }
    if entry.overlap < 0.0 {
        r.errors.push(format!("overlap ({:.1}) must be >= 0", entry.overlap));
    }
    if entry.preutterance < 0.0 {
        r.errors.push(format!(
            "preutterance ({:.1}) must be >= 0",
            entry.preutterance
        ));
    }

    // --- Relationship checks ---
    if entry.overlap > entry.preutterance {
        r.warnings.push(format!(
            "overlap ({:.1}) > preutterance ({:.1}): unusual",
            entry.overlap, entry.preutterance
        ));
    }

    if entry.consonant < entry.preutterance {
        r.warnings.push(format!(
            "consonant ({:.1}) < preutterance ({:.1}): unusual for CV",
            entry.consonant, entry.preutterance
        ));
    }

    // --- Bounds within WAV ---
    if entry.offset > wav_duration_ms {
        r.errors.push(format!(
            "offset ({:.1}) exceeds WAV duration ({:.1})",
            entry.offset, wav_duration_ms
        ));
    }

    let abs_end = entry.cutoff_to_abs_end(wav_duration_ms);
    if abs_end <= entry.offset {
        r.errors.push(format!(
            "effective end ({:.1}) <= offset ({:.1}): zero or negative usable region",
            abs_end, entry.offset
        ));
    }
    if abs_end > wav_duration_ms {
        r.errors.push(format!(
            "effective end ({:.1}) exceeds WAV duration ({:.1})",
            abs_end, wav_duration_ms
        ));
    }

    let usable_len = abs_end - entry.offset;
    if entry.consonant > usable_len {
        r.warnings.push(format!(
            "consonant ({:.1}) > usable region ({:.1}): extends past cutoff",
            entry.consonant, usable_len
        ));
    }

    // --- Alias check ---
    if entry.alias.is_empty() {
        r.warnings.push("empty alias".to_string());
    }

    r
}

/// Apply validation result as status to an entry (mutates).
pub fn apply_validation(entry: &mut OtoEntry, result: &ValidationResult) {
    if !result.errors.is_empty() {
        entry.status = EntryStatus::Error(result.errors.join("; "));
    } else if !result.warnings.is_empty() {
        entry.status = EntryStatus::HasWarnings(result.warnings.clone());
    }
}
