/// Unicode block characters for amplitude rendering (1/8 resolution per row)
const BLOCK_CHARS: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Render a waveform into a 2D grid of characters.
/// Returns a Vec of String lines (top to bottom).
///
/// # Arguments
/// * `samples` - mono audio samples in [-1.0, 1.0]
/// * `width`   - terminal columns available for the waveform
/// * `height`  - terminal rows available for the waveform (not counting axis)
/// * `start_sample` - first sample index to render
/// * `end_sample`   - last sample index to render (exclusive)
pub fn render_waveform(
    samples: &[f32],
    width: usize,
    height: usize,
    start_sample: usize,
    end_sample: usize,
) -> Vec<String> {
    if width == 0 || height == 0 || samples.is_empty() {
        return vec!["(no waveform)".to_string()];
    }

    let end_sample = end_sample.min(samples.len());
    let start_sample = start_sample.min(end_sample);
    let total = end_sample - start_sample;

    if total == 0 {
        return vec![" ".repeat(width); height];
    }

    // --- Compute per-column min/max amplitude ---
    let samples_per_col = total as f32 / width as f32;
    let mut col_data: Vec<(f32, f32)> = Vec::with_capacity(width); // (min, max)

    for col in 0..width {
        let s_start = (start_sample as f32 + col as f32 * samples_per_col) as usize;
        let s_end =
            (start_sample as f32 + (col + 1) as f32 * samples_per_col) as usize + 1;
        let s_end = s_end.min(samples.len());
        let slice = &samples[s_start..s_end];

        if slice.is_empty() {
            col_data.push((0.0, 0.0));
        } else {
            let mn = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            col_data.push((mn.max(-1.0), mx.min(1.0)));
        }
    }

    // --- Render into character grid ---
    // Each row covers a vertical amplitude band.
    // Row 0 = top = positive peak, row height-1 = bottom = negative peak.
    let half = height as f32 / 2.0;
    let mut rows: Vec<Vec<char>> = vec![vec![' '; width]; height];

    for col in 0..width {
        let (mn, mx) = col_data[col];
        // Map amplitude [-1.0, 1.0] to row indices [0, height-1]
        // amplitude 1.0  → row 0
        // amplitude 0.0  → row height/2
        // amplitude -1.0 → row height-1
        let amp_to_row = |amp: f32| -> usize {
            let row = ((-amp + 1.0) * 0.5 * height as f32).floor() as usize;
            row.min(height - 1)
        };

        let row_top = amp_to_row(mx);
        let row_bot = amp_to_row(mn);

        for row in 0..height {
            let ch = if row >= row_top && row <= row_bot {
                if mx > 0.05 {
                    '█'
                } else {
                    '▄'
                }
            } else {
                ' '
            };
            rows[row][col] = ch;
        }

        // Middle zero-crossing indicator
        let mid = (height / 2).min(height - 1);
        if rows[mid][col] == ' ' {
            rows[mid][col] = '·';
        }
    }

    rows.into_iter()
        .map(|r| r.into_iter().collect())
        .collect()
}

/// Render a compact single-row waveform using block chars (▁▂▃▅▇...).
/// Suitable for narrow displays.
pub fn render_waveform_compact(
    samples: &[f32],
    width: usize,
    start_sample: usize,
    end_sample: usize,
) -> String {
    if width == 0 || samples.is_empty() {
        return String::new();
    }

    let end_sample = end_sample.min(samples.len());
    let start_sample = start_sample.min(end_sample);
    let total = end_sample - start_sample;
    if total == 0 {
        return " ".repeat(width);
    }

    let samples_per_col = total as f32 / width as f32;
    let mut result = String::with_capacity(width * 4); // UTF-8 multi-byte

    for col in 0..width {
        let s_start = (start_sample as f32 + col as f32 * samples_per_col) as usize;
        let s_end =
            (start_sample as f32 + (col + 1) as f32 * samples_per_col) as usize + 1;
        let s_end = s_end.min(samples.len());
        let slice = &samples[s_start..s_end];

        if slice.is_empty() {
            result.push(' ');
            continue;
        }

        let rms: f32 = {
            let ss: f32 = slice.iter().map(|s| s * s).sum();
            (ss / slice.len() as f32).sqrt()
        };

        // Map RMS [0, 1.0] to block index
        let idx = ((rms * 8.0) as usize).min(BLOCK_CHARS.len() - 1);
        result.push(BLOCK_CHARS[idx]);
    }

    result
}

/// Build a time axis string for the given range.
/// Returns a line like: "|  100ms   200ms   300ms   400ms |"
pub fn render_time_axis(
    width: usize,
    start_ms: f64,
    end_ms: f64,
) -> String {
    if width < 4 {
        return String::new();
    }

    let dur = end_ms - start_ms;
    if dur <= 0.0 {
        return " ".repeat(width);
    }

    let mut chars: Vec<char> = vec![' '; width];

    // Determine a "nice" tick interval
    let target_ticks = (width / 10).max(2).min(10);
    let raw_interval = dur / target_ticks as f64;
    let tick_interval = nice_interval(raw_interval);

    // Find first tick >= start_ms
    let first_tick = (start_ms / tick_interval).ceil() * tick_interval;
    let mut t = first_tick;
    while t <= end_ms {
        let pos = ((t - start_ms) / dur * width as f64) as usize;
        if pos < width {
            let label = format!("{:.0}ms", t);
            for (i, ch) in label.chars().enumerate() {
                let p = pos.saturating_add(i);
                if p < width {
                    chars[p] = ch;
                }
            }
        }
        t += tick_interval;
    }

    chars.into_iter().collect()
}

fn nice_interval(raw: f64) -> f64 {
    let magnitudes = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0];
    *magnitudes
        .iter()
        .find(|&&m| m >= raw)
        .unwrap_or(&1000.0)
}

/// Render a cursor marker line, showing labeled positions.
/// `markers` is a list of (label, position_ms).
/// Returns two lines: the marker symbols and the label values.
pub fn render_markers(
    width: usize,
    start_ms: f64,
    end_ms: f64,
    markers: &[(&str, f64)],
    active_marker: Option<usize>,
) -> (String, String) {
    let dur = end_ms - start_ms;
    let mut symbol_line: Vec<char> = vec![' '; width];
    let mut value_line: Vec<char> = vec![' '; width];

    for (idx, (label, pos_ms)) in markers.iter().enumerate() {
        if *pos_ms < start_ms || *pos_ms > end_ms {
            continue;
        }
        let col = ((pos_ms - start_ms) / dur * width as f64) as usize;
        let col = col.min(width.saturating_sub(1));

        // Symbol: use a block character, highlight if active
        symbol_line[col] = if Some(idx) == active_marker { '┃' } else { '│' };

        // Label characters
        for (i, ch) in label.chars().enumerate() {
            let p = col.saturating_add(i);
            if p < width {
                value_line[p] = ch;
            }
        }
    }

    (
        symbol_line.into_iter().collect(),
        value_line.into_iter().collect(),
    )
}
