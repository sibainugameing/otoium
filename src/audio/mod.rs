pub mod player;
pub mod waveform;

use anyhow::{bail, Context, Result};
use std::path::Path;

/// Decoded audio data for one WAV file.
#[derive(Debug, Clone)]
pub struct WavData {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Mono-mixed samples in [-1.0, 1.0]
    pub samples: Vec<f32>,
    /// Total duration in milliseconds
    pub duration_ms: f64,
    /// Number of bits per sample in source
    pub bits_per_sample: u16,
}

impl WavData {
    /// Load a WAV file using hound. Handles 8/16/24/32-bit PCM and 32-bit float.
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = hound::WavReader::open(path)
            .with_context(|| format!("Failed to open WAV: {}", path.display()))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels;
        let bits_per_sample = spec.bits_per_sample;

        // Read all samples and normalize to f32 [-1.0, 1.0]
        let raw_samples: Vec<f32> = match (spec.sample_format, bits_per_sample) {
            (hound::SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| anyhow::anyhow!(e)))
                .collect::<Result<Vec<_>>>()?,

            (hound::SampleFormat::Int, 8) => reader
                .samples::<i8>()
                .map(|s| s.map(|v| v as f32 / i8::MAX as f32).map_err(|e| anyhow::anyhow!(e)))
                .collect::<Result<Vec<_>>>()?,

            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32).map_err(|e| anyhow::anyhow!(e)))
                .collect::<Result<Vec<_>>>()?,

            (hound::SampleFormat::Int, 24) => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 8_388_607.0).map_err(|e| anyhow::anyhow!(e)))
                .collect::<Result<Vec<_>>>()?,

            (hound::SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32).map_err(|e| anyhow::anyhow!(e)))
                .collect::<Result<Vec<_>>>()?,

            _ => bail!(
                "Unsupported WAV format: {:?} {}bit",
                spec.sample_format,
                bits_per_sample
            ),
        };

        if raw_samples.is_empty() {
            bail!("WAV file has no samples");
        }

        // Mix down to mono
        let samples: Vec<f32> = if channels == 1 {
            raw_samples
        } else {
            let ch = channels as usize;
            raw_samples
                .chunks(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                .collect()
        };

        let duration_ms = samples.len() as f64 / sample_rate as f64 * 1000.0;

        Ok(Self {
            sample_rate,
            channels,
            samples,
            duration_ms,
            bits_per_sample,
        })
    }

    /// Get the sample at a given time position (ms), with linear interpolation.
    pub fn sample_at_ms(&self, ms: f64) -> f32 {
        let idx = ms / 1000.0 * self.sample_rate as f64;
        let i = idx.floor() as usize;
        if i >= self.samples.len() {
            return 0.0;
        }
        self.samples[i]
    }

    /// Extract a sub-range of samples by time (ms). Clamps to valid range.
    pub fn slice_ms(&self, start_ms: f64, end_ms: f64) -> &[f32] {
        let sr = self.sample_rate as f64;
        let start = ((start_ms / 1000.0 * sr) as usize).min(self.samples.len());
        let end = ((end_ms / 1000.0 * sr) as usize).min(self.samples.len());
        &self.samples[start..end]
    }

    /// Compute RMS amplitude over a window [start_ms, end_ms].
    pub fn rms_ms(&self, start_ms: f64, end_ms: f64) -> f32 {
        let slice = self.slice_ms(start_ms, end_ms);
        if slice.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = slice.iter().map(|s| s * s).sum();
        (sum_sq / slice.len() as f32).sqrt()
    }

    /// Find the first sample (from start) whose RMS in a 10ms window exceeds threshold.
    /// Returns the time in ms, or None if no such point found.
    pub fn find_onset_ms(&self, threshold: f32, window_ms: f64) -> Option<f64> {
        let step_ms = window_ms / 2.0;
        let mut t = 0.0;
        while t + window_ms <= self.duration_ms {
            if self.rms_ms(t, t + window_ms) > threshold {
                return Some(t);
            }
            t += step_ms;
        }
        None
    }

    /// Find the last sample (from end) whose RMS in a 10ms window exceeds threshold.
    /// Returns the time in ms of the offset, or None.
    pub fn find_offset_ms(&self, threshold: f32, window_ms: f64) -> Option<f64> {
        let step_ms = window_ms / 2.0;
        let mut t = self.duration_ms - window_ms;
        while t >= 0.0 {
            if self.rms_ms(t, t + window_ms) > threshold {
                return Some(t + window_ms);
            }
            t -= step_ms;
        }
        None
    }
}
