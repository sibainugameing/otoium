//! Minimal, read-only WAV decoder (PCM 8/16/24/32, float 32/64, WAVE_FORMAT_EXTENSIBLE).
//! Source WAV files are never written by this program.
use crate::error::{Error, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    /// Mono mix-down in [-1, 1].
    pub samples: Vec<f32>,
    /// Fraction of raw samples at or beyond 0.999 full scale.
    pub clipped_ratio: f64,
}

impl Wav {
    pub fn duration_ms(&self) -> f64 {
        self.samples.len() as f64 * 1000.0 / self.sample_rate as f64
    }
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    }
}

pub fn read_wav(path: &Path) -> Result<Wav> {
    let bytes = std::fs::read(path)?;
    parse_wav(&bytes)
}

fn u16le(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}
fn u32le(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
}

pub fn parse_wav(b: &[u8]) -> Result<Wav> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(Error::Wav("not a RIFF/WAVE file".into()));
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // tag, ch, sr, bits
    let mut data: Option<(usize, usize)> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32le(b, pos + 4) as usize;
        let body = pos + 8;
        let avail = b.len() - body;
        let sz = size.min(avail); // tolerate truncated / streamed data sizes
        if id == b"fmt " {
            if sz < 16 {
                return Err(Error::Wav("fmt chunk too small".into()));
            }
            let mut tag = u16le(b, body);
            let ch = u16le(b, body + 2);
            let sr = u32le(b, body + 4);
            let bits = u16le(b, body + 14);
            if tag == 0xFFFE && sz >= 26 {
                tag = u16le(b, body + 24);
            }
            fmt = Some((tag, ch, sr, bits));
        } else if id == b"data" {
            data = Some((body, sz));
        }
        pos = body + sz + (sz & 1);
        if fmt.is_some() && data.is_some() {
            break;
        }
    }
    let (tag, ch, sr, bits) = fmt.ok_or_else(|| Error::Wav("missing fmt chunk".into()))?;
    let (dpos, dlen) = data.ok_or_else(|| Error::Wav("missing data chunk".into()))?;
    if ch == 0 || sr == 0 {
        return Err(Error::Wav("invalid channel count or sample rate".into()));
    }
    let bps = (bits / 8) as usize;
    if bps == 0 {
        return Err(Error::Wav("invalid bit depth".into()));
    }
    let frame = bps * ch as usize;
    let nframes = dlen / frame;
    let mut samples = Vec::with_capacity(nframes);
    let mut clipped = 0usize;
    let mut total = 0usize;
    for f in 0..nframes {
        let mut acc = 0.0f64;
        for c in 0..ch as usize {
            let p = dpos + f * frame + c * bps;
            let v: f64 = match (tag, bits) {
                (1, 8) => (b[p] as f64 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([b[p], b[p + 1]]) as f64 / 32768.0,
                (1, 24) => {
                    let raw = (b[p] as i32) | ((b[p + 1] as i32) << 8) | ((b[p + 2] as i32) << 16);
                    let s = (raw << 8) >> 8; // sign extend 24 -> 32
                    s as f64 / 8388608.0
                }
                (1, 32) => i32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as f64 / 2147483648.0,
                (3, 32) => f32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as f64,
                (3, 64) => {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(&b[p..p + 8]);
                    f64::from_le_bytes(a)
                }
                _ => return Err(Error::Wav(format!("unsupported format tag={tag} bits={bits}"))),
            };
            let v = if v.is_finite() { v } else { 0.0 };
            if v.abs() >= 0.999 {
                clipped += 1;
            }
            total += 1;
            acc += v;
        }
        samples.push((acc / ch as f64) as f32);
    }
    Ok(Wav {
        sample_rate: sr,
        channels: ch,
        samples,
        clipped_ratio: if total > 0 { clipped as f64 / total as f64 } else { 0.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn make_wav16(sr: u32, samples: &[i16]) -> Vec<u8> {
        let mut v = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&sr.to_le_bytes());
        v.extend_from_slice(&(sr * 2).to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    #[test]
    fn parses_16bit_mono() {
        let w = parse_wav(&make_wav16(44100, &[0, 16384, -16384, 32767])).unwrap();
        assert_eq!(w.sample_rate, 44100);
        assert_eq!(w.samples.len(), 4);
        assert!((w.samples[1] - 0.5).abs() < 1e-4);
        assert!(w.clipped_ratio > 0.0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_wav(b"hello world, not a wav").is_err());
    }

    #[test]
    fn tolerates_truncated_data_chunk() {
        let mut b = make_wav16(8000, &[1, 2, 3, 4]);
        b.truncate(b.len() - 3); // cut mid-sample
        let w = parse_wav(&b).unwrap();
        assert_eq!(w.samples.len(), 2); // 5 bytes remain -> 2 whole samples
    }
}
