//! Synthetic signal generators for unit tests (deterministic).
use std::f64::consts::PI;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

pub fn silence(sr: u32, sec: f64) -> Vec<f32> {
    let mut s = 12345u64;
    (0..(sr as f64 * sec) as usize).map(|_| (lcg(&mut s) * 3e-4) as f32).collect()
}

/// High-passed noise (first difference) as a crude unvoiced fricative.
pub fn fricative(sr: u32, sec: f64, amp: f64) -> Vec<f32> {
    let mut s = 777u64;
    let n = (sr as f64 * sec) as usize;
    let raw: Vec<f64> = (0..=n).map(|_| lcg(&mut s)).collect();
    (0..n).map(|i| ((raw[i + 1] - raw[i]) * 0.5 * amp) as f32).collect()
}

/// Harmonic tone with 1/h rolloff and short fades.
pub fn vowel(sr: u32, sec: f64, f0: f64, amp: f64) -> Vec<f32> {
    let n = (sr as f64 * sec) as usize;
    let fade = (sr as f64 * 0.005) as usize;
    let norm: f64 = (1..=12).map(|h| 1.0 / h as f64).sum();
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let mut v = 0.0;
            for h in 1..=12 {
                v += (2.0 * PI * h as f64 * f0 * t).sin() / h as f64;
            }
            let g = if i < fade {
                i as f64 / fade as f64
            } else if i + fade > n {
                (n - i) as f64 / fade as f64
            } else {
                1.0
            };
            (v / norm * amp * g) as f32
        })
        .collect()
}

pub fn concat(parts: &[Vec<f32>]) -> Vec<f32> {
    parts.iter().flatten().copied().collect()
}

/// silence(100ms) + fricative(120ms) + vowel(250ms) + silence(100ms).
/// Ground truth boundaries (ms): 100, 220, 470.
pub fn sh_a(sr: u32) -> Vec<f32> {
    concat(&[silence(sr, 0.1), fricative(sr, 0.12, 0.3), vowel(sr, 0.25, 120.0, 0.5), silence(sr, 0.1)])
}
