//! Acoustic feature extraction: energy, voicing (autocorrelation), HF ratio, spectral flux, ZCR.
use std::f64::consts::PI;

pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place radix-2 FFT. `re.len()` must be a power of two.
pub fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    assert!(n.is_power_of_two() && im.len() == n);
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let ang = 2.0 * PI / (len as f64) * sign;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let s = 1.0 / n as f64;
        for k in 0..n {
            re[k] *= s;
            im[k] *= s;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExtractCfg {
    pub hop_ms: f64,
    pub spec_win_ms: f64,
    pub energy_win_ms: f64,
    pub voicing_win_ms: f64,
}

impl ExtractCfg {
    /// Used by the DP aligner.
    pub fn coarse() -> Self {
        Self { hop_ms: 5.0, spec_win_ms: 20.0, energy_win_ms: 10.0, voicing_win_ms: 30.0 }
    }
    /// Used for boundary refinement (1 ms resolution).
    pub fn fine() -> Self {
        Self { hop_ms: 1.0, spec_win_ms: 10.0, energy_win_ms: 4.0, voicing_win_ms: 30.0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Frame {
    pub t_ms: f64,
    pub energy_db: f64,
    pub voicing: f64,
    pub hf_ratio: f64,
    /// Relative spectral change vs. previous frame, in [0, 1].
    pub flux: f64,
    pub zcr: f64,
}

#[derive(Clone, Debug)]
pub struct FeatureTrack {
    pub hop_ms: f64,
    pub frames: Vec<Frame>,
    pub floor_db: f64,
    pub peak_db: f64,
    pub flux_ref: f64,
    /// True when dynamic range is too small to trust relative thresholds.
    pub degenerate: bool,
}

impl FeatureTrack {
    pub fn snr_db(&self) -> f64 {
        self.peak_db - self.floor_db
    }
    pub fn norm_energy(&self, db: f64) -> f64 {
        ((db - self.floor_db) / (self.peak_db - self.floor_db).max(1.0)).clamp(0.0, 1.3)
    }
    pub fn index_at(&self, t_ms: f64) -> usize {
        ((t_ms / self.hop_ms).round().max(0.0) as usize).min(self.frames.len().saturating_sub(1))
    }
    /// Mean of `f` over frames whose centre lies in [a_ms, b_ms]. Returns 0 if empty.
    pub fn mean_over<F: Fn(&Frame) -> f64>(&self, a_ms: f64, b_ms: f64, f: F) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let a = self.index_at(a_ms.min(b_ms));
        let b = self.index_at(a_ms.max(b_ms));
        let mut s = 0.0;
        let mut n = 0usize;
        for fr in &self.frames[a..=b] {
            s += f(fr);
            n += 1;
        }
        if n == 0 { 0.0 } else { s / n as f64 }
    }
}

fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() - 1) as f64 * p).round() as usize;
    s[idx]
}

pub fn extract(x: &[f32], sr: u32, cfg: ExtractCfg) -> FeatureTrack {
    let srf = sr as f64;
    let len = x.len();
    let hop_s = cfg.hop_ms * srf / 1000.0;
    let n_frames = ((len as f64) / hop_s).floor() as usize + 1;

    // prefix sums of x^2 for O(1) windowed energy
    let mut pre = vec![0.0f64; len + 1];
    for i in 0..len {
        pre[i + 1] = pre[i] + (x[i] as f64) * (x[i] as f64);
    }
    let e_half = ((cfg.energy_win_ms * srf / 1000.0) / 2.0).round().max(1.0) as isize;

    let wl = ((cfg.spec_win_ms * srf / 1000.0).round() as usize).max(16);
    let nfft = next_pow2(wl);
    let half = nfft / 2;
    let hann: Vec<f64> = (0..wl).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (wl as f64 - 1.0)).cos()).collect();
    let bin_hz = srf / nfft as f64;
    let lo_bin = ((50.0 / bin_hz).ceil() as usize).max(1);
    let hf_bin = ((4000.0f64.min(srf * 0.25)) / bin_hz).round() as usize;

    let vl = ((cfg.voicing_win_ms * srf / 1000.0).round() as usize).max(32);
    let vfft = next_pow2(2 * vl);
    let lag_min = ((srf / 500.0).ceil() as usize).max(2);
    let lag_max = ((srf / 60.0) as usize).min(vl / 2).max(lag_min + 1);

    let mut frames = Vec::with_capacity(n_frames);
    let mut prev_mag = vec![0.0f64; half + 1];
    let mut re = vec![0.0f64; nfft];
    let mut im = vec![0.0f64; nfft];
    let mut vre = vec![0.0f64; vfft];
    let mut vim = vec![0.0f64; vfft];
    let mut mags = vec![0.0f64; half + 1];

    for i in 0..n_frames {
        let center = (i as f64 * hop_s).round() as isize;

        // energy (short rectangular window)
        let s = (center - e_half).clamp(0, len as isize) as usize;
        let e = (center + e_half).clamp(0, len as isize) as usize;
        let mean = (pre[e] - pre[s]) / ((e - s).max(1) as f64);
        let energy_db = 10.0 * (mean + 1e-10).log10();

        // spectrum
        let start = center - (wl as isize) / 2;
        for k in 0..nfft {
            re[k] = 0.0;
            im[k] = 0.0;
        }
        let mut zc = 0usize;
        let mut last = 0.0f64;
        for k in 0..wl {
            let idx = start + k as isize;
            let v = if idx >= 0 && (idx as usize) < len { x[idx as usize] as f64 } else { 0.0 };
            re[k] = v * hann[k];
            if k > 0 && (v >= 0.0) != (last >= 0.0) {
                zc += 1;
            }
            last = v;
        }
        fft(&mut re, &mut im, false);
        for k in 0..=half {
            mags[k] = (re[k] * re[k] + im[k] * im[k]).sqrt();
        }
        let mut tot = 0.0;
        let mut hf = 0.0;
        let mut num = 0.0;
        let mut den = 1e-9;
        for k in lo_bin..=half {
            let p = mags[k] * mags[k];
            tot += p;
            if k >= hf_bin {
                hf += p;
            }
            num += (mags[k] - prev_mag[k]).abs();
            den += mags[k] + prev_mag[k];
        }
        let hf_ratio = hf / (tot + 1e-12);
        let flux = if i == 0 { 0.0 } else { (num / den).clamp(0.0, 1.0) };
        prev_mag.copy_from_slice(&mags);

        // voicing: FFT autocorrelation of mean-removed segment
        let vstart = center - (vl as isize) / 2;
        let mut mu = 0.0;
        let mut cnt = 0.0;
        for k in 0..vl {
            let idx = vstart + k as isize;
            if idx >= 0 && (idx as usize) < len {
                mu += x[idx as usize] as f64;
                cnt += 1.0;
            }
        }
        mu /= f64::max(cnt, 1.0);
        for k in 0..vfft {
            vre[k] = 0.0;
            vim[k] = 0.0;
        }
        for k in 0..vl {
            let idx = vstart + k as isize;
            if idx >= 0 && (idx as usize) < len {
                vre[k] = x[idx as usize] as f64 - mu;
            }
        }
        fft(&mut vre, &mut vim, false);
        for k in 0..vfft {
            vre[k] = vre[k] * vre[k] + vim[k] * vim[k];
            vim[k] = 0.0;
        }
        fft(&mut vre, &mut vim, true);
        let r0 = vre[0];
        let mut voicing = 0.0;
        if r0 > 1e-12 {
            for lag in lag_min..=lag_max.min(vfft / 2 - 1) {
                // divide by the overlap fraction so periodic signals score ~1 at every lag
                let r = vre[lag] / (r0 * (1.0 - lag as f64 / vl as f64));
                if r > voicing {
                    voicing = r;
                }
            }
        }

        frames.push(Frame {
            t_ms: i as f64 * cfg.hop_ms,
            energy_db,
            voicing: voicing.clamp(0.0, 1.0),
            hf_ratio,
            flux,
            zcr: zc as f64 / wl as f64,
        });
    }

    let energies: Vec<f64> = frames.iter().map(|f| f.energy_db).collect();
    let fluxes: Vec<f64> = frames.iter().map(|f| f.flux).collect();
    let floor_db = percentile(&energies, 0.05);
    let peak_db = percentile(&energies, 0.95);
    FeatureTrack {
        hop_ms: cfg.hop_ms,
        frames,
        floor_db,
        peak_db,
        flux_ref: percentile(&fluxes, 0.90).max(1e-3),
        degenerate: peak_db - floor_db < 6.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_naive_dft() {
        let n = 16;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 % 5) as f64) - 2.0).collect();
        let mut re = x.clone();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im, false);
        for k in 0..n {
            let (mut sr, mut si) = (0.0, 0.0);
            for (t, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                sr += v * a.cos();
                si += v * a.sin();
            }
            assert!((re[k] - sr).abs() < 1e-9 && (im[k] - si).abs() < 1e-9);
        }
    }

    #[test]
    fn fft_roundtrip() {
        let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut re = x.clone();
        let mut im = vec![0.0; 32];
        fft(&mut re, &mut im, false);
        fft(&mut re, &mut im, true);
        for i in 0..32 {
            assert!((re[i] - x[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn voicing_high_for_harmonic_low_for_noise() {
        let sr = 44100;
        let harm = crate::testutil::vowel(sr, 0.3, 120.0, 0.5);
        let noise = crate::testutil::fricative(sr, 0.3, 0.3);
        let th = extract(&harm, sr, ExtractCfg::coarse());
        let tn = extract(&noise, sr, ExtractCfg::coarse());
        let mh = th.mean_over(50.0, 250.0, |f| f.voicing);
        let mn = tn.mean_over(50.0, 250.0, |f| f.voicing);
        assert!(mh > 0.7, "harmonic voicing {mh}");
        assert!(mn < 0.4, "noise voicing {mn}");
    }
}
