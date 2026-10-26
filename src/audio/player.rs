/// Audio playback module.
///
/// Uses rodio for cross-platform audio output (macOS CoreAudio + Linux ALSA/PulseAudio).
/// Playback is run on a background thread to keep the TUI responsive.
///
/// Design goals:
/// - Non-blocking: play() returns immediately
/// - Stoppable: stop() kills current playback
/// - Position tracking: elapsed_ms() for UI cursor
use anyhow::Result;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Playback state shared between main thread and player thread.
#[derive(Debug, Default)]
struct PlayState {
    is_playing: bool,
    start_offset_ms: f64,
    started_at: Option<Instant>,
}

/// Manages audio output via rodio.
pub struct AudioPlayer {
    state: Arc<Mutex<PlayState>>,
    // rodio sink is Option because audio output may be unavailable
    // (headless CI, no audio device, etc.)
    sink: Option<Arc<Mutex<rodio::Sink>>>,
    _stream: Option<rodio::OutputStream>,
    _stream_handle: Option<rodio::OutputStreamHandle>,
}

impl AudioPlayer {
    /// Create a new player. Returns Ok even if no audio device is available
    /// (in that case, playback silently does nothing).
    pub fn new() -> Self {
        match rodio::OutputStream::try_default() {
            Ok((stream, handle)) => {
                let sink = rodio::Sink::try_new(&handle).ok().map(|s| Arc::new(Mutex::new(s)));
                Self {
                    state: Arc::new(Mutex::new(PlayState::default())),
                    sink,
                    _stream: Some(stream),
                    _stream_handle: Some(handle),
                }
            }
            Err(e) => {
                log::warn!("No audio output device: {e}. Playback disabled.");
                Self {
                    state: Arc::new(Mutex::new(PlayState::default())),
                    sink: None,
                    _stream: None,
                    _stream_handle: None,
                }
            }
        }
    }

    pub fn is_available(&self) -> bool {
        self.sink.is_some()
    }

    /// Play a WAV file from `start_ms` to `end_ms`.
    /// If `end_ms` is None, plays to end of file.
    pub fn play_range(
        &mut self,
        path: &Path,
        start_ms: f64,
        end_ms: Option<f64>,
        wav_data: &super::WavData,
    ) -> Result<()> {
        self.stop();

        let sink = match &self.sink {
            Some(s) => s.clone(),
            None => return Ok(()), // no audio device, silently skip
        };

        let sr = wav_data.sample_rate;
        let start_sample = (start_ms / 1000.0 * sr as f64) as usize;
        let end_sample = match end_ms {
            Some(e) => ((e / 1000.0 * sr as f64) as usize).min(wav_data.samples.len()),
            None => wav_data.samples.len(),
        };

        let samples: Vec<f32> = wav_data.samples[start_sample..end_sample].to_vec();

        // Build a rodio source from raw samples
        let source = rodio::buffer::SamplesBuffer::new(1, sr, samples);

        {
            let mut lock = sink.lock().unwrap();
            lock.stop();
            lock.append(source);
            lock.play();
        }

        {
            let mut state = self.state.lock().unwrap();
            state.is_playing = true;
            state.start_offset_ms = start_ms;
            state.started_at = Some(Instant::now());
        }

        Ok(())
    }

    /// Stop current playback.
    pub fn stop(&mut self) {
        if let Some(sink) = &self.sink {
            let mut lock = sink.lock().unwrap();
            lock.stop();
        }
        let mut state = self.state.lock().unwrap();
        state.is_playing = false;
        state.started_at = None;
    }

    /// Returns true if audio is currently playing.
    pub fn is_playing(&self) -> bool {
        if let Some(sink) = &self.sink {
            let lock = sink.lock().unwrap();
            !lock.is_paused() && !lock.empty()
        } else {
            false
        }
    }

    /// Returns elapsed playback position in ms from start_ms.
    pub fn elapsed_ms(&self) -> Option<f64> {
        let state = self.state.lock().unwrap();
        state.started_at.map(|t| {
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            state.start_offset_ms + elapsed
        })
    }

    pub fn toggle_pause(&self) {
        if let Some(sink) = &self.sink {
            let mut lock = sink.lock().unwrap();
            if lock.is_paused() {
                lock.play();
            } else {
                lock.pause();
            }
        }
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}
