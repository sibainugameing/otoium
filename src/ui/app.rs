use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyEventKind};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::collections::VecDeque;
use std::io::Stdout;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::audio::player::AudioPlayer;
use crate::audio::WavData;
use crate::oto::{EntryStatus, OtoEntry, OtoIni};
use crate::ui::keybindings::{key_to_action, Action};
use crate::ui::screens;
use crate::util::history::History;

/// Which pane currently has keyboard focus.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Focus {
    #[default]
    FileList,
    Waveform,
    Params,
    Search,
    Help,
}

/// The complete mutable state of the application.
pub struct App {
    /// Directory being edited
    pub voicebank_dir: PathBuf,
    /// All WAV files found in the directory
    pub wav_files: Vec<PathBuf>,
    /// Currently selected file index
    pub selected_file: usize,
    /// The loaded oto.ini (None if not yet loaded or doesn't exist)
    pub oto: Option<OtoIni>,
    /// Cached WAV data for the currently displayed file
    pub current_wav: Option<WavData>,
    /// Currently selected entry index within oto.entries
    pub selected_entry: usize,
    /// Which parameter field is focused in the params pane (0-5)
    pub selected_param: usize,
    /// Waveform view window: start time in ms
    pub view_start_ms: f64,
    /// Waveform view window: end time in ms
    pub view_end_ms: f64,
    /// Playback cursor position (ms)
    pub playback_pos_ms: Option<f64>,
    /// Loop range
    pub loop_start_ms: Option<f64>,
    pub loop_end_ms: Option<f64>,
    /// Audio player
    pub player: AudioPlayer,
    /// Currently focused pane
    pub focus: Focus,
    /// Search query (when focus == Search)
    pub search_query: String,
    /// Filtered file indices (empty = no filter active)
    pub search_results: Vec<usize>,
    /// Text input buffer (for direct value editing)
    pub input_buffer: Option<String>,
    /// Undo/redo history
    pub history: History<OtoEntry>,
    /// Status/error message to display
    pub status_msg: Option<(String, bool)>, // (msg, is_error)
    /// Whether to show help overlay
    pub show_help: bool,
    /// Whether app should exit
    pub should_quit: bool,
    /// Progress message (for async operations)
    pub progress: Option<String>,
}

impl App {
    pub fn new(voicebank_dir: PathBuf) -> Result<Self> {
        let wav_files = crate::util::scan_wav_files(&voicebank_dir)?;

        // Try to load oto.ini if it exists
        let oto_path = voicebank_dir.join("oto.ini");
        let oto = if oto_path.exists() {
            match OtoIni::load(&oto_path) {
                Ok(ini) => {
                    log::info!("Loaded {} entries from oto.ini", ini.entries.len());
                    Some(ini)
                }
                Err(e) => {
                    log::error!("Failed to load oto.ini: {e}");
                    Some(OtoIni::new(oto_path)) // empty
                }
            }
        } else {
            Some(OtoIni::new(oto_path))
        };

        let mut app = Self {
            voicebank_dir,
            wav_files,
            selected_file: 0,
            oto,
            current_wav: None,
            selected_entry: 0,
            selected_param: 0,
            view_start_ms: 0.0,
            view_end_ms: 1000.0,
            playback_pos_ms: None,
            loop_start_ms: None,
            loop_end_ms: None,
            player: AudioPlayer::new(),
            focus: Focus::FileList,
            search_query: String::new(),
            search_results: Vec::new(),
            input_buffer: None,
            history: History::new(100),
            status_msg: None,
            show_help: false,
            should_quit: false,
            progress: None,
        };

        app.load_current_wav();
        Ok(app)
    }

    /// Load WAV data for the currently selected file.
    pub fn load_current_wav(&mut self) {
        let path = match self.current_wav_path() {
            Some(p) => p,
            None => {
                self.current_wav = None;
                return;
            }
        };

        match WavData::load(&path) {
            Ok(wav) => {
                self.view_start_ms = 0.0;
                self.view_end_ms = wav.duration_ms;
                self.current_wav = Some(wav);
            }
            Err(e) => {
                log::warn!("Failed to load WAV {}: {e}", path.display());
                self.current_wav = None;
                self.set_status(format!("WAV load error: {e}"), true);
            }
        }
    }

    pub fn current_wav_path(&self) -> Option<PathBuf> {
        self.wav_files.get(self.selected_file).cloned()
    }

    /// Get the entries that match the currently selected WAV file.
    pub fn current_entries(&self) -> Vec<&OtoEntry> {
        let wav_name = self.current_wav_path()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        let wav_name = match wav_name {
            Some(n) => n,
            None => return Vec::new(),
        };
        match &self.oto {
            Some(ini) => ini.entries_for_wav(&wav_name),
            None => Vec::new(),
        }
    }

    /// Get mutable reference to the currently selected entry.
    pub fn current_entry_mut(&mut self) -> Option<&mut OtoEntry> {
        let wav_name = self.current_wav_path()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))?;

        let oto = self.oto.as_mut()?;
        let indices = oto.wav_index.get(&wav_name)?.clone();
        let idx = indices.get(self.selected_entry).copied()?;
        oto.entries.get_mut(idx)
    }

    pub fn current_entry(&self) -> Option<&OtoEntry> {
        let wav_name = self.current_wav_path()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))?;

        let oto = self.oto.as_ref()?;
        let indices = oto.wav_index.get(&wav_name)?;
        let idx = *indices.get(self.selected_entry)?;
        oto.entries.get(idx)
    }

    pub fn current_entry_index(&self) -> Option<usize> {
        let wav_name = self.current_wav_path()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))?;
        let oto = self.oto.as_ref()?;
        let indices = oto.wav_index.get(&wav_name)?;
        indices.get(self.selected_entry).copied()
    }

    pub fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_msg = Some((msg.into(), is_error));
    }

    pub fn clear_status(&mut self) {
        self.status_msg = None;
    }

    /// Save oto.ini with backup.
    pub fn save(&mut self) -> Result<()> {
        if let Some(ini) = &mut self.oto {
            ini.save().context("Failed to save oto.ini")?;
            ini.dirty = false;
            self.set_status("Saved oto.ini (backup created)".to_string(), false);
        }
        Ok(())
    }

    /// Undo last parameter change.
    pub fn undo(&mut self) {
        if let Some(prev) = self.history.undo() {
            if let Some(idx) = self.current_entry_index() {
                if let Some(ini) = &mut self.oto {
                    ini.entries[idx] = prev;
                    ini.dirty = true;
                    self.set_status("Undo".to_string(), false);
                }
            }
        } else {
            self.set_status("Nothing to undo".to_string(), false);
        }
    }

    pub fn redo(&mut self) {
        if let Some(next) = self.history.redo() {
            if let Some(idx) = self.current_entry_index() {
                if let Some(ini) = &mut self.oto {
                    ini.entries[idx] = next;
                    ini.dirty = true;
                    self.set_status("Redo".to_string(), false);
                }
            }
        } else {
            self.set_status("Nothing to redo".to_string(), false);
        }
    }

    /// Push current entry to history before editing.
    pub fn push_history(&mut self) {
        if let Some(entry) = self.current_entry().cloned() {
            self.history.push(entry);
        }
    }

    /// Move boundary for the focused parameter field.
    pub fn adjust_param(&mut self, delta_ms: f64) {
        self.push_history();
        if let Some(entry) = self.current_entry_mut() {
            match self.selected_param {
                0 => entry.offset = (entry.offset + delta_ms).max(0.0),
                1 => entry.preutterance = (entry.preutterance + delta_ms).max(0.0),
                2 => entry.consonant = (entry.consonant + delta_ms).max(0.0),
                3 => entry.cutoff += delta_ms,
                4 => entry.overlap = (entry.overlap + delta_ms).max(0.0),
                _ => {}
            }
            if let Some(ini) = &mut self.oto {
                ini.dirty = true;
            }
        }
    }

    /// Toggle play/stop for the current WAV.
    pub fn toggle_play(&mut self) {
        if self.player.is_playing() {
            self.player.stop();
            self.playback_pos_ms = None;
        } else {
            if let (Some(wav), Some(path)) = (&self.current_wav, self.current_wav_path()) {
                let start = self.loop_start_ms.unwrap_or(self.view_start_ms);
                let end = self.loop_end_ms;
                let wav_clone = wav.clone();
                let _ = self.player.play_range(&path, start, end, &wav_clone);
            }
        }
    }

    /// Update playback cursor from player state.
    pub fn tick_playback(&mut self) {
        if self.player.is_playing() {
            self.playback_pos_ms = self.player.elapsed_ms();
        } else {
            if self.playback_pos_ms.is_some() {
                self.playback_pos_ms = None;
            }
        }
    }

    /// Main event loop.
    pub fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
        loop {
            // Draw
            terminal.draw(|frame| {
                screens::draw_main(frame, self);
            })?;

            // Tick playback position (~60fps)
            if event::poll(Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        self.handle_key(key)?;
                    }
                    Event::Resize(_, _) => {
                        // ratatui handles resize automatically; just redraw
                    }
                    _ => {}
                }
            }

            self.tick_playback();

            if self.should_quit {
                // Prompt if unsaved changes
                if self.oto.as_ref().map_or(false, |o| o.dirty) {
                    // For simplicity, auto-save on quit with a warning
                    // In a fuller implementation, show a confirmation dialog
                    self.set_status("Unsaved changes! Press Q again to quit without saving, S to save.".to_string(), true);
                    self.should_quit = false; // require second press – handled below
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> Result<()> {
        let action = key_to_action(key);

        // Help overlay intercepts all keys
        if self.show_help {
            match action {
                Action::Help | Action::Cancel | Action::Quit => self.show_help = false,
                _ => {}
            }
            return Ok(());
        }

        // Text input mode
        if let Some(ref mut buf) = self.input_buffer {
            match action {
                Action::Confirm => {
                    let text = buf.clone();
                    self.input_buffer = None;
                    self.apply_text_input(&text);
                }
                Action::Cancel => {
                    self.input_buffer = None;
                }
                Action::TypeChar(c) => {
                    buf.push(c);
                }
                _ => {
                    // backspace
                    if key.code == crossterm::event::KeyCode::Backspace {
                        buf.pop();
                    }
                }
            }
            return Ok(());
        }

        // Search mode
        if self.focus == Focus::Search {
            match action {
                Action::Confirm => {
                    self.execute_search();
                    self.focus = Focus::FileList;
                }
                Action::Cancel => {
                    self.search_query.clear();
                    self.search_results.clear();
                    self.focus = Focus::FileList;
                }
                Action::TypeChar(c) => {
                    self.search_query.push(c);
                    self.execute_search();
                }
                _ => {
                    if key.code == crossterm::event::KeyCode::Backspace {
                        self.search_query.pop();
                        self.execute_search();
                    }
                }
            }
            return Ok(());
        }

        // Normal mode
        match action {
            Action::Quit => self.should_quit = true,
            Action::Help => self.show_help = true,
            Action::FocusNext => self.cycle_focus(1),
            Action::FocusPrev => self.cycle_focus(-1),
            Action::Save => {
                if let Err(e) = self.save() {
                    self.set_status(format!("Save failed: {e}"), true);
                }
            }
            Action::Undo => self.undo(),
            Action::Redo => self.redo(),
            Action::PlayStop => self.toggle_play(),
            Action::Search => {
                self.focus = Focus::Search;
                self.search_query.clear();
            }

            // Pane-specific actions
            _ => match self.focus {
                Focus::FileList => self.handle_filelist_action(action),
                Focus::Waveform => self.handle_waveform_action(action),
                Focus::Params => self.handle_params_action(action),
                _ => {}
            },
        }

        Ok(())
    }

    fn cycle_focus(&mut self, dir: i32) {
        let panes = [Focus::FileList, Focus::Waveform, Focus::Params];
        let current = panes.iter().position(|p| *p == self.focus).unwrap_or(0);
        let next = ((current as i32 + dir).rem_euclid(panes.len() as i32)) as usize;
        self.focus = panes[next].clone();
    }

    fn handle_filelist_action(&mut self, action: Action) {
        match action {
            Action::MoveUp => {
                if self.selected_file > 0 {
                    self.selected_file -= 1;
                    self.selected_entry = 0;
                    self.load_current_wav();
                }
            }
            Action::MoveDown => {
                let max = self.wav_files.len().saturating_sub(1);
                if self.selected_file < max {
                    self.selected_file += 1;
                    self.selected_entry = 0;
                    self.load_current_wav();
                }
            }
            Action::PageUp => {
                self.selected_file = self.selected_file.saturating_sub(10);
                self.selected_entry = 0;
                self.load_current_wav();
            }
            Action::PageDown => {
                let max = self.wav_files.len().saturating_sub(1);
                self.selected_file = (self.selected_file + 10).min(max);
                self.selected_entry = 0;
                self.load_current_wav();
            }
            Action::Confirm => {
                // Switch to waveform pane
                self.focus = Focus::Waveform;
            }
            Action::AutoEstimate => {
                self.run_auto_estimate();
            }
            Action::Validate => {
                self.run_validate();
            }
            _ => {}
        }
    }

    fn handle_waveform_action(&mut self, action: Action) {
        let step_ms = 5.0;
        let fine_ms = 1.0;
        match action {
            Action::MoveLeft => self.scroll_view(-step_ms),
            Action::MoveRight => self.scroll_view(step_ms),
            Action::MoveLeftFine => self.scroll_view(-fine_ms),
            Action::MoveRightFine => self.scroll_view(fine_ms),
            Action::IncrValue => self.zoom_view(0.8),   // zoom in
            Action::DecrValue => self.zoom_view(1.25),  // zoom out
            Action::SetLoopStart => {
                self.loop_start_ms = Some(self.view_start_ms + (self.view_end_ms - self.view_start_ms) / 2.0);
            }
            Action::SetLoopEnd => {
                self.loop_end_ms = Some(self.view_start_ms + (self.view_end_ms - self.view_start_ms) * 0.75);
            }
            Action::MoveUp => {
                // Switch to previous entry for this WAV
                if self.selected_entry > 0 {
                    self.selected_entry -= 1;
                }
            }
            Action::MoveDown => {
                let count = self.current_entries().len();
                if self.selected_entry + 1 < count {
                    self.selected_entry += 1;
                }
            }
            _ => {}
        }
    }

    fn handle_params_action(&mut self, action: Action) {
        match action {
            Action::MoveUp => {
                if self.selected_param > 0 {
                    self.selected_param -= 1;
                }
            }
            Action::MoveDown => {
                if self.selected_param < 5 {
                    self.selected_param += 1;
                }
            }
            Action::MoveLeft | Action::DecrValue => self.adjust_param(-5.0),
            Action::MoveRight | Action::IncrValue => self.adjust_param(5.0),
            Action::MoveLeftFine => self.adjust_param(-1.0),
            Action::MoveRightFine => self.adjust_param(1.0),
            Action::Confirm => {
                // Start text editing
                if let Some(entry) = self.current_entry() {
                    let current_val = self.param_value_str(entry);
                    self.input_buffer = Some(current_val);
                }
            }
            _ => {}
        }
    }

    fn param_value_str(&self, entry: &OtoEntry) -> String {
        match self.selected_param {
            0 => format!("{:.3}", entry.offset),
            1 => format!("{:.3}", entry.preutterance),
            2 => format!("{:.3}", entry.consonant),
            3 => format!("{:.3}", entry.cutoff),
            4 => format!("{:.3}", entry.overlap),
            5 => entry.alias.clone(),
            _ => String::new(),
        }
    }

    fn apply_text_input(&mut self, text: &str) {
        self.push_history();
        if let Some(entry) = self.current_entry_mut() {
            match self.selected_param {
                5 => {
                    // Alias editing
                    entry.alias = text.to_string();
                }
                _ => {
                    if let Ok(val) = text.trim().parse::<f64>() {
                        match self.selected_param {
                            0 => entry.offset = val.max(0.0),
                            1 => entry.preutterance = val.max(0.0),
                            2 => entry.consonant = val.max(0.0),
                            3 => entry.cutoff = val,
                            4 => entry.overlap = val.max(0.0),
                            _ => {}
                        }
                        if let Some(ini) = &mut self.oto {
                            ini.dirty = true;
                        }
                        self.set_status(format!("Set to {}", val), false);
                    } else {
                        self.set_status(format!("Invalid number: {text}"), true);
                    }
                }
            }
        }
    }

    fn scroll_view(&mut self, delta_ms: f64) {
        let dur = self.view_end_ms - self.view_start_ms;
        let max_end = self.current_wav.as_ref().map_or(self.view_end_ms, |w| w.duration_ms);
        let new_start = (self.view_start_ms + delta_ms).clamp(0.0, max_end - dur);
        self.view_start_ms = new_start;
        self.view_end_ms = new_start + dur;
    }

    fn zoom_view(&mut self, factor: f64) {
        let center = (self.view_start_ms + self.view_end_ms) / 2.0;
        let half_dur = (self.view_end_ms - self.view_start_ms) / 2.0 * factor;
        let half_dur = half_dur.clamp(10.0, 30_000.0);
        let max_end = self.current_wav.as_ref().map_or(60_000.0, |w| w.duration_ms);
        self.view_start_ms = (center - half_dur).max(0.0);
        self.view_end_ms = (center + half_dur).min(max_end);
    }

    fn execute_search(&mut self) {
        let q = self.search_query.to_lowercase();
        if q.is_empty() {
            self.search_results.clear();
            return;
        }
        self.search_results = self
            .wav_files
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.to_lowercase().contains(&q))
            })
            .map(|(i, _)| i)
            .collect();
    }

    fn run_auto_estimate(&mut self) {
        self.set_status("Auto-estimate: running simple silence detection…".to_string(), false);
        if let (Some(wav), Some(path)) = (&self.current_wav.clone(), self.current_wav_path()) {
            let wav_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            let onset = wav.find_onset_ms(0.02, 10.0).unwrap_or(0.0);
            let offset = wav.find_offset_ms(0.02, 10.0).unwrap_or(wav.duration_ms);

            // Check if entry exists; if not, create one
            let has_entry = self.oto.as_ref()
                .and_then(|ini| ini.wav_index.get(&wav_name))
                .map_or(false, |v| !v.is_empty());

            if !has_entry {
                // Create a new entry
                let stem = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let mut entry = OtoEntry::new_empty(wav_name.clone(), stem);

                // Simple heuristic: consonant region is first ~30% of voiced region
                let voiced_dur = offset - onset;
                entry.offset = onset;
                entry.preutterance = voiced_dur * 0.3;
                entry.consonant = voiced_dur * 0.5;
                entry.cutoff = OtoEntry::abs_end_to_cutoff(offset, wav.duration_ms);
                entry.overlap = voiced_dur * 0.1;
                entry.status = EntryStatus::Estimated { confidence: 0.5 };

                if let Some(ini) = &mut self.oto {
                    ini.entries.push(entry);
                    ini.dirty = true;
                    ini.rebuild_index();
                }
            }

            self.set_status(
                format!("Estimated: onset={:.0}ms offset={:.0}ms", onset, offset),
                false,
            );
        }
    }

    fn run_validate(&mut self) {
        if let (Some(entry), Some(wav)) = (self.current_entry().cloned(), &self.current_wav) {
            let result = crate::oto::validator::validate_entry(&entry, wav.duration_ms);
            if result.is_ok() {
                self.set_status("Validation: OK".to_string(), false);
            } else {
                let msgs: Vec<_> = result.errors.iter().chain(result.warnings.iter()).cloned().collect();
                self.set_status(format!("Validation: {}", msgs.join(" | ")), !result.errors.is_empty());
            }
        } else {
            self.set_status("No entry to validate".to_string(), false);
        }
    }
}
