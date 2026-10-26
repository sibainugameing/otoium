/// Centralized key binding definitions.
/// All key→action mappings are defined here for easy reconfiguration.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// High-level application actions, independent of physical keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    // Navigation
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    MoveLeftFine,     // Shift+Left
    MoveRightFine,    // Shift+Right
    PageUp,
    PageDown,
    GotoFirst,
    GotoLast,

    // Pane management
    FocusNext,        // Tab
    FocusPrev,        // Shift+Tab

    // Editing
    Confirm,          // Enter
    Cancel,           // Esc
    IncrValue,        // +
    DecrValue,        // -
    TypeChar(char),   // any printable character

    // File operations
    Save,             // S
    Search,           // /

    // Playback
    PlayStop,         // Space
    SetLoopStart,     // [
    SetLoopEnd,       // ]

    // Processing
    AutoEstimate,     // A
    Validate,         // V
    BatchEdit,        // B

    // History
    Undo,             // U or Ctrl+Z
    Redo,             // Ctrl+R

    // App
    Help,             // ?
    Quit,             // Q or Ctrl+C
    Resize,           // terminal resize event (synthetic)

    // Unknown / unhandled
    Unknown,
}

/// Map a crossterm KeyEvent to an Action.
pub fn key_to_action(event: KeyEvent) -> Action {
    use KeyCode::*;
    use KeyModifiers as M;

    let mods = event.modifiers;
    let ctrl = mods.contains(M::CONTROL);
    let shift = mods.contains(M::SHIFT);
    let bare = mods == M::NONE || mods == M::SHIFT;

    match event.code {
        Up if bare => Action::MoveUp,
        Down if bare => Action::MoveDown,
        Left if !shift => Action::MoveLeft,
        Right if !shift => Action::MoveRight,
        Left if shift => Action::MoveLeftFine,
        Right if shift => Action::MoveRightFine,
        PageUp => Action::PageUp,
        PageDown => Action::PageDown,
        Home => Action::GotoFirst,
        End => Action::GotoLast,

        Tab if !shift => Action::FocusNext,
        BackTab => Action::FocusPrev,       // Shift+Tab often sent as BackTab

        Enter => Action::Confirm,
        Esc => Action::Cancel,

        Char('+') | Char('=') if bare => Action::IncrValue,
        Char('-') if bare => Action::DecrValue,

        Char('s') | Char('S') if bare => Action::Save,
        Char('/') if bare => Action::Search,
        Char(' ') => Action::PlayStop,
        Char('[') if bare => Action::SetLoopStart,
        Char(']') if bare => Action::SetLoopEnd,

        Char('a') | Char('A') if bare => Action::AutoEstimate,
        Char('v') | Char('V') if bare => Action::Validate,
        Char('b') | Char('B') if bare => Action::BatchEdit,

        Char('u') | Char('U') if bare => Action::Undo,
        Char('z') if ctrl => Action::Undo,
        Char('r') if ctrl => Action::Redo,

        Char('?') => Action::Help,
        Char('q') | Char('Q') if bare => Action::Quit,
        Char('c') if ctrl => Action::Quit,

        Char(c) if bare && (c.is_alphanumeric() || c.is_ascii_punctuation()) => {
            Action::TypeChar(c)
        }

        _ => Action::Unknown,
    }
}

/// Help text for the status bar (brief).
pub fn status_bar_help(pane: &str) -> &'static str {
    match pane {
        "files" => "[↑↓] Select  [Tab] Focus  [/] Search  [Space] Play  [S] Save  [Q] Quit",
        "waveform" => "[←→] Move  [Shift+←→] Fine  [Space] Play  [[] Loop  [Tab] Focus",
        "params" => "[↑↓] Field  [+-] Adjust  [Enter] Edit  [Tab] Focus  [U] Undo",
        "search" => "[Enter] Confirm  [Esc] Cancel",
        _ => "[Tab] Focus  [Space] Play  [S] Save  [Q] Quit  [?] Help",
    }
}

/// Full help text for the help screen.
pub const HELP_TEXT: &str = r#"
OTO Assist – Keyboard Reference
═══════════════════════════════

NAVIGATION
  ↑ / ↓           Select file or parameter field
  ← / →           Move boundary / playback position
  Shift + ← / →   Fine adjustment (1ms steps)
  Page Up / Down   Scroll file list
  Home / End       Jump to first / last file
  Tab              Switch focus between panes
  Shift+Tab        Switch focus backwards

EDITING
  Enter            Confirm / begin text input
  Esc              Cancel / exit text input
  + / =            Increase value
  -                Decrease value
  U  /  Ctrl+Z     Undo
  Ctrl+R           Redo

PLAYBACK
  Space            Play / Stop current file
  [                Set loop start at playhead
  ]                Set loop end at playhead

FILE OPERATIONS
  /                Search files
  S                Save oto.ini (creates backup)
  B                Batch edit mode

PROCESSING
  A                Auto-estimate boundaries for current file
  V                Validate current entry

DISPLAY
  ?                Show / hide this help
  Q  /  Ctrl+C     Quit

NOTE: All processing is local. No files are sent to any server.
"#;
