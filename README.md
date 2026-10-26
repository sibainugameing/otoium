<p align="center">
    <img width="160" alt="Image" alt="Project Logo" src="https://github.com/user-attachments/assets/6decc58f-dfff-42c4-afb7-3e209b5c3bba" />
</p>
# otoium

for utau oto.ini

# OTO Assist

A Rust-based terminal TUI application for creating and editing UTAU voicebank oto.ini configurations.

## Verified Environments

| Environment | Status |
|------|------|
| Ubuntu 22.04 LTS | Target platform (verification required) |
| Apple Silicon Mac (macOS 13+) | Target platform (verification required) |

> **Note:** This repository was created in a network-restricted environment. Please run `cargo build` and tests in your own environment.

---

## Setup

### Ubuntu / Debian

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Install audio development libraries
sudo apt update
sudo apt install -y \
    libasound2-dev \
    pkg-config \
    build-essential

# Clone the repository and build
git clone <repo-url> oto_assist
cd oto_assist
cargo build --release

# Launch
./target/release/oto_assist /path/to/voicebank
```

### macOS (Apple Silicon / Intel)

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build (no additional libraries required because CoreAudio is built into macOS)
git clone <repo-url> oto_assist
cd oto_assist
cargo build --release

# Launch
./target/release/oto_assist /path/to/voicebank
```

---

## Launching the Application

```bash
# Launch with a target directory
oto_assist /path/to/voicebank

# Launch in the current directory
oto_assist .

# Launch with debug logging enabled
RUST_LOG=debug oto_assist /path/to/voicebank
```

---

## Keyboard Shortcuts

| Key | Action |
|------|------|
| `↑` / `↓` | Select files or parameter fields |
| `←` / `→` | Move the waveform cursor / adjust values (5 ms) |
| `Shift+←` / `Shift+→` | Fine adjustment (1 ms) |
| `Tab` | Move to the next pane |
| `Shift+Tab` | Move to the previous pane |
| `Enter` | Confirm / enter text input mode |
| `Esc` | Cancel |
| `+` / `-` | Increase / decrease numeric values |
| `Space` | Play / stop |
| `[` / `]` | Set loop range boundaries |
| `/` | Search files |
| `S` | Save (automatically creates a backup) |
| `U` / `Ctrl+Z` | Undo |
| `Ctrl+R` | Redo |
| `A` | Automatic boundary estimation (currently silence detection only) |
| `V` | Validate the current entry |
| `?` | Display help |
| `Q` / `Ctrl+C` | Quit |

---

## Screen Layout

```
OTO Assist | Project: my_voicebank [modified]
Directory: /path/to/my_voicebank
WAV: 139  Entries: 42  Errors: 2

┌ Files [139] ────┬ Waveform: _あ.wav ──────────────────────────────┐
│ ▶ _あ.wav     · │  │                          │         │         │
│   _い.wav     ✓ │  ·                          ·         ·         │
│   _う.wav     ~ │  █████▇▅▃▂▁·················                    │
│   _え.wav     ! │  │                          │         │         │
│   _お.wav     · │  O     V   P       C                  X        │
├─────────────────┘  0   10  70      80                 950ms      │
├─────────────────────────────────────────────────────────────────────┤
│ ▶ Alias          あ                                                  │
│   Offset         10.000 ms                                          │
│   Preutterance   70.000 ms                                          │
│   Consonant      80.000 ms                                          │
│   Cutoff        -50.000 ms                                          │
│   Overlap        10.000 ms                                          │
├─────────────────────────────────────────────────────────────────────┤
│ [↑↓] Field  [+-] Adjust  [Enter] Edit  [Tab] Focus  [U] Undo       │
└─────────────────────────────────────────────────────────────────────┘
```

### Marker Symbols

| Symbol | Meaning |
|------|------|
| `O` | Offset |
| `P` | Preutterance |
| `C` | Consonant |
| `X` | Cutoff (effective blank position) |
| `V` | Overlap start (preutterance - overlap) |
| `►` | Playback cursor |

### File List Status Symbols

| Symbol | Meaning |
|------|------|
| `·` | Unconfigured or unchecked |
| `~` | Automatically estimated |
| `✓` | Confirmed |
| `!` | Error detected |

---

## oto.ini Specification Notes

```
wavfile.wav=alias,offset,consonant,cutoff,preutterance,overlap
```

**Note: Pay attention to the sign of `cutoff`.**

| `cutoff` Value | Meaning |
|-------------|------|
| Positive (e.g., `800.0`) | Milliseconds from the beginning of the WAV file (absolute position) |
| Negative (e.g., `-50.0`) | Milliseconds measured backward from the end of the WAV file |
| `0.0` | End of the WAV file |

This application internally prioritizes negative values (relative to the end of the file).

---

## Dependencies and Licenses

| Crate | Version | License | Purpose |
|----------|-----------|-----------|------|
| ratatui | 0.26 | MIT | TUI framework |
| crossterm | 0.27 | MIT | Terminal interaction |
| hound | 3.5 | MIT | WAV file reading |
| symphonia | 0.5 | MPL-2.0 | Additional audio formats |
| rodio | 0.17 | MIT/Apache-2.0 | Audio playback |
| rustfft | 6.1 | MIT | FFT (for spectral analysis) |
| ndarray | 0.15 | MIT/Apache-2.0 | Array operations |
| encoding_rs | 0.8 | Apache-2.0/MIT | Shift-JIS ↔ UTF-8 conversion |
| serde / serde_json | 1 | MIT/Apache-2.0 | Configuration storage |
| anyhow / thiserror | 1 | MIT/Apache-2.0 | Error handling |
| unicode-width | 0.1 | MIT/Apache-2.0 | CJK character width calculation |
| chrono | 0.4 | MIT/Apache-2.0 | Timestamps |

symphonia is licensed under MPL-2.0. When distributing binaries, comply with the requirements of the MPL-2.0 license.

---

## Running Tests

```bash
# Run all tests
cargo test

# Test specific modules
cargo test oto::parser
cargo test util::history

# Show detailed output
cargo test -- --nocapture
```

---

## Implemented Features (Through Phase 2)

- [x] Specify a directory and list WAV files
- [x] Load oto.ini (automatic UTF-8 / Shift-JIS detection)
- [x] Save oto.ini (with backups and atomic writes)
- [x] Convert cutoff values between positive and negative representations (implemented according to the specification)
- [x] Three-pane TUI layout (file list / waveform / parameters)
- [x] Render WAV waveforms in the terminal using Unicode block characters
- [x] Display the time axis and markers (O / P / C / X / V)
- [x] Edit five parameters and the alias
- [x] Keyboard controls (navigation / fine adjustment / direct input)
- [x] Undo / Redo (up to 100 steps)
- [x] Automatic backups before saving
- [x] File search ( `/` key)
- [x] Automatic estimation (basic silence-detection implementation)
- [x] Validation
- [x] Help screen
- [x] Terminal resizing support
- [x] Restore terminal state on exit
- [x] Restore terminal state through a panic-recovery hook

---

## Unimplemented Features and Known Limitations

| Feature | Status | Notes |
|------|------|------|
| Audio playback | Requires testing | Silently disabled if rodio initialization fails |
| HMM phoneme alignment | Not implemented | Requires a pretrained model. License and data sources have not been decided |
| VCV / CVVC alias generation | Not implemented | Phoneme parser not yet implemented |
| Spectrogram display | Not implemented | rustfft is integrated, but rendering is not implemented |
| Batch editing mode | Not implemented | Phase 3 |
| Template functionality | Not implemented | Phase 3 |
| Duplicate alias detection UI | Not implemented | Detection logic is implemented |
| Accuracy evaluation | Not performed | Requires ground-truth boundary data |

---

## Automatic Oto Configuration Algorithm (Current Implementation)

### Phase 1: Silence Detection (Implemented)

1. Calculate RMS energy using 10 ms windows.
2. Scan from the beginning and identify the first position exceeding the threshold (0.02) as the onset.
3. Scan backward from the end and identify the first position exceeding the threshold as the offset.
4. Assign parameter values based on proportions of the audio segment:
   - overlap = 10% of the audio segment duration
   - preutterance = 30% of the audio segment duration
   - consonant = 50% of the audio segment duration
   - cutoff = value relative to the end of the WAV file

**Note:** This is not a "high-accuracy" implementation. Since the error against ground-truth boundaries has not been measured, no numerical accuracy claims are made.

### Phase 2 and Beyond: HMM Phoneme Alignment (Not Implemented)

The following areas are under investigation, with reference to the implementations of SHIRO and Moresampler:

- Pretrained data for Japanese phoneme models (HMM / HSMM)
- Implementation of a Viterbi decoder
- Boundary correction rules based on phoneme classes
- Compatibility verification with Moresampler (differences from the original implementation require further verification)
