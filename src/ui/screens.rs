use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use unicode_width::UnicodeWidthStr;

use crate::audio::waveform::{render_markers, render_time_axis, render_waveform};
use crate::oto::{EntryStatus, OtoEntry};
use crate::ui::app::{App, Focus};
use crate::ui::keybindings::{status_bar_help, HELP_TEXT};

// ─────────────────────────────────────────────────────────────────────────────
// Color palette (all monochrome-safe: use styles without relying on color alone)
// ─────────────────────────────────────────────────────────────────────────────

const STYLE_SELECTED: Style = Style::new()
    .fg(Color::Black)
    .bg(Color::Yellow)
    .add_modifier(Modifier::BOLD);

const STYLE_FOCUSED_BORDER: Style = Style::new().fg(Color::Cyan);
const STYLE_NORMAL_BORDER: Style = Style::new().fg(Color::DarkGray);
const STYLE_ERROR: Style = Style::new().fg(Color::Red).add_modifier(Modifier::BOLD);
const STYLE_WARN: Style = Style::new().fg(Color::Yellow);
const STYLE_OK: Style = Style::new().fg(Color::Green);
const STYLE_ESTIMATED: Style = Style::new().fg(Color::Magenta);
const STYLE_DIRTY: Style = Style::new().fg(Color::Yellow).add_modifier(Modifier::ITALIC);

// ─────────────────────────────────────────────────────────────────────────────
// Top-level drawing entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn draw_main(frame: &mut Frame, app: &App) {
    let area = frame.size();

    // Show help overlay if requested
    if app.show_help {
        draw_help_overlay(frame, area);
        return;
    }

    // Minimum usable size
    if area.width < 40 || area.height < 10 {
        let msg = Paragraph::new("Terminal too small.\nPlease resize to at least 40×10.")
            .style(Style::default().fg(Color::Red));
        frame.render_widget(msg, area);
        return;
    }

    // Layout:
    // ┌ header (3 lines) ─────────────────────────────────────────────┐
    // ├ files (left) │ waveform (center/right)                        │
    // ├ params (bottom band)                                           │
    // └ status bar (1 line) ──────────────────────────────────────────┘

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),     // main content
            Constraint::Length(8),  // params
            Constraint::Length(2),  // status
        ])
        .split(area);

    draw_header(frame, app, chunks[0]);

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(75),
        ])
        .split(chunks[1]);

    draw_file_list(frame, app, main_chunks[0]);
    draw_waveform(frame, app, main_chunks[1]);
    draw_params(frame, app, chunks[2]);
    draw_status_bar(frame, app, chunks[3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Header
// ─────────────────────────────────────────────────────────────────────────────

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let dirty_marker = if app.oto.as_ref().map_or(false, |o| o.dirty) {
        " [modified]"
    } else {
        ""
    };

    let project_name = app
        .voicebank_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");

    let wav_count = app.wav_files.len();
    let entry_count = app.oto.as_ref().map_or(0, |o| o.entries.len());
    let error_count = app
        .oto
        .as_ref()
        .map_or(0, |o| {
            o.entries
                .iter()
                .filter(|e| matches!(e.status, EntryStatus::Error(_)))
                .count()
        });

    let lines = vec![
        Line::from(vec![
            Span::styled("otoium", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" | Project: {project_name}{dirty_marker}")),
        ]),
        Line::from(format!(
            "Directory: {}",
            app.voicebank_dir.display()
        )),
        Line::from(format!(
            "WAV: {wav_count}  Entries: {entry_count}  Errors: {error_count}"
        )),
    ];

    let header = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::BOTTOM));
    frame.render_widget(header, area);
}

// ─────────────────────────────────────────────────────────────────────────────
// File list
// ─────────────────────────────────────────────────────────────────────────────

fn draw_file_list(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == Focus::FileList;
    let border_style = if focused {
        STYLE_FOCUSED_BORDER
    } else {
        STYLE_NORMAL_BORDER
    };

    // Build the display list
    let files_to_show: Vec<(usize, &std::path::PathBuf)> = if !app.search_results.is_empty() {
        app.search_results
            .iter()
            .map(|&i| (i, &app.wav_files[i]))
            .collect()
    } else {
        app.wav_files.iter().enumerate().collect()
    };

    let items: Vec<ListItem> = files_to_show
        .iter()
        .map(|(orig_idx, path)| {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");

            // Check entry status for this file
            let wav_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let status_char = match app.oto.as_ref().and_then(|ini| ini.wav_index.get(wav_name)) {
                None => ' ',
                Some(indices) if indices.is_empty() => ' ',
                Some(indices) => {
                    let any_error = indices.iter().any(|&i| {
                        matches!(app.oto.as_ref().unwrap().entries[i].status, EntryStatus::Error(_))
                    });
                    let any_confirmed = indices.iter().any(|&i| {
                        matches!(app.oto.as_ref().unwrap().entries[i].status, EntryStatus::Confirmed)
                    });
                    let any_estimated = indices.iter().any(|&i| {
                        matches!(app.oto.as_ref().unwrap().entries[i].status, EntryStatus::Estimated { .. })
                    });
                    if any_error { '!' }
                    else if any_confirmed { '✓' }
                    else if any_estimated { '~' }
                    else { '·' }
                }
            };

            let prefix = if *orig_idx == app.selected_file { "▶ " } else { "  " };

            // Truncate name to fit
            let max_name_width = (area.width as usize).saturating_sub(6);
            let display_name = if name.width() > max_name_width {
                format!("{}…", &name[..max_name_width.saturating_sub(1)])
            } else {
                name.to_string()
            };

            let style = if *orig_idx == app.selected_file && focused {
                STYLE_SELECTED
            } else if *orig_idx == app.selected_file {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let status_style = match status_char {
                '!' => STYLE_ERROR,
                '✓' => STYLE_OK,
                '~' => STYLE_ESTIMATED,
                _ => Style::default(),
            };

            ListItem::new(Line::from(vec![
                Span::styled(format!("{prefix}{display_name}"), style),
                Span::styled(format!(" {status_char}"), status_style),
            ]))
        })
        .collect();

    let title = if !app.search_results.is_empty() {
        format!("Files [{}/{}]", app.search_results.len(), app.wav_files.len())
    } else {
        format!("Files [{}]", app.wav_files.len())
    };

    let mut state = ListState::default();
    state.select(Some(app.selected_file));

    let list = List::new(items)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .highlight_style(STYLE_SELECTED);

    frame.render_stateful_widget(list, area, &mut state);
}

// ─────────────────────────────────────────────────────────────────────────────
// Waveform panel
// ─────────────────────────────────────────────────────────────────────────────

fn draw_waveform(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == Focus::Waveform;
    let border_style = if focused {
        STYLE_FOCUSED_BORDER
    } else {
        STYLE_NORMAL_BORDER
    };

    let wav_name = app
        .current_wav_path()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();

    let block = Block::default()
        .title(format!("Waveform: {wav_name}"))
        .borders(Borders::ALL)
        .border_style(border_style);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 3 || inner.width < 10 {
        return;
    }

    let width = inner.width as usize;
    let start_ms = app.view_start_ms;
    let end_ms = app.view_end_ms;

    match &app.current_wav {
        None => {
            let msg = Paragraph::new("No WAV loaded");
            frame.render_widget(msg, inner);
        }
        Some(wav) => {
            let sr = wav.sample_rate as f64;
            let start_sample = (start_ms / 1000.0 * sr) as usize;
            let end_sample = (end_ms / 1000.0 * sr) as usize;

            // Reserve rows for: markers (1), waveform body, time axis (1)
            let waveform_height = (inner.height as usize).saturating_sub(2);
            let waveform_lines = render_waveform(
                &wav.samples,
                width,
                waveform_height,
                start_sample,
                end_sample,
            );

            // Build marker list from current entry
            let mut markers: Vec<(&str, f64)> = Vec::new();
            if let Some(entry) = app.current_entry() {
                markers.push(("O", entry.offset));
                markers.push(("P", entry.offset + entry.preutterance));
                markers.push(("C", entry.offset + entry.consonant));
                let abs_end = entry.cutoff_to_abs_end(wav.duration_ms);
                markers.push(("X", abs_end));
                let overlap_start = entry.offset + entry.preutterance - entry.overlap;
                if overlap_start >= 0.0 {
                    markers.push(("V", overlap_start));
                }
            }
            if let Some(pos) = app.playback_pos_ms {
                markers.push(("►", pos));
            }

            let (sym_line, val_line) =
                render_markers(width, start_ms, end_ms, &markers, None);
            let time_axis = render_time_axis(width, start_ms, end_ms);

            // Render all lines
            let mut all_lines: Vec<Line> = Vec::new();
            all_lines.push(Line::from(sym_line));
            for wl in &waveform_lines {
                all_lines.push(Line::from(wl.as_str()));
            }
            all_lines.push(Line::from(Span::styled(val_line, Style::default().fg(Color::Cyan))));
            all_lines.push(Line::from(Span::styled(time_axis, Style::default().fg(Color::DarkGray))));

            let paragraph = Paragraph::new(Text::from(all_lines));
            frame.render_widget(paragraph, inner);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter editing panel
// ─────────────────────────────────────────────────────────────────────────────

fn draw_params(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == Focus::Params;
    let border_style = if focused {
        STYLE_FOCUSED_BORDER
    } else {
        STYLE_NORMAL_BORDER
    };

    let block = Block::default()
        .title("Parameters")
        .borders(Borders::ALL)
        .border_style(border_style);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let param_names = ["Alias", "Offset", "Preutterance", "Consonant", "Cutoff", "Overlap"];
    let units = ["", " ms", " ms", " ms", " ms", " ms"];

    let entry = app.current_entry();
    let wav_dur = app.current_wav.as_ref().map(|w| w.duration_ms);

    let values: Vec<String> = match entry {
        None => param_names.iter().map(|_| "-".to_string()).collect(),
        Some(e) => vec![
            e.alias.clone(),
            format!("{:.3}", e.offset),
            format!("{:.3}", e.preutterance),
            format!("{:.3}", e.consonant),
            format!("{:.3}", e.cutoff),
            format!("{:.3}", e.overlap),
        ],
    };

    let mut lines: Vec<Line> = Vec::new();

    for (i, (name, unit)) in param_names.iter().zip(units.iter()).enumerate() {
        let val = if let (Some(buf), true) = (&app.input_buffer, app.selected_param == i && focused) {
            format!("{}_", buf) // show cursor in editing mode
        } else {
            format!("{}{}", values[i], unit)
        };

        let is_selected = app.selected_param == i;
        let style = if is_selected && focused {
            STYLE_SELECTED
        } else if is_selected {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let prefix = if is_selected { "▶ " } else { "  " };
        let label = format!("{prefix}{name:<14} {val}");
        lines.push(Line::from(Span::styled(label, style)));
    }

    // Validation status
    if let (Some(entry), Some(dur)) = (entry, wav_dur) {
        let result = crate::oto::validator::validate_entry(entry, dur);
        if !result.errors.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("✗ {}", result.errors.join(" | ")),
                STYLE_ERROR,
            )));
        } else if !result.warnings.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("⚠ {}", result.warnings.join(" | ")),
                STYLE_WARN,
            )));
        }
    }

    let para = Paragraph::new(Text::from(lines));
    frame.render_widget(para, inner);
}

// ─────────────────────────────────────────────────────────────────────────────
// Status bar
// ─────────────────────────────────────────────────────────────────────────────

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    // Top line: status/error message or search prompt
    // Bottom line: key hints

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    // Status message or search
    let status_widget = if app.focus == Focus::Search {
        Paragraph::new(Line::from(vec![
            Span::raw("Search: "),
            Span::styled(&app.search_query, Style::default().add_modifier(Modifier::UNDERLINED)),
            Span::raw("_"),
        ]))
    } else if let Some((msg, is_err)) = &app.status_msg {
        let style = if *is_err { STYLE_ERROR } else { STYLE_OK };
        Paragraph::new(Span::styled(msg.as_str(), style))
    } else if app.player.is_playing() {
        let pos = app.playback_pos_ms.unwrap_or(0.0);
        Paragraph::new(Span::styled(
            format!("▶ Playing… {:.0}ms", pos),
            STYLE_OK,
        ))
    } else {
        Paragraph::new("")
    };
    frame.render_widget(status_widget, chunks[0]);

    // Key hints
    let pane_name = match app.focus {
        Focus::FileList => "files",
        Focus::Waveform => "waveform",
        Focus::Params => "params",
        Focus::Search => "search",
        Focus::Help => "help",
    };
    let hints = Paragraph::new(Line::from(Span::styled(
        status_bar_help(pane_name),
        Style::default().fg(Color::DarkGray),
    )));
    frame.render_widget(hints, chunks[1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Help overlay
// ─────────────────────────────────────────────────────────────────────────────

fn draw_help_overlay(frame: &mut Frame, area: Rect) {
    // Center a popup
    let popup_width = area.width.min(70);
    let popup_height = area.height.min(35);
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let help = Paragraph::new(HELP_TEXT)
        .block(
            Block::default()
                .title("Help – [?] or [Esc] to close")
                .borders(Borders::ALL)
                .border_style(STYLE_FOCUSED_BORDER),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(help, popup_area);
}
