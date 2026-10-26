mod audio;
mod auto;
mod oto;
mod ui;
mod util;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;

use ui::app::App;

fn main() -> Result<()> {
    // Initialize logger (RUST_LOG=debug for verbose output)
    env_logger::init();

    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    let voicebank_dir = if args.len() > 1 {
        std::path::PathBuf::from(&args[1])
    } else {
        std::env::current_dir()?
    };

    if !voicebank_dir.exists() {
        eprintln!("Error: directory does not exist: {}", voicebank_dir.display());
        std::process::exit(1);
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app with panic hook to restore terminal
    let result = run_app(&mut terminal, voicebank_dir);

    // Always restore terminal on exit
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Application error: {e}");
        std::process::exit(1);
    }

    Ok(())
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    voicebank_dir: std::path::PathBuf,
) -> Result<()> {
    let mut app = App::new(voicebank_dir)?;

    // Install panic hook so terminal is restored even on panic
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    app.run(terminal)
}
