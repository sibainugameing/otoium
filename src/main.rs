use otoium::batch::{run_auto, Progress};
use otoium::eval::compare;
use otoium::format::BankFormat;
use otoium::oto::OtoFile;
use otoium::pipeline::{Config, Engine};
use otoium::store::MergeMode;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

fn usage() -> ! {
    eprintln!(
        "otoium (engine only; TUI not implemented)\n\n\
         USAGE:\n  otoium auto <voicebank_dir> [--mode append|update|replace] [--recursive] [--jobs N]\n\
         \x20                 [--dry-run] [--overwrite-manual] [--format cv|vcv|cvvc]\n\
         \x20                 [--positive-cutoff] [--report FILE] [--json FILE]\n\
         \x20 otoium eval <generated oto.ini> <reference oto.ini> [--large-ms N]\n"
    );
    std::process::exit(2);
}

fn arg(a: &[String], i: usize) -> &String {
    a.get(i).unwrap_or_else(|| usage())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("auto") => cmd_auto(&args[1..]),
        Some("eval") => cmd_eval(&args[1..]),
        _ => usage(),
    }
}

fn cmd_auto(a: &[String]) {
    let mut dir: Option<PathBuf> = None;
    let mut cfg = Config::default();
    let (mut report, mut json) = (None::<PathBuf>, None::<PathBuf>);
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--mode" => {
                cfg.store.mode = match arg(a, i + 1).as_str() {
                    "append" => MergeMode::Append,
                    "update" => MergeMode::Update,
                    "replace" => MergeMode::Replace,
                    _ => usage(),
                };
                i += 1;
            }
            "--recursive" => cfg.recursive = true,
            "--dry-run" => cfg.store.dry_run = true,
            "--overwrite-manual" => cfg.store.overwrite_manual = true,
            "--positive-cutoff" => cfg.policy.negative_cutoff = false,
            "--jobs" => {
                cfg.jobs = arg(a, i + 1).parse().unwrap_or_else(|_| usage());
                i += 1;
            }
            "--format" => {
                cfg.format_hint = Some(match arg(a, i + 1).as_str() {
                    "cv" => BankFormat::Cv,
                    "vcv" => BankFormat::Vcv,
                    "cvvc" => BankFormat::Cvvc,
                    _ => usage(),
                });
                i += 1;
            }
            "--report" => {
                report = Some(PathBuf::from(arg(a, i + 1)));
                i += 1;
            }
            "--json" => {
                json = Some(PathBuf::from(arg(a, i + 1)));
                i += 1;
            }
            s if !s.starts_with("--") && dir.is_none() => dir = Some(PathBuf::from(s)),
            _ => usage(),
        }
        i += 1;
    }
    let dir = dir.unwrap_or_else(|| usage());
    let cancel = AtomicBool::new(false);
    let engine = Engine::acoustic();
    let progress = |p: &Progress| {
        if p.stage == "done" {
            eprintln!(
                "[{}/{}] ok {} warn {} fail {} | {} | {:.1}s",
                p.done, p.total, p.ok, p.warn, p.fail, p.current, p.elapsed_secs
            );
        }
    };
    match run_auto(&dir, &cfg, &engine, &cancel, &progress) {
        Ok(rep) => {
            let text = rep.render_text();
            println!("{text}");
            if let Some(p) = report {
                if let Err(e) = std::fs::write(&p, &text) {
                    eprintln!("could not write report: {e}");
                }
            }
            if let Some(p) = json {
                if let Err(e) = std::fs::write(&p, rep.render_json()) {
                    eprintln!("could not write json: {e}");
                }
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_eval(a: &[String]) {
    let mut paths = Vec::new();
    let mut large = 20.0;
    let mut i = 0;
    while i < a.len() {
        if a[i] == "--large-ms" {
            large = a.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            i += 1;
        } else {
            paths.push(a[i].clone());
        }
        i += 1;
    }
    if paths.len() != 2 {
        usage();
    }
    let load = |p: &str| match std::fs::read(p) {
        Ok(b) => OtoFile::parse(&b),
        Err(e) => {
            eprintln!("cannot read {p}: {e}");
            std::process::exit(1);
        }
    };
    let (g, r) = (load(&paths[0]), load(&paths[1]));
    let rep = compare(&g, &r, large);
    println!("{}", rep.render(r.entries().count()));
}
