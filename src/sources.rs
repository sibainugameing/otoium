//! Pronunciation sources besides the file name: index.csv.
//! ASSUMPTION (unverified against Moresampler docs): each line is `<wav file name>,<alias or pronunciation>`;
//! extra columns are ignored; lines starting with '#' are skipped. The format is therefore best-effort.
use crate::sjis;
use std::collections::HashMap;
use std::path::Path;

pub fn read_index_csv(path: &Path) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return map,
    };
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(&bytes);
    let text = match std::str::from_utf8(body) {
        Ok(s) => s.to_string(),
        Err(_) => match sjis::decode(body) {
            Some(s) => s,
            None => return map,
        },
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        if cols.len() < 2 || cols[0].is_empty() || cols[1].is_empty() {
            continue;
        }
        let mut f = cols[0].to_lowercase();
        if !f.ends_with(".wav") {
            f.push_str(".wav");
        }
        let list = map.entry(f).or_default();
        if !list.iter().any(|a| a == cols[1]) {
            list.push(cols[1].to_string());
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_basic_index() {
        let dir = std::env::temp_dir().join(format!("otoium_idx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("index.csv");
        std::fs::write(&p, "# c\nKA.wav,ka\nki,き,extra\n\nbad\n").unwrap();
        let m = read_index_csv(&p);
        assert_eq!(m["ka.wav"], vec!["ka"]);
        assert_eq!(m["ki.wav"], vec!["き"]);
        assert_eq!(m.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
