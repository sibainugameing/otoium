//! oto.ini model with lossless handling of unparsable lines and comments.
//! Line format: `file.wav=alias,offset,consonant,cutoff,preutterance,overlap`
use crate::error::{Error, Result};
use crate::sjis;

#[derive(Debug, Clone, PartialEq)]
pub struct OtoEntry {
    pub file: String,
    pub alias: String,
    pub offset: f64,
    pub consonant: f64,
    pub cutoff: f64,
    pub preutter: f64,
    pub overlap: f64,
}

pub fn fmt_num(v: f64) -> String {
    let mut s = format!("{:.3}", v);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".into();
    }
    s
}

impl OtoEntry {
    pub fn key(&self) -> (String, String) {
        (self.file.clone(), self.alias.clone())
    }

    pub fn to_line(&self) -> String {
        format!(
            "{}={},{},{},{},{},{}",
            self.file,
            self.alias,
            fmt_num(self.offset),
            fmt_num(self.consonant),
            fmt_num(self.cutoff),
            fmt_num(self.preutter),
            fmt_num(self.overlap)
        )
    }

    pub fn parse_line(line: &str) -> Option<OtoEntry> {
        let (file, rest) = line.split_once('=')?;
        if file.trim().is_empty() || file.starts_with('#') || file.starts_with(';') {
            return None;
        }
        let f: Vec<&str> = rest.split(',').collect();
        if f.len() != 6 {
            return None;
        }
        let num = |s: &str| -> Option<f64> {
            let t = s.trim();
            if t.is_empty() { Some(0.0) } else { t.parse::<f64>().ok().filter(|v| v.is_finite()) }
        };
        Some(OtoEntry {
            file: file.to_string(),
            alias: f[0].to_string(),
            offset: num(f[1])?,
            consonant: num(f[2])?,
            cutoff: num(f[3])?,
            preutter: num(f[4])?,
            overlap: num(f[5])?,
        })
    }

    /// Equality after the 3-decimal rounding used when writing.
    pub fn same_values(&self, o: &OtoEntry) -> bool {
        let t = 0.0006;
        self.file == o.file
            && self.alias == o.alias
            && (self.offset - o.offset).abs() < t
            && (self.consonant - o.consonant).abs() < t
            && (self.cutoff - o.cutoff).abs() < t
            && (self.preutter - o.preutter).abs() < t
            && (self.overlap - o.overlap).abs() < t
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Pure ASCII so far; becomes `default_encoding` when non-ASCII text is added.
    Ascii,
    Utf8,
    Utf8Bom,
    ShiftJis,
}

#[derive(Debug, Clone)]
pub enum Line {
    Entry(OtoEntry),
    /// Comments, blank lines and lines that are not valid entries; preserved verbatim.
    Text(String),
    /// Lines that could not be decoded; preserved byte-for-byte.
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct OtoFile {
    pub lines: Vec<Line>,
    pub encoding: Encoding,
    pub eol: String,
    pub final_newline: bool,
}

impl OtoFile {
    pub fn new(encoding: Encoding) -> Self {
        Self { lines: Vec::new(), encoding, eol: "\r\n".into(), final_newline: true }
    }

    pub fn parse(bytes: &[u8]) -> OtoFile {
        let bom = bytes.starts_with(&[0xEF, 0xBB, 0xBF]);
        let body = if bom { &bytes[3..] } else { bytes };
        let encoding = if bom {
            Encoding::Utf8Bom
        } else if body.is_ascii() {
            Encoding::Ascii
        } else if std::str::from_utf8(body).is_ok() {
            Encoding::Utf8
        } else {
            Encoding::ShiftJis
        };
        let final_newline = body.ends_with(b"\n");
        let mut parts: Vec<&[u8]> = body.split(|&b| b == b'\n').collect();
        if final_newline || body.is_empty() {
            parts.pop();
        }
        let eol = if parts.first().map(|p| p.ends_with(b"\r")).unwrap_or(true) { "\r\n" } else { "\n" };
        let mut lines = Vec::with_capacity(parts.len());
        for p in parts {
            let p = p.strip_suffix(b"\r").unwrap_or(p);
            let text = match encoding {
                Encoding::ShiftJis => sjis::decode(p),
                _ => String::from_utf8(p.to_vec()).ok(),
            };
            match text {
                Some(t) => match OtoEntry::parse_line(&t) {
                    Some(e) => lines.push(Line::Entry(e)),
                    None => lines.push(Line::Text(t)),
                },
                None => lines.push(Line::Bytes(p.to_vec())),
            }
        }
        OtoFile { lines, encoding, eol: eol.into(), final_newline }
    }

    pub fn entries(&self) -> impl Iterator<Item = &OtoEntry> {
        self.lines.iter().filter_map(|l| if let Line::Entry(e) = l { Some(e) } else { None })
    }

    pub fn is_ascii(&self) -> bool {
        self.lines.iter().all(|l| match l {
            Line::Entry(e) => e.to_line().is_ascii(),
            Line::Text(t) => t.is_ascii(),
            Line::Bytes(b) => b.is_ascii(),
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        if self.encoding == Encoding::Utf8Bom {
            out.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
        }
        let n = self.lines.len();
        for (i, l) in self.lines.iter().enumerate() {
            match l {
                Line::Bytes(b) => out.extend_from_slice(b),
                Line::Entry(_) | Line::Text(_) => {
                    let t = match l {
                        Line::Entry(e) => e.to_line(),
                        Line::Text(t) => t.clone(),
                        Line::Bytes(_) => unreachable!(),
                    };
                    match self.encoding {
                        Encoding::ShiftJis => out.extend(
                            sjis::encode(&t).ok_or_else(|| Error::Oto(format!("cannot encode in Shift_JIS: {t}")))?,
                        ),
                        _ => out.extend_from_slice(t.as_bytes()),
                    }
                }
            }
            if i + 1 < n || self.final_newline {
                out.extend_from_slice(self.eol.as_bytes());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_format() {
        assert_eq!(fmt_num(12.0), "12");
        assert_eq!(fmt_num(12.3456), "12.346");
        assert_eq!(fmt_num(-0.0001), "0");
        assert_eq!(fmt_num(-300.5), "-300.5");
    }

    #[test]
    fn parse_and_roundtrip_utf8() {
        let src = "# comment\r\nka.wav=ka,45.3,120,-300,80,30\r\n\r\nbad line\r\n";
        let f = OtoFile::parse(src.as_bytes());
        assert_eq!(f.encoding, Encoding::Ascii);
        assert_eq!(f.entries().count(), 1);
        assert_eq!(f.to_bytes().unwrap(), src.as_bytes());
    }

    #[test]
    fn shift_jis_roundtrip_preserves_undecodable() {
        let mut raw = Vec::new();
        raw.extend(sjis::encode("あ.wav=あ,1,2,-3,4,5").unwrap());
        raw.extend_from_slice(b"\r\n");
        raw.extend_from_slice(&[0x8A, 0xBF, b'=', b'x', b'\r', b'\n']); // kanji line: undecodable
        let f = OtoFile::parse(&raw);
        assert_eq!(f.encoding, Encoding::ShiftJis);
        assert_eq!(f.entries().count(), 1);
        assert_eq!(f.entries().next().unwrap().alias, "あ");
        assert_eq!(f.to_bytes().unwrap(), raw);
    }

    #[test]
    fn empty_fields_are_zero_and_commas_validated() {
        let e = OtoEntry::parse_line("a.wav=,,,,,").unwrap();
        assert_eq!(e.offset, 0.0);
        assert!(OtoEntry::parse_line("a.wav=a,1,2").is_none());
        assert!(OtoEntry::parse_line("a.wav=a,x,2,3,4,5").is_none());
    }
}
