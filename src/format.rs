//! Alias parsing, voicebank format detection (CV / VCV / CVVC) and expected-alias coverage.
use crate::error::{Error, Result};
use crate::kana::{is_nucleus, parse_token, split_tokens};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    /// One mora (or single phoneme): "ka", "a".
    Plain,
    /// Vowel + CV: "a ka".
    VcvLike,
    /// Vowel + consonant only: "a k".
    VcLike,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankFormat {
    Cv,
    Vcv,
    Cvvc,
}

#[derive(Debug, Clone)]
pub struct ParsedAlias {
    pub tokens: Vec<Vec<String>>,
    pub phonemes: Vec<String>,
    pub long: bool,
    pub kind: AliasKind,
}

pub fn parse_alias(s: &str) -> Result<ParsedAlias> {
    let toks = split_tokens(s);
    if toks.is_empty() {
        return Err(Error::Pron(format!("alias '{s}' has no pronunciation tokens")));
    }
    let mut tokens = Vec::new();
    let mut long = false;
    for t in &toks {
        let p = parse_token(t)?;
        long |= p.long;
        if p.phonemes.is_empty() {
            return Err(Error::Pron(format!("token '{t}' produced no phonemes")));
        }
        tokens.push(p.phonemes);
    }
    let phonemes: Vec<String> = tokens.iter().flatten().cloned().collect();
    let kind = match tokens.len() {
        1 => AliasKind::Plain,
        2 => {
            let a = &tokens[0];
            let b = &tokens[1];
            if a.len() == 1 && is_nucleus(&a[0]) {
                if is_nucleus(b.last().map(|s| s.as_str()).unwrap_or("")) {
                    AliasKind::VcvLike
                } else {
                    AliasKind::VcLike
                }
            } else {
                AliasKind::Other
            }
        }
        _ => AliasKind::Other,
    };
    Ok(ParsedAlias { tokens, phonemes, long, kind })
}

#[derive(Debug, Clone)]
pub struct FormatHypothesis {
    pub format: BankFormat,
    pub score: f64,
}

/// Rank format hypotheses from alias structure only.
/// NOTE: scoring weights are heuristic and unvalidated; acoustic hypothesis comparison is NOT implemented.
pub fn detect_format(aliases: &[ParsedAlias]) -> Vec<FormatHypothesis> {
    let n = aliases.len().max(1) as f64;
    let frac = |k: AliasKind| aliases.iter().filter(|a| a.kind == k).count() as f64 / n;
    let (p, v, c) = (frac(AliasKind::Plain), frac(AliasKind::VcvLike), frac(AliasKind::VcLike));
    let mut h = vec![
        FormatHypothesis { format: BankFormat::Cv, score: (p - v - c).max(0.0) + 0.01 },
        FormatHypothesis { format: BankFormat::Vcv, score: v * 1.2 },
        FormatHypothesis { format: BankFormat::Cvvc, score: if p > 0.05 { c * 1.2 } else { c * 0.8 } },
    ];
    h.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    h
}

const CV_MORAS: &[&str] = &[
    "a", "i", "u", "e", "o", "ka", "ki", "ku", "ke", "ko", "sa", "shi", "su", "se", "so", "ta", "chi", "tsu", "te",
    "to", "na", "ni", "nu", "ne", "no", "ha", "hi", "fu", "he", "ho", "ma", "mi", "mu", "me", "mo", "ya", "yu", "yo",
    "ra", "ri", "ru", "re", "ro", "wa", "wo", "N", "ga", "gi", "gu", "ge", "go", "za", "ji", "zu", "ze", "zo", "da",
    "de", "do", "ba", "bi", "bu", "be", "bo", "pa", "pi", "pu", "pe", "po", "kya", "kyu", "kyo", "sha", "shu", "sho",
    "cha", "chu", "cho", "nya", "nyu", "nyo", "hya", "hyu", "hyo", "mya", "myu", "myo", "rya", "ryu", "ryo", "gya",
    "gyu", "gyo", "ja", "ju", "jo", "bya", "byu", "byo", "pya", "pyu", "pyo",
];
const VC_CONSONANTS: &[&str] = &[
    "k", "s", "sh", "t", "ch", "ts", "n", "h", "m", "y", "r", "w", "g", "z", "j", "d", "b", "p", "ky", "ny", "hy",
    "my", "ry", "gy", "by", "py",
];
const VCV_VOWELS: &[&str] = &["a", "i", "u", "e", "o", "N"];

/// Expected phoneme sequences (space-joined keys) for a standard Japanese bank of the given format.
pub fn expected_phoneme_keys(format: BankFormat) -> Vec<String> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let cv: Vec<Vec<String>> = CV_MORAS.iter().filter_map(|m| parse_token(m).ok().map(|p| p.phonemes)).collect();
    match format {
        BankFormat::Cv => out.extend(cv),
        BankFormat::Vcv => {
            for v in VCV_VOWELS {
                for m in &cv {
                    let mut seq = vec![v.to_string()];
                    seq.extend(m.iter().cloned());
                    out.push(seq);
                }
            }
        }
        BankFormat::Cvvc => {
            out.extend(cv);
            for v in VCV_VOWELS {
                for c in VC_CONSONANTS {
                    out.push(vec![v.to_string(), c.to_string()]);
                }
            }
        }
    }
    out.into_iter().map(|s| s.join(" ")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        assert_eq!(parse_alias("ka").unwrap().kind, AliasKind::Plain);
        assert_eq!(parse_alias("a ka").unwrap().kind, AliasKind::VcvLike);
        assert_eq!(parse_alias("あ か").unwrap().kind, AliasKind::VcvLike);
        assert_eq!(parse_alias("a k").unwrap().kind, AliasKind::VcLike);
        assert_eq!(parse_alias("- ka").unwrap().kind, AliasKind::Plain);
        assert!(parse_alias("   ").is_err());
    }

    #[test]
    fn detection() {
        let v: Vec<_> = ["a ka", "i ki", "u ku", "e ke", "o ko"].iter().map(|s| parse_alias(s).unwrap()).collect();
        assert_eq!(detect_format(&v)[0].format, BankFormat::Vcv);
        let c: Vec<_> = ["ka", "ki", "ku", "ke"].iter().map(|s| parse_alias(s).unwrap()).collect();
        assert_eq!(detect_format(&c)[0].format, BankFormat::Cv);
        let m: Vec<_> = ["ka", "ki", "a k", "i k"].iter().map(|s| parse_alias(s).unwrap()).collect();
        assert_eq!(detect_format(&m)[0].format, BankFormat::Cvvc);
    }
}
