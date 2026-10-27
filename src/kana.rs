//! Japanese pronunciation parsing: hiragana / katakana / romaji -> phoneme sequences.
//! Handles youon (きゃ), sokuon (っ), hatsuon (ん), chouon (ー), foreign-sound combos (ふぁ, てぃ, ...).
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhClass {
    Vowel,
    Plosive,
    Affricate,
    Fricative,
    Nasal,
    Liquid,
    Glide,
    /// Geminate closure (sokuon).
    Closure,
}

pub fn classify(sym: &str) -> Option<PhClass> {
    use PhClass::*;
    Some(match sym {
        "a" | "i" | "u" | "e" | "o" => Vowel,
        "N" | "m" | "n" | "ny" | "my" => Nasal,
        "r" | "ry" => Liquid,
        "y" | "w" => Glide,
        "ts" | "ch" | "j" => Affricate,
        "s" | "sh" | "h" | "hy" | "f" | "fy" | "z" | "v" => Fricative,
        "cl" => Closure,
        "k" | "ky" | "g" | "gy" | "t" | "ty" | "d" | "dy" | "p" | "py" | "b" | "by" => Plosive,
        _ => return None,
    })
}

pub fn is_voiced(sym: &str) -> bool {
    matches!(
        sym,
        "a" | "i" | "u" | "e" | "o" | "N" | "m" | "n" | "ny" | "my" | "r" | "ry" | "y" | "w" | "g" | "gy"
            | "d" | "dy" | "b" | "by" | "z" | "j" | "v"
    )
}

/// Vowel or moraic nasal (mora nucleus).
pub fn is_nucleus(sym: &str) -> bool {
    matches!(classify(sym), Some(PhClass::Vowel)) || sym == "N"
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pron {
    pub phonemes: Vec<String>,
    /// A prolonged-sound mark (ー) was present. It does not add phonemes.
    pub long: bool,
}

fn to_hira(c: char) -> char {
    let u = c as u32;
    if (0x30A1..=0x30F6).contains(&u) {
        char::from_u32(u - 0x60).unwrap_or(c)
    } else {
        c
    }
}

pub fn is_kana_char(c: char) -> bool {
    let u = c as u32;
    (0x3041..=0x3096).contains(&u) || (0x30A1..=0x30FA).contains(&u) || c == 'ー'
}

fn base(c: char) -> Option<&'static str> {
    Some(match c {
        'あ' => "a", 'い' => "i", 'う' => "u", 'え' => "e", 'お' => "o",
        'か' => "ka", 'き' => "ki", 'く' => "ku", 'け' => "ke", 'こ' => "ko",
        'さ' => "sa", 'し' => "shi", 'す' => "su", 'せ' => "se", 'そ' => "so",
        'た' => "ta", 'ち' => "chi", 'つ' => "tsu", 'て' => "te", 'と' => "to",
        'な' => "na", 'に' => "ni", 'ぬ' => "nu", 'ね' => "ne", 'の' => "no",
        'は' => "ha", 'ひ' => "hi", 'ふ' => "fu", 'へ' => "he", 'ほ' => "ho",
        'ま' => "ma", 'み' => "mi", 'む' => "mu", 'め' => "me", 'も' => "mo",
        'や' => "ya", 'ゆ' => "yu", 'よ' => "yo",
        'ら' => "ra", 'り' => "ri", 'る' => "ru", 'れ' => "re", 'ろ' => "ro",
        'わ' => "wa", 'を' => "wo",
        'が' => "ga", 'ぎ' => "gi", 'ぐ' => "gu", 'げ' => "ge", 'ご' => "go",
        'ざ' => "za", 'じ' => "ji", 'ず' => "zu", 'ぜ' => "ze", 'ぞ' => "zo",
        'だ' => "da", 'ぢ' => "ji", 'づ' => "zu", 'で' => "de", 'ど' => "do",
        'ば' => "ba", 'び' => "bi", 'ぶ' => "bu", 'べ' => "be", 'ぼ' => "bo",
        'ぱ' => "pa", 'ぴ' => "pi", 'ぷ' => "pu", 'ぺ' => "pe", 'ぽ' => "po",
        'ゔ' => "vu",
        'ぁ' => "a", 'ぃ' => "i", 'ぅ' => "u", 'ぇ' => "e", 'ぉ' => "o",
        _ => return None,
    })
}

fn is_rv(c: char) -> bool {
    matches!(c, 'a' | 'i' | 'u' | 'e' | 'o')
}

fn combine(base: &str, small: char) -> Option<String> {
    match small {
        'ゃ' | 'ゅ' | 'ょ' => {
            let v = match small {
                'ゃ' => 'a',
                'ゅ' => 'u',
                _ => 'o',
            };
            if base == "te" || base == "de" {
                return Some(format!("{}y{}", &base[..1], v));
            }
            if base.len() >= 2 && base.ends_with('i') {
                let stem = &base[..base.len() - 1];
                if matches!(stem, "sh" | "ch" | "j") {
                    Some(format!("{stem}{v}"))
                } else {
                    Some(format!("{stem}y{v}"))
                }
            } else {
                None
            }
        }
        'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' => {
            let v = match small {
                'ぁ' => 'a',
                'ぃ' => 'i',
                'ぅ' => 'u',
                'ぇ' => 'e',
                _ => 'o',
            };
            if base == "u" {
                return Some(format!("w{v}"));
            }
            if base == "i" && v == 'e' {
                return Some("ye".to_string());
            }
            let mut stem = base.to_string();
            while stem.chars().last().map(is_rv).unwrap_or(false) {
                stem.pop();
            }
            if stem.is_empty() {
                None
            } else {
                Some(format!("{stem}{v}"))
            }
        }
        _ => None,
    }
}

const CL2: [&str; 14] = ["ch", "sh", "ts", "ky", "gy", "ny", "hy", "my", "ry", "by", "py", "dy", "ty", "fy"];
const CL1: [&str; 17] = ["k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m", "r", "y", "w", "f", "v", "j"];

pub fn romaji_to_phonemes(tok: &str) -> Result<Vec<String>> {
    if tok == "cl" {
        return Ok(vec!["cl".to_string()]);
    }
    let s: Vec<char> = tok.chars().collect();
    let n = s.len();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    let low = |c: char| c.to_ascii_lowercase();
    while i < n {
        let ch = s[i];
        if ch == 'N' {
            out.push("N".into());
            i += 1;
            continue;
        }
        let c = low(ch);
        if is_rv(c) {
            out.push(c.to_string());
            i += 1;
            continue;
        }
        let next = s.get(i + 1).map(|x| low(*x));
        if c == 'n' && !matches!(next, Some(x) if is_rv(x) || x == 'y') {
            out.push("N".into());
            i += 1;
            if next == Some('n') {
                let after = s.get(i + 1).map(|x| low(*x));
                if !matches!(after, Some(x) if is_rv(x) || x == 'y') {
                    i += 1;
                }
            }
            continue;
        }
        if let Some(nx) = next {
            let tch = c == 't' && nx == 'c' && s.get(i + 2).map(|x| low(*x)) == Some('h');
            if (nx == c && c != 'n') || tch {
                out.push("cl".into());
                i += 1;
                continue;
            }
        }
        let two = match next {
            Some(nx) => format!("{c}{nx}"),
            None => String::new(),
        };
        if CL2.contains(&two.as_str()) {
            out.push(two);
            i += 2;
        } else if CL1.contains(&c.to_string().as_str()) {
            out.push(c.to_string());
            i += 1;
        } else {
            return Err(Error::Pron(format!("unrecognised romaji '{tok}' at '{ch}'")));
        }
    }
    Ok(out)
}

pub fn parse_kana_token(tok: &str) -> Result<Pron> {
    let chars: Vec<char> = tok.chars().map(to_hira).collect();
    let mut p = Pron::default();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        match c {
            'っ' => p.phonemes.push("cl".into()),
            'ー' => p.long = true,
            'ん' => p.phonemes.push("N".into()),
            _ => {
                let b = base(c).ok_or_else(|| Error::Pron(format!("unknown kana '{c}' in '{tok}'")))?;
                let mut r = b.to_string();
                if let Some(&nx) = chars.get(i + 1) {
                    if let Some(com) = combine(&r, nx) {
                        r = com;
                        i += 1;
                    }
                }
                p.phonemes.extend(romaji_to_phonemes(&r)?);
            }
        }
        i += 1;
    }
    Ok(p)
}

/// Parse a single whitespace-free token (kana or romaji).
pub fn parse_token(tok: &str) -> Result<Pron> {
    if tok.chars().any(is_kana_char) {
        parse_kana_token(tok)
    } else {
        Ok(Pron { phonemes: romaji_to_phonemes(tok)?, long: false })
    }
}

/// Split an alias / file stem into pronunciation tokens. "-" (word-initial mark) and "R" are ignored.
pub fn split_tokens(s: &str) -> Vec<&str> {
    s.split(|c: char| c.is_whitespace() || c == '_' || c == '\u{3000}')
        .filter(|t| !t.is_empty() && *t != "-" && *t != "R" && *t != "・")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ph(s: &str) -> Vec<String> {
        parse_token(s).unwrap().phonemes
    }

    #[test]
    fn basic_and_youon() {
        assert_eq!(ph("か"), vec!["k", "a"]);
        assert_eq!(ph("きゃ"), vec!["ky", "a"]);
        assert_eq!(ph("しゃ"), vec!["sh", "a"]);
        assert_eq!(ph("じゅ"), vec!["j", "u"]);
        assert_eq!(ph("ちゃ"), vec!["ch", "a"]);
        assert_eq!(ph("つ"), vec!["ts", "u"]);
    }

    #[test]
    fn katakana_and_special() {
        assert_eq!(ph("カ"), vec!["k", "a"]);
        assert_eq!(ph("ファ"), vec!["f", "a"]);
        assert_eq!(ph("ティ"), vec!["t", "i"]);
        assert_eq!(ph("ウィ"), vec!["w", "i"]);
        assert_eq!(ph("ん"), vec!["N"]);
        assert_eq!(ph("っか"), vec!["cl", "k", "a"]);
        let p = parse_token("かー").unwrap();
        assert_eq!(p.phonemes, vec!["k", "a"]);
        assert!(p.long);
    }

    #[test]
    fn romaji() {
        assert_eq!(ph("ka"), vec!["k", "a"]);
        assert_eq!(ph("kya"), vec!["ky", "a"]);
        assert_eq!(ph("shi"), vec!["sh", "i"]);
        assert_eq!(ph("tsu"), vec!["ts", "u"]);
        assert_eq!(ph("n"), vec!["N"]);
        assert_eq!(ph("N"), vec!["N"]);
        assert_eq!(ph("kka"), vec!["cl", "k", "a"]);
        assert_eq!(ph("k"), vec!["k"]);
        assert!(parse_token("qxz").is_err());
    }

    #[test]
    fn tokens() {
        assert_eq!(split_tokens("a ka"), vec!["a", "ka"]);
        assert_eq!(split_tokens("- ka"), vec!["ka"]);
        assert_eq!(split_tokens("あ_か"), vec!["あ", "か"]);
    }
}
