//! Partial Shift_JIS (CP932) codec: ASCII, half-width katakana, hiragana, katakana, full-width
//! alphanumerics and a few punctuation marks. Anything else is reported as unsupported so callers can
//! preserve raw bytes instead of corrupting them. Kanji are NOT supported (no table is embedded).

pub fn decode(b: &[u8]) -> Option<String> {
    let mut s = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c < 0x80 {
            s.push(c as char);
            i += 1;
        } else if (0xA1..=0xDF).contains(&c) {
            s.push(char::from_u32(0xFF61 + (c as u32 - 0xA1))?);
            i += 1;
        } else if (0x81..=0x9F).contains(&c) || (0xE0..=0xFC).contains(&c) {
            let c2 = *b.get(i + 1)?;
            s.push(map_dbcs(((c as u32) << 8) | c2 as u32)?);
            i += 2;
        } else {
            return None;
        }
    }
    Some(s)
}

fn map_dbcs(code: u32) -> Option<char> {
    let cp = match code {
        0x8140 => 0x3000,
        0x8141 => 0x3001,
        0x8142 => 0x3002,
        0x8145 => 0x30FB,
        0x815B => 0x30FC,
        0x824F..=0x8258 => 0xFF10 + (code - 0x824F),
        0x8260..=0x8279 => 0xFF21 + (code - 0x8260),
        0x8281..=0x829A => 0xFF41 + (code - 0x8281),
        0x829F..=0x82F1 => 0x3041 + (code - 0x829F),
        0x8340..=0x837E => 0x30A1 + (code - 0x8340),
        0x8380..=0x8396 => 0x30A1 + (code - 0x8341),
        _ => return None,
    };
    char::from_u32(cp)
}

fn encode_char(c: char, out: &mut Vec<u8>) -> bool {
    let u = c as u32;
    let code: u32 = match u {
        0..=0x7F => {
            out.push(u as u8);
            return true;
        }
        0xFF61..=0xFF9F => {
            out.push((0xA1 + (u - 0xFF61)) as u8);
            return true;
        }
        0x3000 => 0x8140,
        0x3001 => 0x8141,
        0x3002 => 0x8142,
        0x30FB => 0x8145,
        0x30FC => 0x815B,
        0xFF10..=0xFF19 => 0x824F + (u - 0xFF10),
        0xFF21..=0xFF3A => 0x8260 + (u - 0xFF21),
        0xFF41..=0xFF5A => 0x8281 + (u - 0xFF41),
        0x3041..=0x3093 => 0x829F + (u - 0x3041),
        0x30A1..=0x30DF => 0x8340 + (u - 0x30A1),
        0x30E0..=0x30F6 => 0x8341 + (u - 0x30A1),
        _ => return false,
    };
    out.push((code >> 8) as u8);
    out.push((code & 0xFF) as u8);
    true
}

pub fn encode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        if !encode_char(c, &mut out) {
            return None;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_kana() {
        for s in ["あいうえお", "カキクケコ", "ぁヴん", "ミムヶ", "ka_あ.wav", "ー、。"] {
            let e = encode(s).unwrap_or_else(|| panic!("encode {s}"));
            assert_eq!(decode(&e).unwrap(), s);
        }
    }

    #[test]
    fn known_bytes() {
        assert_eq!(encode("あ").unwrap(), vec![0x82, 0xA0]);
        assert_eq!(encode("ア").unwrap(), vec![0x83, 0x41]);
        assert_eq!(encode("ー").unwrap(), vec![0x81, 0x5B]);
    }

    #[test]
    fn kanji_unsupported() {
        assert!(encode("漢").is_none());
        assert!(decode(&[0x8A, 0xBF]).is_none());
    }
}
