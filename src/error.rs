use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Wav(String),
    Oto(String),
    Pron(String),
    Align(String),
    Store(String),
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Wav(s) => write!(f, "WAV error: {s}"),
            Error::Oto(s) => write!(f, "oto.ini error: {s}"),
            Error::Pron(s) => write!(f, "pronunciation error: {s}"),
            Error::Align(s) => write!(f, "alignment error: {s}"),
            Error::Store(s) => write!(f, "store error: {s}"),
            Error::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
