//! Tokenizer abstraction (Phase 4).
//!
//! When the `tokenizer` feature is enabled, this module loads a real
//! HuggingFace tokenizer (`tokenizer.json`) via the [`tokenizers`] crate.
//! When disabled (the default), it falls back to a deterministic
//! byte-level tokenizer that maps every input byte to its u8 value as a
//! token id (vocab_size = 256). The fallback exists so the rest of the
//! server (HTTP API, request scheduling, generation loop) can be built
//! and tested without pulling in a heavy native-code dep.
//!
//! Both implementations expose the same minimal interface used by the
//! generation loop:
//! - [`Tokenizer::encode`]
//! - [`Tokenizer::decode`]
//! - [`Tokenizer::vocab_size`]

use std::path::Path;

/// Errors a tokenizer can produce.
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Backend(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "tokenizer io error: {e}"),
            TokenizerError::Backend(m) => write!(f, "tokenizer backend error: {m}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

/// Type-erased tokenizer the server uses.
pub enum Tokenizer {
    /// Deterministic byte-level fallback. Vocab is 0..=255; every byte of
    /// the input is one token. Decode is just `String::from_utf8_lossy`.
    Bytes,
    #[cfg(feature = "tokenizer")]
    Hf(tokenizers::Tokenizer),
}

impl Tokenizer {
    /// Always succeeds. Use when no `tokenizer.json` is available.
    pub fn bytes() -> Self {
        Tokenizer::Bytes
    }

    /// Try to load a HuggingFace `tokenizer.json` from disk. Falls back
    /// to the byte tokenizer when the `tokenizer` feature is disabled
    /// or the file isn't there.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        #[cfg(feature = "tokenizer")]
        {
            let inner = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| TokenizerError::Backend(e.to_string()))?;
            return Ok(Tokenizer::Hf(inner));
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            // Behave deterministically when the user asks for a tokenizer
            // file but the backend is not compiled in: surface the
            // missing-feature condition rather than silently downgrading.
            let _ = path;
            Err(TokenizerError::Backend(
                "tokenizer feature is disabled at compile time; rebuild with \
                 `--features tokenizer` to load tokenizer.json".to_string(),
            ))
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Bytes => 256,
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => t.get_vocab_size(true),
        }
    }

    pub fn encode(&self, input: &str) -> Result<Vec<u32>, TokenizerError> {
        match self {
            Tokenizer::Bytes => Ok(input.bytes().map(|b| b as u32).collect()),
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => {
                let enc = t
                    .encode(input, false)
                    .map_err(|e| TokenizerError::Backend(e.to_string()))?;
                Ok(enc.get_ids().to_vec())
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        match self {
            Tokenizer::Bytes => {
                let bytes: Vec<u8> = ids
                    .iter()
                    .map(|&id| (id & 0xFF) as u8)
                    .collect();
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => t
                .decode(ids, true)
                .map_err(|e| TokenizerError::Backend(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trips_ascii() {
        let t = Tokenizer::bytes();
        let ids = t.encode("hello").unwrap();
        assert_eq!(ids, vec![104, 101, 108, 108, 111]);
        let s = t.decode(&ids).unwrap();
        assert_eq!(s, "hello");
        assert_eq!(t.vocab_size(), 256);
    }

    #[test]
    fn byte_tokenizer_handles_utf8_lossily() {
        let t = Tokenizer::bytes();
        let ids = t.encode("héllo").unwrap();
        // "h" + 2 bytes for é + "llo"
        assert_eq!(ids.len(), 6);
        // Round-trip succeeds (the two bytes for é form a valid UTF-8 sequence).
        let s = t.decode(&ids).unwrap();
        assert_eq!(s, "héllo");
    }

    #[cfg(not(feature = "tokenizer"))]
    #[test]
    fn missing_tokenizer_feature_returns_error_for_file_load() {
        let path = std::path::PathBuf::from("/nonexistent/tokenizer.json");
        match Tokenizer::from_file(&path) {
            Err(TokenizerError::Backend(msg)) => assert!(msg.contains("tokenizer feature")),
            other => panic!("expected Backend error about disabled feature, got {}",
                match other { Ok(_) => "Ok(_)".to_string(), Err(e) => format!("Err({e})") }),
        }
    }
}
