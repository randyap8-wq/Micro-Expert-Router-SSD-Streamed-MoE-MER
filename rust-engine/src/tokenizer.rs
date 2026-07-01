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

    /// Largest token id this tokenizer can emit, including added and
    /// special tokens. For the byte fallback this is always 255. For a
    /// HuggingFace tokenizer it is the maximum id over the full vocabulary
    /// (`with_added_tokens = true`), which covers reserved/special ids that
    /// may sit above the base-vocabulary count.
    pub fn max_token_id(&self) -> u32 {
        match self {
            Tokenizer::Bytes => 255,
            #[cfg(feature = "tokenizer")]
            Tokenizer::Hf(t) => t.get_vocab(true).values().copied().max().unwrap_or(0),
        }
    }

    /// Validate that every token id this tokenizer can emit is addressable
    /// by a model whose output/embedding vocabulary is `model_vocab_size`.
    ///
    /// The invariant is `max_token_id < model_vocab_size` (ids are
    /// zero-based). This deliberately checks the maximum *emittable* id —
    /// including added and special tokens — rather than requiring the raw
    /// base-vocabulary count to equal `model_vocab_size`, because real
    /// checkpoints routinely pad the embedding table beyond the tokenizer's
    /// base vocab and reserve high ids for special tokens.
    pub fn validate_vocab_compat(&self, model_vocab_size: usize) -> Result<(), TokenizerError> {
        let max_id = self.max_token_id() as usize;
        if max_id >= model_vocab_size {
            return Err(TokenizerError::Backend(format!(
                "tokenizer can emit token id {max_id} but model vocab_size is \
                 {model_vocab_size}; every token id must be < model vocab_size"
            )));
        }
        Ok(())
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

    #[test]
    fn byte_tokenizer_max_token_id_is_255() {
        assert_eq!(Tokenizer::bytes().max_token_id(), 255);
    }

    #[test]
    fn vocab_compat_accepts_model_larger_than_max_token_id() {
        // Byte tokenizer emits ids 0..=255; a model with vocab_size 256
        // addresses all of them.
        assert!(Tokenizer::bytes().validate_vocab_compat(256).is_ok());
        assert!(Tokenizer::bytes().validate_vocab_compat(100_000).is_ok());
    }

    #[test]
    fn vocab_compat_rejects_max_token_id_at_or_above_vocab_size() {
        // vocab_size == max_id fails (ids are zero-based, so 255 needs
        // vocab_size >= 256); anything smaller also fails.
        let err = Tokenizer::bytes().validate_vocab_compat(255).unwrap_err();
        match err {
            TokenizerError::Backend(m) => assert!(m.contains("255") && m.contains("vocab_size")),
            other => panic!("expected Backend error, got {other}"),
        }
        assert!(Tokenizer::bytes().validate_vocab_compat(10).is_err());
    }
}
