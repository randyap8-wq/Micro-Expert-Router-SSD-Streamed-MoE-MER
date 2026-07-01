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

/// Incremental streaming decoder (hardening pass, F2).
///
/// The streaming path previously re-decoded the *entire* cumulative
/// completion after every token and diffed against a cloned cumulative
/// string — `O(tokens²)` total decode work plus one full-string clone
/// per token. `StreamDecoder` instead keeps a **bounded look-behind
/// window** of recent token ids, just large enough for UTF-8 /
/// byte-fallback / BPE boundary correctness:
///
/// * Each pushed token decodes only the window (a handful of ids), not
///   the whole completion.
/// * A trailing run of U+FFFD replacement characters — the signature
///   of a UTF-8 sequence still split across byte-fallback tokens — is
///   **held back** until a later token completes (or disproves) the
///   sequence, so multi-byte characters are never emitted torn.
/// * Once the window's decode has been fully emitted, the window is
///   trimmed to its final token (kept as decoder context for
///   BPE/metaspace joining) and the bookkeeping prefix is re-derived,
///   so the window never grows with the stream.
/// * A hard cap ([`Self::MAX_LOOKBEHIND_TOKENS`]) force-flushes
///   pathological streams that never resolve (e.g. an endless run of
///   invalid bytes), bounding both memory and per-token decode cost.
///
/// [`Self::finish`] flushes any held-back text at end of stream.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    /// Bounded look-behind window of the most recent token ids.
    ids: Vec<u32>,
    /// Prefix of `decode(&ids)` that has already been emitted.
    emitted: String,
}

impl StreamDecoder {
    /// Hard cap on the look-behind window. A well-formed UTF-8
    /// scalar spans at most 4 bytes (≤ 4 byte-fallback tokens), so 64
    /// is generous margin for merged BPE pieces while keeping the
    /// worst-case per-token decode cost small and constant.
    pub const MAX_LOOKBEHIND_TOKENS: usize = 64;

    pub fn new() -> Self {
        Self::default()
    }

    /// Number of ids currently held in the look-behind window
    /// (bounded by [`Self::MAX_LOOKBEHIND_TOKENS`]).
    pub fn lookbehind_len(&self) -> usize {
        self.ids.len()
    }

    /// Feed one generated token id; returns the newly stable decoded
    /// text (possibly empty while a multi-byte sequence is pending).
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<String, TokenizerError> {
        self.ids.push(id);
        let s = tokenizer.decode(&self.ids)?;
        // Hold back a trailing replacement-character run: it may be an
        // incomplete UTF-8 sequence that the next byte-fallback token
        // completes. Force-flush at the window cap so an endless run
        // of genuinely invalid bytes cannot grow the window forever.
        let force = self.ids.len() >= Self::MAX_LOOKBEHIND_TOKENS;
        let hold = if force {
            0
        } else {
            trailing_replacement_len(&s)
        };
        let safe_end = s.len() - hold;
        let delta = if s.len() >= self.emitted.len() && s.starts_with(self.emitted.as_str()) {
            if safe_end > self.emitted.len() {
                s[self.emitted.len()..safe_end].to_string()
            } else {
                String::new()
            }
        } else {
            // The decoder revised earlier characters (rare, but
            // possible with BPE cleanup rules): emit the window's
            // revised text and resynchronise, mirroring the legacy
            // cumulative-diff fallback.
            s[..safe_end].to_string()
        };
        if hold == 0 {
            // Fully emitted: trim the window to its last token (kept
            // as decode context for BPE joining) and re-derive the
            // emitted prefix so the invariant `emitted` ⊑ `decode(ids)`
            // holds for the trimmed window too.
            if self.ids.len() > 1 {
                let last = *self.ids.last().expect("just pushed");
                self.ids.clear();
                self.ids.push(last);
            }
            self.emitted = tokenizer.decode(&self.ids)?;
        } else {
            self.emitted = s[..safe_end].to_string();
        }
        Ok(delta)
    }

    /// Flush any held-back (possibly incomplete) text at end of
    /// stream and reset the decoder.
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String, TokenizerError> {
        if self.ids.is_empty() {
            return Ok(String::new());
        }
        let s = tokenizer.decode(&self.ids)?;
        let delta = if s.len() >= self.emitted.len() && s.starts_with(self.emitted.as_str()) {
            s[self.emitted.len()..].to_string()
        } else {
            s
        };
        self.ids.clear();
        self.emitted.clear();
        Ok(delta)
    }
}

/// Byte length of the trailing run of U+FFFD replacement characters in
/// `s` (each is 3 bytes in UTF-8). Used to hold back potentially
/// incomplete UTF-8 sequences during incremental decoding.
fn trailing_replacement_len(s: &str) -> usize {
    let mut hold = 0usize;
    for c in s.chars().rev() {
        if c == char::REPLACEMENT_CHARACTER {
            hold += c.len_utf8();
        } else {
            break;
        }
    }
    hold
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

    /// F2: multi-byte UTF-8 characters split across byte-fallback
    /// tokens are held back until complete and then emitted whole —
    /// never as torn replacement characters.
    #[test]
    fn stream_decoder_holds_back_split_multibyte_utf8() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        // "é" = 0xC3 0xA9; "€" = 0xE2 0x82 0xAC.
        assert_eq!(d.push(&t, 0xC3).unwrap(), "");
        assert_eq!(d.push(&t, 0xA9).unwrap(), "é");
        assert_eq!(d.push(&t, b'x' as u32).unwrap(), "x");
        assert_eq!(d.push(&t, 0xE2).unwrap(), "");
        assert_eq!(d.push(&t, 0x82).unwrap(), "");
        assert_eq!(d.push(&t, 0xAC).unwrap(), "€");
        assert_eq!(d.finish(&t).unwrap(), "");
    }

    /// F2: a genuinely invalid intermediate byte sequence is
    /// eventually emitted as replacement characters once later valid
    /// text resolves it — the stream neither stalls nor drops text.
    #[test]
    fn stream_decoder_resolves_invalid_intermediate_bytes() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&t, b'a' as u32).unwrap(), "a");
        // Stray continuation byte: held back (could be a prefix of a
        // longer sequence from the decoder's perspective).
        assert_eq!(d.push(&t, 0xA9).unwrap(), "");
        // A following ASCII byte proves it invalid; both are emitted.
        let out = d.push(&t, b'b' as u32).unwrap();
        assert_eq!(out, "\u{FFFD}b");
        assert_eq!(d.finish(&t).unwrap(), "");
    }

    /// F2: a trailing incomplete sequence at end of stream is flushed
    /// by `finish` (as a replacement character) rather than dropped.
    #[test]
    fn stream_decoder_finish_flushes_trailing_incomplete_sequence() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&t, b'h' as u32).unwrap(), "h");
        assert_eq!(d.push(&t, 0xE2).unwrap(), "");
        assert_eq!(d.push(&t, 0x82).unwrap(), "");
        assert_eq!(d.finish(&t).unwrap(), "\u{FFFD}");
        // Decoder is reusable after finish.
        assert_eq!(d.push(&t, b'i' as u32).unwrap(), "i");
    }

    /// F2: long streams — the concatenated deltas equal the one-shot
    /// decode of the full id sequence, and the look-behind window
    /// stays bounded (no O(tokens²) re-decode, no unbounded state).
    #[test]
    fn stream_decoder_long_stream_matches_full_decode_with_bounded_window() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        // Mixed ASCII + multi-byte content, repeated well past any
        // window size.
        let text = "héllo wörld €42 ✓ ".repeat(200);
        let ids = t.encode(&text).unwrap();
        assert!(ids.len() > 4 * StreamDecoder::MAX_LOOKBEHIND_TOKENS);
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&d.push(&t, id).unwrap());
            assert!(
                d.lookbehind_len() <= StreamDecoder::MAX_LOOKBEHIND_TOKENS,
                "look-behind window must stay bounded"
            );
        }
        out.push_str(&d.finish(&t).unwrap());
        assert_eq!(out, t.decode(&ids).unwrap());
        assert_eq!(out, text);
    }

    /// F2: an adversarial endless run of invalid bytes cannot grow the
    /// window past the hard cap; the force-flush emits the pending
    /// replacement characters and keeps streaming.
    #[test]
    fn stream_decoder_force_flushes_endless_invalid_run_at_cap() {
        let t = Tokenizer::bytes();
        let mut d = StreamDecoder::new();
        let mut emitted = String::new();
        for _ in 0..(3 * StreamDecoder::MAX_LOOKBEHIND_TOKENS) {
            emitted.push_str(&d.push(&t, 0xC3).unwrap());
            assert!(d.lookbehind_len() <= StreamDecoder::MAX_LOOKBEHIND_TOKENS);
        }
        emitted.push_str(&d.finish(&t).unwrap());
        assert!(
            emitted.contains('\u{FFFD}'),
            "invalid bytes must eventually surface as replacement characters"
        );
    }

    /// F2 (merged BPE pieces): with the HF tokenizer backend, pieces
    /// that decode differently in context than in isolation still
    /// stream correctly — the concatenated deltas equal the one-shot
    /// decode of the full sequence.
    #[cfg(feature = "tokenizer")]
    #[test]
    fn stream_decoder_matches_full_decode_for_hf_bpe() {
        // Minimal in-memory BPE tokenizer with a metaspace-style
        // decoder: "▁" marks word boundaries and decodes to a space in
        // context but is stripped at sequence start.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": true},
            "post_processor": null,
            "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {"\u2581": 0, "h": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7,
                           "he": 8, "ll": 9, "hell": 10, "hello": 11, "\u2581w": 12, "or": 13,
                           "orl": 14, "\u2581world": 15, "\u2581hello": 16},
                "merges": ["h e", "l l", "he ll", "hell o", "\u2581 w", "o r", "or l",
                            "\u2581w orl", "\u2581worl d", "\u2581 hello"]
            }
        }"#;
        let inner = tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("valid tokenizer");
        let t = Tokenizer::Hf(inner);
        let ids = t.encode("hello world").unwrap();
        assert!(ids.len() >= 2, "expected merged BPE pieces");
        let mut d = StreamDecoder::new();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&d.push(&t, id).unwrap());
        }
        out.push_str(&d.finish(&t).unwrap());
        assert_eq!(out, t.decode(&ids).unwrap());
    }
}
