//! TOML configuration for the production server (gist Phase 8).
//!
//! Replaces the long-tail of CLI flags with a single config file. The
//! existing CLI subcommands (`gen-data`, `run`) keep working unchanged
//! — `serve --config <path>` is the new entry point that reads this
//! struct.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP bind address, e.g. `127.0.0.1:8080`.
    #[serde(default = "default_bind")]
    pub bind: String,

    /// Maximum tokens any one request is allowed to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_bind() -> String { "127.0.0.1:8080".to_string() }
fn default_max_tokens() -> usize { 256 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Directory containing `expert_*.bin` files and (optionally)
    /// `metadata.json` and `tokenizer.json`.
    pub data_dir: PathBuf,

    /// Number of experts per layer.
    pub num_experts: u32,

    /// Top-K experts activated per token.
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Hidden / residual-stream dimension.
    pub d_model: usize,

    /// FFN intermediate dimension.
    pub d_ff: usize,

    /// Bytes per expert file (must be a multiple of `block_align`).
    pub expert_size: usize,

    /// Number of transformer layers (1 for the legacy single-layer mode,
    /// 32 for full Mixtral).
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
}

fn default_top_k() -> usize { 2 }
fn default_num_layers() -> usize { 1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfigToml {
    /// LRU cache slots **per layer**.
    #[serde(default = "default_cache_slots")]
    pub cache_slots: usize,

    /// O_DIRECT block alignment.
    #[serde(default = "default_block_align")]
    pub block_align: usize,

    /// Disable O_DIRECT (required on tmpfs / macOS / CI).
    #[serde(default)]
    pub no_direct: bool,

    /// Predictive prefetcher fanout (0 disables prefetching entirely).
    #[serde(default = "default_predict_fanout")]
    pub predict_fanout: usize,

    /// Don't prefetch below this transition probability.
    #[serde(default = "default_predict_min_prob")]
    pub predict_min_prob: f64,
}

fn default_cache_slots() -> usize { 4 }
fn default_block_align() -> usize { 4096 }
fn default_predict_fanout() -> usize { 2 }
fn default_predict_min_prob() -> f64 { 0.05 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Optional path to a HuggingFace `tokenizer.json`. If omitted, the
    /// engine falls back to a deterministic byte tokenizer.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self { path: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub storage: StorageConfigToml,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let body = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let cfg: Config = toml::from_str(&body).map_err(|e| ConfigError::Parse(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.num_experts == 0 {
            return Err(ConfigError::Invalid("model.num_experts must be > 0".into()));
        }
        if self.model.top_k == 0 || self.model.top_k as u32 > self.model.num_experts {
            return Err(ConfigError::Invalid(
                "model.top_k must be in 1..=num_experts".into(),
            ));
        }
        if self.model.d_model == 0 || self.model.d_ff == 0 {
            return Err(ConfigError::Invalid(
                "model.d_model and model.d_ff must be > 0".into(),
            ));
        }
        if self.model.num_layers == 0 {
            return Err(ConfigError::Invalid("model.num_layers must be > 0".into()));
        }
        if !self.storage.block_align.is_power_of_two() || self.storage.block_align == 0 {
            return Err(ConfigError::Invalid(
                "storage.block_align must be a positive power of two".into(),
            ));
        }
        if self.model.expert_size % self.storage.block_align != 0 {
            return Err(ConfigError::Invalid(format!(
                "model.expert_size ({}) must be a multiple of storage.block_align ({})",
                self.model.expert_size, self.storage.block_align
            )));
        }
        if self.server.max_tokens == 0 {
            return Err(ConfigError::Invalid("server.max_tokens must be > 0".into()));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io { path: PathBuf, source: std::io::Error },
    Parse(String),
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io { path, source } => write!(f, "config io ({}): {source}", path.display()),
            ConfigError::Parse(m) => write!(f, "config parse: {m}"),
            ConfigError::Invalid(m) => write!(f, "config invalid: {m}"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_cfg() -> Config {
        Config {
            server: ServerConfig { bind: "127.0.0.1:8080".into(), max_tokens: 64 },
            model: ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 64,
                d_ff: 256,
                expert_size: 4096,
                num_layers: 1,
            },
            storage: StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: false,
                predict_fanout: 2,
                predict_min_prob: 0.05,
            },
            tokenizer: TokenizerConfig::default(),
        }
    }

    #[test]
    fn valid_config_passes_validation() {
        minimal_cfg().validate().expect("valid");
    }

    #[test]
    fn rejects_top_k_greater_than_num_experts() {
        let mut c = minimal_cfg();
        c.model.top_k = 99;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_misaligned_expert_size() {
        let mut c = minimal_cfg();
        c.model.expert_size = 5000; // not a multiple of 4096
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_non_power_of_two_block_align() {
        let mut c = minimal_cfg();
        c.storage.block_align = 4097;
        assert!(c.validate().is_err());
    }

    #[test]
    fn round_trips_through_toml() {
        let c = minimal_cfg();
        let s = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        back.validate().unwrap();
        assert_eq!(back.model.num_experts, c.model.num_experts);
        assert_eq!(back.server.bind, c.server.bind);
    }
}
