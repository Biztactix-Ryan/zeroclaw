//! Extism WASM plugin embedding provider
//!
//! Bridges ZeroClaw's `EmbeddingProvider` trait to an extism plugin that
//! implements the MiniLM or similar ONNX embedding model.
//!
//! The plugin `.wasm` binary is loaded from disk and the plugin directory
//! (containing `model.onnx`, `vocab.txt`, etc.) is mounted into the WASI
//! sandbox at `/model` so the guest can read model assets at runtime without
//! baking them into the binary.

use std::sync::Mutex;

use anyhow::Context;
use async_trait::async_trait;
use extism::Plugin;
use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingProvider;

/// Input for embed_text call
#[derive(Debug, Serialize)]
struct EmbedInput {
    text: String,
}

/// Output from embed_text
#[derive(Debug, Deserialize)]
struct EmbedOutput {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    dimensions: usize,
}

/// Batch input
#[derive(Debug, Serialize)]
struct BatchEmbedInput {
    texts: Vec<String>,
}

/// Batch output
#[derive(Debug, Deserialize)]
struct BatchEmbedOutput {
    embeddings: Vec<Vec<f32>>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    dimensions: usize,
    #[allow(dead_code)]
    count: usize,
}

/// Extism-based embedding provider using a WASM plugin
///
/// The plugin must expose (via `extism-pdk` `#[plugin_fn]`):
/// - `init()`          → empty on success
/// - `embed_text(JSON)` → JSON `EmbedOutput`
/// - `embed_batch(JSON)` → JSON `BatchEmbedOutput`
pub struct ExtismEmbedding {
    plugin_name: String,
    dimensions: usize,
    plugin: Mutex<Plugin>,
}

impl ExtismEmbedding {
    /// Create a new extism embedding provider.
    ///
    /// `plugin_name` — name used for plugin directory lookup and config.
    /// `wasm_path`   — optional explicit path to the `.wasm` file.
    ///
    /// The plugin directory (the parent of the `.wasm` file) is mounted
    /// read-only into the WASI guest at `/model` so it can load
    /// `model.onnx`, `vocab.txt`, etc.
    pub fn new(plugin_name: &str, wasm_path: Option<&str>) -> Result<Self, anyhow::Error> {
        let dims = match plugin_name {
            n if n.contains("minilm") || n.contains("mini-lm") => 384,
            n if n.contains("bge-small") => 512,
            n if n.contains("bge") => 768,
            n if n.contains("e5-small") => 512,
            n if n.contains("e5") => 768,
            _ => 384,
        };

        let (wasm_bytes, plugin_dir) = load_wasm(plugin_name, wasm_path)?;

        // Mount the plugin directory read-only at /model inside the WASI sandbox
        let host_path = format!("ro:{}", plugin_dir.display());
        let manifest = extism::Manifest::new([extism::Wasm::data(wasm_bytes)])
            .with_allowed_path(host_path, "/model");

        let plugin =
            Plugin::new(&manifest, [], true).context("failed to instantiate extism plugin")?;

        tracing::info!(
            plugin = %plugin_name,
            dims,
            model_dir = %plugin_dir.display(),
            "extism embedding plugin loaded"
        );

        Ok(Self {
            plugin_name: plugin_name.to_string(),
            dimensions: dims,
            plugin: Mutex::new(plugin),
        })
    }

    /// Call init on the plugin (loads model + tokenizer inside the guest).
    pub fn initialize(&self) -> Result<(), anyhow::Error> {
        let mut plugin = self
            .plugin
            .lock()
            .map_err(|e| anyhow::anyhow!("plugin lock poisoned: {e}"))?;

        plugin
            .call::<&[u8], &[u8]>("init", &[])
            .context("plugin init failed")?;

        Ok(())
    }

    fn embed_via_plugin(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let mut plugin = self
            .plugin
            .lock()
            .map_err(|e| anyhow::anyhow!("plugin lock poisoned: {e}"))?;

        if texts.len() == 1 {
            let input = serde_json::to_string(&EmbedInput {
                text: texts[0].clone(),
            })?;

            let output_json: String = plugin
                .call("embed_text", &input)
                .context("embed_text call failed")?;

            let output: EmbedOutput = serde_json::from_str(&output_json)
                .context("failed to parse embed_text JSON output")?;

            if output.embedding.is_empty() {
                return Ok(vec![]);
            }

            Ok(vec![output.embedding])
        } else {
            let input = serde_json::to_string(&BatchEmbedInput {
                texts: texts.to_vec(),
            })?;

            let output_json: String = plugin
                .call("embed_batch", &input)
                .context("embed_batch call failed")?;

            let output: BatchEmbedOutput = serde_json::from_str(&output_json)
                .context("failed to parse embed_batch JSON output")?;

            Ok(output.embeddings)
        }
    }
}

#[async_trait]
impl EmbeddingProvider for ExtismEmbedding {
    fn name(&self) -> &str {
        &self.plugin_name
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        self.embed_via_plugin(&owned)
    }

    async fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut results = self.embed(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding result"))
    }
}

/// Load WASM bytes and resolve the plugin directory (for WASI mount).
///
/// Returns `(wasm_bytes, plugin_dir)`.
fn load_wasm(
    plugin_name: &str,
    wasm_path: Option<&str>,
) -> Result<(Vec<u8>, std::path::PathBuf), anyhow::Error> {
    if let Some(path) = wasm_path {
        let p = std::path::Path::new(path);
        if p.exists() {
            let dir = p
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf();
            tracing::info!(plugin = %plugin_name, path = %p.display(), "loading extism embedding plugin");
            let bytes =
                std::fs::read(p).with_context(|| format!("failed to read {}", p.display()))?;
            return Ok((bytes, dir));
        }
        anyhow::bail!("WASM file not found at explicit path: {path}");
    }

    let plugin_dir = find_plugin_dir(plugin_name)?;
    let wasm_file = plugin_dir.join("zeroclaw-embedding.wasm");
    let alt_wasm = plugin_dir.join(format!("{plugin_name}.wasm"));

    let path = if wasm_file.exists() {
        wasm_file
    } else if alt_wasm.exists() {
        alt_wasm
    } else {
        anyhow::bail!(
            "WASM file not found for plugin '{}' in {:?} (tried: {:?}, {:?})",
            plugin_name,
            plugin_dir,
            wasm_file,
            alt_wasm
        );
    };

    tracing::info!(
        plugin = %plugin_name,
        path = %path.display(),
        "loading extism embedding plugin"
    );

    let bytes =
        std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok((bytes, plugin_dir))
}

/// Find the plugin directory via ZEROCLAW_PLUGINS env or default locations
fn find_plugin_dir(plugin_name: &str) -> Result<std::path::PathBuf, anyhow::Error> {
    let search_paths: Vec<std::path::PathBuf> = [
        std::env::var("ZEROCLAW_PLUGINS")
            .ok()
            .map(std::path::PathBuf::from),
        Some(std::path::PathBuf::from("/opt/zeroclaw/plugins")),
        Some(std::path::PathBuf::from("./plugins")),
        dirs::plugin_dir(),
    ]
    .into_iter()
    .flatten()
    .collect();

    for base in search_paths {
        let candidate = base.join(plugin_name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    anyhow::bail!("plugin '{plugin_name}' not found in any search path")
}

/// Directory helper (cross-platform)
mod dirs {
    use std::path::PathBuf;

    pub fn plugin_dir() -> Option<PathBuf> {
        #[cfg(target_os = "linux")]
        {
            Some(PathBuf::from("/var/lib/zeroclaw/plugins"))
        }
        #[cfg(target_os = "macos")]
        {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join("Library/Application Support/Zeroclaw/plugins"))
        }
        #[cfg(target_os = "windows")]
        {
            std::env::var("APPDATA")
                .ok()
                .map(|h| PathBuf::from(h).join("ZeroClaw/plugins"))
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            None
        }
    }
}
