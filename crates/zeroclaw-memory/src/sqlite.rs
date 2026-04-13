use super::embeddings::EmbeddingProvider;
use super::traits::{
    ChunkInput, ChunkResult, ChunkSearchOpts, DocumentInput, ExportFilter, IndexResult,
    KnowledgeStore, Memory, MemoryCategory, MemoryEntry, SearchScope, Source, SourceRegistration,
    SourceStats,
};
use super::vector;
use anyhow::Context;
use async_trait::async_trait;
use chrono::Local;
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use uuid::Uuid;
use zeroclaw_config::schema::SearchMode;

/// Maximum allowed open timeout (seconds) to avoid unreasonable waits.
const SQLITE_OPEN_TIMEOUT_CAP_SECS: u64 = 300;

/// SQLite-backed persistent memory — the brain
///
/// Full-stack search engine:
/// - **Vector DB**: embeddings stored as BLOB, cosine similarity search
/// - **Keyword Search**: FTS5 virtual table with BM25 scoring
/// - **Hybrid Merge**: weighted fusion of vector + keyword results
/// - **Embedding Cache**: LRU-evicted cache to avoid redundant API calls
/// - **Safe Reindex**: temp DB → seed → sync → atomic swap → rollback
pub struct SqliteMemory {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    db_path: PathBuf,
    embedder: Arc<dyn EmbeddingProvider>,
    vector_weight: f32,
    keyword_weight: f32,
    cache_max: usize,
    search_mode: SearchMode,
    // Phase 4: local embedder for multi-model indexing (MAX-fusion)
    local_embedder: Option<Arc<dyn EmbeddingProvider>>,
    // TurboQuant: embedding quantization mode
    quantization: vector::BitWidth,
}

impl SqliteMemory {
    pub fn new(workspace_dir: &Path) -> anyhow::Result<Self> {
        Self::with_embedder(
            workspace_dir,
            Arc::new(super::embeddings::NoopEmbedding),
            0.7,
            0.3,
            10_000,
            None,
            SearchMode::default(),
        )
    }

    /// Like `new`, but stores data in `{db_name}.db` instead of `brain.db`.
    pub fn new_named(workspace_dir: &Path, db_name: &str) -> anyhow::Result<Self> {
        let db_path = workspace_dir.join("memory").join(format!("{db_name}.db"));
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Self::open_connection(&db_path, None)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous  = NORMAL;
             PRAGMA mmap_size    = 8388608;
             PRAGMA cache_size   = -2000;
             PRAGMA temp_store   = MEMORY;",
        )?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
            embedder: Arc::new(super::embeddings::NoopEmbedding),
            vector_weight: 0.7,
            keyword_weight: 0.3,
            cache_max: 10_000,
            search_mode: SearchMode::default(),
            local_embedder: None,
            quantization: vector::BitWidth::Int8,
        })
    }

    /// Build SQLite memory with optional open timeout.
    ///
    /// If `open_timeout_secs` is `Some(n)`, opening the database is limited to `n` seconds
    /// (capped at 300). Useful when the DB file may be locked or on slow storage.
    /// `None` = wait indefinitely (default).
    pub fn with_embedder(
        workspace_dir: &Path,
        embedder: Arc<dyn EmbeddingProvider>,
        vector_weight: f32,
        keyword_weight: f32,
        cache_max: usize,
        open_timeout_secs: Option<u64>,
        search_mode: SearchMode,
    ) -> anyhow::Result<Self> {
        let db_path = workspace_dir.join("memory").join("brain.db");

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Self::open_connection(&db_path, open_timeout_secs)?;

        // ── Production-grade PRAGMA tuning ──────────────────────
        // WAL mode: concurrent reads during writes, crash-safe
        // normal sync: 2× write speed, still durable on WAL
        // mmap 8 MB: let the OS page-cache serve hot reads
        // cache 2 MB: keep ~500 hot pages in-process
        // temp_store memory: temp tables never hit disk
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous  = NORMAL;
             PRAGMA mmap_size    = 8388608;
             PRAGMA cache_size   = -2000;
             PRAGMA temp_store   = MEMORY;",
        )?;

        Self::init_schema(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
            embedder,
            vector_weight,
            keyword_weight,
            cache_max,
            search_mode,
            local_embedder: None,
            quantization: vector::BitWidth::Int8,
        })
    }

    /// Open SQLite connection, optionally with a timeout (for locked/slow storage).
    fn open_connection(
        db_path: &Path,
        open_timeout_secs: Option<u64>,
    ) -> anyhow::Result<Connection> {
        let path_buf = db_path.to_path_buf();

        let conn = if let Some(secs) = open_timeout_secs {
            let capped = secs.min(SQLITE_OPEN_TIMEOUT_CAP_SECS);
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                let result = Connection::open(&path_buf);
                let _ = tx.send(result);
            });
            match rx.recv_timeout(Duration::from_secs(capped)) {
                Ok(Ok(c)) => c,
                Ok(Err(e)) => return Err(e).context("SQLite failed to open database"),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    anyhow::bail!("SQLite connection open timed out after {} seconds", capped);
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("SQLite open thread exited unexpectedly");
                }
            }
        } else {
            Connection::open(&path_buf).context("SQLite failed to open database")?
        };

        Ok(conn)
    }

    /// Initialize all tables: memories, FTS5, `embedding_cache`
    fn init_schema(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            "-- Core memories table
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                key         TEXT NOT NULL UNIQUE,
                content     TEXT NOT NULL,
                category    TEXT NOT NULL DEFAULT 'core',
                embedding   BLOB,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);

            -- FTS5 full-text search (BM25 scoring)
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                key, content, content=memories, content_rowid=rowid
            );

            -- FTS5 triggers: keep in sync with memories table
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, key, content)
                VALUES (new.rowid, new.key, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, content)
                VALUES ('delete', old.rowid, old.key, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, content)
                VALUES ('delete', old.rowid, old.key, old.content);
                INSERT INTO memories_fts(rowid, key, content)
                VALUES (new.rowid, new.key, new.content);
            END;

            -- Embedding cache with LRU eviction
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding    BLOB NOT NULL,
                created_at   TEXT NOT NULL,
                accessed_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_accessed ON embedding_cache(accessed_at);",
        )?;

        // Migration: add session_id column if not present (safe to run repeatedly)
        let schema_sql: String = conn
            .prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'")?
            .query_row([], |row| row.get::<_, String>(0))?;

        if !schema_sql.contains("session_id") {
            conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN session_id TEXT;
                 CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);",
            )?;
        }

        // Migration: add namespace column
        if !schema_sql.contains("namespace") {
            conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN namespace TEXT DEFAULT 'default';
                 CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);",
            )?;
        }

        // Migration: add importance column
        if !schema_sql.contains("importance") {
            conn.execute_batch("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5;")?;
        }

        // Migration: add superseded_by column
        if !schema_sql.contains("superseded_by") {
            conn.execute_batch("ALTER TABLE memories ADD COLUMN superseded_by TEXT;")?;
        }

        // Migration: add hit_count and is_pinned to embedding_cache
        let cache_exists = conn
            .prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name='embedding_cache'")
            .and_then(|mut s| s.query_row([], |row| row.get::<_, String>(0)))
            .ok();
        if let Some(ref cache_schema) = cache_exists {
            if !cache_schema.contains("hit_count") {
                let _ = conn.execute_batch(
                    "ALTER TABLE embedding_cache ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0;",
                );
            }
            if !cache_schema.contains("is_pinned") {
                let _ = conn.execute_batch(
                    "ALTER TABLE embedding_cache ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0;",
                );
            }
        }

        // ── Knowledge layer schema ──────────────────────────────────
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sources (
                source_id       TEXT PRIMARY KEY,
                display_name    TEXT NOT NULL,
                uri_scheme      TEXT NOT NULL,
                plugin_version  TEXT,
                config          TEXT,
                registered_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                last_sync_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS source_documents (
                doc_id          TEXT PRIMARY KEY,
                source_id       TEXT NOT NULL,
                source_uri      TEXT NOT NULL,
                title           TEXT,
                content_hash    TEXT,
                chunk_count     INTEGER NOT NULL DEFAULT 0,
                total_tokens    INTEGER NOT NULL DEFAULT 0,
                indexed_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                metadata        TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
                UNIQUE (source_id, source_uri)
            );
            CREATE INDEX IF NOT EXISTS idx_docs_source ON source_documents(source_id);
            CREATE INDEX IF NOT EXISTS idx_docs_uri   ON source_documents(source_uri);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT NOT NULL,
                source_id       TEXT NOT NULL,
                content         TEXT NOT NULL,
                start_line      INTEGER,
                end_line        INTEGER,
                chunk_index     INTEGER NOT NULL DEFAULT 0,
                section_path    TEXT,
                embedding       BLOB,
                embedding_model TEXT,
                created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY (doc_id)    REFERENCES source_documents(doc_id)  ON DELETE CASCADE,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)       ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc   ON chunks(doc_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, content=chunks, content_rowid=rowid
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_embedding_id TEXT PRIMARY KEY,
                chunk_id           TEXT NOT NULL,
                model_id           TEXT NOT NULL,
                embedding          BLOB NOT NULL,
                dimensions         INTEGER NOT NULL,
                created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                UNIQUE (chunk_id, model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_emb_chunk ON chunk_embeddings(chunk_id);

            CREATE TABLE IF NOT EXISTS embedding_cost_ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id        TEXT NOT NULL,
                tokens_used     INTEGER NOT NULL,
                cost_usd        REAL NOT NULL,
                period          TEXT NOT NULL,
                recorded_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_cost_period ON embedding_cost_ledger(period);

            CREATE TABLE IF NOT EXISTS reembed_queue (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id        TEXT NOT NULL,
                model_id        TEXT NOT NULL,
                priority        INTEGER NOT NULL DEFAULT 0,
                retries         INTEGER NOT NULL DEFAULT 0,
                queued_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                UNIQUE (chunk_id, model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_reembed_priority ON reembed_queue(priority DESC, queued_at ASC);",
        )?;

        Ok(())
    }

    fn category_to_str(cat: &MemoryCategory) -> String {
        match cat {
            MemoryCategory::Core => "core".into(),
            MemoryCategory::Daily => "daily".into(),
            MemoryCategory::Conversation => "conversation".into(),
            MemoryCategory::Custom(name) => name.clone(),
        }
    }

    fn str_to_category(s: &str) -> MemoryCategory {
        match s {
            "core" => MemoryCategory::Core,
            "daily" => MemoryCategory::Daily,
            "conversation" => MemoryCategory::Conversation,
            other => MemoryCategory::Custom(other.to_string()),
        }
    }

    /// Deterministic content hash for embedding cache.
    /// Uses SHA-256 (truncated) instead of DefaultHasher, which is
    /// explicitly documented as unstable across Rust versions.
    fn content_hash(text: &str) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(text.as_bytes());
        // First 8 bytes → 16 hex chars, matching previous format length
        format!(
            "{:016x}",
            u64::from_be_bytes(
                hash[..8]
                    .try_into()
                    .expect("SHA-256 always produces >= 8 bytes")
            )
        )
    }

    /// Provide access to the connection for advanced queries (e.g. retrieval pipeline).
    pub fn connection(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
    }

    // ── Cost Ledger & Budget Enforcement ─────────────────────────────

    /// Record an embedding cost in the ledger.
    pub fn record_embedding_cost(&self, model_id: &str, tokens_used: usize, cost_per_1k: f64) {
        let cost_usd = cost_per_1k * (tokens_used as f64 / 1000.0);
        let period = chrono::Utc::now().format("%Y-%m").to_string();
        let conn = self.conn.lock();
        let _ = conn.execute(
            "INSERT INTO embedding_cost_ledger (model_id, tokens_used, cost_usd, period)
             VALUES (?1, ?2, ?3, ?4)",
            params![model_id, tokens_used as i64, cost_usd, period],
        );
    }

    /// Check if the current period's spend is within budget.
    pub fn check_embedding_budget(&self, daily_limit: f64, monthly_limit: f64) -> anyhow::Result<()> {
        let now = chrono::Utc::now();
        let period = now.format("%Y-%m").to_string();
        let today = now.format("%Y-%m-%d").to_string();
        let conn = self.conn.lock();

        if monthly_limit > 0.0 {
            let spend: f64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) FROM embedding_cost_ledger WHERE period = ?1",
                    params![period],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);
            if spend >= monthly_limit {
                anyhow::bail!("monthly embedding budget exceeded: ${spend:.4} / ${monthly_limit:.2}");
            }
        }

        if daily_limit > 0.0 {
            let spend: f64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) FROM embedding_cost_ledger WHERE recorded_at >= ?1",
                    params![format!("{today}T00:00:00Z")],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);
            if spend >= daily_limit {
                anyhow::bail!("daily embedding budget exceeded: ${spend:.4} / ${daily_limit:.2}");
            }
        }

        Ok(())
    }

    /// Get current period spend summary.
    pub fn embedding_spend_summary(&self) -> (f64, f64) {
        let now = chrono::Utc::now();
        let period = now.format("%Y-%m").to_string();
        let today = now.format("%Y-%m-%d").to_string();
        let conn = self.conn.lock();

        let monthly: f64 = conn
            .query_row(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM embedding_cost_ledger WHERE period = ?1",
                params![period],
                |row| row.get(0),
            )
            .unwrap_or(0.0);

        let daily: f64 = conn
            .query_row(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM embedding_cost_ledger WHERE recorded_at >= ?1",
                params![format!("{today}T00:00:00Z")],
                |row| row.get(0),
            )
            .unwrap_or(0.0);

        (daily, monthly)
    }

    /// Get embedding from cache, or compute + cache it
    pub async fn get_or_compute_embedding(&self, text: &str) -> anyhow::Result<Option<Vec<f32>>> {
        if self.embedder.dimensions() == 0 {
            return Ok(None); // Noop embedder
        }

        let hash = Self::content_hash(text);
        let now = Local::now().to_rfc3339();

        // Check cache (offloaded to blocking thread)
        let conn = self.conn.clone();
        let hash_c = hash.clone();
        let now_c = now.clone();
        let cached = tokio::task::spawn_blocking(move || -> anyhow::Result<Option<Vec<f32>>> {
            let conn = conn.lock();
            let mut stmt =
                conn.prepare("SELECT embedding FROM embedding_cache WHERE content_hash = ?1")?;
            let blob: Option<Vec<u8>> = stmt.query_row(params![hash_c], |row| row.get(0)).ok();
            if let Some(bytes) = blob {
                conn.execute(
                    "UPDATE embedding_cache SET accessed_at = ?1 WHERE content_hash = ?2",
                    params![now_c, hash_c],
                )?;
                return Ok(Some(vector::smart_decode(&bytes)));
            }
            Ok(None)
        })
        .await??;

        if cached.is_some() {
            return Ok(cached);
        }

        // Compute embedding (async I/O)
        let embedding = self.embedder.embed_one(text).await?;
        let bytes = vector::encode_embedding(&embedding, self.quantization);

        // Store in cache + LRU eviction (offloaded to blocking thread)
        let conn = self.conn.clone();
        #[allow(clippy::cast_possible_wrap)]
        let cache_max = self.cache_max as i64;
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.lock();
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, created_at, accessed_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![hash, bytes, now, now],
            )?;
            conn.execute(
                "DELETE FROM embedding_cache WHERE content_hash IN (
                    SELECT content_hash FROM embedding_cache
                    ORDER BY accessed_at ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM embedding_cache) - ?1)
                )",
                params![cache_max],
            )?;
            Ok(())
        })
        .await??;

        Ok(Some(embedding))
    }

    /// FTS5 BM25 keyword search
    pub fn fts5_search(
        conn: &Connection,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        // Escape FTS5 special chars and build query
        let fts_query: String = query
            .split_whitespace()
            .map(|w| format!("\"{w}\""))
            .collect::<Vec<_>>()
            .join(" OR ");

        if fts_query.is_empty() {
            return Ok(Vec::new());
        }

        let sql = "SELECT m.id, bm25(memories_fts) as score
                   FROM memories_fts f
                   JOIN memories m ON m.rowid = f.rowid
                   WHERE memories_fts MATCH ?1
                   ORDER BY score
                   LIMIT ?2";

        let mut stmt = conn.prepare(sql)?;
        #[allow(clippy::cast_possible_wrap)]
        let limit_i64 = limit as i64;

        let rows = stmt.query_map(params![fts_query, limit_i64], |row| {
            let id: String = row.get(0)?;
            let score: f64 = row.get(1)?;
            // BM25 returns negative scores (lower = better), negate for ranking
            #[allow(clippy::cast_possible_truncation)]
            Ok((id, (-score) as f32))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Vector similarity search: scan embeddings and compute cosine similarity.
    ///
    /// Optional `category` and `session_id` filters reduce full-table scans
    /// when the caller already knows the scope of relevant memories.
    pub fn vector_search(
        conn: &Connection,
        query_embedding: &[f32],
        limit: usize,
        category: Option<&str>,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        let mut sql = "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL".to_string();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(cat) = category {
            let _ = write!(sql, " AND category = ?{idx}");
            param_values.push(Box::new(cat.to_string()));
            idx += 1;
        }
        if let Some(sid) = session_id {
            let _ = write!(sql, " AND session_id = ?{idx}");
            param_values.push(Box::new(sid.to_string()));
        }

        let mut stmt = conn.prepare(&sql)?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(AsRef::as_ref).collect();
        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob))
        })?;

        let mut scored: Vec<(String, f32)> = Vec::new();
        for row in rows {
            let (id, blob) = row?;
            let emb = vector::smart_decode(&blob);
            let sim = vector::cosine_similarity(query_embedding, &emb);
            if sim > 0.0 {
                scored.push((id, sim));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    /// Safe reindex: rebuild FTS5 + embeddings with rollback on failure
    #[allow(dead_code)]
    pub async fn reindex(&self) -> anyhow::Result<usize> {
        // Step 1: Rebuild FTS5
        {
            let conn = self.conn.clone();
            tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                let conn = conn.lock();
                conn.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")?;
                Ok(())
            })
            .await??;
        }

        // Step 2: Re-embed all memories that lack embeddings
        if self.embedder.dimensions() == 0 {
            return Ok(0);
        }

        let conn = self.conn.clone();
        let entries: Vec<(String, String)> = tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt =
                conn.prepare("SELECT id, content FROM memories WHERE embedding IS NULL")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            Ok::<_, anyhow::Error>(rows.filter_map(std::result::Result::ok).collect())
        })
        .await??;

        let mut count = 0;
        let quantization = self.quantization;
        for (id, content) in &entries {
            if let Ok(Some(emb)) = self.get_or_compute_embedding(content).await {
                let bytes = vector::encode_embedding(&emb, quantization);
                let conn = self.conn.clone();
                let id = id.clone();
                tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                    let conn = conn.lock();
                    conn.execute(
                        "UPDATE memories SET embedding = ?1 WHERE id = ?2",
                        params![bytes, id],
                    )?;
                    Ok(())
                })
                .await??;
                count += 1;
            }
        }

        Ok(count)
    }

    /// List memories by time range (used when query is empty).
    async fn recall_by_time_only(
        &self,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        let conn = self.conn.clone();
        let sid = session_id.map(String::from);
        let since_owned = since.map(String::from);
        let until_owned = until.map(String::from);

        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<MemoryEntry>> {
            let conn = conn.lock();
            let since_ref = since_owned.as_deref();
            let until_ref = until_owned.as_deref();

            let mut sql =
                "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by FROM memories \
                           WHERE superseded_by IS NULL AND 1=1"
                    .to_string();
            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            let mut idx = 1;

            if let Some(sid) = sid.as_deref() {
                let _ = write!(sql, " AND session_id = ?{idx}");
                param_values.push(Box::new(sid.to_string()));
                idx += 1;
            }
            if let Some(s) = since_ref {
                let _ = write!(sql, " AND created_at >= ?{idx}");
                param_values.push(Box::new(s.to_string()));
                idx += 1;
            }
            if let Some(u) = until_ref {
                let _ = write!(sql, " AND created_at <= ?{idx}");
                param_values.push(Box::new(u.to_string()));
                idx += 1;
            }
            let _ = write!(sql, " ORDER BY updated_at DESC LIMIT ?{idx}");
            #[allow(clippy::cast_possible_wrap)]
            param_values.push(Box::new(limit as i64));

            let mut stmt = conn.prepare(&sql)?;
            let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(AsRef::as_ref).collect();
            let rows = stmt.query_map(params_ref.as_slice(), |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    key: row.get(1)?,
                    content: row.get(2)?,
                    category: Self::str_to_category(&row.get::<_, String>(3)?),
                    timestamp: row.get(4)?,
                    session_id: row.get(5)?,
                    score: None,
                    namespace: row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "default".into()),
                    importance: row.get(7)?,
                    superseded_by: row.get(8)?,
                })
            })?;

            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            Ok(results)
        })
        .await?
    }
}
#[async_trait]
impl KnowledgeStore for SqliteMemory {
    async fn register_source(&self, source: SourceRegistration) -> anyhow::Result<()> {
        let source_id = source.source_id.trim();
        let uri_scheme = source.uri_scheme.trim();

        if source_id.is_empty() {
            anyhow::bail!("source_id cannot be empty");
        }
        if source_id.len() > 64 {
            anyhow::bail!("source_id must be 64 characters or fewer");
        }
        if source_id.bytes().any(|b| b.is_ascii_whitespace()) {
            anyhow::bail!("source_id cannot contain whitespace");
        }
        if uri_scheme.is_empty() {
            anyhow::bail!("uri_scheme cannot be empty");
        }

        let config_json = source.config.map(|v| v.to_string());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| {
                chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                    .unwrap()
                    .to_rfc3339()
            })
            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());

        let source_id = source.source_id.clone();
        let display_name = source.display_name.clone();
        let uri_scheme = source.uri_scheme.clone();
        let plugin_version = source.plugin_version.clone();

        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.lock();
            conn.execute(
                "INSERT OR REPLACE INTO sources
                 (source_id, display_name, uri_scheme, plugin_version, config, registered_at, last_sync_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    source_id,
                    display_name,
                    uri_scheme,
                    plugin_version,
                    config_json,
                    now.clone(),
                    now,
                ],
            )?;
            Ok(())
        })
        .await?
    }

    async fn index_document(&self, doc: DocumentInput) -> anyhow::Result<IndexResult> {
        let embedder = self.embedder.clone();
        let local_embedder = self.local_embedder.clone();
        let local_model_id = local_embedder
            .as_ref()
            .map(|e| format!("{}:local", e.name()));

        let source_id = doc.source_id.clone();
        let source_uri = doc.source_uri.clone();
        let content_hash = doc.content_hash.clone();
        let title = doc.title;
        let metadata = doc.metadata;

        for chunk in &doc.chunks {
            let trimmed = chunk.content.trim();
            if trimmed.is_empty() {
                anyhow::bail!("chunk at index {} has empty content", chunk.chunk_index);
            }
            let tokens = trimmed.split_whitespace().count();
            if tokens < 10 {
                anyhow::bail!(
                    "chunk at index {} has {} tokens (min 10 required): '{}...'",
                    chunk.chunk_index,
                    tokens,
                    &trimmed[..trimmed.len().min(80)]
                );
            }
            if tokens > 1000 {
                println!(
                    "[WARN] chunk at index {} has {} tokens (>1000, will proceed): '{}...'",
                    chunk.chunk_index,
                    tokens,
                    &trimmed[..trimmed.len().min(80)]
                );
            }
        }

        let contents: Vec<&str> = doc.chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings: Vec<Vec<f32>> = if contents.is_empty() {
            vec![]
        } else {
            embedder.embed(&contents).await?
        };

        let local_embeddings: Vec<Vec<f32>> = if let Some(ref le) = local_embedder {
            if contents.is_empty() {
                vec![]
            } else {
                le.embed(&contents).await.unwrap_or_else(|_| vec![])
            }
        } else {
            vec![]
        };

        let chunk_entries: Vec<(String, ChunkInput, Option<Vec<f32>>, Option<Vec<f32>>)> = doc
            .chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let emb = if i < embeddings.len() && !embeddings[i].is_empty() {
                    Some(embeddings[i].clone())
                } else {
                    None
                };
                let local_emb = if i < local_embeddings.len() && !local_embeddings[i].is_empty() {
                    Some(local_embeddings[i].clone())
                } else {
                    None
                };
                (Uuid::new_v4().to_string(), chunk, emb, local_emb)
            })
            .collect();

        let conn = self.conn.clone();
        let quantization = self.quantization;
        tokio::task::spawn_blocking(move || -> anyhow::Result<IndexResult> {
            let conn = conn.lock();

            let source_exists: bool = conn
                .query_row(
                    "SELECT 1 FROM sources WHERE source_id = ?1",
                    params![source_id],
                    |_| Ok(true),
                )
                .unwrap_or(false);
            if !source_exists {
                anyhow::bail!("source_id '{}' not registered; call register_source() first", source_id);
            }

            let existing_doc: Option<(String, String)> = conn
                .query_row(
                    "SELECT doc_id, content_hash FROM source_documents WHERE source_id = ?1 AND source_uri = ?2",
                    params![source_id, source_uri],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .ok();

            let (doc_id, old_hash) = match existing_doc {
                Some((id, hash)) => (id, Some(hash)),
                None => (Uuid::new_v4().to_string(), None),
            };

            let skip = content_hash.as_ref().is_some_and(|ch| {
                old_hash.as_ref().is_some_and(|oh| ch == oh)
            });

            if skip {
                return Ok(IndexResult {
                    indexed: false,
                    chunks_created: 0,
                    skipped: true,
                });
            }

            let _ = conn.execute_batch("PRAGMA foreign_keys=OFF");

            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?1",
                params![doc_id],
            )?;

            let now = chrono::Utc::now().to_rfc3339();
            let chunk_count = chunk_entries.len();
            let total_tokens = chunk_count * 100;
            let embedder_name = embedder.name();

            for (chunk_id, chunk, emb_opt, local_emb_opt) in chunk_entries {
                let emb_blob =
                    emb_opt.map(|v| vector::encode_embedding(&v, quantization));
                conn.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, source_id, content, start_line, end_line, chunk_index, section_path, embedding, embedding_model)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        chunk_id,
                        doc_id,
                        source_id,
                        chunk.content,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.chunk_index,
                        chunk.section_path,
                        emb_blob,
                        embedder_name,
                    ],
                )?;

                if let (Some(local_emb), Some(model_id)) = (local_emb_opt, local_model_id.as_deref()) {
                    let local_blob = vector::encode_embedding(&local_emb, quantization);
                    let dims = local_emb.len();
                    conn.execute(
                        "INSERT OR REPLACE INTO chunk_embeddings
                         (chunk_embedding_id, chunk_id, model_id, embedding, dimensions, created_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            Uuid::new_v4().to_string(),
                            chunk_id,
                            model_id,
                            local_blob,
                            dims,
                            now,
                        ],
                    )?;
                }
            }

            conn.execute(
                "INSERT OR REPLACE INTO source_documents
                 (doc_id, source_id, source_uri, title, content_hash, chunk_count, total_tokens, indexed_at, updated_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    doc_id,
                    source_id,
                    source_uri,
                    title,
                    content_hash,
                    chunk_count,
                    total_tokens,
                    now,
                    now,
                    metadata.map(|m| m.to_string()),
                ],
            )?;

            Ok(IndexResult {
                indexed: true,
                chunks_created: chunk_count,
                skipped: false,
            })
        })
        .await?
    }

    async fn remove_document(&self, source_uri: &str) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let source_uri = source_uri.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.lock();
            conn.execute(
                "DELETE FROM source_documents WHERE source_uri = ?1",
                params![source_uri],
            )?;
            Ok(())
        })
        .await?
    }

    async fn source_stats(&self, source_id: &str) -> anyhow::Result<SourceStats> {
        let conn = self.conn.clone();
        let source_id = source_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<SourceStats> {
            let conn = conn.lock();

            let exists: bool = conn
                .query_row(
                    "SELECT 1 FROM sources WHERE source_id = ?1",
                    params![source_id],
                    |_| Ok(true),
                )
                .unwrap_or(false);
            if !exists {
                anyhow::bail!("source_id '{}' not found", source_id);
            }

            let (doc_count, chunk_count, last_indexed): (i64, i64, Option<String>) = conn
                .query_row(
                    "SELECT COUNT(*), COALESCE(SUM(chunk_count),0), MAX(indexed_at)
                     FROM source_documents WHERE source_id = ?1",
                    params![source_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap_or((0, 0, None));

            Ok(SourceStats {
                doc_count: doc_count as usize,
                chunk_count: chunk_count as usize,
                last_indexed_at: last_indexed,
            })
        })
        .await?
    }

    async fn search_chunks(
        &self,
        query: &str,
        opts: ChunkSearchOpts,
    ) -> anyhow::Result<Vec<ChunkResult>> {
        let embedder = self.embedder.clone();
        let vector_weight = self.vector_weight;
        let keyword_weight = self.keyword_weight;
        let relevance_decay_alpha: Option<f64> = None;

        // Embed query for vector search (None if no embedder configured)
        let query_vec: Option<Arc<Vec<f32>>> = if embedder.dimensions() == 0 {
            None
        } else {
            embedder.embed_one(query).await.ok().map(Arc::new)
        };

        // MAX-fusion: embed query with local model too (if configured)
        let local_query_vec: Option<Arc<Vec<f32>>> = if let Some(ref le) = self.local_embedder {
            le.embed_one(query).await.ok().map(Arc::new)
        } else {
            None
        };
        let query_str = query.to_string();
        let opts_clone = ChunkSearchOpts {
            scope: match opts.scope {
                SearchScope::Source(id) => SearchScope::Source(id),
                SearchScope::Sources(ids) => {
                    SearchScope::Sources(ids)
                }
                SearchScope::All => SearchScope::All,
            },
            limit: opts.limit,
            since: opts.since.clone(),
            section_filter: opts.section_filter.clone(),
        };
        let opts_scope = match opts_clone.scope {
            SearchScope::Source(id) => {
                format!("AND c.source_id = '{}'", id.replace("'", "''"))
            }
            SearchScope::Sources(ids) => {
                let list = ids
                    .iter()
                    .map(|s| format!("'{}'", s.replace("'", "''")))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("AND c.source_id IN ({})", list)
            }
            SearchScope::All => String::new(),
        };
        let opts_since = opts_clone.since;
        let opts_section = opts_clone.section_filter;
        let opts_limit = opts_clone.limit;

        let conn = self.conn.clone();
        let local_qv = local_query_vec;
        let query_vec_opt = query_vec;
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<ChunkResult>> {
            let conn = conn.lock();

            let since_filter = opts_since
                .as_ref()
                .map(|s| format!("AND c.created_at >= '{}'", s.replace("'", "''")))
                .unwrap_or_default();

            let section_filter_sql = opts_section
                .as_ref()
                .map(|s| format!("AND c.section_path LIKE '%{}%'", s.replace("'", "''")))
                .unwrap_or_default();

            let fts_query = query_str
                .split_whitespace()
                .map(|w| format!("\"{}*\"", w.replace("'", "''")))
                .collect::<Vec<_>>()
                .join(" ");

            if fts_query.is_empty() {
                return Ok(vec![]);
            }

            let sql = format!(
                "SELECT
                    c.chunk_id,
                    c.content,
                    c.source_id,
                    c.section_path,
                    d.source_uri,
                    c.embedding,
                    c.created_at,
                    bm25(chunks_fts) AS bm25_score
                 FROM chunks_fts f
                 JOIN chunks c ON c.rowid = f.rowid
                 JOIN source_documents d ON d.doc_id = c.doc_id
                 WHERE chunks_fts MATCH '{fts_query}' {opts_scope} {since_filter} {section_filter_sql}
                 LIMIT {limit}",
                fts_query = fts_query,
                opts_scope = opts_scope,
                since_filter = since_filter,
                section_filter_sql = section_filter_sql,
                limit = opts_limit.max(100)
            );

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map([], |row| {
                let chunk_id: String = row.get(0)?;
                let content: String = row.get(1)?;
                let source_id: String = row.get(2)?;
                let section_path: Option<String> = row.get(3)?;
                let source_uri: String = row.get(4)?;
                let embedding_blob: Option<Vec<u8>> = row.get(5)?;
                let created_at: String = row.get(6)?;
                let bm25_raw: f64 = row.get(7)?;

                Ok((
                    chunk_id,
                    content,
                    source_id,
                    section_path,
                    source_uri,
                    embedding_blob,
                    created_at,
                    bm25_raw,
                ))
            })?;

            let mut results: Vec<(f32, ChunkResult)> = Vec::new();

            for row in rows.filter_map(|r| r.ok()) {
                let (
                    chunk_id,
                    content,
                    source_id,
                    section_path,
                    source_uri,
                    embedding_blob,
                    created_at,
                    bm25_raw,
                ) = row;

                // Primary model cosine similarity
                let primary_sim = match (&query_vec_opt, &embedding_blob) {
                    (Some(qv), Some(blob)) => {
                        let emb = vector::smart_decode(blob);
                        vector::cosine_similarity(qv, &emb)
                    }
                    _ => 0.0,
                };

                // MAX-fusion: check chunk_embeddings for additional model scores
                let cosine_sim = if local_qv.is_some() {
                    let mut max_sim = primary_sim;

                    // Look up local model embedding for this chunk
                    let local_emb: Option<Vec<u8>> = conn
                        .prepare_cached(
                            "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?1 LIMIT 1",
                        )
                        .ok()
                        .and_then(|mut stmt| {
                            stmt.query_row(params![chunk_id], |row| row.get(0)).ok()
                        });

                    if let (Some(blob), Some(lqv)) = (local_emb, &local_qv) {
                        let emb = vector::smart_decode(&blob);
                        let local_sim = vector::cosine_similarity(lqv, &emb);
                        if local_sim > max_sim {
                            max_sim = local_sim;
                        }
                    }

                    max_sim
                } else {
                    primary_sim
                };

                let bm25_norm = ((-bm25_raw.abs()) / 20.0).exp() as f32;

                let raw_score = vector_weight * cosine_sim + keyword_weight * bm25_norm;

                let alpha = relevance_decay_alpha.unwrap_or(0.0) as f32;
                let decay = if alpha > 0.0 {
                    let created = chrono::DateTime::parse_from_rfc3339(&created_at)
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now());
                    let now = chrono::Utc::now();
                    let days = (now - created).num_days() as f32;
                    1.0 / (1.0 + alpha * days)
                } else {
                    1.0
                };

                let final_score = raw_score * decay;

                results.push((
                    final_score,
                    ChunkResult {
                        chunk_id,
                        content,
                        source_id,
                        source_uri,
                        section_path,
                        score: final_score,
                    },
                ));
            }

            results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Less));
            let chunks: Vec<ChunkResult> = results
                .into_iter()
                .take(opts.limit)
                .map(|(_, chunk)| chunk)
                .collect();

            Ok(chunks)
        })
        .await?
    }

    async fn list_sources(&self) -> anyhow::Result<Vec<Source>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Source>> {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT source_id, display_name, uri_scheme, registered_at, last_sync_at FROM sources",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(Source {
                    source_id: row.get(0)?,
                    display_name: row.get(1)?,
                    uri_scheme: row.get(2)?,
                    registered_at: row.get(3)?,
                    last_sync_at: row.get(4)?,
                })
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await?
    }

    async fn health(&self) -> anyhow::Result<bool> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            let conn = conn.lock();
            conn.execute("SELECT 1 FROM sources LIMIT 1", [])?;
            Ok(true)
        })
        .await?
    }
}

#[async_trait]
impl Memory for SqliteMemory {
    fn name(&self) -> &str {
        "sqlite"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> anyhow::Result<()> {
        // Compute embedding (async, before blocking work)
        let quantization = self.quantization;
        let embedding_bytes = self
            .get_or_compute_embedding(content)
            .await?
            .map(|emb| vector::encode_embedding(&emb, quantization));

        let conn = self.conn.clone();
        let key = key.to_string();
        let content = content.to_string();
        let sid = session_id.map(String::from);

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.lock();
            let now = Local::now().to_rfc3339();
            let cat = Self::category_to_str(&category);
            let id = Uuid::new_v4().to_string();

            conn.execute(
                "INSERT INTO memories (id, key, content, category, embedding, created_at, updated_at, session_id, namespace, importance)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'default', 0.5)
                 ON CONFLICT(key) DO UPDATE SET
                    content = excluded.content,
                    category = excluded.category,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at,
                    session_id = excluded.session_id",
                params![id, key, content, cat, embedding_bytes, now, now, sid],
            )?;
            Ok(())
        })
        .await?
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        // Time-only query: list by time range when no keywords
        if query.trim().is_empty() {
            return self
                .recall_by_time_only(limit, session_id, since, until)
                .await;
        }

        // Compute query embedding only when needed (skip for BM25-only mode)
        let query_embedding = if self.search_mode == SearchMode::Bm25 {
            None
        } else {
            self.get_or_compute_embedding(query).await?
        };

        let conn = self.conn.clone();
        let query = query.to_string();
        let sid = session_id.map(String::from);
        let since_owned = since.map(String::from);
        let until_owned = until.map(String::from);
        let vector_weight = self.vector_weight;
        let keyword_weight = self.keyword_weight;
        let search_mode = self.search_mode.clone();

        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<MemoryEntry>> {
            let conn = conn.lock();
            let session_ref = sid.as_deref();
            let since_ref = since_owned.as_deref();
            let until_ref = until_owned.as_deref();

            // FTS5 BM25 keyword search (skip for embedding-only mode)
            let keyword_results = if search_mode == SearchMode::Embedding {
                Vec::new()
            } else {
                Self::fts5_search(&conn, &query, limit * 2).unwrap_or_default()
            };

            // Vector similarity search (skip for BM25-only mode)
            let vector_results = if search_mode == SearchMode::Bm25 {
                Vec::new()
            } else if let Some(ref qe) = query_embedding {
                Self::vector_search(&conn, qe, limit * 2, None, session_ref).unwrap_or_default()
            } else {
                Vec::new()
            };

            // Merge results based on search mode
            let merged = if vector_results.is_empty() {
                keyword_results
                    .iter()
                    .map(|(id, score)| vector::ScoredResult {
                        id: id.clone(),
                        vector_score: None,
                        keyword_score: Some(*score),
                        final_score: *score,
                    })
                    .collect::<Vec<_>>()
            } else if keyword_results.is_empty() {
                vector_results
                    .iter()
                    .map(|(id, score)| vector::ScoredResult {
                        id: id.clone(),
                        vector_score: Some(*score),
                        keyword_score: None,
                        final_score: *score,
                    })
                    .collect::<Vec<_>>()
            } else {
                vector::hybrid_merge(
                    &vector_results,
                    &keyword_results,
                    vector_weight,
                    keyword_weight,
                    limit,
                )
            };

            // Fetch full entries for merged results in a single query
            // instead of N round-trips (N+1 pattern).
            let mut results = Vec::new();
            if !merged.is_empty() {
                let placeholders: String = (1..=merged.len())
                    .map(|i| format!("?{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by \
                     FROM memories WHERE superseded_by IS NULL AND id IN ({placeholders})"
                );
                let mut stmt = conn.prepare(&sql)?;
                let id_params: Vec<Box<dyn rusqlite::types::ToSql>> = merged
                    .iter()
                    .map(|s| Box::new(s.id.clone()) as Box<dyn rusqlite::types::ToSql>)
                    .collect();
                let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                    id_params.iter().map(AsRef::as_ref).collect();
                let rows = stmt.query_map(params_ref.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, Option<f64>>(7)?,
                        row.get::<_, Option<String>>(8)?,
                    ))
                })?;

                let mut entry_map = std::collections::HashMap::new();
                for row in rows {
                    let (id, key, content, cat, ts, sid, ns, imp, sup) = row?;
                    entry_map.insert(id, (key, content, cat, ts, sid, ns, imp, sup));
                }

                for scored in &merged {
                    if let Some((key, content, cat, ts, sid, ns, imp, sup)) = entry_map.remove(&scored.id) {
                        if let Some(s) = since_ref
                            && ts.as_str() < s {
                                continue;
                            }
                        if let Some(u) = until_ref
                            && ts.as_str() > u {
                                continue;
                            }
                        let entry = MemoryEntry {
                            id: scored.id.clone(),
                            key,
                            content,
                            category: Self::str_to_category(&cat),
                            timestamp: ts,
                            session_id: sid,
                            score: Some(f64::from(scored.final_score)),
                            namespace: ns.unwrap_or_else(|| "default".into()),
                            importance: imp,
                            superseded_by: sup,
                        };
                        if let Some(filter_sid) = session_ref
                            && entry.session_id.as_deref() != Some(filter_sid) {
                                continue;
                            }
                        results.push(entry);
                    }
                }
            }

            // If hybrid returned nothing, fall back to LIKE search.
            if results.is_empty() {
                const MAX_LIKE_KEYWORDS: usize = 8;
                let keywords: Vec<String> = query
                    .split_whitespace()
                    .take(MAX_LIKE_KEYWORDS)
                    .map(|w| format!("%{w}%"))
                    .collect();
                if !keywords.is_empty() {
                    let conditions: Vec<String> = keywords
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            format!("(content LIKE ?{} OR key LIKE ?{})", i * 2 + 1, i * 2 + 2)
                        })
                        .collect();
                    let where_clause = conditions.join(" OR ");
                    let mut param_idx = keywords.len() * 2 + 1;
                    let mut time_conditions = String::new();
                    if since_ref.is_some() {
                        let _ = write!(time_conditions, " AND created_at >= ?{param_idx}");
                        param_idx += 1;
                    }
                    if until_ref.is_some() {
                        let _ = write!(time_conditions, " AND created_at <= ?{param_idx}");
                        param_idx += 1;
                    }
                    let sql = format!(
                        "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by FROM memories
                         WHERE superseded_by IS NULL AND ({where_clause}){time_conditions}
                         ORDER BY updated_at DESC
                         LIMIT ?{param_idx}"
                    );
                    let mut stmt = conn.prepare(&sql)?;
                    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                    for kw in &keywords {
                        param_values.push(Box::new(kw.clone()));
                        param_values.push(Box::new(kw.clone()));
                    }
                    if let Some(s) = since_ref {
                        param_values.push(Box::new(s.to_string()));
                    }
                    if let Some(u) = until_ref {
                        param_values.push(Box::new(u.to_string()));
                    }
                    #[allow(clippy::cast_possible_wrap)]
                    param_values.push(Box::new(limit as i64));
                    let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                        param_values.iter().map(AsRef::as_ref).collect();
                    let rows = stmt.query_map(params_ref.as_slice(), |row| {
                        Ok(MemoryEntry {
                            id: row.get(0)?,
                            key: row.get(1)?,
                            content: row.get(2)?,
                            category: Self::str_to_category(&row.get::<_, String>(3)?),
                            timestamp: row.get(4)?,
                            session_id: row.get(5)?,
                            score: Some(1.0),
                            namespace: row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "default".into()),
                            importance: row.get(7)?,
                            superseded_by: row.get(8)?,
                        })
                    })?;
                    for row in rows {
                        let entry = row?;
                        if let Some(sid) = session_ref
                            && entry.session_id.as_deref() != Some(sid) {
                                continue;
                            }
                        results.push(entry);
                    }
                }
            }

            results.truncate(limit);
            Ok(results)
        })
        .await?
    }

    async fn get(&self, key: &str) -> anyhow::Result<Option<MemoryEntry>> {
        let conn = self.conn.clone();
        let key = key.to_string();

        tokio::task::spawn_blocking(move || -> anyhow::Result<Option<MemoryEntry>> {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by FROM memories WHERE key = ?1",
            )?;

            let mut rows = stmt.query_map(params![key], |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    key: row.get(1)?,
                    content: row.get(2)?,
                    category: Self::str_to_category(&row.get::<_, String>(3)?),
                    timestamp: row.get(4)?,
                    session_id: row.get(5)?,
                    score: None,
                    namespace: row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "default".into()),
                    importance: row.get(7)?,
                    superseded_by: row.get(8)?,
                })
            })?;

            match rows.next() {
                Some(Ok(entry)) => Ok(Some(entry)),
                _ => Ok(None),
            }
        })
        .await?
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        const DEFAULT_LIST_LIMIT: i64 = 1000;

        let conn = self.conn.clone();
        let category = category.cloned();
        let sid = session_id.map(String::from);

        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<MemoryEntry>> {
            let conn = conn.lock();
            let session_ref = sid.as_deref();
            let mut results = Vec::new();

            let row_mapper = |row: &rusqlite::Row| -> rusqlite::Result<MemoryEntry> {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    key: row.get(1)?,
                    content: row.get(2)?,
                    category: Self::str_to_category(&row.get::<_, String>(3)?),
                    timestamp: row.get(4)?,
                    session_id: row.get(5)?,
                    score: None,
                    namespace: row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "default".into()),
                    importance: row.get(7)?,
                    superseded_by: row.get(8)?,
                })
            };

            if let Some(ref cat) = category {
                let cat_str = Self::category_to_str(cat);
                let mut stmt = conn.prepare(
                    "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by FROM memories
                     WHERE superseded_by IS NULL AND category = ?1 ORDER BY updated_at DESC LIMIT ?2",
                )?;
                let rows = stmt.query_map(params![cat_str, DEFAULT_LIST_LIMIT], row_mapper)?;
                for row in rows {
                    let entry = row?;
                    if let Some(sid) = session_ref
                        && entry.session_id.as_deref() != Some(sid) {
                            continue;
                        }
                    results.push(entry);
                }
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by FROM memories
                     WHERE superseded_by IS NULL ORDER BY updated_at DESC LIMIT ?1",
                )?;
                let rows = stmt.query_map(params![DEFAULT_LIST_LIMIT], row_mapper)?;
                for row in rows {
                    let entry = row?;
                    if let Some(sid) = session_ref
                        && entry.session_id.as_deref() != Some(sid) {
                            continue;
                        }
                    results.push(entry);
                }
            }

            Ok(results)
        })
        .await?
    }

    async fn forget(&self, key: &str) -> anyhow::Result<bool> {
        let conn = self.conn.clone();
        let key = key.to_string();

        tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            let conn = conn.lock();
            let affected = conn.execute("DELETE FROM memories WHERE key = ?1", params![key])?;
            Ok(affected > 0)
        })
        .await?
    }

    async fn purge_namespace(&self, namespace: &str) -> anyhow::Result<usize> {
        let conn = self.conn.clone();
        let namespace = namespace.to_string();

        tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let conn = conn.lock();
            let affected = conn.execute(
                "DELETE FROM memories WHERE category = ?1",
                params![namespace],
            )?;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Ok(affected)
        })
        .await?
    }

    async fn purge_session(&self, session_id: &str) -> anyhow::Result<usize> {
        let conn = self.conn.clone();
        let session_id = session_id.to_string();

        tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let conn = conn.lock();
            let affected = conn.execute(
                "DELETE FROM memories WHERE session_id = ?1",
                params![session_id],
            )?;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Ok(affected)
        })
        .await?
    }

    async fn count(&self) -> anyhow::Result<usize> {
        let conn = self.conn.clone();

        tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let conn = conn.lock();
            let count: i64 =
                conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Ok(count as usize)
        })
        .await?
    }

    async fn health_check(&self) -> bool {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || conn.lock().execute_batch("SELECT 1").is_ok())
            .await
            .unwrap_or(false)
    }

    async fn export(&self, filter: &ExportFilter) -> anyhow::Result<Vec<MemoryEntry>> {
        let conn = self.conn.clone();
        let filter = filter.clone();

        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<MemoryEntry>> {
            let conn = conn.lock();
            let mut sql =
                "SELECT id, key, content, category, created_at, session_id, namespace, importance, superseded_by \
                 FROM memories WHERE 1=1"
                    .to_string();
            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            let mut idx = 1;

            if let Some(ref ns) = filter.namespace {
                let _ = write!(sql, " AND namespace = ?{idx}");
                param_values.push(Box::new(ns.clone()));
                idx += 1;
            }
            if let Some(ref sid) = filter.session_id {
                let _ = write!(sql, " AND session_id = ?{idx}");
                param_values.push(Box::new(sid.clone()));
                idx += 1;
            }
            if let Some(ref cat) = filter.category {
                let _ = write!(sql, " AND category = ?{idx}");
                param_values.push(Box::new(Self::category_to_str(cat)));
                idx += 1;
            }
            if let Some(ref since) = filter.since {
                let _ = write!(sql, " AND created_at >= ?{idx}");
                param_values.push(Box::new(since.clone()));
                idx += 1;
            }
            if let Some(ref until) = filter.until {
                let _ = write!(sql, " AND created_at <= ?{idx}");
                param_values.push(Box::new(until.clone()));
                let _ = idx;
            }
            sql.push_str(" ORDER BY created_at ASC");

            let mut stmt = conn.prepare(&sql)?;
            let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(AsRef::as_ref).collect();
            let rows = stmt.query_map(params_ref.as_slice(), |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    key: row.get(1)?,
                    content: row.get(2)?,
                    category: Self::str_to_category(&row.get::<_, String>(3)?),
                    timestamp: row.get(4)?,
                    session_id: row.get(5)?,
                    score: None,
                    namespace: row.get::<_, Option<String>>(6)?.unwrap_or_else(|| "default".into()),
                    importance: row.get(7)?,
                    superseded_by: row.get(8)?,
                })
            })?;

            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            Ok(results)
        })
        .await?
    }

    async fn recall_namespaced(
        &self,
        namespace: &str,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        let entries = self
            .recall(query, limit * 2, session_id, since, until)
            .await?;
        let filtered: Vec<MemoryEntry> = entries
            .into_iter()
            .filter(|e| e.namespace == namespace)
            .take(limit)
            .collect();
        Ok(filtered)
    }

    async fn store_with_metadata(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        namespace: Option<&str>,
        importance: Option<f64>,
    ) -> anyhow::Result<()> {
        let quantization = self.quantization;
        let embedding_bytes = self
            .get_or_compute_embedding(content)
            .await?
            .map(|emb| vector::encode_embedding(&emb, quantization));

        let conn = self.conn.clone();
        let key = key.to_string();
        let content = content.to_string();
        let sid = session_id.map(String::from);
        let ns = namespace.unwrap_or("default").to_string();
        let imp = importance.unwrap_or(0.5);

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.lock();
            let now = Local::now().to_rfc3339();
            let cat = Self::category_to_str(&category);
            let id = Uuid::new_v4().to_string();

            conn.execute(
                "INSERT INTO memories (id, key, content, category, embedding, created_at, updated_at, session_id, namespace, importance)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                 ON CONFLICT(key) DO UPDATE SET
                    content = excluded.content,
                    category = excluded.category,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at,
                    session_id = excluded.session_id,
                    namespace = excluded.namespace,
                    importance = excluded.importance",
                params![id, key, content, cat, embedding_bytes, now, now, sid, ns, imp],
            )?;
            Ok(())
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_sqlite() -> (TempDir, SqliteMemory) {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();
        (tmp, mem)
    }

    #[tokio::test]
    async fn sqlite_name() {
        let (_tmp, mem) = temp_sqlite();
        assert_eq!(mem.name(), "sqlite");
    }

    #[tokio::test]
    async fn sqlite_health() {
        let (_tmp, mem) = temp_sqlite();
        assert!(mem.health_check().await);
    }

    #[tokio::test]
    async fn sqlite_store_and_get() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("user_lang", "Prefers Rust", MemoryCategory::Core, None)
            .await
            .unwrap();

        let entry = mem.get("user_lang").await.unwrap();
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.key, "user_lang");
        assert_eq!(entry.content, "Prefers Rust");
        assert_eq!(entry.category, MemoryCategory::Core);
    }

    #[tokio::test]
    async fn sqlite_store_upsert() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("pref", "likes Rust", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("pref", "loves Rust", MemoryCategory::Core, None)
            .await
            .unwrap();

        let entry = mem.get("pref").await.unwrap().unwrap();
        assert_eq!(entry.content, "loves Rust");
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_recall_keyword() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "Rust is fast and safe", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "Python is interpreted", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store(
            "c",
            "Rust has zero-cost abstractions",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

        let results = mem.recall("Rust", 10, None, None, None).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|r| r.content.to_lowercase().contains("rust"))
        );
    }

    #[tokio::test]
    async fn sqlite_recall_multi_keyword() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "Rust is fast", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "Rust is safe and fast", MemoryCategory::Core, None)
            .await
            .unwrap();

        let results = mem.recall("fast safe", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty());
        // Entry with both keywords should score higher
        assert!(results[0].content.contains("safe") && results[0].content.contains("fast"));
    }

    #[tokio::test]
    async fn sqlite_recall_no_match() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "Rust rocks", MemoryCategory::Core, None)
            .await
            .unwrap();
        let results = mem
            .recall("javascript", 10, None, None, None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn sqlite_forget() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("temp", "temporary data", MemoryCategory::Conversation, None)
            .await
            .unwrap();
        assert_eq!(mem.count().await.unwrap(), 1);

        let removed = mem.forget("temp").await.unwrap();
        assert!(removed);
        assert_eq!(mem.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn sqlite_forget_nonexistent() {
        let (_tmp, mem) = temp_sqlite();
        let removed = mem.forget("nope").await.unwrap();
        assert!(!removed);
    }

    #[tokio::test]
    async fn sqlite_list_all() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "one", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "two", MemoryCategory::Daily, None)
            .await
            .unwrap();
        mem.store("c", "three", MemoryCategory::Conversation, None)
            .await
            .unwrap();

        let all = mem.list(None, None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn sqlite_list_by_category() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "core1", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "core2", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("c", "daily1", MemoryCategory::Daily, None)
            .await
            .unwrap();

        let core = mem.list(Some(&MemoryCategory::Core), None).await.unwrap();
        assert_eq!(core.len(), 2);

        let daily = mem.list(Some(&MemoryCategory::Daily), None).await.unwrap();
        assert_eq!(daily.len(), 1);
    }

    #[tokio::test]
    async fn sqlite_count_empty() {
        let (_tmp, mem) = temp_sqlite();
        assert_eq!(mem.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn sqlite_get_nonexistent() {
        let (_tmp, mem) = temp_sqlite();
        assert!(mem.get("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sqlite_db_persists() {
        let tmp = TempDir::new().unwrap();

        {
            let mem = SqliteMemory::new(tmp.path()).unwrap();
            mem.store("persist", "I survive restarts", MemoryCategory::Core, None)
                .await
                .unwrap();
        }

        // Reopen
        let mem2 = SqliteMemory::new(tmp.path()).unwrap();
        let entry = mem2.get("persist").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().content, "I survive restarts");
    }

    #[tokio::test]
    async fn sqlite_category_roundtrip() {
        let (_tmp, mem) = temp_sqlite();
        let categories = [
            MemoryCategory::Core,
            MemoryCategory::Daily,
            MemoryCategory::Conversation,
            MemoryCategory::Custom("project".into()),
        ];

        for (i, cat) in categories.iter().enumerate() {
            mem.store(&format!("k{i}"), &format!("v{i}"), cat.clone(), None)
                .await
                .unwrap();
        }

        for (i, cat) in categories.iter().enumerate() {
            let entry = mem.get(&format!("k{i}")).await.unwrap().unwrap();
            assert_eq!(&entry.category, cat);
        }
    }

    // ── FTS5 search tests ────────────────────────────────────────

    #[tokio::test]
    async fn fts5_bm25_ranking() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "a",
            "Rust is a systems programming language",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.store(
            "b",
            "Python is great for scripting",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.store(
            "c",
            "Rust and Rust and Rust everywhere",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

        let results = mem.recall("Rust", 10, None, None, None).await.unwrap();
        assert!(results.len() >= 2);
        // All results should contain "Rust"
        for r in &results {
            assert!(
                r.content.to_lowercase().contains("rust"),
                "Expected 'rust' in: {}",
                r.content
            );
        }
    }

    #[tokio::test]
    async fn fts5_multi_word_query() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "The quick brown fox jumps", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "A lazy dog sleeps", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("c", "The quick dog runs fast", MemoryCategory::Core, None)
            .await
            .unwrap();

        let results = mem.recall("quick dog", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty());
        // "The quick dog runs fast" matches both terms
        assert!(results[0].content.contains("quick"));
    }

    #[tokio::test]
    async fn recall_empty_query_returns_recent_entries() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "data", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Empty query = time-only mode: returns recent entries
        let results = mem.recall("", 10, None, None, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "a");
    }

    #[tokio::test]
    async fn recall_whitespace_query_returns_recent_entries() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "data", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Whitespace-only query = time-only mode: returns recent entries
        let results = mem.recall("   ", 10, None, None, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "a");
    }

    // ── Embedding cache tests ────────────────────────────────────

    #[test]
    fn content_hash_deterministic() {
        let h1 = SqliteMemory::content_hash("hello world");
        let h2 = SqliteMemory::content_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_different_inputs() {
        let h1 = SqliteMemory::content_hash("hello");
        let h2 = SqliteMemory::content_hash("world");
        assert_ne!(h1, h2);
    }

    // ── Schema tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn schema_has_fts5_table() {
        let (_tmp, mem) = temp_sqlite();
        let conn = mem.conn.lock();
        // FTS5 table should exist
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn schema_has_embedding_cache() {
        let (_tmp, mem) = temp_sqlite();
        let conn = mem.conn.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embedding_cache'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn schema_memories_has_embedding_column() {
        let (_tmp, mem) = temp_sqlite();
        let conn = mem.conn.lock();
        // Check that embedding column exists by querying it
        let result = conn.execute_batch("SELECT embedding FROM memories LIMIT 0");
        assert!(result.is_ok());
    }

    // ── FTS5 sync trigger tests ──────────────────────────────────

    #[tokio::test]
    async fn fts5_syncs_on_insert() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "test_key",
            "unique_searchterm_xyz",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

        let conn = mem.conn.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"unique_searchterm_xyz\"'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn fts5_syncs_on_delete() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "del_key",
            "deletable_content_abc",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.forget("del_key").await.unwrap();

        let conn = mem.conn.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"deletable_content_abc\"'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn fts5_syncs_on_update() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "upd_key",
            "original_content_111",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.store("upd_key", "updated_content_222", MemoryCategory::Core, None)
            .await
            .unwrap();

        let conn = mem.conn.lock();
        // Old content should not be findable
        let old: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"original_content_111\"'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(old, 0);

        // New content should be findable
        let new: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"updated_content_222\"'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(new, 1);
    }

    // ── Open timeout tests ────────────────────────────────────────

    #[test]
    fn open_with_timeout_succeeds_when_fast() {
        let tmp = TempDir::new().unwrap();
        let embedder = Arc::new(super::super::embeddings::NoopEmbedding);
        let mem = SqliteMemory::with_embedder(
            tmp.path(),
            embedder,
            0.7,
            0.3,
            1000,
            Some(5),
            SearchMode::default(),
        );
        assert!(
            mem.is_ok(),
            "open with 5s timeout should succeed on fast path"
        );
        assert_eq!(mem.unwrap().name(), "sqlite");
    }

    #[tokio::test]
    async fn open_with_timeout_store_recall_unchanged() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::with_embedder(
            tmp.path(),
            Arc::new(super::super::embeddings::NoopEmbedding),
            0.7,
            0.3,
            1000,
            Some(2),
            SearchMode::default(),
        )
        .unwrap();
        mem.store(
            "timeout_key",
            "value with timeout",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        let entry = mem.get("timeout_key").await.unwrap().unwrap();
        assert_eq!(entry.content, "value with timeout");
    }

    // ── With-embedder constructor test ───────────────────────────

    #[test]
    fn with_embedder_noop() {
        let tmp = TempDir::new().unwrap();
        let embedder = Arc::new(super::super::embeddings::NoopEmbedding);
        let mem = SqliteMemory::with_embedder(
            tmp.path(),
            embedder,
            0.7,
            0.3,
            1000,
            None,
            SearchMode::default(),
        );
        assert!(mem.is_ok());
        assert_eq!(mem.unwrap().name(), "sqlite");
    }

    // ── Reindex test ─────────────────────────────────────────────

    #[tokio::test]
    async fn reindex_rebuilds_fts() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("r1", "reindex test alpha", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("r2", "reindex test beta", MemoryCategory::Core, None)
            .await
            .unwrap();

        // Reindex should succeed (noop embedder → 0 re-embedded)
        let count = mem.reindex().await.unwrap();
        assert_eq!(count, 0);

        // FTS should still work after rebuild
        let results = mem.recall("reindex", 10, None, None, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    // ── Recall limit test ────────────────────────────────────────

    #[tokio::test]
    async fn recall_respects_limit() {
        let (_tmp, mem) = temp_sqlite();
        for i in 0..20 {
            mem.store(
                &format!("k{i}"),
                &format!("common keyword item {i}"),
                MemoryCategory::Core,
                None,
            )
            .await
            .unwrap();
        }

        let results = mem
            .recall("common keyword", 5, None, None, None)
            .await
            .unwrap();
        assert!(results.len() <= 5);
    }

    // ── Score presence test ──────────────────────────────────────

    #[tokio::test]
    async fn recall_results_have_scores() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("s1", "scored result test", MemoryCategory::Core, None)
            .await
            .unwrap();

        let results = mem.recall("scored", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.score.is_some(), "Expected score on result: {:?}", r.key);
        }
    }

    // ── Edge cases: FTS5 special characters ──────────────────────

    #[tokio::test]
    async fn recall_with_quotes_in_query() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("q1", "He said hello world", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Quotes in query should not crash FTS5
        let results = mem.recall("\"hello\"", 10, None, None, None).await.unwrap();
        // May or may not match depending on FTS5 escaping, but must not error
        assert!(results.len() <= 10);
    }

    #[tokio::test]
    async fn recall_with_asterisk_in_query() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a1", "wildcard test content", MemoryCategory::Core, None)
            .await
            .unwrap();
        let results = mem.recall("wild*", 10, None, None, None).await.unwrap();
        assert!(results.len() <= 10);
    }

    #[tokio::test]
    async fn recall_with_parentheses_in_query() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("p1", "function call test", MemoryCategory::Core, None)
            .await
            .unwrap();
        let results = mem
            .recall("function()", 10, None, None, None)
            .await
            .unwrap();
        assert!(results.len() <= 10);
    }

    #[tokio::test]
    async fn recall_with_sql_injection_attempt() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("safe", "normal content", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Should not crash or leak data
        let results = mem
            .recall("'; DROP TABLE memories; --", 10, None, None, None)
            .await
            .unwrap();
        assert!(results.len() <= 10);
        // Table should still exist
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    // ── Edge cases: store ────────────────────────────────────────

    #[tokio::test]
    async fn store_empty_content() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("empty", "", MemoryCategory::Core, None)
            .await
            .unwrap();
        let entry = mem.get("empty").await.unwrap().unwrap();
        assert_eq!(entry.content, "");
    }

    #[tokio::test]
    async fn store_empty_key() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("", "content for empty key", MemoryCategory::Core, None)
            .await
            .unwrap();
        let entry = mem.get("").await.unwrap().unwrap();
        assert_eq!(entry.content, "content for empty key");
    }

    #[tokio::test]
    async fn store_very_long_content() {
        let (_tmp, mem) = temp_sqlite();
        let long_content = "x".repeat(100_000);
        mem.store("long", &long_content, MemoryCategory::Core, None)
            .await
            .unwrap();
        let entry = mem.get("long").await.unwrap().unwrap();
        assert_eq!(entry.content.len(), 100_000);
    }

    #[tokio::test]
    async fn store_unicode_and_emoji() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "emoji_key_🦀",
            "こんにちは 🚀 Ñoño",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        let entry = mem.get("emoji_key_🦀").await.unwrap().unwrap();
        assert_eq!(entry.content, "こんにちは 🚀 Ñoño");
    }

    #[tokio::test]
    async fn store_content_with_newlines_and_tabs() {
        let (_tmp, mem) = temp_sqlite();
        let content = "line1\nline2\ttab\rcarriage\n\nnewparagraph";
        mem.store("whitespace", content, MemoryCategory::Core, None)
            .await
            .unwrap();
        let entry = mem.get("whitespace").await.unwrap().unwrap();
        assert_eq!(entry.content, content);
    }

    // ── Edge cases: recall ───────────────────────────────────────

    #[tokio::test]
    async fn recall_single_character_query() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "x marks the spot", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Single char may not match FTS5 but LIKE fallback should work
        let results = mem.recall("x", 10, None, None, None).await.unwrap();
        // Should not crash; may or may not find results
        assert!(results.len() <= 10);
    }

    #[tokio::test]
    async fn recall_limit_zero() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "some content", MemoryCategory::Core, None)
            .await
            .unwrap();
        let results = mem.recall("some", 0, None, None, None).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn recall_limit_one() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "matching content alpha", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "matching content beta", MemoryCategory::Core, None)
            .await
            .unwrap();
        let results = mem
            .recall("matching content", 1, None, None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn recall_matches_by_key_not_just_content() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "rust_preferences",
            "User likes systems programming",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        // "rust" appears in key but not content — LIKE fallback checks key too
        let results = mem.recall("rust", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty(), "Should match by key");
    }

    #[tokio::test]
    async fn recall_unicode_query() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("jp", "日本語のテスト", MemoryCategory::Core, None)
            .await
            .unwrap();
        let results = mem.recall("日本語", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty());
    }

    // ── Edge cases: schema idempotency ───────────────────────────

    #[tokio::test]
    async fn schema_idempotent_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mem = SqliteMemory::new(tmp.path()).unwrap();
            mem.store("k1", "v1", MemoryCategory::Core, None)
                .await
                .unwrap();
        }
        // Open again — init_schema runs again on existing DB
        let mem2 = SqliteMemory::new(tmp.path()).unwrap();
        let entry = mem2.get("k1").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().content, "v1");
        // Store more data — should work fine
        mem2.store("k2", "v2", MemoryCategory::Daily, None)
            .await
            .unwrap();
        assert_eq!(mem2.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn schema_triple_open() {
        let tmp = TempDir::new().unwrap();
        let _m1 = SqliteMemory::new(tmp.path()).unwrap();
        let _m2 = SqliteMemory::new(tmp.path()).unwrap();
        let m3 = SqliteMemory::new(tmp.path()).unwrap();
        assert!(m3.health_check().await);
    }

    // ── Edge cases: forget + FTS5 consistency ────────────────────

    #[tokio::test]
    async fn forget_then_recall_no_ghost_results() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "ghost",
            "phantom memory content",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.forget("ghost").await.unwrap();
        let results = mem
            .recall("phantom memory", 10, None, None, None)
            .await
            .unwrap();
        assert!(
            results.is_empty(),
            "Deleted memory should not appear in recall"
        );
    }

    #[tokio::test]
    async fn forget_and_re_store_same_key() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("cycle", "version 1", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.forget("cycle").await.unwrap();
        mem.store("cycle", "version 2", MemoryCategory::Core, None)
            .await
            .unwrap();
        let entry = mem.get("cycle").await.unwrap().unwrap();
        assert_eq!(entry.content, "version 2");
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    // ── Edge cases: reindex ──────────────────────────────────────

    #[tokio::test]
    async fn reindex_empty_db() {
        let (_tmp, mem) = temp_sqlite();
        let count = mem.reindex().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn reindex_twice_is_safe() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("r1", "reindex data", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.reindex().await.unwrap();
        let count = mem.reindex().await.unwrap();
        assert_eq!(count, 0); // Noop embedder → nothing to re-embed
        // Data should still be intact
        let results = mem.recall("reindex", 10, None, None, None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    // ── Edge cases: content_hash ─────────────────────────────────

    #[test]
    fn content_hash_empty_string() {
        let h = SqliteMemory::content_hash("");
        assert!(!h.is_empty());
        assert_eq!(h.len(), 16); // 16 hex chars
    }

    #[test]
    fn content_hash_unicode() {
        let h1 = SqliteMemory::content_hash("🦀");
        let h2 = SqliteMemory::content_hash("🦀");
        assert_eq!(h1, h2);
        let h3 = SqliteMemory::content_hash("🚀");
        assert_ne!(h1, h3);
    }

    #[test]
    fn content_hash_long_input() {
        let long = "a".repeat(1_000_000);
        let h = SqliteMemory::content_hash(&long);
        assert_eq!(h.len(), 16);
    }

    // ── Edge cases: category helpers ─────────────────────────────

    #[test]
    fn category_roundtrip_custom_with_spaces() {
        let cat = MemoryCategory::Custom("my custom category".into());
        let s = SqliteMemory::category_to_str(&cat);
        assert_eq!(s, "my custom category");
        let back = SqliteMemory::str_to_category(&s);
        assert_eq!(back, cat);
    }

    #[test]
    fn category_roundtrip_empty_custom() {
        let cat = MemoryCategory::Custom(String::new());
        let s = SqliteMemory::category_to_str(&cat);
        assert_eq!(s, "");
        let back = SqliteMemory::str_to_category(&s);
        assert_eq!(back, MemoryCategory::Custom(String::new()));
    }

    // ── Edge cases: list ─────────────────────────────────────────

    #[tokio::test]
    async fn list_custom_category() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "c1",
            "custom1",
            MemoryCategory::Custom("project".into()),
            None,
        )
        .await
        .unwrap();
        mem.store(
            "c2",
            "custom2",
            MemoryCategory::Custom("project".into()),
            None,
        )
        .await
        .unwrap();
        mem.store("c3", "other", MemoryCategory::Core, None)
            .await
            .unwrap();

        let project = mem
            .list(Some(&MemoryCategory::Custom("project".into())), None)
            .await
            .unwrap();
        assert_eq!(project.len(), 2);
    }

    #[tokio::test]
    async fn list_empty_db() {
        let (_tmp, mem) = temp_sqlite();
        let all = mem.list(None, None).await.unwrap();
        assert!(all.is_empty());
    }

    // ── Bulk deletion tests ───────────────────────────────────────

    #[tokio::test]
    async fn sqlite_purge_namespace_removes_all_matching_entries() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a1", "data1", MemoryCategory::Custom("ns1".into()), None)
            .await
            .unwrap();
        mem.store("a2", "data2", MemoryCategory::Custom("ns1".into()), None)
            .await
            .unwrap();
        mem.store("b1", "data3", MemoryCategory::Custom("ns2".into()), None)
            .await
            .unwrap();

        let count = mem.purge_namespace("ns1").await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_purge_namespace_preserves_other_namespaces() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a1", "data1", MemoryCategory::Custom("ns1".into()), None)
            .await
            .unwrap();
        mem.store("b1", "data2", MemoryCategory::Custom("ns2".into()), None)
            .await
            .unwrap();
        mem.store("c1", "data3", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("d1", "data4", MemoryCategory::Daily, None)
            .await
            .unwrap();

        let count = mem.purge_namespace("ns1").await.unwrap();
        assert_eq!(count, 1);
        assert_eq!(mem.count().await.unwrap(), 3);

        let remaining = mem.list(None, None).await.unwrap();
        assert!(
            remaining
                .iter()
                .all(|e| e.category != MemoryCategory::Custom("ns1".into()))
        );
    }

    #[tokio::test]
    async fn sqlite_purge_namespace_returns_count() {
        let (_tmp, mem) = temp_sqlite();
        for i in 0..5 {
            mem.store(
                &format!("k{i}"),
                "data",
                MemoryCategory::Custom("target".into()),
                None,
            )
            .await
            .unwrap();
        }

        let count = mem.purge_namespace("target").await.unwrap();
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn sqlite_purge_session_removes_all_matching_entries() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a1", "data1", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("a2", "data2", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("b1", "data3", MemoryCategory::Core, Some("sess-b"))
            .await
            .unwrap();

        let count = mem.purge_session("sess-a").await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_purge_session_preserves_other_sessions() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a1", "data1", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("b1", "data2", MemoryCategory::Core, Some("sess-b"))
            .await
            .unwrap();
        mem.store("c1", "data3", MemoryCategory::Core, None)
            .await
            .unwrap();

        let count = mem.purge_session("sess-a").await.unwrap();
        assert_eq!(count, 1);
        assert_eq!(mem.count().await.unwrap(), 2);

        let remaining = mem.list(None, None).await.unwrap();
        assert!(
            remaining
                .iter()
                .all(|e| e.session_id.as_deref() != Some("sess-a"))
        );
    }

    #[tokio::test]
    async fn sqlite_purge_session_returns_count() {
        let (_tmp, mem) = temp_sqlite();
        for i in 0..3 {
            mem.store(
                &format!("k{i}"),
                "data",
                MemoryCategory::Core,
                Some("target-sess"),
            )
            .await
            .unwrap();
        }

        let count = mem.purge_session("target-sess").await.unwrap();
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn sqlite_purge_namespace_empty_namespace_is_noop() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "data", MemoryCategory::Core, None)
            .await
            .unwrap();

        let count = mem.purge_namespace("").await.unwrap();
        assert_eq!(count, 0);
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_purge_session_empty_session_is_noop() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "data", MemoryCategory::Core, Some("sess"))
            .await
            .unwrap();

        let count = mem.purge_session("").await.unwrap();
        assert_eq!(count, 0);
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    // ── Session isolation ─────────────────────────────────────────

    #[tokio::test]
    async fn store_and_recall_with_session_id() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("k1", "session A fact", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("k2", "session B fact", MemoryCategory::Core, Some("sess-b"))
            .await
            .unwrap();
        mem.store("k3", "no session fact", MemoryCategory::Core, None)
            .await
            .unwrap();

        // Recall with session-a filter returns only session-a entry
        let results = mem
            .recall("fact", 10, Some("sess-a"), None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "k1");
        assert_eq!(results[0].session_id.as_deref(), Some("sess-a"));
    }

    #[tokio::test]
    async fn recall_no_session_filter_returns_all() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("k1", "alpha fact", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("k2", "beta fact", MemoryCategory::Core, Some("sess-b"))
            .await
            .unwrap();
        mem.store("k3", "gamma fact", MemoryCategory::Core, None)
            .await
            .unwrap();

        // Recall without session filter returns all matching entries
        let results = mem.recall("fact", 10, None, None, None).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn cross_session_recall_isolation() {
        let (_tmp, mem) = temp_sqlite();
        mem.store(
            "secret",
            "session A secret data",
            MemoryCategory::Core,
            Some("sess-a"),
        )
        .await
        .unwrap();

        // Session B cannot see session A data
        let results = mem
            .recall("secret", 10, Some("sess-b"), None, None)
            .await
            .unwrap();
        assert!(results.is_empty());

        // Session A can see its own data
        let results = mem
            .recall("secret", 10, Some("sess-a"), None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn list_with_session_filter() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("k1", "a1", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("k2", "a2", MemoryCategory::Conversation, Some("sess-a"))
            .await
            .unwrap();
        mem.store("k3", "b1", MemoryCategory::Core, Some("sess-b"))
            .await
            .unwrap();
        mem.store("k4", "none1", MemoryCategory::Core, None)
            .await
            .unwrap();

        // List with session-a filter
        let results = mem.list(None, Some("sess-a")).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|e| e.session_id.as_deref() == Some("sess-a"))
        );

        // List with session-a + category filter
        let results = mem
            .list(Some(&MemoryCategory::Core), Some("sess-a"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "k1");
    }

    #[tokio::test]
    async fn schema_migration_idempotent_on_reopen() {
        let tmp = TempDir::new().unwrap();

        // First open: creates schema + migration
        {
            let mem = SqliteMemory::new(tmp.path()).unwrap();
            mem.store("k1", "before reopen", MemoryCategory::Core, Some("sess-x"))
                .await
                .unwrap();
        }

        // Second open: migration runs again but is idempotent
        {
            let mem = SqliteMemory::new(tmp.path()).unwrap();
            let results = mem
                .recall("reopen", 10, Some("sess-x"), None, None)
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].key, "k1");
            assert_eq!(results[0].session_id.as_deref(), Some("sess-x"));
        }
    }

    // ── §4.1 Concurrent write contention tests ──────────────

    #[tokio::test]
    async fn sqlite_concurrent_writes_no_data_loss() {
        let (_tmp, mem) = temp_sqlite();
        let mem = std::sync::Arc::new(mem);

        let mut handles = Vec::new();
        for i in 0..10 {
            let mem = std::sync::Arc::clone(&mem);
            handles.push(tokio::spawn(async move {
                mem.store(
                    &format!("concurrent_key_{i}"),
                    &format!("value_{i}"),
                    MemoryCategory::Core,
                    None,
                )
                .await
                .unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let count = mem.count().await.unwrap();
        assert_eq!(
            count, 10,
            "all 10 concurrent writes must succeed without data loss"
        );
    }

    #[tokio::test]
    async fn sqlite_concurrent_read_write_no_panic() {
        let (_tmp, mem) = temp_sqlite();
        let mem = std::sync::Arc::new(mem);

        // Pre-populate
        mem.store("shared_key", "initial", MemoryCategory::Core, None)
            .await
            .unwrap();

        let mut handles = Vec::new();

        // Concurrent reads
        for _ in 0..5 {
            let mem = std::sync::Arc::clone(&mem);
            handles.push(tokio::spawn(async move {
                let _ = mem.get("shared_key").await.unwrap();
            }));
        }

        // Concurrent writes
        for i in 0..5 {
            let mem = std::sync::Arc::clone(&mem);
            handles.push(tokio::spawn(async move {
                mem.store(
                    &format!("key_{i}"),
                    &format!("val_{i}"),
                    MemoryCategory::Core,
                    None,
                )
                .await
                .unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Should have 6 total entries (1 pre-existing + 5 new)
        assert_eq!(mem.count().await.unwrap(), 6);
    }

    // ── Export (GDPR Art. 20) tests ─────────────────────────

    #[tokio::test]
    async fn export_no_filter_returns_all_entries() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "one", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "two", MemoryCategory::Daily, None)
            .await
            .unwrap();
        mem.store("c", "three", MemoryCategory::Conversation, None)
            .await
            .unwrap();

        let filter = ExportFilter::default();
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn export_with_namespace_filter() {
        let (_tmp, mem) = temp_sqlite();
        mem.store_with_metadata(
            "a",
            "ns1 data",
            MemoryCategory::Core,
            None,
            Some("ns1"),
            None,
        )
        .await
        .unwrap();
        mem.store_with_metadata(
            "b",
            "ns2 data",
            MemoryCategory::Core,
            None,
            Some("ns2"),
            None,
        )
        .await
        .unwrap();

        let filter = ExportFilter {
            namespace: Some("ns1".into()),
            ..Default::default()
        };
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].namespace, "ns1");
    }

    #[tokio::test]
    async fn export_with_session_id_filter() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "sess-a data", MemoryCategory::Core, Some("sess-a"))
            .await
            .unwrap();
        mem.store("b", "sess-b data", MemoryCategory::Core, Some("sess-b"))
            .await
            .unwrap();

        let filter = ExportFilter {
            session_id: Some("sess-a".into()),
            ..Default::default()
        };
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "a");
    }

    #[tokio::test]
    async fn export_with_category_filter() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "core data", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "daily data", MemoryCategory::Daily, None)
            .await
            .unwrap();

        let filter = ExportFilter {
            category: Some(MemoryCategory::Core),
            ..Default::default()
        };
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].category, MemoryCategory::Core);
    }

    #[tokio::test]
    async fn export_with_time_range() {
        let (_tmp, mem) = temp_sqlite();
        // Store entries — created_at is set to Local::now() by store()
        mem.store("a", "old data", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "new data", MemoryCategory::Core, None)
            .await
            .unwrap();

        // Export with a time range that covers everything
        let filter = ExportFilter {
            since: Some("2000-01-01T00:00:00Z".into()),
            until: Some("2099-12-31T23:59:59Z".into()),
            ..Default::default()
        };
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 2);

        // Export with a time range in the far future (no results)
        let filter = ExportFilter {
            since: Some("2099-01-01T00:00:00Z".into()),
            ..Default::default()
        };
        let results = mem.export(&filter).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn export_with_combined_filters() {
        let (_tmp, mem) = temp_sqlite();
        mem.store_with_metadata(
            "a",
            "match",
            MemoryCategory::Core,
            Some("sess-a"),
            Some("ns1"),
            None,
        )
        .await
        .unwrap();
        mem.store_with_metadata(
            "b",
            "no match ns",
            MemoryCategory::Core,
            Some("sess-a"),
            Some("ns2"),
            None,
        )
        .await
        .unwrap();
        mem.store_with_metadata(
            "c",
            "no match sess",
            MemoryCategory::Core,
            None,
            Some("ns1"),
            None,
        )
        .await
        .unwrap();

        let filter = ExportFilter {
            namespace: Some("ns1".into()),
            session_id: Some("sess-a".into()),
            category: Some(MemoryCategory::Core),
            since: Some("2000-01-01T00:00:00Z".into()),
            until: Some("2099-12-31T23:59:59Z".into()),
        };
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "a");
    }

    #[tokio::test]
    async fn export_empty_database_returns_empty_vec() {
        let (_tmp, mem) = temp_sqlite();
        let filter = ExportFilter::default();
        let results = mem.export(&filter).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn export_ordering_is_chronological() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("first", "data1", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Small delay to ensure different timestamps
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        mem.store("second", "data2", MemoryCategory::Core, None)
            .await
            .unwrap();

        let filter = ExportFilter::default();
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results[0].timestamp <= results[1].timestamp,
            "Export must be ordered by created_at ASC"
        );
    }

    #[tokio::test]
    async fn export_preserves_field_integrity() {
        let (_tmp, mem) = temp_sqlite();
        mem.store_with_metadata(
            "roundtrip_key",
            "roundtrip content",
            MemoryCategory::Custom("custom_cat".into()),
            Some("sess-rt"),
            Some("ns-rt"),
            Some(0.9),
        )
        .await
        .unwrap();

        let filter = ExportFilter::default();
        let results = mem.export(&filter).await.unwrap();
        assert_eq!(results.len(), 1);
        let e = &results[0];
        assert_eq!(e.key, "roundtrip_key");
        assert_eq!(e.content, "roundtrip content");
        assert_eq!(e.category, MemoryCategory::Custom("custom_cat".into()));
        assert_eq!(e.session_id.as_deref(), Some("sess-rt"));
        assert_eq!(e.namespace, "ns-rt");
        assert_eq!(e.importance, Some(0.9));
    }

    // ── §4.2 Reindex / corruption recovery tests ────────────

    #[tokio::test]
    async fn sqlite_reindex_preserves_data() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("a", "Rust is fast", MemoryCategory::Core, None)
            .await
            .unwrap();
        mem.store("b", "Python is interpreted", MemoryCategory::Core, None)
            .await
            .unwrap();

        mem.reindex().await.unwrap();

        let count = mem.count().await.unwrap();
        assert_eq!(count, 2, "reindex must preserve all entries");

        let entry = mem.get("a").await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().content, "Rust is fast");
    }

    #[tokio::test]
    async fn sqlite_reindex_idempotent() {
        let (_tmp, mem) = temp_sqlite();
        mem.store("x", "test data", MemoryCategory::Core, None)
            .await
            .unwrap();

        // Multiple reindex calls should be safe
        mem.reindex().await.unwrap();
        mem.reindex().await.unwrap();
        mem.reindex().await.unwrap();

        assert_eq!(mem.count().await.unwrap(), 1);
    }

    // ── SearchMode tests ─────────────────────────────────────────

    #[tokio::test]
    async fn search_mode_bm25_only() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::with_embedder(
            tmp.path(),
            Arc::new(super::super::embeddings::NoopEmbedding),
            0.7,
            0.3,
            1000,
            None,
            SearchMode::Bm25,
        )
        .unwrap();
        mem.store(
            "lang",
            "User prefers Rust programming",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();
        mem.store("food", "User likes pizza", MemoryCategory::Core, None)
            .await
            .unwrap();

        let results = mem.recall("Rust", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty(), "BM25 mode should find keyword matches");
        assert!(
            results.iter().any(|e| e.content.contains("Rust")),
            "BM25 should match on keyword 'Rust'"
        );
    }

    #[tokio::test]
    async fn search_mode_embedding_only() {
        let tmp = TempDir::new().unwrap();
        // NoopEmbedding returns None, so embedding-only mode will fall back to LIKE
        let mem = SqliteMemory::with_embedder(
            tmp.path(),
            Arc::new(super::super::embeddings::NoopEmbedding),
            0.7,
            0.3,
            1000,
            None,
            SearchMode::Embedding,
        )
        .unwrap();
        mem.store(
            "lang",
            "User prefers Rust programming",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

        // With NoopEmbedding, vector search returns empty, and FTS is skipped.
        // The recall method falls back to LIKE search.
        let results = mem.recall("Rust", 10, None, None, None).await.unwrap();
        // LIKE fallback should still find it
        assert!(
            results.iter().any(|e| e.content.contains("Rust")),
            "Embedding mode with noop should fall back to LIKE and still find results"
        );
    }

    #[tokio::test]
    async fn search_mode_hybrid_default() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();
        // Default search mode should be Hybrid
        assert_eq!(mem.search_mode, SearchMode::Hybrid);

        mem.store(
            "lang",
            "User prefers Rust programming",
            MemoryCategory::Core,
            None,
        )
        .await
        .unwrap();

        let results = mem.recall("Rust", 10, None, None, None).await.unwrap();
        assert!(!results.is_empty(), "Hybrid mode should find results");
    }

    // ── NoopEmbedding: memory works without embedding provider ──

    #[tokio::test]
    async fn noop_store_and_recall_works() {
        let (_tmp, mem) = temp_sqlite();
        // NoopEmbedding is the default — store should work
        mem.store("lang", "Rust is great", MemoryCategory::Core, None)
            .await
            .unwrap();
        // Recall should still work via BM25 keyword search
        let results = mem.recall("Rust", 5, None, None, None).await.unwrap();
        assert!(!results.is_empty(), "recall should work without embeddings via BM25");
        assert_eq!(results[0].key, "lang");
    }

    #[tokio::test]
    async fn noop_knowledge_store_index_and_search() {
        let (_tmp, mem) = temp_sqlite();
        // Register source + index document should work even without embeddings
        mem.register_source(SourceRegistration {
            source_id: "noop-src".into(),
            display_name: "NoopTest".into(),
            uri_scheme: "test://".into(),
            plugin_version: None,
            config: None,
        })
        .await
        .unwrap();

        let result = mem
            .index_document(DocumentInput {
                source_id: "noop-src".into(),
                source_uri: "test://noop-doc".into(),
                title: Some("Noop Doc".into()),
                content_hash: Some("noop1".into()),
                chunks: vec![ChunkInput {
                    content: "This document is indexed without any embedding provider configured and should still be searchable via keyword matching".into(),
                    start_line: Some(1),
                    end_line: Some(1),
                    chunk_index: 0,
                    section_path: None,
                }],
                metadata: None,
            })
            .await
            .unwrap();
        assert!(result.indexed);
        assert_eq!(result.chunks_created, 1);

        // search_chunks should work via BM25 (no vector component)
        let results = mem
            .search_chunks(
                "keyword matching",
                ChunkSearchOpts {
                    scope: SearchScope::All,
                    limit: 10,
                    since: None,
                    section_filter: None,
                },
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "search should work without embeddings via BM25");
        assert!(results[0].content.contains("keyword matching"));
    }

    // ── Cost ledger & budget tests ──────────────────────────────

    #[test]
    fn cost_ledger_records_and_sums() {
        let (_tmp, mem) = temp_sqlite();
        mem.record_embedding_cost("openai:text-embedding-3-small", 1000, 0.02);
        mem.record_embedding_cost("openai:text-embedding-3-small", 500, 0.02);
        let (daily, monthly) = mem.embedding_spend_summary();
        assert!((monthly - 0.03).abs() < 0.001);
        assert!((daily - 0.03).abs() < 0.001);
    }

    #[test]
    fn budget_check_passes_when_under_limit() {
        let (_tmp, mem) = temp_sqlite();
        mem.record_embedding_cost("model", 100, 0.01);
        assert!(mem.check_embedding_budget(1.0, 10.0).is_ok());
    }

    #[test]
    fn budget_check_monthly_exceeded() {
        let (_tmp, mem) = temp_sqlite();
        mem.record_embedding_cost("model", 10_000, 0.01);
        let result = mem.check_embedding_budget(0.0, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn budget_check_daily_exceeded() {
        let (_tmp, mem) = temp_sqlite();
        mem.record_embedding_cost("model", 10_000, 0.01);
        let result = mem.check_embedding_budget(0.05, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn budget_check_zero_means_unlimited() {
        let (_tmp, mem) = temp_sqlite();
        mem.record_embedding_cost("model", 1_000_000, 1.0);
        assert!(mem.check_embedding_budget(0.0, 0.0).is_ok());
    }
}
