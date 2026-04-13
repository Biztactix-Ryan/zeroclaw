//! Background re-embed worker for Phase 4 multi-model embeddings.
//!
//! Processes the `reembed_queue` table: picks up chunks that need embedding
//! with a new model, embeds them, stores in `chunk_embeddings`, and records
//! cost in the ledger.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use rusqlite::{params, Connection};
use tokio::sync::watch;

use super::embeddings::EmbeddingProvider;
use super::vector;

/// A single queue item to process.
struct QueueItem {
    id: i64,
    chunk_id: String,
    model_id: String,
    retries: i64,
}

/// Background worker that processes the reembed queue.
pub struct ReembedWorker {
    conn: Arc<Mutex<Connection>>,
    embedder: Arc<dyn EmbeddingProvider>,
    shutdown: watch::Receiver<bool>,
    quantization: vector::BitWidth,
    daily_limit: f64,
    monthly_limit: f64,
    cost_per_1k: f64,
    batch_size: usize,
    max_retries: i64,
}

impl ReembedWorker {
    pub fn new(
        conn: Arc<Mutex<Connection>>,
        embedder: Arc<dyn EmbeddingProvider>,
        shutdown: watch::Receiver<bool>,
        quantization: vector::BitWidth,
        daily_limit: f64,
        monthly_limit: f64,
        cost_per_1k: f64,
    ) -> Self {
        Self {
            conn,
            embedder,
            shutdown,
            quantization,
            daily_limit,
            monthly_limit,
            cost_per_1k,
            batch_size: 10,
            max_retries: 3,
        }
    }

    /// Run the worker loop until shutdown signal.
    pub async fn run(&mut self) {
        loop {
            // Check shutdown
            if *self.shutdown.borrow() {
                tracing::info!("reembed worker: shutdown signal received");
                break;
            }

            // Check budget
            if !self.within_budget() {
                tracing::warn!("reembed worker: budget exceeded, pausing");
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(300)) => {},
                    _ = self.shutdown.changed() => break,
                }
                continue;
            }

            // Fetch batch from queue
            let items = self.fetch_batch();
            if items.is_empty() {
                // Nothing to do — wait before polling again
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {},
                    _ = self.shutdown.changed() => break,
                }
                continue;
            }

            for item in items {
                if *self.shutdown.borrow() {
                    break;
                }

                match self.process_item(&item).await {
                    Ok(()) => {
                        self.dequeue(item.id);
                        tracing::debug!(
                            chunk_id = %item.chunk_id,
                            model = %item.model_id,
                            "reembed: success"
                        );
                    }
                    Err(e) => {
                        if item.retries + 1 >= self.max_retries {
                            tracing::error!(
                                chunk_id = %item.chunk_id,
                                model = %item.model_id,
                                retries = item.retries + 1,
                                "reembed: max retries exceeded, removing from queue: {e}"
                            );
                            self.dequeue(item.id);
                        } else {
                            tracing::warn!(
                                chunk_id = %item.chunk_id,
                                model = %item.model_id,
                                retry = item.retries + 1,
                                "reembed: retry after error: {e}"
                            );
                            self.increment_retry(item.id);
                        }
                    }
                }
            }
        }
    }

    /// Process a single queue item: read chunk content, embed, store.
    async fn process_item(&self, item: &QueueItem) -> anyhow::Result<()> {
        // Read chunk content
        let content = {
            let conn = self.conn.lock();
            conn.query_row(
                "SELECT content FROM chunks WHERE chunk_id = ?1",
                params![item.chunk_id],
                |row| row.get::<_, String>(0),
            )
            .map_err(|e| anyhow::anyhow!("chunk not found: {e}"))?
        };

        // Embed
        let embedding = self.embedder.embed_one(&content).await?;
        let tokens = content.split_whitespace().count();

        // Store in chunk_embeddings
        let blob = vector::encode_embedding(&embedding, self.quantization);
        let dims = embedding.len();
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR REPLACE INTO chunk_embeddings
                 (chunk_embedding_id, chunk_id, model_id, embedding, dimensions)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    uuid::Uuid::new_v4().to_string(),
                    item.chunk_id,
                    item.model_id,
                    blob,
                    dims,
                ],
            )?;
        }

        // Record cost
        self.record_cost(&item.model_id, tokens);

        Ok(())
    }

    fn fetch_batch(&self) -> Vec<QueueItem> {
        let conn = self.conn.lock();
        let Ok(mut stmt) = conn.prepare(
            "SELECT id, chunk_id, model_id, retries FROM reembed_queue
             ORDER BY priority DESC, queued_at ASC
             LIMIT ?1",
        ) else {
            return vec![];
        };

        let Ok(rows) = stmt.query_map(params![self.batch_size as i64], |row| {
            Ok(QueueItem {
                id: row.get(0)?,
                chunk_id: row.get(1)?,
                model_id: row.get(2)?,
                retries: row.get(3)?,
            })
        }) else {
            return vec![];
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    fn dequeue(&self, id: i64) {
        let conn = self.conn.lock();
        let _ = conn.execute("DELETE FROM reembed_queue WHERE id = ?1", params![id]);
    }

    fn increment_retry(&self, id: i64) {
        let conn = self.conn.lock();
        let _ = conn.execute(
            "UPDATE reembed_queue SET retries = retries + 1 WHERE id = ?1",
            params![id],
        );
    }

    fn within_budget(&self) -> bool {
        if self.daily_limit <= 0.0 && self.monthly_limit <= 0.0 {
            return true;
        }

        let now = chrono::Utc::now();
        let period = now.format("%Y-%m").to_string();
        let today = now.format("%Y-%m-%d").to_string();
        let conn = self.conn.lock();

        if self.monthly_limit > 0.0 {
            let spend: f64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) FROM embedding_cost_ledger WHERE period = ?1",
                    params![period],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);
            if spend >= self.monthly_limit {
                return false;
            }
        }

        if self.daily_limit > 0.0 {
            let spend: f64 = conn
                .query_row(
                    "SELECT COALESCE(SUM(cost_usd), 0.0) FROM embedding_cost_ledger WHERE recorded_at >= ?1",
                    params![format!("{today}T00:00:00Z")],
                    |row| row.get(0),
                )
                .unwrap_or(0.0);
            if spend >= self.daily_limit {
                return false;
            }
        }

        true
    }

    fn record_cost(&self, model_id: &str, tokens: usize) {
        let cost_usd = self.cost_per_1k * (tokens as f64 / 1000.0);
        let period = chrono::Utc::now().format("%Y-%m").to_string();
        let conn = self.conn.lock();
        let _ = conn.execute(
            "INSERT INTO embedding_cost_ledger (model_id, tokens_used, cost_usd, period)
             VALUES (?1, ?2, ?3, ?4)",
            params![model_id, tokens as i64, cost_usd, period],
        );
    }
}

/// Enqueue chunks for re-embedding with a new model.
pub fn enqueue_chunks(
    conn: &Connection,
    chunk_ids: &[String],
    model_id: &str,
    priority: i32,
) -> anyhow::Result<usize> {
    let mut count = 0;
    for chunk_id in chunk_ids {
        let result = conn.execute(
            "INSERT OR IGNORE INTO reembed_queue (chunk_id, model_id, priority)
             VALUES (?1, ?2, ?3)",
            params![chunk_id, model_id, priority],
        );
        if let Ok(n) = result {
            count += n;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Arc<Mutex<Connection>>) {
        let tmp = TempDir::new().unwrap();
        let mem = crate::sqlite::SqliteMemory::new(tmp.path()).unwrap();
        let conn = mem.connection().clone();
        (tmp, conn)
    }

    #[test]
    fn enqueue_and_count() {
        let (_tmp, conn) = setup();
        let conn_guard = conn.lock();

        // Insert a source + doc + chunk so FK constraints are satisfied
        let now = chrono::Utc::now().to_rfc3339();
        conn_guard
            .execute(
                "INSERT INTO sources (source_id, display_name, uri_scheme, registered_at)
                 VALUES ('s1', 'S1', 'test://', ?1)",
                params![now],
            )
            .unwrap();
        conn_guard
            .execute(
                "INSERT INTO source_documents (doc_id, source_id, source_uri, chunk_count, total_tokens, indexed_at, updated_at)
                 VALUES ('d1', 's1', 'test://d1', 1, 50, ?1, ?1)",
                params![now],
            )
            .unwrap();
        conn_guard
            .execute(
                "INSERT INTO chunks (chunk_id, doc_id, source_id, content, chunk_index, created_at)
                 VALUES ('c1', 'd1', 's1', 'test content for embedding', 0, ?1)",
                params![now],
            )
            .unwrap();

        let enqueued =
            enqueue_chunks(&conn_guard, &["c1".into()], "local:minilm", 1).unwrap();
        assert_eq!(enqueued, 1);

        // Duplicate enqueue should be ignored (UNIQUE constraint)
        let dup = enqueue_chunks(&conn_guard, &["c1".into()], "local:minilm", 1).unwrap();
        assert_eq!(dup, 0);

        // Different model for same chunk should work
        let diff =
            enqueue_chunks(&conn_guard, &["c1".into()], "local:bge-small", 1).unwrap();
        assert_eq!(diff, 1);

        let count: i64 = conn_guard
            .query_row("SELECT COUNT(*) FROM reembed_queue", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn worker_budget_check() {
        let (_tmp, conn) = setup();
        let (tx, rx) = watch::channel(false);

        let worker = ReembedWorker::new(
            conn.clone(),
            Arc::new(crate::embeddings::NoopEmbedding),
            rx,
            vector::BitWidth::Int8,
            0.05, // $0.05 daily limit
            1.00, // $1.00 monthly limit
            0.01, // $0.01 per 1k tokens
        );

        // No spend yet — should be within budget
        assert!(worker.within_budget());

        // Record spend that exceeds daily limit
        {
            let conn = conn.lock();
            let period = chrono::Utc::now().format("%Y-%m").to_string();
            conn.execute(
                "INSERT INTO embedding_cost_ledger (model_id, tokens_used, cost_usd, period)
                 VALUES ('model', 10000, 0.10, ?1)",
                params![period],
            )
            .unwrap();
        }

        // Should now be over daily limit
        assert!(!worker.within_budget());

        drop(tx);
    }
}
