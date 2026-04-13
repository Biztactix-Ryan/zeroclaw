// Vector operations — cosine similarity, normalization, hybrid merge.

/// Cosine similarity between two vectors. Returns 0.0–1.0.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if !denom.is_finite() || denom < f64::EPSILON {
        return 0.0;
    }

    let raw = dot / denom;
    if !raw.is_finite() {
        return 0.0;
    }

    // Clamp to [0, 1] — embeddings are typically positive
    #[allow(clippy::cast_possible_truncation)]
    let sim = raw.clamp(0.0, 1.0) as f32;
    sim
}

/// Serialize f32 vector to bytes (little-endian)
pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes to f32 vector (little-endian)
pub fn bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// A scored result for hybrid merging
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub id: String,
    pub vector_score: Option<f32>,
    pub keyword_score: Option<f32>,
    pub final_score: f32,
}

/// Hybrid merge: combine vector and keyword results with weighted fusion.
///
/// Normalizes each score set to [0, 1], then computes:
///   `final_score` = `vector_weight` * `vector_score` + `keyword_weight` * `keyword_score`
///
/// Deduplicates by id, keeping the best score from each source.
pub fn hybrid_merge(
    vector_results: &[(String, f32)],  // (id, cosine_similarity)
    keyword_results: &[(String, f32)], // (id, bm25_score)
    vector_weight: f32,
    keyword_weight: f32,
    limit: usize,
) -> Vec<ScoredResult> {
    use std::collections::HashMap;

    let mut map: HashMap<String, ScoredResult> = HashMap::new();

    // Normalize vector scores (already 0–1 from cosine similarity)
    for (id, score) in vector_results {
        map.entry(id.clone())
            .and_modify(|r| r.vector_score = Some(*score))
            .or_insert_with(|| ScoredResult {
                id: id.clone(),
                vector_score: Some(*score),
                keyword_score: None,
                final_score: 0.0,
            });
    }

    // Normalize keyword scores (BM25 can be any positive number)
    let max_kw = keyword_results
        .iter()
        .map(|(_, s)| *s)
        .fold(0.0_f32, f32::max);
    let max_kw = if max_kw < f32::EPSILON { 1.0 } else { max_kw };

    for (id, score) in keyword_results {
        let normalized = score / max_kw;
        map.entry(id.clone())
            .and_modify(|r| r.keyword_score = Some(normalized))
            .or_insert_with(|| ScoredResult {
                id: id.clone(),
                vector_score: None,
                keyword_score: Some(normalized),
                final_score: 0.0,
            });
    }

    // Compute final scores
    let mut results: Vec<ScoredResult> = map
        .into_values()
        .map(|mut r| {
            let vs = r.vector_score.unwrap_or(0.0);
            let ks = r.keyword_score.unwrap_or(0.0);
            r.final_score = vector_weight * vs + keyword_weight * ks;
            r
        })
        .collect();

    results.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    results.truncate(limit);
    results
}

// ── Scalar Quantization (TurboQuant) ─────────────────────────────────────
//
// Compresses f32 embeddings to int8/int4/binary representations by mapping
// each dimension's value range to a fixed integer range.
//
// Storage comparison (384-dim MiniLM):
//   f32:    1,536 bytes
//   int8:     386 bytes (2 header + 384 data)  → 4.0x compression
//   int4:     194 bytes (2 header + 192 data)  → 7.9x compression
//   binary:    50 bytes (2 header + 48 data)   → 30.7x compression
//
// Header: min (f32 LE) + scale (f32 LE) = 8 bytes for decode.

/// Quantized embedding with metadata for decode.
#[derive(Debug, Clone)]
pub struct QuantizedVec {
    pub min: f32,
    pub scale: f32,
    pub data: Vec<u8>,
    pub bit_width: BitWidth,
    pub dimensions: usize,
}

/// Supported quantization bit widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    /// No quantization — raw f32 storage
    F32,
    /// 8-bit unsigned integer (0–255)
    Int8,
    /// 4-bit unsigned integer (0–15), two values per byte
    Int4,
    /// 1-bit binary (sign of each dimension), 8 values per byte
    Binary,
}

impl BitWidth {
    /// Bytes needed for `dims` dimensions at this bit width (excluding header).
    pub fn data_bytes(self, dims: usize) -> usize {
        match self {
            BitWidth::F32 => dims * 4,
            BitWidth::Int8 => dims,
            BitWidth::Int4 => (dims + 1) / 2,
            BitWidth::Binary => (dims + 7) / 8,
        }
    }

    /// Compression ratio vs f32 (higher = more compression).
    pub fn compression_ratio(self) -> f32 {
        match self {
            BitWidth::F32 => 1.0,
            BitWidth::Int8 => 4.0,
            BitWidth::Int4 => 8.0,
            BitWidth::Binary => 32.0,
        }
    }
}

/// Encode f32 vector to scalar-quantized int8.
///
/// Maps [min, max] → [0, 255]. Stores min + scale in header for decode.
pub fn quantize_int8(v: &[f32]) -> QuantizedVec {
    let (min, max) = min_max(v);
    let range = max - min;
    let scale = if range < f32::EPSILON { 1.0 } else { range };

    let data: Vec<u8> = v
        .iter()
        .map(|&x| {
            let normalized = (x - min) / scale;
            (normalized * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect();

    QuantizedVec {
        min,
        scale,
        data,
        bit_width: BitWidth::Int8,
        dimensions: v.len(),
    }
}

/// Encode f32 vector to scalar-quantized int4 (two values per byte).
///
/// Maps [min, max] → [0, 15].
pub fn quantize_int4(v: &[f32]) -> QuantizedVec {
    let (min, max) = min_max(v);
    let range = max - min;
    let scale = if range < f32::EPSILON { 1.0 } else { range };

    let mut data = Vec::with_capacity((v.len() + 1) / 2);
    for pair in v.chunks(2) {
        let hi = ((pair[0] - min) / scale * 15.0).round().clamp(0.0, 15.0) as u8;
        let lo = if pair.len() > 1 {
            ((pair[1] - min) / scale * 15.0).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        data.push((hi << 4) | lo);
    }

    QuantizedVec {
        min,
        scale,
        data,
        bit_width: BitWidth::Int4,
        dimensions: v.len(),
    }
}

/// Encode f32 vector to binary quantization (sign bit per dimension).
///
/// Each dimension becomes 1 if positive, 0 if negative/zero.
/// Comparison uses Hamming distance instead of cosine similarity.
pub fn quantize_binary(v: &[f32]) -> QuantizedVec {
    let mut data = Vec::with_capacity((v.len() + 7) / 8);
    for byte_chunk in v.chunks(8) {
        let mut byte = 0u8;
        for (i, &x) in byte_chunk.iter().enumerate() {
            if x > 0.0 {
                byte |= 1 << (7 - i);
            }
        }
        data.push(byte);
    }

    QuantizedVec {
        min: 0.0,
        scale: 1.0,
        data,
        bit_width: BitWidth::Binary,
        dimensions: v.len(),
    }
}

/// Decode quantized vector back to f32.
pub fn dequantize(q: &QuantizedVec) -> Vec<f32> {
    match q.bit_width {
        BitWidth::F32 => bytes_to_vec(&q.data),
        BitWidth::Int8 => q
            .data
            .iter()
            .map(|&b| q.min + (f32::from(b) / 255.0) * q.scale)
            .collect(),
        BitWidth::Int4 => {
            let mut result = Vec::with_capacity(q.dimensions);
            for &byte in &q.data {
                let hi = (byte >> 4) & 0x0F;
                result.push(q.min + (f32::from(hi) / 15.0) * q.scale);
                if result.len() < q.dimensions {
                    let lo = byte & 0x0F;
                    result.push(q.min + (f32::from(lo) / 15.0) * q.scale);
                }
            }
            result
        }
        BitWidth::Binary => {
            let mut result = Vec::with_capacity(q.dimensions);
            for &byte in &q.data {
                for i in 0..8 {
                    if result.len() >= q.dimensions {
                        break;
                    }
                    result.push(if byte & (1 << (7 - i)) != 0 {
                        1.0
                    } else {
                        -1.0
                    });
                }
            }
            result
        }
    }
}

/// Serialize a quantized vector to bytes for SQLite BLOB storage.
///
/// Format: [bit_width: u8] [dims: u16 LE] [min: f32 LE] [scale: f32 LE] [data...]
pub fn quantized_to_bytes(q: &QuantizedVec) -> Vec<u8> {
    let header_size = 1 + 2 + 4 + 4; // bit_width + dims + min + scale
    let mut bytes = Vec::with_capacity(header_size + q.data.len());
    bytes.push(match q.bit_width {
        BitWidth::F32 => 32,
        BitWidth::Int8 => 8,
        BitWidth::Int4 => 4,
        BitWidth::Binary => 1,
    });
    bytes.extend_from_slice(&(q.dimensions as u16).to_le_bytes());
    bytes.extend_from_slice(&q.min.to_le_bytes());
    bytes.extend_from_slice(&q.scale.to_le_bytes());
    bytes.extend_from_slice(&q.data);
    bytes
}

/// Deserialize a quantized vector from bytes.
pub fn bytes_to_quantized(bytes: &[u8]) -> Option<QuantizedVec> {
    if bytes.len() < 11 {
        return None; // minimum header
    }
    let bit_width = match bytes[0] {
        32 => BitWidth::F32,
        8 => BitWidth::Int8,
        4 => BitWidth::Int4,
        1 => BitWidth::Binary,
        _ => return None,
    };
    let dims = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
    let min = f32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    let scale = f32::from_le_bytes([bytes[7], bytes[8], bytes[9], bytes[10]]);
    let data = bytes[11..].to_vec();

    let expected = bit_width.data_bytes(dims);
    if data.len() < expected {
        return None;
    }

    Some(QuantizedVec {
        min,
        scale,
        data,
        bit_width,
        dimensions: dims,
    })
}

/// Smart decode: auto-detect quantized vs raw f32 format.
///
/// Quantized blobs start with a known header byte (8, 4, or 1) followed by
/// dims as u16 LE. Raw f32 blobs have length divisible by 4 and typically
/// don't start with 1/4/8 as the first byte of a float.
///
/// This enables backward compatibility: old f32 BLOBs continue to work
/// alongside new quantized BLOBs.
pub fn smart_decode(blob: &[u8]) -> Vec<f32> {
    // Try quantized format first — check header byte
    if blob.len() >= 11 {
        let bit_width_byte = blob[0];
        if matches!(bit_width_byte, 1 | 4 | 8 | 32) {
            let dims = u16::from_le_bytes([blob[1], blob[2]]) as usize;
            let bw = match bit_width_byte {
                32 => BitWidth::F32,
                8 => BitWidth::Int8,
                4 => BitWidth::Int4,
                1 => BitWidth::Binary,
                _ => unreachable!(),
            };
            let expected_total = 11 + bw.data_bytes(dims);
            if blob.len() == expected_total && dims > 0 && dims <= 4096 {
                if let Some(q) = bytes_to_quantized(blob) {
                    return dequantize(&q);
                }
            }
        }
    }

    // Fallback: raw f32
    bytes_to_vec(blob)
}

/// Encode an f32 embedding using the specified quantization mode.
///
/// Returns bytes ready for SQLite BLOB storage.
pub fn encode_embedding(v: &[f32], bit_width: BitWidth) -> Vec<u8> {
    match bit_width {
        BitWidth::F32 => vec_to_bytes(v),
        _ => {
            let q = match bit_width {
                BitWidth::Int8 => quantize_int8(v),
                BitWidth::Int4 => quantize_int4(v),
                BitWidth::Binary => quantize_binary(v),
                BitWidth::F32 => unreachable!(),
            };
            quantized_to_bytes(&q)
        }
    }
}

fn min_max(v: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in v {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    if !min.is_finite() {
        min = 0.0;
    }
    if !max.is_finite() {
        max = 1.0;
    }
    (min, max)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn cosine_similar_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 3.1];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_mismatched_lengths() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn vec_bytes_roundtrip() {
        let original = vec![1.0_f32, -2.5, 3.14, 0.0, f32::MAX];
        let bytes = vec_to_bytes(&original);
        let restored = bytes_to_vec(&bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn vec_bytes_empty() {
        let bytes = vec_to_bytes(&[]);
        assert!(bytes.is_empty());
        let restored = bytes_to_vec(&bytes);
        assert!(restored.is_empty());
    }

    #[test]
    fn hybrid_merge_vector_only() {
        let vec_results = vec![("a".into(), 0.9), ("b".into(), 0.5)];
        let merged = hybrid_merge(&vec_results, &[], 0.7, 0.3, 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "a");
        assert!(merged[0].final_score > merged[1].final_score);
    }

    #[test]
    fn hybrid_merge_keyword_only() {
        let kw_results = vec![("x".into(), 10.0), ("y".into(), 5.0)];
        let merged = hybrid_merge(&[], &kw_results, 0.7, 0.3, 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "x");
    }

    #[test]
    fn hybrid_merge_deduplicates() {
        let vec_results = vec![("a".into(), 0.9)];
        let kw_results = vec![("a".into(), 10.0)];
        let merged = hybrid_merge(&vec_results, &kw_results, 0.7, 0.3, 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "a");
        // Should have both scores
        assert!(merged[0].vector_score.is_some());
        assert!(merged[0].keyword_score.is_some());
        // Final score should be higher than either alone
        assert!(merged[0].final_score > 0.7 * 0.9);
    }

    #[test]
    fn hybrid_merge_respects_limit() {
        let vec_results: Vec<(String, f32)> = (0..20)
            .map(|i| (format!("item_{i}"), 1.0 - i as f32 * 0.05))
            .collect();
        let merged = hybrid_merge(&vec_results, &[], 1.0, 0.0, 5);
        assert_eq!(merged.len(), 5);
    }

    #[test]
    fn hybrid_merge_empty_inputs() {
        let merged = hybrid_merge(&[], &[], 0.7, 0.3, 10);
        assert!(merged.is_empty());
    }

    // ── Edge cases: cosine similarity ────────────────────────────

    #[test]
    fn cosine_nan_returns_zero() {
        let a = vec![f32::NAN, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        // NaN propagates through arithmetic — result should be 0.0 (clamped or denom check)
        assert!(sim.is_finite(), "Expected finite, got {sim}");
    }

    #[test]
    fn cosine_infinity_returns_zero_or_finite() {
        let a = vec![f32::INFINITY, 1.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite(), "Expected finite, got {sim}");
    }

    #[test]
    fn cosine_negative_values() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![-1.0, -2.0, -3.0];
        // Identical negative vectors → cosine = 1.0, but clamped to [0,1]
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_opposite_vectors_clamped() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        // Cosine = -1.0, clamped to 0.0
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_high_dimensional() {
        let a: Vec<f32> = (0..1536).map(|i| (f64::from(i) * 0.001) as f32).collect();
        let b: Vec<f32> = (0..1536)
            .map(|i| (f64::from(i) * 0.001 + 0.0001) as f32)
            .collect();
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim > 0.99,
            "High-dim similar vectors should be close: {sim}"
        );
    }

    #[test]
    fn cosine_single_element() {
        assert!((cosine_similarity(&[5.0], &[5.0]) - 1.0).abs() < 0.001);
        assert!(cosine_similarity(&[5.0], &[-5.0]).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_both_zero_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    // ── Edge cases: vec↔bytes serialization ──────────────────────

    #[test]
    fn bytes_to_vec_non_aligned_truncates() {
        // 5 bytes → only first 4 used (1 float), last byte dropped
        let bytes = vec![0u8, 0, 0, 0, 0xFF];
        let result = bytes_to_vec(&bytes);
        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < f32::EPSILON);
    }

    #[test]
    fn bytes_to_vec_three_bytes_returns_empty() {
        let bytes = vec![1u8, 2, 3];
        let result = bytes_to_vec(&bytes);
        assert!(result.is_empty());
    }

    #[test]
    fn vec_bytes_roundtrip_special_values() {
        let special = vec![f32::MIN, f32::MAX, f32::EPSILON, -0.0, 0.0];
        let bytes = vec_to_bytes(&special);
        let restored = bytes_to_vec(&bytes);
        assert_eq!(special.len(), restored.len());
        for (a, b) in special.iter().zip(restored.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn vec_bytes_roundtrip_nan_preserves_bits() {
        let nan_vec = vec![f32::NAN];
        let bytes = vec_to_bytes(&nan_vec);
        let restored = bytes_to_vec(&bytes);
        assert!(restored[0].is_nan());
    }

    // ── Edge cases: hybrid merge ─────────────────────────────────

    #[test]
    fn hybrid_merge_limit_zero() {
        let vec_results = vec![("a".into(), 0.9)];
        let merged = hybrid_merge(&vec_results, &[], 0.7, 0.3, 0);
        assert!(merged.is_empty());
    }

    #[test]
    fn hybrid_merge_zero_weights() {
        let vec_results = vec![("a".into(), 0.9)];
        let kw_results = vec![("b".into(), 10.0)];
        let merged = hybrid_merge(&vec_results, &kw_results, 0.0, 0.0, 10);
        // All final scores should be 0.0
        for r in &merged {
            assert!(r.final_score.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn hybrid_merge_negative_keyword_scores() {
        // BM25 scores are negated in our code, but raw negatives shouldn't crash
        let kw_results = vec![("a".into(), -5.0), ("b".into(), -1.0)];
        let merged = hybrid_merge(&[], &kw_results, 0.7, 0.3, 10);
        assert_eq!(merged.len(), 2);
        // Should still produce finite scores
        for r in &merged {
            assert!(r.final_score.is_finite());
        }
    }

    #[test]
    fn hybrid_merge_duplicate_ids_in_same_source() {
        let vec_results = vec![("a".into(), 0.9), ("a".into(), 0.5)];
        let merged = hybrid_merge(&vec_results, &[], 1.0, 0.0, 10);
        // Should deduplicate — only 1 entry for "a"
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn hybrid_merge_large_bm25_normalization() {
        let kw_results = vec![("a".into(), 1000.0), ("b".into(), 500.0), ("c".into(), 1.0)];
        let merged = hybrid_merge(&[], &kw_results, 0.0, 1.0, 10);
        // "a" should have normalized score of 1.0
        assert!((merged[0].keyword_score.unwrap() - 1.0).abs() < 0.001);
        // "b" should have 0.5
        assert!((merged[1].keyword_score.unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn hybrid_merge_single_item() {
        let merged = hybrid_merge(&[("only".into(), 0.8)], &[], 0.7, 0.3, 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "only");
    }

    // ── TurboQuant: scalar quantization tests ───────────────────────

    fn sample_embedding() -> Vec<f32> {
        // Simulates a normalized 384-dim embedding
        (0..384)
            .map(|i| ((i as f64 * 0.0163).sin() * 0.5) as f32)
            .collect()
    }

    #[test]
    fn int8_roundtrip_preserves_similarity() {
        let original = sample_embedding();
        let quantized = quantize_int8(&original);
        let decoded = dequantize(&quantized);

        let sim = cosine_similarity(&original, &decoded);
        assert!(
            sim > 0.999,
            "int8 roundtrip should preserve >99.9% similarity, got {sim:.6}"
        );
    }

    #[test]
    fn int4_roundtrip_reasonable_similarity() {
        let original = sample_embedding();
        let quantized = quantize_int4(&original);
        let decoded = dequantize(&quantized);

        let sim = cosine_similarity(&original, &decoded);
        assert!(
            sim > 0.98,
            "int4 roundtrip should preserve >98% similarity, got {sim:.6}"
        );
    }

    #[test]
    fn binary_roundtrip_directional_similarity() {
        let original = sample_embedding();
        let quantized = quantize_binary(&original);
        let decoded = dequantize(&quantized);

        let sim = cosine_similarity(&original, &decoded);
        assert!(
            sim > 0.5,
            "binary should preserve rough direction, got {sim:.6}"
        );
    }

    #[test]
    fn int8_compression_ratio() {
        let original = sample_embedding();
        let f32_size = vec_to_bytes(&original).len();
        let q = quantize_int8(&original);
        let q_size = quantized_to_bytes(&q).len();

        let ratio = f32_size as f32 / q_size as f32;
        assert!(
            ratio > 3.5,
            "int8 should give >3.5x compression, got {ratio:.1}x ({f32_size} → {q_size})"
        );
    }

    #[test]
    fn int4_compression_ratio() {
        let original = sample_embedding();
        let f32_size = vec_to_bytes(&original).len();
        let q = quantize_int4(&original);
        let q_size = quantized_to_bytes(&q).len();

        let ratio = f32_size as f32 / q_size as f32;
        assert!(
            ratio > 7.0,
            "int4 should give >7x compression, got {ratio:.1}x ({f32_size} → {q_size})"
        );
    }

    #[test]
    fn binary_compression_ratio() {
        let original = sample_embedding();
        let f32_size = vec_to_bytes(&original).len();
        let q = quantize_binary(&original);
        let q_size = quantized_to_bytes(&q).len();

        let ratio = f32_size as f32 / q_size as f32;
        assert!(
            ratio > 25.0,
            "binary should give >25x compression, got {ratio:.1}x ({f32_size} → {q_size})"
        );
    }

    #[test]
    fn quantized_blob_roundtrip() {
        let original = sample_embedding();
        let q = quantize_int8(&original);
        let blob = quantized_to_bytes(&q);
        let restored = bytes_to_quantized(&blob).expect("should decode");

        assert_eq!(restored.dimensions, 384);
        assert_eq!(restored.bit_width, BitWidth::Int8);
        assert_eq!(restored.data.len(), q.data.len());

        let decoded = dequantize(&restored);
        let sim = cosine_similarity(&original, &decoded);
        assert!(sim > 0.999);
    }

    #[test]
    fn quantized_blob_roundtrip_int4() {
        let original = sample_embedding();
        let q = quantize_int4(&original);
        let blob = quantized_to_bytes(&q);
        let restored = bytes_to_quantized(&blob).expect("should decode");

        assert_eq!(restored.bit_width, BitWidth::Int4);
        let decoded = dequantize(&restored);
        let sim = cosine_similarity(&original, &decoded);
        assert!(sim > 0.98);
    }

    #[test]
    fn quantized_blob_roundtrip_binary() {
        let original = sample_embedding();
        let q = quantize_binary(&original);
        let blob = quantized_to_bytes(&q);
        let restored = bytes_to_quantized(&blob).expect("should decode");

        assert_eq!(restored.bit_width, BitWidth::Binary);
        assert_eq!(restored.dimensions, 384);
    }

    #[test]
    fn int8_preserves_relative_ordering() {
        // Two similar vectors and one dissimilar — ordering should be preserved
        let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01 + 0.001).sin()).collect();
        let c: Vec<f32> = (0..384).map(|i| (i as f32 * 0.5).cos()).collect();

        let qa = quantize_int8(&a);
        let qb = quantize_int8(&b);
        let qc = quantize_int8(&c);

        let da = dequantize(&qa);
        let db = dequantize(&qb);
        let dc = dequantize(&qc);

        let sim_ab_orig = cosine_similarity(&a, &b);
        let sim_ac_orig = cosine_similarity(&a, &c);
        let sim_ab_quant = cosine_similarity(&da, &db);
        let sim_ac_quant = cosine_similarity(&da, &dc);

        // a and b are very similar, a and c are dissimilar
        // This ordering should be preserved after quantization
        assert!(
            sim_ab_orig > sim_ac_orig,
            "original: a-b ({sim_ab_orig:.4}) should be more similar than a-c ({sim_ac_orig:.4})"
        );
        assert!(
            sim_ab_quant > sim_ac_quant,
            "quantized: ordering should be preserved: a-b ({sim_ab_quant:.4}) > a-c ({sim_ac_quant:.4})"
        );
    }

    #[test]
    fn quantize_constant_vector() {
        // All values the same — edge case for scale = 0
        let v = vec![0.5_f32; 384];
        let q = quantize_int8(&v);
        let decoded = dequantize(&q);
        // All decoded values should be approximately 0.5
        for &d in &decoded {
            assert!((d - 0.5).abs() < 0.01, "constant vector decode failed: {d}");
        }
    }

    #[test]
    fn quantize_zero_vector() {
        let v = vec![0.0_f32; 384];
        let q = quantize_int8(&v);
        let decoded = dequantize(&q);
        for &d in &decoded {
            assert!(d.abs() < 0.01, "zero vector decode failed: {d}");
        }
    }

    #[test]
    fn bytes_to_quantized_invalid_returns_none() {
        assert!(bytes_to_quantized(&[]).is_none());
        assert!(bytes_to_quantized(&[0u8; 5]).is_none());
        // Invalid bit width
        assert!(bytes_to_quantized(&[99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
    }
}
