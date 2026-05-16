//! # mmr-rerank
//!
//! Maximal Marginal Relevance (Carbonell & Goldstein, 1998) reranker.
//!
//! Given relevance scores `rel[i] = sim(query, doc_i)` and a pairwise
//! similarity matrix `pair[i][j] = sim(doc_i, doc_j)`, returns the
//! top-k indices that balance query relevance against novelty:
//!
//! ```text
//! pick = argmax_{i not picked}  λ * rel[i] - (1 - λ) * max_{j picked} pair[i][j]
//! ```
//!
//! ## Example
//!
//! ```
//! use mmr_rerank::mmr;
//! let rel = vec![0.9, 0.85, 0.6, 0.55];
//! // 4 docs; pair[i][j] = pairwise similarity
//! let pair = vec![
//!     vec![1.0, 0.95, 0.10, 0.10],
//!     vec![0.95, 1.0, 0.10, 0.10],
//!     vec![0.10, 0.10, 1.0, 0.95],
//!     vec![0.10, 0.10, 0.95, 1.0],
//! ];
//! // λ = 0.5 trades off relevance and diversity equally.
//! let picks = mmr(&rel, &pair, 0.5, 2);
//! // Expect 0 (top relevance), then 2 (low pair sim with 0).
//! assert_eq!(picks, vec![0, 2]);
//! ```

#![deny(missing_docs)]

/// Pick `k` indices in MMR order. `lambda ∈ [0, 1]`.
pub fn mmr(rel: &[f32], pair: &[Vec<f32>], lambda: f32, k: usize) -> Vec<usize> {
    let n = rel.len();
    assert_eq!(pair.len(), n, "pair rows must equal rel length");
    let mut picked: Vec<usize> = Vec::with_capacity(k.min(n));
    let mut remaining: Vec<usize> = (0..n).collect();

    while picked.len() < k && !remaining.is_empty() {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx_in_rem = 0;
        for (slot, &cand) in remaining.iter().enumerate() {
            let max_pair = picked
                .iter()
                .map(|&p| pair[cand][p])
                .fold(0.0_f32, f32::max);
            let score = lambda * rel[cand] - (1.0 - lambda) * max_pair;
            if score > best_score {
                best_score = score;
                best_idx_in_rem = slot;
            }
        }
        let winner = remaining.remove(best_idx_in_rem);
        picked.push(winner);
    }

    picked
}
