# mmr-rerank

[![crates.io](https://img.shields.io/crates/v/mmr-rerank.svg)](https://crates.io/crates/mmr-rerank)

Maximal Marginal Relevance (Carbonell & Goldstein 1998) reranker.
Balances query relevance against pairwise novelty so retrieved chunks
don't all say the same thing.

```rust
use mmr_rerank::mmr;
let rel = vec![0.9, 0.85, 0.6, 0.55];
let pair = vec![
    vec![1.0, 0.95, 0.10, 0.10],
    vec![0.95, 1.0, 0.10, 0.10],
    vec![0.10, 0.10, 1.0, 0.95],
    vec![0.10, 0.10, 0.95, 1.0],
];
let picks = mmr(&rel, &pair, 0.5, 2);
// [0, 2] — top relevance, then diverse from it
```

Zero deps. MIT or Apache-2.0.
