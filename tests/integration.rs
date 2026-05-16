use mmr_rerank::mmr;

#[test]
fn picks_top_relevance_first() {
    let rel = vec![0.9, 0.85, 0.6, 0.55];
    let pair = vec![
        vec![1.0, 0.95, 0.10, 0.10],
        vec![0.95, 1.0, 0.10, 0.10],
        vec![0.10, 0.10, 1.0, 0.95],
        vec![0.10, 0.10, 0.95, 1.0],
    ];
    let picks = mmr(&rel, &pair, 0.5, 1);
    assert_eq!(picks, vec![0]);
}

#[test]
fn diversity_steers_second_pick() {
    let rel = vec![0.9, 0.85, 0.6, 0.55];
    let pair = vec![
        vec![1.0, 0.95, 0.10, 0.10],
        vec![0.95, 1.0, 0.10, 0.10],
        vec![0.10, 0.10, 1.0, 0.95],
        vec![0.10, 0.10, 0.95, 1.0],
    ];
    // With lambda 0.5, MMR should prefer index 2 (more diverse) over 1.
    let picks = mmr(&rel, &pair, 0.5, 2);
    assert_eq!(picks, vec![0, 2]);
}

#[test]
fn lambda_one_is_pure_relevance() {
    let rel = vec![0.9, 0.85, 0.6, 0.55];
    let pair = vec![
        vec![1.0, 0.95, 0.10, 0.10],
        vec![0.95, 1.0, 0.10, 0.10],
        vec![0.10, 0.10, 1.0, 0.95],
        vec![0.10, 0.10, 0.95, 1.0],
    ];
    let picks = mmr(&rel, &pair, 1.0, 4);
    assert_eq!(picks, vec![0, 1, 2, 3]);
}

#[test]
fn lambda_zero_is_pure_diversity() {
    let rel = vec![0.9, 0.85, 0.6, 0.55];
    let pair = vec![
        vec![1.0, 0.95, 0.10, 0.10],
        vec![0.95, 1.0, 0.10, 0.10],
        vec![0.10, 0.10, 1.0, 0.95],
        vec![0.10, 0.10, 0.95, 1.0],
    ];
    let picks = mmr(&rel, &pair, 0.0, 2);
    // First pick is the most-relevant on ties (all "0 max-pair" are equal,
    // so first cand wins). Second pick should be far from the first.
    assert_eq!(picks[0], 0);
    assert!(picks[1] == 2 || picks[1] == 3);
}

#[test]
fn k_caps_at_remaining() {
    let rel = vec![0.9, 0.5];
    let pair = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let picks = mmr(&rel, &pair, 0.5, 10);
    assert_eq!(picks.len(), 2);
}
