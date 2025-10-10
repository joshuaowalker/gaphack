# Global Gap Metric Design

## Objective

Implement a **global gap metric** to track convergence progress during iterative refinement (Pass 2). This metric provides a direct indicator of progress toward our optimization objective: maximizing the barcode gap between intra-cluster and inter-cluster distances.

Unlike AMI (which measures clustering stability), the global gap metric measures **clustering quality** in terms of barcode separation.

## Key Design Decisions

### 1. Computation Timing: Start of Iteration

**Decision**: Compute global gap metrics at the **start** of each iteration, using the proximity graph that was just built for `current_clusters`.

**Rationale**:
- Zero additional cost (graph already exists)
- Avoids doubling graph construction time (~10-20 seconds)
- Metrics describe the "input" state that refinement is trying to improve
- Can track improvement by comparing metrics iteration-to-iteration

**Trade-off**: Metrics are "lagging" by one iteration (describe pre-refinement state), but this is acceptable for convergence tracking.

### 2. K-Nearest Neighbors for Inter-Cluster Distances

**Decision**: Use `K` nearest neighbor clusters (configurable constant) for inter-cluster distance calculation.

**Default**: `K=3`

**Rationale**:
- **K=1 is conservative**: Focuses on hardest case (closest neighbor), directly aligned with separation goal
- **K>1 hedges against medoid limitations**: Medoid-based proximity may underestimate true cluster overlap at edges
- **K=3 balances robustness vs. cost**: 3× the inter-cluster distances of K=1, still very manageable

**Implementation**:
```python
# In cluster_refinement.py
GLOBAL_GAP_K_NEIGHBORS = 3  # Tunable constant
```

### 3. Singleton Handling

**Decision**: For singleton clusters, use intra-cluster distance = `[0]` (i.e., perfect internal cohesion).

**Rationale**:
- Conceptually cleaner: singleton has perfect cohesion by definition
- Gap then purely measures separation from neighbors
- Alternative (using `min_split`) would artificially deflate singleton gaps by subtracting a positive value

**Implementation note**: When computing intra-cluster p95, a singleton returns 0.

### 4. Per-Cluster Gap Calculation

For each cluster C:

1. **Intra-cluster distances**:
   - All pairwise distances within C: `{d(s_i, s_j) for all i,j in C where i < j}`
   - Singleton case: empty list → percentile returns 0

2. **Inter-cluster distances**:
   - Get K nearest neighbor clusters from proximity graph
   - Pool all pairwise distances: `{d(s_i, s_j) for s_i in C, s_j in neighbor_cluster, for all K neighbors}`

3. **Cluster gap**:
   ```python
   intra_upper = percentile(intra_distances, target_percentile)  # e.g., p95
   inter_lower = percentile(inter_distances, 100 - target_percentile)  # e.g., p5
   gap_C = inter_lower - intra_upper
   ```

### 5. Global Metrics

Aggregate per-cluster gaps into four global metrics:

```python
global_gap_metrics = {
    # Mean gap across all clusters (unweighted)
    'mean_gap': mean(gap_C for all C),

    # Mean gap weighted by cluster size (sequence-centric view)
    'weighted_gap': sum(gap_C * len(C) for all C) / total_sequences,

    # Fraction of clusters with positive gap
    'gap_coverage': len([C for C if gap_C > 0]) / len(all_clusters),

    # Fraction of sequences in positive-gap clusters
    'gap_coverage_sequences': sum(len(C) for C if gap_C > 0) / total_sequences
}
```

**Rationale for multiple metrics**:
- `mean_gap`: Simple average, equal weight per cluster
- `weighted_gap`: Prioritizes large clusters (most sequences)
- `gap_coverage`: Robust to outliers, clear interpretation
- `gap_coverage_sequences`: Combines coverage concept with sequence weighting

## Computational Cost Analysis

**Typical Pass 2 iteration** (1,500 clusters, avg size 2-3 sequences):

1. **Intra-cluster distances**:
   - Small clusters (n=2-3): ~3 distances per cluster
   - Total: ~4,500 distances

2. **Inter-cluster distances** (K=3):
   - ~3 sequences/cluster × 3 sequences/neighbor × 3 neighbors = ~27 distances per cluster
   - Total: ~40,500 distances

3. **Grand total**: ~45,000 distance computations

**Scoped MSA approach**:
- Each cluster creates MSA for itself + K=3 neighbors
- Average scope size: ~4 clusters × 2.5 sequences = ~10 sequences per MSA
- 1,500 clusters × ~10 sequence MSA alignments
- **Estimated overhead**: ~5-10 seconds per iteration (reasonable for convergence tracking)

**Why scoped MSA instead of global distance provider**:
- Global MSA across all sequences would be prohibitively expensive
- Scoped MSA provides better alignments for partial marker coverage
- Each cluster's gap requires only local context (cluster + neighbors)

## Expected Behavior

### Healthy Convergence Pattern
```
Iter 1: mean_gap=0.015, coverage=45%
Iter 2: mean_gap=0.019, coverage=58%
Iter 3: mean_gap=0.023, coverage=67%
Iter 4: mean_gap=0.024, coverage=69%  # Plateau
Iter 5: mean_gap=0.024, coverage=70%  # Diminishing returns
```

**Interpretation**: Gap increasing and coverage increasing → refinement is improving separation.

### Stuck Pattern
```
Iter 5: mean_gap=0.021, coverage=65%
Iter 6: mean_gap=0.020, coverage=66%  # Slight fluctuation
Iter 7: mean_gap=0.021, coverage=65%  # Oscillating
Iter 8: mean_gap=0.020, coverage=66%  # No progress
```

**Interpretation**: Metrics oscillating or flat → refinement is churning without improvement.

### Convergence Indicators

**Strong signal to stop**:
- `gap_coverage` > 90% and stable for 2+ iterations
- `mean_gap` not increasing for 3+ iterations
- AMI ≈ 1.0 (no structural changes)

**Continue refining**:
- `gap_coverage` < 80%
- `mean_gap` still increasing
- AMI < 0.99 (significant changes still occurring)

## Logging Format

```
Pass 2 iteration 5: 1456 clusters
Global gap (pre-refinement): mean=0.0234, weighted=0.0241, coverage=67.3% (2891/4296 seqs)
... [refinement operations] ...
Pass 2 Iter 5 Summary:
  Clusters: 1446 -> 1456 (+10)
  AMI: 0.984
  [existing statistics]
```

**Key elements**:
- Logged at start of iteration (pre-refinement)
- Shows both mean and weighted gap
- Shows coverage as percentage and absolute counts
- Appears before refinement summary for clear temporal ordering

## Alternative Approaches Considered

### 1. Compute at End of Iteration
Build second K-NN graph for `next_clusters` to get immediate post-refinement metrics.

**Rejected because**: Doubles graph construction cost (~10-20 seconds), too expensive for every iteration.

### 2. Only Nearest Cluster (K=1)
Use only the single nearest cluster for inter-cluster distances.

**Considered but enhanced**: K=1 is most conservative, but K=3 hedges against medoid-based distance underestimating true overlap. Made K configurable.

### 3. Multi-Percentile Global Gap
Compute global gap at multiple percentiles (p50, p75, p90, p95) and combine.

**Rejected for now**: Requires collecting ALL intra/inter distances globally (expensive), or loses per-cluster granularity. Current approach provides per-cluster gaps which are more diagnostic.

### 4. Sample-Based Approximation
For large clusters, sample N sequences instead of computing all O(n²) pairs.

**Deferred**: Current cluster sizes (2-3 sequences) don't warrant sampling. Can revisit if large clusters become common.

## Implementation Location

**Primary file**: `gaphack/cluster_refinement.py`

**New function**: `compute_global_gap_metrics()`
- Input: clusters dict, proximity graph, sequences, headers, target_percentile
- Output: dict with 4 global metrics
- **Key implementation detail**: Creates scoped MSA-based distance provider for each cluster + K neighbors
  - Avoids cost-prohibitive global MSA across all sequences
  - Provides better alignments for sequences covering different marker regions

**Integration point**: `pass2_iterative_merge()`
- Called immediately after building proximity graph
- Before seed prioritization and refinement loop
- Stored in iteration_stats for checkpointing

## Testing Strategy

1. **Unit test**: `compute_global_gap_metrics()` with synthetic clusters
   - Verify singleton handling
   - Verify coverage calculations
   - Verify weighted vs unweighted mean

2. **Integration test**: Run small refinement iteration
   - Verify metrics improve or stabilize
   - Verify no performance degradation

3. **Real data validation**: Run on Russula subset
   - Verify metrics correlate with known convergence points
   - Verify computational overhead is negligible

## Future Enhancements

1. **Adaptive K**: Could adjust K based on cluster density (dense regions → lower K)
2. **Gap distribution tracking**: Track histogram of per-cluster gaps to identify problematic clusters
3. **Early stopping criterion**: Automatically stop when metrics plateau
4. **Per-cluster gap reporting**: Output clusters with poor gaps for manual inspection

---

*Document created: 2025-10-10*
*Status: Design approved, ready for implementation*
