# Gap Optimization Objective: Discussion Summary

## Current Approach

The current gapHACk algorithm optimizes for a **single global percentile gap**:
- **Intra-cluster upper bound**: p95 (95th percentile) of all intra-cluster distances
- **Inter-cluster lower bound**: p5 (5th percentile) of all inter-cluster distances
- **Gap**: `inter_lower - intra_upper`

At each iteration, evaluates all valid merge pairs and selects the merge that produces the **largest gap**.

## Identified Limitations

1. **Edge-focused**: Only examines distribution edges, not the mass of distributions
2. **Insensitive to intermediate effects**: Large clusters may benefit from merges that improve separation at lower percentiles, but these aren't captured
3. **Outlier vulnerability**: While percentiles help, a few misplaced sequences in large clusters may not affect the p95 boundary
4. **Binary view**: Either a gap exists or it doesn't—doesn't capture *quality* of separation

## Proposed Alternatives

### Alternative 1: Per-Sequence Gap (Original Proposal)
For each sequence, calculate its personal barcode gap:
- Intra: p95 of distances to same-cluster sequences
- Inter: p5 of distances to all other-cluster sequences
- Per-sequence gap: `inter_lower - intra_upper`

**Global objective**: Mean of all per-sequence gaps

**Advantages**:
- Every sequence "votes" on clustering quality
- Detects misplaced sequences (poor personal gap)
- Rewards homogeneous clusters
- Conceptually aligned with silhouette coefficient (Rousseeuw 1987)

**Disadvantages**:
- Singleton handling unclear (no intra-cluster distances)
- Small cluster instability (few distances → noisy percentiles)
- **Computational cost**: 100-1000× more expensive (requires per-sequence distance lists)
- Incremental calculation complex (can't easily use global caches)

### Alternative 2: Multi-Percentile Global Gap (Recommended)
Calculate gaps at multiple percentiles and combine with weights:
```
gap = w1*gap_p50 + w2*gap_p75 + w3*gap_p90 + w4*gap_p95
```
Default weights: [0.2, 0.3, 0.3, 0.2]

**Advantages**:
- Sensitive to distribution mass, not just edges
- Still uses global distributions → easy incremental calculation
- Tunable via weights (optimize empirically)
- Only ~4× computational overhead (acceptable)

**Disadvantages**:
- Requires weight tuning
- Less intuitive than per-sequence approach

### Alternative 3: Area Under Gap Curve
Calculate gap at many percentiles (10, 20, ..., 90) and integrate:
```
gap = sum(gap_pX for X in [10, 20, 30, ..., 90]) / 9
```

**Advantages**:
- Comprehensive measure across entire distribution
- No arbitrary weight tuning
- Related to Kolmogorov-Smirnov statistic

**Disadvantages**:
- More percentile calculations (~9× overhead)
- May over-weight tails

### Alternative 4: Silhouette-Inspired Gap
Adapt silhouette coefficient to gap framework:
```
for each sequence i:
    a_i = p95 of distances to same-cluster sequences
    b_i = p5 of distances to nearest other cluster  # Not all clusters
    gap_i = b_i - a_i
global_gap = mean(gap_i)
```

**Advantages**:
- Well-studied in clustering literature
- More focused than full per-sequence (nearest cluster only)

**Disadvantages**:
- Still requires per-sequence calculation
- "Nearest other cluster" adds complexity

## Recommended Implementation Strategy

### Phase 1: Add Metrics (1-2 hours)
Implement Alternative 2 (multi-percentile) and Alternative 1 (per-sequence) as **tracking metrics only**:
- Don't use for optimization yet
- Track alongside current objective in `gap_history`
- Compare correlation with ground truth quality (ARI, homogeneity)

### Phase 2: Experimental Comparison (Few days)
Run experiments on Russula dataset comparing:
- Current single-percentile objective
- Multi-percentile objective (various weight schemes)
- Per-sequence objective (if Phase 1 shows promise)

Metrics to compare:
- Final clustering quality (ARI, homogeneity, completeness)
- Convergence behavior (iterations to stability)
- Intermediate clustering quality during optimization

### Phase 3: Production Implementation (If warranted)
If experiments show clear improvement:
- Add `gap_objective` parameter to `GapOptimizedClustering`
- Optimize incremental calculation for chosen objective
- Update tests and documentation

## Key References

**Silhouette Coefficient**:
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.
- DOI: 10.1016/0377-0427(87)90125-7

**Key insight**: Per-sequence gap is conceptually similar to silhouette coefficient, but uses percentile-based distances rather than means (more robust to outliers in DNA barcoding).

## Decision

**Prioritize convergence/seed prioritization work first**, then return to optimization objectives with:
1. Stable baseline for comparison
2. Faster experimental iteration
3. Better understanding of system behavior

---

*Last updated: 2025-10-09*
