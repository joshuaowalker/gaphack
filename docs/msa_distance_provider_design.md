# MSA-Based Distance Provider Design

## Overview

This document describes the design for implementing MSA (Multiple Sequence Alignment) based distance computation throughout gaphack. The goal is to achieve maximum consistency in distance calculations by using a shared alignment space (via SPOA) rather than independent pairwise alignments.

## Motivation

### Current Problem
- **Inconsistent alignments**: Each pairwise distance uses independent alignment, leading to different gap placements for the same sequences in different contexts
- **Triangle inequality violations**: Independent alignments can produce non-metric distances
- **Redundant computation**: Same sequence pairs may be aligned multiple times in different contexts

### Proposed Solution
- **Shared alignment space**: Use SPOA to create MSA once, score all pairwise distances from the same alignment
- **Biological consistency**: All sequences in a neighborhood share alignment context
- **Performance**: SPOA once (~0.05s for 100 seqs) + fast scoring (~0.1ms/pair) is faster than many independent pairwise alignments (~1ms/pair)

## Performance Characteristics

### SPOA Scaling
Tested with 400bp sequences at varying diversity levels:

| Sequences | Diversity | Time | Throughput |
|-----------|-----------|------|------------|
| 100       | 1%        | 0.062s | 1,623 seq/s |
| 100       | 5%        | 0.080s | 1,251 seq/s |
| 100       | 10%       | 0.100s | 1,004 seq/s |
| 500       | 0%        | 0.187s | 2,668 seq/s |

### Performance Comparison
For 100 sequences with 50% density (2,500 distance queries):

| Method | SPOA | Scoring | Total |
|--------|------|---------|-------|
| **Current (LazyDistanceProvider)** | - | 2,500 × 1ms | **2.5s** |
| **Proposed (MSACachedDistanceProvider)** | 0.05s | 2,500 × 0.1ms | **0.3s** |

**Expected speedup: ~8x for dense distance queries**

## Architecture

### New Component: MSACachedDistanceProvider

```python
class MSACachedDistanceProvider(DistanceProvider):
    """Distance provider that caches MSA and computes distances on-demand.

    This provider runs SPOA once to create a multiple sequence alignment,
    then computes pairwise distances on-demand from the shared alignment space.
    Falls back to pairwise alignment if SPOA fails.
    """

    def __init__(self, sequences: List[str], headers: List[str]):
        """Initialize with sequences and create MSA.

        Args:
            sequences: List of DNA sequences
            headers: List of sequence headers (for debugging)
        """

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Get distance between two sequences using cached MSA."""

    def get_distances_from_sequence(self, idx: int, targets: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to multiple targets."""

    def ensure_distances_computed(self, indices: Set[int]) -> None:
        """No-op for MSA provider - all distances available from MSA."""

    def build_distance_matrix(self) -> np.ndarray:
        """Build full distance matrix from cached MSA."""
```

### Implementation Details

```python
class MSACachedDistanceProvider(DistanceProvider):
    def __init__(self, sequences: List[str], headers: List[str]):
        self.sequences = sequences
        self.headers = headers
        self.n = len(sequences)
        self._distance_cache = {}  # Cache computed distances

        # Run SPOA once and cache aligned sequences
        logging.debug(f"Creating MSA for {self.n} sequences using SPOA")
        aligned = run_spoa_msa(sequences)

        if aligned is None:
            # SPOA failed - fall back to pairwise alignment
            logging.warning(f"SPOA failed for {self.n} sequences, falling back to pairwise")
            self._use_pairwise = True
            self._pairwise_provider = LazyDistanceProvider(
                sequences,
                alignment_method="adjusted",
                end_skip_distance=0,
                normalize_homopolymers=True,
                handle_iupac_overlap=True,
                normalize_indels=True,
                max_repeat_motif_length=0
            )
        else:
            # SPOA succeeded - use MSA-based scoring
            self._use_pairwise = False
            self.aligned_sequences = replace_terminal_gaps(aligned)
            logging.debug(f"MSA created successfully, alignment length: {len(aligned[0])}")

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Get distance between two sequences."""
        if idx1 == idx2:
            return 0.0

        # Check cache
        cache_key = (min(idx1, idx2), max(idx1, idx2))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # Compute distance
        if self._use_pairwise:
            distance = self._pairwise_provider.get_distance(idx1, idx2)
        else:
            distance = compute_msa_distance(
                self.aligned_sequences[idx1],
                self.aligned_sequences[idx2]
            )

        # Cache and return
        self._distance_cache[cache_key] = distance
        return distance

    def get_distances_from_sequence(self, idx: int, targets: Set[int]) -> Dict[int, float]:
        """Get distances from one sequence to multiple targets."""
        if self._use_pairwise:
            return self._pairwise_provider.get_distances_from_sequence(idx, targets)

        return {target_idx: self.get_distance(idx, target_idx)
                for target_idx in targets}

    def ensure_distances_computed(self, indices: Set[int]) -> None:
        """No-op for MSA provider - all distances available from MSA."""
        pass

    def build_distance_matrix(self) -> np.ndarray:
        """Build full distance matrix from cached MSA."""
        matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = self.get_distance(i, j)
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix
```

## Use Cases and Integration Points

### 1. Classic gaphack (Already Implemented ✓)
**Location**: `cli.py:224`, `calculate_distance_matrix()`
**Status**: Already uses SPOA via `calculate_distance_matrix()`
**No changes needed**

### 2. gaphack-analyze (Already Implemented ✓)
**Location**: `analyze.py:33, 78`
**Status**: Already uses SPOA via `calculate_distance_matrix()`
**No changes needed**

### 3. BLAST Neighborhood Pruning (New Implementation)
**Location**: `decompose.py:507`, `_prune_neighborhood_by_distance()`
**Current**: Uses global `LazyDistanceProvider` with `SubsetDistanceProvider`
**Proposed**:
```python
def _prune_neighborhood_by_distance(self, ...):
    # Create MSA-based provider for neighborhood
    neighborhood_distance_provider = MSACachedDistanceProvider(
        neighborhood_sequences,
        neighborhood_headers
    )

    # Compute distances from targets to all sequences
    for seq_idx in range(len(neighborhood_sequences)):
        if seq_idx not in target_indices:
            distances_to_targets = neighborhood_distance_provider.get_distances_from_sequence(
                seq_idx, set(target_indices)
            )
            min_dist = min(distances_to_targets.values())
            # ... rest of N+N logic
```

**Benefits**:
- All neighborhood sequences share alignment context
- ~8x faster for typical neighborhood of 100-200 sequences
- Eliminates triangle inequality violations within neighborhood

### 4. Target Clustering (New Implementation)
**Location**: `target_clustering.py:298, 357, 365`
**Current**: Uses passed-in `distance_provider` (typically `SubsetDistanceProvider`)
**Proposed**: Accept `MSACachedDistanceProvider` from caller
**No changes to target_clustering.py needed** - it already works with any `DistanceProvider`

Caller changes in `decompose.py:511-523`:
```python
# Instead of SubsetDistanceProvider, use MSACachedDistanceProvider
neighborhood_distance_provider = MSACachedDistanceProvider(
    pruned_sequences,
    pruned_headers
)

# Target clustering works as-is
target_cluster_indices, remaining_indices, clustering_metrics = self.target_clustering.cluster(
    neighborhood_distance_provider, pruned_target_indices, pruned_sequences
)
```

### 5. Conflict Resolution (New Implementation)
**Location**: `cluster_refinement.py:186-203`, `apply_full_gaphack_to_scope_with_metadata()`
**Current**: Uses `ScopedDistanceProvider` wrapping global provider
**Proposed**:
```python
def apply_full_gaphack_to_scope_with_metadata(...):
    # Instead of ScopedDistanceProvider, create MSA provider for scope
    msa_provider = MSACachedDistanceProvider(scope_sequences, scope_headers)

    # Build distance matrix from MSA
    distance_matrix = msa_provider.build_distance_matrix()

    # Apply full gapHACk clustering (unchanged)
    clusterer = GapOptimizedClustering(...)
    final_clusters, singletons, metadata = clusterer.cluster(distance_matrix)
```

**Benefits**:
- Conflict scopes typically 50-300 sequences - perfect size for SPOA
- Consistent alignment across all sequences in conflict component
- Eliminates alignment inconsistencies that caused conflicts

### 6. Close Cluster Refinement (New Implementation)
**Location**: `cluster_refinement.py:186-203` (same function as conflicts)
**Status**: Same changes as conflict resolution
**No additional work needed**

### 7. Cluster Graph Construction (New Implementation)
**Location**: `cluster_graph.py:249`, medoid distance calculation
**Current**: Uses global `LazyDistanceProvider.get_distance()`
**Proposed**:
```python
def _build_graph_from_blast(...):
    # For each cluster, collect its BLAST neighbors
    # Create MSA for cluster + its neighbors
    # Compute medoid distances from MSA

    # Pseudocode:
    for cluster_id in clusters:
        neighbor_cluster_ids = get_blast_neighbors(cluster_id)
        all_cluster_ids = [cluster_id] + neighbor_cluster_ids

        # Collect all sequences from these clusters
        scope_sequences = []
        scope_headers = []
        for cid in all_cluster_ids:
            scope_sequences.extend(get_cluster_sequences(cid))
            scope_headers.extend(get_cluster_headers(cid))

        # Create MSA provider for this scope
        msa_provider = MSACachedDistanceProvider(scope_sequences, scope_headers)

        # Compute medoid distances
        for neighbor_id in neighbor_cluster_ids:
            distance = msa_provider.get_distance(
                medoid_idx[cluster_id],
                medoid_idx[neighbor_id]
            )
```

**Note**: This requires more significant refactoring - may be lower priority

### 8. Triangle Inequality Filtering (Potentially Remove)
**Location**: `triangle_filtering.py`
**Current**: Detects and filters alignment failures
**Proposed**: May become unnecessary with MSA-based approach
**Action**: Monitor and potentially deprecate after validation

## Experimental Parameters

All distance computations (both MSA-based and pairwise fallback) use the experimental parameters:

```python
end_skip_distance = 0
normalize_homopolymers = True
handle_iupac_overlap = True
normalize_indels = True
max_repeat_motif_length = 0
ambiguity_penalty = 0.0  # Currently disabled
```

These are enforced in both `compute_pairwise_distance()` and `compute_msa_distance()`.

## Implementation Plan

### Phase 1: Core Infrastructure (High Priority)
1. ✅ Implement helper functions (already done):
   - `run_spoa_msa()`
   - `replace_terminal_gaps()`
   - `filter_msa_positions()`
   - `compute_msa_distance()`

2. **Implement `MSACachedDistanceProvider`** (this document):
   - Create new class in `lazy_distances.py`
   - Implement all `DistanceProvider` interface methods
   - Add comprehensive tests

3. **Update `calculate_distance_matrix()`** (already done ✓):
   - Use SPOA for full matrix computation
   - Graceful fallback to pairwise

### Phase 2: Neighborhood Operations (High Priority)
4. **Integrate into neighborhood pruning**:
   - Modify `_prune_neighborhood_by_distance()` to use `MSACachedDistanceProvider`
   - Update distance computation to use MSA provider

5. **Integrate into target clustering**:
   - Update caller in `decompose.py` to create `MSACachedDistanceProvider`
   - Target clustering code requires no changes

### Phase 3: Refinement Operations (Medium Priority)
6. **Integrate into conflict resolution**:
   - Modify `apply_full_gaphack_to_scope_with_metadata()`
   - Use MSA provider instead of `ScopedDistanceProvider`

7. **Integrate into close cluster refinement**:
   - Same changes as conflict resolution
   - Already covered by same function

### Phase 4: Advanced Features (Lower Priority)
8. **Cluster graph construction**:
   - Refactor to use MSA providers for medoid neighborhoods
   - More complex - requires careful design

9. **Triangle inequality filtering**:
   - Monitor effectiveness with MSA approach
   - Consider deprecation if no longer needed

### Phase 5: Validation and Optimization
10. **Performance testing**:
    - Benchmark against current implementation
    - Validate ~8x speedup expectations

11. **Quality validation**:
    - Test on Russula dataset
    - Verify clustering quality metrics maintained or improved
    - Confirm reduction in triangle inequality violations

12. **Documentation**:
    - Update CLAUDE.md with new architecture
    - Document when to use MSA vs pairwise providers

## Testing Strategy

### Unit Tests
- Test `MSACachedDistanceProvider` with small sequence sets
- Test fallback to pairwise when SPOA fails
- Test cache correctness
- Test all interface methods

### Integration Tests
- Test neighborhood pruning with MSA provider
- Test target clustering with MSA provider
- Test conflict resolution with MSA provider

### Performance Tests
- Benchmark SPOA scaling (already done)
- Benchmark end-to-end performance on test datasets
- Compare memory usage

### Quality Tests
- Run full Russula dataset (1,429 sequences)
- Compare clustering quality metrics (ARI, homogeneity, completeness)
- Verify no degradation in biological coherence

## Risks and Mitigations

### Risk 1: SPOA Failure on Diverse Sequences
**Mitigation**: Graceful fallback to `LazyDistanceProvider`
**Status**: Already implemented in design

### Risk 2: Memory Usage for Large MSAs
**Concern**: Storing full MSA for 1000 sequences
**Mitigation**:
- BLAST neighborhoods rarely exceed 1000 sequences
- N+N pruning reduces to ~100-200 sequences
- MSA strings are lightweight compared to distance cache

### Risk 3: SPOA Not Available
**Mitigation**: Fallback already implemented
**Future**: Consider bundling SPOA or using alternative MSA tools

### Risk 4: Lower Quality Alignments
**Concern**: SPOA may produce worse alignments than pairwise for some pairs
**Mitigation**:
- BLAST pre-filters for similar sequences
- Extensive testing on real data
- Can revert if quality degrades

## Success Criteria

1. **Performance**: ≥5x speedup for neighborhood operations (target: 8x)
2. **Quality**: Clustering metrics (ARI, homogeneity, completeness) maintained or improved
3. **Consistency**: Reduction in triangle inequality violations
4. **Reliability**: Graceful fallback works in all failure cases
5. **Test Coverage**: All new code covered by unit and integration tests

## Future Enhancements

### Dynamic MSA Provider Selection
Automatically choose between MSA and pairwise based on:
- Sequence count (MSA better for 10+ sequences)
- Expected similarity (BLAST neighborhoods → MSA)
- Diversity estimate (high diversity → pairwise)

### Incremental MSA Updates
For iterative algorithms, support adding sequences to existing MSA rather than recomputing from scratch.

### Alternative MSA Tools
Evaluate other MSA tools (MAFFT, Clustal Omega) for different use cases:
- MAFFT: Better for diverse sequences
- Clustal Omega: Better for large sequence sets
- SPOA: Fast for similar sequences (current choice)

### GPU Acceleration
Investigate GPU-accelerated MSA tools for very large neighborhoods (1000+ sequences).

## References

- SPOA: https://github.com/rvaser/spoa
- Current implementation: `utils.py:215-439` (MSA helpers and `calculate_distance_matrix()`)
- Distance provider interface: `lazy_distances.py:12-28`
- Example usage: `msalign.py:268-304` (inspiration for MSA-based scoring)
