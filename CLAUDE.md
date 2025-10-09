# Development Notes for Claude

This file contains development notes and future considerations for the gapHACk project.

## Quick Reference

### Architecture Overview
- **Distance Provider Architecture**: Three-tier system (global cache, scoped providers, precomputed) - see [Performance Considerations](#performance-considerations)
- **Cluster Graph**: BLAST K-NN graph for O(K) proximity queries - see [Cluster Graph Infrastructure](#cluster-graph-infrastructure)
- **Conflict Resolution**: AssignmentTracker + DFS component analysis - see [Conflict Resolution Architecture](#conflict-resolution-architecture)
- **Result Tracking**: Comprehensive reproducibility via DecomposeResults - see [Result Tracking and Reproducibility](#result-tracking-and-reproducibility)

### Key Files by Functionality
- **Core clustering**: `core.py` (main algorithm + DistanceCache)
- **Target mode**: `target_clustering.py`
- **Decompose orchestration**: `decompose.py` (main DecomposeClustering class, AssignmentTracker)
- **Target selection**: `target_selection.py` (TargetSelector, NearbyTargetSelector, BlastResultMemory)
- **Resume and finalization**: `resume.py` (resume_decompose, finalize_decompose, refinement stages)
- **Post-processing**: `cluster_refinement.py` (conflict resolution, close cluster refinement)
- **Distance computation**: `lazy_distances.py` (LazyDistanceProvider, PrecomputedDistanceProvider)
- **Scoped operations**: `scoped_distances.py` (ScopedDistanceProvider for refinement)
- **Cluster proximity**: `cluster_graph.py` (ClusterGraph with BLAST K-NN)
- **BLAST integration**: `blast_neighborhood.py`

### Testing
- **317 passing tests**, 18,958 lines of test code
- **Real biological data**: 1,429 Russula sequences with ground truth
- **Performance baselines**: See [Performance Baselines](#performance-baselines)
- **Quality metrics**: ARI >0.85, Homogeneity >0.90, Completeness >0.85
- **Documentation**: `tests/PHASE4_SUMMARY.md`

## Algorithm Development History

### Gap-Optimized Clustering Evolution

The gap-optimized clustering algorithm has evolved through several iterations:

1. **Original Greedy (Distance-based)**: Used inter-cluster distance as heuristic, calculated gaps only for evaluation
2. **Beam Search Implementation**: Used gap as heuristic with configurable beam width to explore multiple search paths
3. **Current Gap-based Greedy**: Uses gap as heuristic in a greedy search, running to completion and tracking the best gap encountered (removed early termination and min_gap_size parameter)

### Future Algorithm Considerations

**Beam Search Re-implementation**:
The beam search approach was removed in favor of the simpler gap-based greedy algorithm. However, beam search could be valuable for future exploration if we need:
- Better solution quality (beam search can find gaps ~10% better than distance-based greedy)
- Exploration of multiple search paths simultaneously
- Analysis of search space characteristics

Key implementation details for future reference:
- Beam search maintained top-k partitions at each level based on gap scores
- Used partition caching to avoid recalculating the same configurations
- Explored 35-40% fewer partitions than exhaustive search while finding equivalent solutions
- Default beam width of 10 provided good balance of quality vs. performance

The beam search implementation can be found in git history if needed for future algorithm research.

## gaphack-decompose Implementation Notes

### Context Requirements for Gap Calculation

The barcode gap calculation fundamentally requires both intra-cluster and inter-cluster distances. Without sufficient inter-cluster context, the algorithm cannot determine if a gap exists. This manifests in two key areas:

#### N+N Neighborhood Pruning
- Takes N sequences within max_lump (potential cluster members)
- Takes N additional sequences beyond max_lump (inter-cluster context)
- This 1:1 ratio ensures balanced representation for percentile calculations
- Located in: `decompose.py::_prune_neighborhood_by_distance()`

#### Iterative Context Expansion in Refinement
- When refining close clusters, initial scope often lacks inter-cluster context
- Clusters within max_lump distance would all merge without external context
- Solution: iteratively add clusters beyond threshold until positive gap achieved
- Located in: `cluster_refinement.py::expand_context_for_gap_optimization()`

### Technical Debt and Known Issues

#### Target Clustering Linkage Consistency
- `_find_closest_to_target()` hardcodes 95th percentile linkage instead of using `self.target_percentile`
- Should be updated to use the configured percentile for consistency
- Location: `target_clustering.py:309` - change `np.percentile(filtered_distances, 95)` to `np.percentile(filtered_distances, self.target_percentile)`
- Current behavior works correctly with default 95th percentile but inconsistent with other percentile settings

### Performance Considerations

#### Distance Provider Architecture
The codebase implements a three-tier distance computation system for efficient large-scale clustering:

**1. Global Distance Provider** (`lazy_distances.py::LazyDistanceProvider`)
- Computes distances on-demand using configured alignment parameters
- Maintains global cache (`_distance_cache`) of all computed distances
- Tracks unique computations to monitor cache effectiveness
- Prevents redundant distance calculations across entire decompose run

**2. Scoped Distance Provider** (`scoped_distances.py::ScopedDistanceProvider`)
- Maps local indices within refinement scopes to global indices
- Enables classic gapHACk to work on sequence subsets while reusing global cache
- Maintains bidirectional mappings: `scope_to_global` and `global_to_scope`
- Includes local cache for frequently accessed computations within scope
- Critical for conflict resolution and close cluster refinement performance

**3. Precomputed Distance Provider** (`lazy_distances.py::PrecomputedDistanceProvider`)
- Wraps full precomputed distance matrices for classic gaphack usage
- No-op for `ensure_distances_computed()` since all distances exist
- Used by core algorithm and when full matrix is available

This architecture is critical for performance with 100K+ sequences, as it:
- Avoids redundant distance calculations across all refinement operations
- Enables efficient subset operations without recomputing distances
- Supports incremental clustering with persistent caching

#### Distance Cache in Core Algorithm
The multiprocessing implementation uses a specialized `DistanceCache` class (`core.py:171`):
- Maintains persistent worker caches across parallel operations
- Uses offset-based coordinate mapping for efficient workload distribution
- Enables ~4x speedup with 8 cores (200→800 steps/second)
- Reduces initialization overhead by reusing worker state

#### BLAST Memory Pool
**Purpose**: Stores full BLAST neighborhoods before pruning (`decompose.py:151::BlastResultMemory`)
- Enables "nearby" target selection for better clustering coherence
- Reduces random jumps across sequence space during iterative clustering
- Maintains biological continuity by selecting next target from existing neighborhoods
- Implements spiral target selector for systematic coverage when operating without explicit targets

**Target Selection Strategy**:
- Without explicit targets: selects from BLAST memory pool based on proximity to existing clusters
- Prevents random sampling that could fragment biologically related groups
- Ensures iterative clustering maintains spatial coherence in sequence space

## Post-Processing and Refinement Implementation

### Conflict Resolution Architecture
**Purpose**: Ensures mutually exclusive and collectively exhaustive (MECE) clustering after iterative decompose

**Key Components**:
- `AssignmentTracker` (`decompose.py:65-110`): Tracks sequence assignments across iterations
  - Maintains `assignments` dict: `seq_id -> [(cluster_id, iteration), ...]`
  - Detects conflicts when sequences assigned to multiple clusters
  - Separates single-assignment sequences from conflicts

- `find_conflict_components()` (`cluster_refinement.py:69-100`): Groups conflicts using graph analysis
  - Builds cluster adjacency graph where edges represent shared sequences
  - Uses DFS to find connected components of conflicted clusters
  - Each component is resolved independently with full gapHACk

**Refinement Configuration** (`cluster_refinement.py:29-58`):
- `max_full_gaphack_size` (300): Maximum sequences for full gapHACk refinement
- `preferred_scope_size` (250): Target scope size for optimal performance
- `expansion_size_buffer` (50): Reserve capacity for context expansion
- `conflict_expansion_threshold`: Distance threshold for conflict scope expansion
- `close_cluster_expansion_threshold`: Distance threshold for close cluster refinement
- `incremental_search_distance`: Search distance for incremental updates (future use)
- `max_closest_clusters` (5): Limits for incremental refinement (future use)

### Result Tracking and Reproducibility
The `DecomposeResults` dataclass (`decompose.py:45-63`) provides comprehensive tracking:

**Core Results**:
- `clusters`: Non-conflicted sequences only (clean output)
- `all_clusters`: All sequences including conflicts (for debugging)
- `conflicts`: Sequences assigned to multiple clusters
- `unassigned`: Sequences never processed

**Reproducibility Tracking**:
- `processing_stages`: List of `ProcessingStageInfo` objects tracking each refinement stage
  - `clusters_before` and `clusters_after`: State before/after each stage
  - `components_processed`: Details about each component refined
  - `summary_stats`: Aggregate statistics (counts, changes)
- `active_to_final_mapping`: Maps internal cluster IDs to final output IDs
- `command_line`: Full command used to run decompose
- `start_time`: ISO timestamp of run start

**Verification**:
- `verification_results`: Comprehensive conflict verification after all processing
- Ensures MECE property maintained throughout pipeline

### Cluster Graph Infrastructure
**Purpose**: Efficient proximity queries for finding nearby clusters during refinement (`cluster_graph.py`)

**ClusterGraph Implementation**:
- **BLAST-based K-NN graph**: Uses BLAST to find K nearest neighbors for each cluster medoid
- **Complexity**: O(K) rather than O(C) for proximity queries (C = total clusters)
- **Medoid caching**: Computes and caches cluster medoids (sequence with minimum total distance to others)
- **Dynamic updates**: Graph can be updated as clusters merge or split during refinement

**Key Parameters**:
- `k_neighbors` (default: 20): Number of nearest neighbors to maintain per cluster
- `blast_evalue` (default: 1e-5): BLAST e-value threshold
- `blast_identity` (default: 90.0): Minimum BLAST identity percentage

**Usage in Post-Processing**:
- Finds clusters within distance threshold for close cluster refinement
- Enables efficient "expand context" operations by quickly finding nearby clusters
- Scalable to 100K+ sequences with thousands of clusters

## Testing Infrastructure

### Comprehensive Test Suite
The project includes 317 passing tests with 18,958 lines of test code:

**Test Organization** (custom pytest markers):
- `@pytest.mark.integration`: End-to-end pipeline tests
- `@pytest.mark.performance`: Performance regression tests
- `@pytest.mark.memory`: Memory usage and scaling tests
- `@pytest.mark.scalability`: Algorithm complexity validation
- `@pytest.mark.quality`: Biological relevance and quality metrics

### Real Biological Validation
**Russula Dataset**: 1,429 fungal ITS sequences with 143 ground truth groups
- Power-law size distribution (largest group: 105 sequences)
- Taxonomic and geographic annotations for coherence analysis
- Located in: `examples/data/Russula_INxx.fasta`

**Test Subsets** (for faster iteration):
- `tests/test_data/russula_50.fasta` (50 sequences)
- `tests/test_data/russula_100.fasta` (100 sequences)
- `tests/test_data/russula_200.fasta` (200 sequences)
- `tests/test_data/russula_300.fasta` (300 sequences)
- `tests/test_data/russula_500.fasta` (500 sequences)

### Performance Baselines
**Established Thresholds** (for regression detection):
- 100 sequences: <60 seconds
- 300 sequences: <300 seconds (5 minutes)
- 500 sequences: <600 seconds (10 minutes)
- Full dataset (1,429): <600 seconds with optimizations
- Memory usage: <4GB for full dataset

### Quality Metrics
**Empirically Validated Thresholds**:
- **ARI (Adjusted Rand Index)**: >0.85 (empirical: 0.948)
- **Homogeneity**: >0.90 (empirical: 0.972)
- **Completeness**: >0.85 (empirical: 0.969)
- **Taxonomic coherence**: Mean purity >85%
- **Geographic signal**: 30-90% (balanced expectation)

**See**: `tests/PHASE4_SUMMARY.md` for comprehensive testing documentation

### Testing Best Practices
- Gap-based greedy with gap heuristic is functionally equivalent to beam search with beam_width=1
- Current approach finds optimal gap within the greedy search path (not globally optimal)
- Runs to completion rather than early termination, ensuring full exploration of valid merge space
- N+N pruning effectiveness can be tested by comparing gaps with different context ratios
- Context expansion effectiveness validated by tracking gap improvement across iterations

## Future Work and Experimental Features

### Incremental Clustering (Partial Implementation)
The refinement system includes infrastructure for incremental/streaming clustering:
- `incremental_search_distance` parameter in `RefinementConfig`
- `max_closest_clusters` limit for local refinement scope
- Designed for scenarios where new sequences are added to existing clustering

**Current Status**: Infrastructure present but not fully implemented
**Use Case**: Online clustering where sequences arrive over time
**Implementation Location**: `cluster_refinement.py` (configuration only)

**Design Considerations**:
- New sequences can be assigned to existing clusters via proximity search
- Only nearby clusters need reconsideration (O(K) via ClusterGraph)
- Maintains global clustering quality while processing incrementally
- Would require additional logic for determining when full refinement is needed

### Algorithm Research Opportunities

**Beam Search Re-exploration**:
- Current greedy approach finds optimal gap within its path (not globally optimal)
- Beam search implementation exists in git history
- Could provide ~10% better gaps with beam width 10
- Trade-off: 2-3x slower but explores multiple paths
- Consider if: dataset has complex taxonomy requiring better solution quality

**Alternative Heuristics**:
- Current: gap-based heuristic (maximize barcode gap at each step)
- Alternative 1: Mixed heuristic (distance + gap weighted combination)
- Alternative 2: Lookahead heuristic (consider next N merges before deciding)
- Alternative 3: Hierarchical heuristic (optimize different percentiles at different levels)

**Refinement Scope Optimization**:
- Current: Fixed thresholds for scope expansion
- Alternative: Adaptive thresholds based on cluster density
- Could reduce over-refinement in sparse regions
- Could increase refinement in dense regions (subspecies complexes)

### Known Limitations and Improvement Opportunities

**Current Constraints**:
1. **Scope size limits**: Max 300 sequences for full gapHACk refinement
   - Larger scopes could improve quality but increase runtime
   - Consider hierarchical refinement for larger scopes

2. **Fixed K for K-NN graph**: Currently k=20 for all datasets
   - Dense datasets might benefit from higher K
   - Sparse datasets might benefit from lower K
   - Could adapt K based on dataset characteristics

3. **Target percentile consistency**: Hardcoded 95th in `target_clustering.py:309`
   - Should respect `target_percentile` parameter
   - Simple fix but affects behavior with non-default percentiles

### Performance Optimization Opportunities

**Potential Improvements**:
1. **Parallel BLAST operations**: Currently sequential BLAST calls
   - Could parallelize neighborhood discovery across multiple targets
   - Would improve decompose iteration speed

2. **Incremental medoid updates**: Currently recomputes all medoids
   - After small refinements, most medoids unchanged
   - Could track which clusters changed and update selectively

3. **Approximate nearest neighbor search**: Currently exact BLAST K-NN
   - LSH or other ANN methods could speed up very large datasets
   - Trade-off: approximation vs. speed

4. **GPU acceleration**: Distance calculations are embarrassingly parallel
   - Could offload to GPU for 10-100x speedup on large matrices
   - Useful for datasets with 10K+ sequences in single refinement scope