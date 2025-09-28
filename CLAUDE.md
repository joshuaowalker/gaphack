# Development Notes for Claude

This file contains development notes and future considerations for the gapHACk project.

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

(No current technical debt items)

### Performance Considerations

#### Distance Provider Architecture
- Global distance provider caches all computed distances
- Subset providers map local indices to global cache
- Prevents redundant distance calculations across iterations
- Critical for performance with 100K+ sequences

#### BLAST Memory Pool
- Stores full BLAST neighborhoods before pruning
- Enables "nearby" target selection for better clustering coherence
- Reduces random jumps across sequence space
- Located in: `decompose.py::BlastResultMemory`

## Testing Notes

- Gap-based greedy with gap heuristic is functionally equivalent to beam search with beam_width=1
- Current approach finds optimal gap within the greedy search path (not globally optimal)
- Runs to completion rather than early termination, ensuring full exploration of valid merge space
- N+N pruning effectiveness can be tested by comparing gaps with different context ratios
- Context expansion effectiveness validated by tracking gap improvement across iterations