# gaphack-decompose Design Document

## Overview

**gaphack-decompose** implements iterative neighborhood-based clustering for large datasets by using BLAST to identify sequence neighborhoods, then applying target mode clustering to grow clusters incrementally.

## Philosophy

Instead of clustering all sequences simultaneously (which becomes computationally prohibitive for large datasets), decompose the problem into manageable neighborhoods:

1. **BLAST-based neighborhoods**: Find similar sequences for each target
2. **Target mode clustering**: Grow one cluster per iteration using existing target clustering logic  
3. **Iterative processing**: Continue until coverage criteria are met
4. **Post-processing validation**: Handle multi-cluster assignments and validate results

## Requirements

### Core Functionality
- **Tool name**: `gaphack-decompose` (better than `gaphack-iterate`)
- **Input**: Large FASTA file for decomposition
- **Output**: Multiple cluster files with collision detection and warnings
- **Dependencies**: NCBI BLAST+ toolkit, existing gapHACk target mode clustering

### Target Selection Strategies

#### Supervised Mode
- **Input**: Additional FASTA file containing target sequences to cover
- **Behavior**: Each target sequence attempted in separate iteration until all assigned
- **Stopping**: When all target sequences have been assigned to clusters

#### Unsupervised Mode
- **Random strategy**: Select random unassigned sequence as next target
- **Spiral/neighborhood strategy**: After each clustering, select unclustered sequence from BLAST results as next target
- **Stopping**: Based on number of clusters created OR number of sequences assigned (user configurable)

### BLAST Integration

#### Parameters (based on blast_utils.py)
- **Max hits**: 500 per target (sufficient for expected datasets)
- **Multi-threading**: Use BLAST's built-in threading (`-num_threads`)
- **Identity padding**: Pad identity percentage to ensure sufficient results
- **Hash-based mapping**: Map sequences using hash values for efficient lookup
- **Batch queries**: Support multiple queries in single BLAST invocation

#### Error Detection  
- **Diversity check**: If all/supermajority of retrieved sequences assigned to same cluster, expand BLAST search
- **Coverage validation**: Ensure neighborhood has sufficient diversity for meaningful clustering

### Multi-cluster Assignment Handling
- **Allow multi-assignment**: Sequences can initially be assigned to multiple clusters
- **Post-processing warnings**: Issue warnings for sequences assigned to multiple clusters
- **Validation pass**: Separate validation after all clustering phases complete

### Data Management

#### In-memory Index
- **Assignment tracking**: Map sequence_id → list of cluster_ids  
- **Collision detection**: Track and report multi-cluster assignments
- **Status tracking**: Track which sequences have been processed

#### File Management
- **Unique output names**: Ensure cluster file names don't collide across iterations
- **Naming convention**: `{base}.decompose_cluster_{iteration:03d}_{cluster_id:03d}.fasta`
- **Multi-assignment files**: `{base}.decompose_conflicts.fasta` for sequences in multiple clusters

## Architecture

### Core Components

```
gaphack-decompose
├── DecomposeClustering (main orchestrator)
├── BlastNeighborhoodFinder (BLAST wrapper, adapted from blast_utils.py)  
├── TargetSelectionStrategy (supervised/unsupervised selection)
├── AssignmentTracker (in-memory sequence→cluster mapping)
├── ValidationProcessor (post-processing conflict detection)
└── CLI interface (parameter handling and progress reporting)
```

### Key Classes

#### DecomposeClustering
```python
class DecomposeClustering:
    def decompose(self, input_fasta: str, targets_fasta: Optional[str], 
                  strategy: str, max_clusters: Optional[int], 
                  max_sequences: Optional[int]) -> DecomposeResults
```

#### BlastNeighborhoodFinder  
```python  
class BlastNeighborhoodFinder:
    def find_neighborhood(self, target_sequence: str, 
                         all_sequences: List[Tuple[str, str]],
                         max_hits: int = 500) -> List[str]  # sequence IDs
```

#### AssignmentTracker
```python
class AssignmentTracker:
    def assign_sequence(self, seq_id: str, cluster_id: str, iteration: int)
    def get_conflicts(self) -> Dict[str, List[str]]  # seq_id → cluster_ids
    def is_assigned(self, seq_id: str) -> bool
```

## Implementation Phases

### Phase 1: Basic Infrastructure
1. Adapt blast_utils.py for sequence-to-sequence (not OTU-based) BLAST
2. Implement DecomposeClustering orchestrator  
3. Create basic supervised mode with simple target selection
4. Integrate with existing TargetModeClustering

### Phase 2: Strategy Implementation
1. Add unsupervised random target selection
2. Implement spiral/neighborhood target selection
3. Add stopping criteria and configuration options
4. Implement assignment tracking and conflict detection  

### Phase 3: Robustness & Validation
1. Add BLAST diversity checking and expansion logic
2. Implement post-processing validation
3. Add comprehensive error handling and progress reporting
4. Performance optimization and testing

## Configuration Parameters

### CLI Parameters
```bash
gaphack-decompose input.fasta [options]

--targets FASTA           # Supervised mode: target sequences to cover
--strategy {random,spiral} # Unsupervised strategy (default: random)
--max-clusters N          # Stop after N clusters (unsupervised)  
--max-sequences N         # Stop after N sequences assigned (unsupervised)
--blast-max-hits N        # BLAST max hits per query (default: 500)
--blast-threads N         # BLAST thread count (default: auto)
--blast-evalue FLOAT      # BLAST e-value threshold (default: 1e-5)
--min-identity FLOAT      # BLAST identity threshold (default: auto-calculated)
-o OUTPUT_BASE            # Output file base path
```

### Integration with Existing Parameters
- All existing gapHACk parameters apply to target clustering: `--min-split`, `--max-lump`, `--target-percentile`, etc.

## Output Format

### Files Generated
```
{base}.decompose_cluster_001_001.fasta    # Iteration 1, cluster 1  
{base}.decompose_cluster_002_001.fasta    # Iteration 2, cluster 1
{base}.decompose_conflicts.fasta          # Multi-assigned sequences
{base}.decompose_unassigned.fasta         # Never assigned sequences  
{base}.decompose_report.json              # Summary statistics and warnings
```

### Reporting
- **Iteration summary**: Sequences processed, clusters formed, conflicts detected
- **Final statistics**: Total clusters, assignment coverage, conflict rate
- **Warnings**: Multi-assignments, low-diversity neighborhoods, expansion events

## Future Extensions

### Advanced Strategies
- **Density-based selection**: Target sequences in sparsely covered regions
- **Quality-based selection**: Prefer high-quality sequences as targets
- **Hierarchical decomposition**: Multi-level neighborhood exploration

### Performance Optimizations  
- **Persistent BLAST databases**: Cache databases across runs
- **Incremental processing**: Resume interrupted decomposition runs
- **Memory management**: Stream processing for very large datasets

## Notes

This design leverages existing gapHACk target mode clustering while adding BLAST-based neighborhood discovery. The modular architecture allows incremental development and testing of individual components.

Key design decisions:
- **Separate validation pass**: Simplifies core logic, enables comprehensive conflict reporting
- **Hash-based sequence mapping**: Efficient lookup and collision detection  
- **Pluggable target selection**: Easy to extend with new strategies
- **Consistent file naming**: Prevents collisions, enables batch processing of results