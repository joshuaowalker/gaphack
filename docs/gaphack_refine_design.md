# gaphack-refine Design Document

**Version**: 1.0
**Date**: 2025-01-05
**Status**: Implementation Ready

## Overview

`gaphack-refine` is a standalone tool for applying conflict resolution and close cluster refinement to existing cluster assignments. It enables experimentation with different initial clustering algorithms (gaphack-decompose, vsearch, cd-hit, etc.) followed by standardized refinement using the gapHACk framework.

## Design Principles

1. **Modularity**: Decouples initial clustering from refinement stages
2. **Compatibility**: Works with output from any clustering tool that produces FASTA files
3. **Simplicity**: No state management or checkpointing (refinement is fast enough to restart)
4. **Consistency**: Uses same refinement algorithms as gaphack-decompose
5. **Chaining**: Output format matches input format, enabling iterative refinement

## Command Line Interface

### Basic Usage

```bash
gaphack-refine --input-dir <cluster_directory> \
               --output-dir <output_directory> \
               [refinement options] \
               [algorithm parameters]
```

### Required Arguments

```
--input-dir PATH          Directory containing cluster FASTA files (one per cluster)
                         Each FASTA file represents one cluster
                         Filename (without .fasta) becomes cluster ID

--output-dir PATH         Output directory for refined clusters
                         Creates timestamped subdirectory by default
```

### Refinement Stage Controls

```
--refine-close-clusters FLOAT
                         Enable close cluster refinement with distance threshold
                         Typical value: same as --max-lump (e.g., 0.02)
                         Use 0.0 or omit to disable (default: 0.0, disabled)

Note: Conflict resolution is ALWAYS performed as the first stage.
      This is required because overlapping clusters indicate they are too close
      and must be deconflicted before any other refinement.
```

### Algorithm Parameters

```
--min-split FLOAT        Minimum distance to split clusters (default: 0.005)
--max-lump FLOAT         Maximum distance to lump clusters (default: 0.02)
--target-percentile INT  Percentile for gap optimization (default: 95)
```

### Advanced Refinement Parameters

```
--max-scope-size INT     Maximum sequences for full gapHACk refinement (default: 300)
                         Components larger than this will be skipped with warning

--expansion-threshold FLOAT
                         Distance threshold for scope expansion during close cluster refinement
                         Default: 1.2 * close_threshold (auto-calculated)
                         Higher values include more context clusters
```

### Proximity Graph Parameters

```
--search-method {blast,vsearch}
                         Search method for building proximity graph (default: blast)
                         Used for close cluster refinement

--knn-neighbors INT      K for K-NN cluster graph (default: 20)
                         More neighbors = better proximity detection but slower

--blast-evalue FLOAT     BLAST e-value threshold (default: 1e-5)
--min-identity FLOAT     Minimum sequence identity % (default: auto)
```

### Output Options

```
--renumber               Renumber clusters by size in output (default: True)
                         Largest cluster becomes cluster_00001.fasta

--preserve-ids           Preserve input cluster IDs (don't renumber)
                         Useful for tracking clusters through pipeline

--no-timestamp           Write directly to output-dir instead of timestamped subdirectory
                         Overwrites existing files if present
```

### Other Options

```
--show-progress          Show progress bars during refinement (default: True)
--log-level {DEBUG,INFO,WARNING,ERROR}
                         Logging verbosity (default: INFO)
```

## Input Format

### Directory Structure

```
input_dir/
  cluster_001.fasta       # Cluster with ID "cluster_001"
  cluster_002.fasta       # Cluster with ID "cluster_002"
  ...
  cluster_NNN.fasta       # Cluster with ID "cluster_NNN"
  unassigned.fasta        # (optional) Unassigned sequences - preserved but not refined
```

### Naming Conventions

- **Cluster ID extraction**: Filename without `.fasta` extension
  - `cluster_001.fasta` → cluster ID: `cluster_001`
  - `my_cluster_A.fasta` → cluster ID: `my_cluster_A`
  - `vsearch_otu_42.fasta` → cluster ID: `vsearch_otu_42`

- **Special files**:
  - `unassigned.fasta`: Preserved in output, excluded from refinement
  - Any other `.fasta` file is treated as a cluster

### FASTA Content Requirements

- Each cluster FASTA file must contain:
  - Valid DNA/RNA sequences (A, C, G, T/U, N, ambiguity codes)
  - Sequence headers (IDs)
  - At least one sequence (empty files trigger warning and are skipped)

- **Duplicate sequences across clusters**: Detected as conflicts and resolved in stage 1
- **Duplicate headers within a cluster**: Treated as separate sequences with same ID
- **Sequence content**: All sequences from cluster FASTAs are loaded and used for MSA-based distance calculation

## Workflow Logic

### Stage 0: Load and Validate

```python
1. Load all cluster FASTA files from input directory
   - Read all *.fasta files (except unassigned.fasta)
   - Build clusters = {cluster_id: [sequence_headers]}
   - Build unified sequence and header lists from all cluster files
   - Separately load unassigned.fasta if present

2. Validate input
   - At least 2 clusters required (1 cluster = nothing to refine)
   - All sequences valid (DNA/RNA alphabet)
   - No completely empty clusters (warn and skip)

3. Detect conflicts automatically
   - Scan for sequence headers in multiple clusters
   - Build conflict map: {seq_header: [cluster_id1, cluster_id2, ...]}

4. Report initial state
   Total clusters: N
   Total sequences: M (X unique headers)
   Conflicts detected: Y sequences in Z clusters
   Unassigned sequences: W (if unassigned.fasta present)
```

### Stage 1: Conflict Resolution (Always Applied)

```python
IF conflicts detected:
  1. Group conflicts into connected components (DFS on cluster adjacency graph)

  2. For each conflict component:
     - Extract minimal scope (conflicted clusters only, no expansion)
     - If scope size ≤ max_scope_size:
       * Create MSA-based distance provider for scope sequences
       * Apply full gapHACk clustering
       * Replace original conflicted clusters with gapHACk result
     - Else:
       * Warn: "Skipping oversized conflict component (N sequences > max_scope_size)"
       * Leave clusters unchanged

  3. Track cluster transformations
     - clusters_before: Original conflict component clusters
     - clusters_after: New refined clusters
     - sequences_affected: Count

  4. Verify conflicts resolved
     - Comprehensive conflict scan
     - Report any remaining conflicts

ELSE:
  Skip conflict resolution (log: "No conflicts detected")
```

### Stage 2: Close Cluster Refinement (Optional)

```python
IF --refine-close-clusters > 0.0:
  1. Build BLAST/vsearch K-NN proximity graph
     - Compute cluster medoids (sequence with min total distance)
     - BLAST/vsearch K-NN search (k=knn_neighbors) for each medoid
     - Build graph: cluster → [(neighbor_cluster, distance), ...]

  2. Find close cluster pairs
     - Query graph for pairs within close_threshold
     - close_pairs = [(cluster1, cluster2, distance), ...]

  3. Group close pairs into connected components (DFS)

  4. For each close cluster component:
     - Start with core clusters in component
     - Iteratively expand context until positive gap achieved:
       * Apply full gapHACk to current scope
       * Measure gap size from best configuration
       * If gap < 0.001: add distant context clusters (1.5x, 2.0x, 2.5x max_lump)
       * Repeat up to 5 iterations
     - Replace original clusters with refined result

  5. Track cluster transformations
     - close_pairs_found: Count
     - components_processed: Count
     - clusters_before → clusters_after for each component

ELSE:
  Skip close cluster refinement (log: "Close cluster refinement disabled")
```

### Stage 3: Final Verification and Output

```python
1. Comprehensive conflict verification
   - Scan all final clusters for multi-assigned sequences
   - Ensure MECE property (mutually exclusive, collectively exhaustive)
   - Critical failure if conflicts remain

2. Optionally renumber clusters (default: True, unless --preserve-ids)
   - Sort clusters by size (largest first)
   - Renumber: cluster_00001, cluster_00002, ...
   - Generate mapping: {original_id: final_id}

3. Write output clusters
   - Create output directory (timestamped unless --no-timestamp)
   - Write one FASTA per cluster: cluster_NNNNN.fasta
   - Write unassigned.fasta if present in input
   - Write cluster_mapping.txt (original_id → final_id)

4. Generate summary report (refine_summary.txt)
   - Input/output paths and parameters
   - Initial state (clusters, sequences, conflicts)
   - Stage 1 results (conflict resolution)
   - Stage 2 results (close cluster refinement, if applied)
   - Final state (clusters, sequences, MECE verification)
   - Cluster size distribution
   - Processing time for each stage
```

## Output Format

### Default: Timestamped Subdirectory

```
output_dir/
  20250105_143022/
    cluster_00001.fasta         # Largest cluster
    cluster_00002.fasta         # 2nd largest
    ...
    cluster_NNNNN.fasta         # Smallest cluster
    unassigned.fasta            # (if present in input)
    cluster_mapping.txt         # original_id → final_id mapping
    refine_summary.txt          # Detailed summary report
  latest -> 20250105_143022/    # Symlink to most recent
```

### With `--no-timestamp`

```
output_dir/
  cluster_00001.fasta
  cluster_00002.fasta
  ...
  cluster_mapping.txt
  refine_summary.txt
```

### cluster_mapping.txt Format

```
# gaphack-refine cluster ID mapping
# Generated: 2025-01-05 14:30:22
# Input: initial_clusters/
# Output: refined_output/20250105_143022/

# Original_ID → Deconflicted_ID → Refined_ID → Final_ID

cluster_001 → cluster_001_C → cluster_001_C_R1 → cluster_00001
cluster_002 → cluster_002_C → cluster_002_C_R1 → cluster_00002
cluster_003 → [merged into cluster_001_C_R1] → cluster_00001
vsearch_otu_5 → vsearch_otu_5_C → vsearch_otu_5_C_R1 → cluster_00003
...
```

**Note**: Intermediate IDs (_C for conflict resolution, _R1 for refinement) are tracked internally but final output uses clean sequential numbering unless `--preserve-ids` is specified.

### refine_summary.txt Format

```
gaphack-refine Summary Report
=============================
Run Date: 2025-01-05 14:30:22
Version: 1.0.0
Command: gaphack-refine --input-dir initial/clusters/ --output-dir refined/ --refine-close-clusters 0.02

Input
-----
Input directory: initial/clusters/
Total input files: 143 cluster FASTA files + 1 unassigned.fasta
Total sequences: 1,429
Unique sequence headers: 1,406 (23 duplicates detected)
Input clusters: 143
Unassigned sequences: 0

Initial Conflicts Detected
--------------------------
Conflicted sequences: 23
Conflicted clusters: 47
Conflict components: 8

Algorithm Parameters
--------------------
min_split: 0.005
max_lump: 0.020
target_percentile: 95
max_scope_size: 300
search_method: blast
knn_neighbors: 20

Stage 1: Conflict Resolution
-----------------------------
Status: REQUIRED (conflicts detected)
Method: Full gapHACk with minimal scope (no expansion)

Conflict Components Processed: 8
  Component 1: 6 clusters (45 sequences) → 4 clusters
  Component 2: 12 clusters (87 sequences) → 9 clusters
  Component 3: 4 clusters (23 sequences) → 3 clusters
  Component 4: 2 clusters (15 sequences) → 1 cluster (merged)
  Component 5: 5 clusters (34 sequences) → 4 clusters
  Component 6: 8 clusters (56 sequences) → 6 clusters
  Component 7: 3 clusters (18 sequences) → 2 clusters
  Component 8: 7 clusters (48 sequences) → 5 clusters

Clusters before: 143
Clusters after: 138
Conflicts resolved: 23 → 0 (100% resolution)
Duration: 12.3 seconds

Stage 2: Close Cluster Refinement
----------------------------------
Status: ENABLED (--refine-close-clusters 0.020)
Threshold: 0.020
Expansion threshold: 0.024 (auto-calculated: 1.2 × close_threshold)

Proximity graph construction:
  K-NN neighbors per cluster: 20
  Search method: BLAST
  Graph construction time: 8.5 seconds

Close pairs found: 15 pairs
Connected components: 6

Close Cluster Components Processed: 6
  Component 1: 3 clusters (28 sequences) → 2 clusters
    Context expansion: 1 iteration, gap achieved: 0.0023
  Component 2: 4 clusters (35 sequences) → 3 clusters
    Context expansion: 2 iterations, gap achieved: 0.0018
  Component 3: 2 clusters (18 sequences) → 1 cluster (merged)
    Context expansion: 0 iterations, gap achieved: 0.0012
  Component 4: 5 clusters (42 sequences) → 4 clusters
    Context expansion: 1 iteration, gap achieved: 0.0031
  Component 5: 2 clusters (16 sequences) → 1 cluster (merged)
    Context expansion: 0 iterations, gap achieved: 0.0015
  Component 6: 4 clusters (31 sequences) → 3 clusters
    Context expansion: 2 iterations, gap achieved: 0.0019

Clusters before: 138
Clusters after: 132
Clusters merged: 6
Duration: 45.7 seconds

Final Verification
------------------
Final clusters: 132
Total sequences: 1,429
Conflicts remaining: 0
MECE property: SATISFIED ✓

Cluster Size Distribution
--------------------------
  1-10 sequences: 45 clusters (34.1%)
  11-50 sequences: 62 clusters (47.0%)
  51-100 sequences: 20 clusters (15.2%)
  100+ sequences: 5 clusters (3.8%)

Largest cluster: 105 sequences (cluster_00001)
Smallest cluster: 1 sequence (cluster_00132)
Median cluster size: 8 sequences
Mean cluster size: 10.8 sequences

Output
------
Output directory: refined/20250105_143022/
Cluster files: 132 FASTA files
Unassigned file: Yes (0 sequences)
Cluster mapping: cluster_mapping.txt
Renumbering: Enabled (by size, largest first)

Processing Summary
------------------
Total processing time: 66.5 seconds
  - Input loading: 0.7 seconds
  - Conflict resolution: 12.3 seconds
  - Proximity graph: 8.5 seconds
  - Close cluster refinement: 45.7 seconds
  - Output writing: 0.3 seconds

Status: SUCCESS
Final cluster count: 132 (reduced from 143, -7.7%)
Conflicts resolved: 23 → 0 (100%)
```

## Implementation Architecture

### Module Structure

**New file**: `gaphack/refine_cli.py`

```python
"""Command-line interface for gaphack-refine."""

def load_clusters_from_directory(input_dir: Path) -> Tuple[
    Dict[str, List[str]],  # clusters: cluster_id → sequence_headers
    List[str],              # all_sequences
    List[str],              # all_headers
    List[str]               # unassigned_headers
]:
    """Load all cluster FASTA files from directory.

    Returns:
        - clusters: Dict mapping cluster_id to list of sequence headers
        - all_sequences: Unified list of all sequences from all clusters
        - all_headers: Unified list of all headers from all clusters
        - unassigned_headers: Headers from unassigned.fasta (if present)
    """

def detect_conflicts(clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Detect sequences assigned to multiple clusters.

    Returns:
        Dict mapping sequence_header → list of cluster_ids containing it
    """

def generate_cluster_mapping_report(
    original_clusters: Dict[str, List[str]],
    deconflicted_clusters: Dict[str, List[str]],
    refined_clusters: Dict[str, List[str]],
    final_clusters: Dict[str, List[str]],
    output_path: Path
) -> None:
    """Generate cluster_mapping.txt showing ID transformations."""

def generate_refinement_summary(
    input_dir: Path,
    output_dir: Path,
    parameters: Dict,
    initial_state: Dict,
    stage1_info: ProcessingStageInfo,
    stage2_info: Optional[ProcessingStageInfo],
    final_state: Dict,
    timing: Dict
) -> str:
    """Generate detailed summary report.

    Returns formatted summary text for refine_summary.txt
    """

def write_output_clusters(
    clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    unassigned_headers: List[str],
    output_dir: Path,
    renumber: bool = True
) -> Dict[str, str]:
    """Write cluster FASTA files to output directory.

    Returns:
        Mapping of original_cluster_id → final_cluster_id
    """

def main():
    """Main entry point for gaphack-refine CLI."""
    parser = argparse.ArgumentParser(
        description="Refine existing clusters using conflict resolution and close cluster refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES
    )

    # Add all arguments as specified above
    # Parse arguments
    # Validate inputs
    # Load clusters
    # Detect conflicts
    # Run Stage 1: Conflict resolution (always)
    # Run Stage 2: Close cluster refinement (if requested)
    # Verify final results
    # Write outputs
    # Generate summary report
```

### Reused Modules

- `cluster_refinement.py`:
  - `resolve_conflicts()`
  - `refine_close_clusters()`
  - `RefinementConfig`
  - `verify_no_conflicts()`

- `cluster_graph.py`:
  - `ClusterGraph`

- `decompose.py`:
  - `ClusterIDGenerator`
  - `ProcessingStageInfo`

- `utils.py`:
  - `load_sequences_from_fasta()` (for individual cluster files)

- `distance_providers.py`:
  - `MSACachedDistanceProvider` (used internally by refinement stages)

### Entry Point

**setup.py** or **pyproject.toml**:
```python
entry_points={
    'console_scripts': [
        'gaphack=gaphack.cli:main',
        'gaphack-decompose=gaphack.decompose_cli:main',
        'gaphack-refine=gaphack.refine_cli:main',  # NEW
    ]
}
```

## Error Handling

### Input Validation Errors

```python
# Missing or invalid input directory
FileNotFoundError: "Input directory not found: {input_dir}"

# No cluster FASTA files found
ValueError: "No cluster files (*.fasta) found in {input_dir}"

# Output directory issues
FileExistsError: "Output directory exists: {output_dir} (use --no-timestamp to overwrite)"
PermissionError: "Cannot write to output directory: {output_dir}"

# Invalid FASTA format
BioPython.SeqIO.ParseError: "Invalid FASTA format in {filename}: {error}"

# Insufficient clusters
ValueError: "At least 2 clusters required for refinement (found {n})"
```

### Processing Warnings

```python
# Empty clusters
Warning: "Empty cluster file ignored: {filename}"

# Oversized conflict components
Warning: "Skipping oversized conflict component: {n} sequences > max_scope_size ({max})"
Warning: "  Affected clusters: {cluster_ids}"
Warning: "  Consider increasing --max-scope-size or pre-processing this component"

# No close clusters found
Info: "No close cluster pairs found within threshold {threshold}"
Info: "Skipping close cluster refinement stage"

# Remaining conflicts after resolution
Error: "Conflict resolution incomplete: {n} conflicts remain"
Error: "This may indicate oversized components were skipped"
```

### Final Verification Failures

```python
# MECE property violation
Error: "CRITICAL: Final verification failed - MECE property violated"
Error: "  Conflicted sequences: {n}"
Error: "  This indicates a refinement algorithm failure"
Error: "  Please report this issue with input data"
```

## Usage Examples

### Example 1: Refine gaphack-decompose Output

```bash
# Initial clustering
gaphack-decompose --input seqs.fasta \
                  --output-dir initial_run \
                  --max-clusters 100

# Refine with conflict resolution only
gaphack-refine --input-dir initial_run/clusters/latest/ \
               --output-dir refined_conflicts_only

# Refine with both stages
gaphack-refine --input-dir initial_run/clusters/latest/ \
               --output-dir refined_full \
               --refine-close-clusters 0.02
```

### Example 2: Refine vsearch Output

```bash
# vsearch clustering (external tool)
vsearch --cluster_fast seqs.fasta \
        --id 0.97 \
        --clusters vsearch_clusters/cluster_

# Refine vsearch clusters
gaphack-refine --input-dir vsearch_clusters/ \
               --output-dir vsearch_refined \
               --refine-close-clusters 0.02 \
               --max-lump 0.03 \
               --min-split 0.01
```

### Example 3: Iterative Refinement (Chained Runs)

```bash
# Round 1: Conservative refinement
gaphack-refine --input-dir initial_clusters/ \
               --output-dir round1 \
               --refine-close-clusters 0.015

# Round 2: More aggressive refinement
gaphack-refine --input-dir round1/latest/ \
               --output-dir round2 \
               --refine-close-clusters 0.025 \
               --max-scope-size 500

# Round 3: Final polishing
gaphack-refine --input-dir round2/latest/ \
               --output-dir final \
               --refine-close-clusters 0.02
```

### Example 4: Preserve Original IDs

```bash
# Refine but keep original cluster IDs (no renumbering)
gaphack-refine --input-dir my_clusters/ \
               --output-dir refined_preserved \
               --preserve-ids \
               --refine-close-clusters 0.02
```

### Example 5: Custom Algorithm Parameters

```bash
# Refine with custom gapHACk parameters
gaphack-refine --input-dir clusters/ \
               --output-dir refined_custom \
               --refine-close-clusters 0.025 \
               --min-split 0.003 \
               --max-lump 0.025 \
               --target-percentile 90 \
               --max-scope-size 500
```

## Testing Strategy

### Unit Tests

```python
# test_refine_cli.py

def test_load_clusters_from_directory():
    """Test loading cluster FASTAs from directory."""

def test_load_clusters_with_unassigned():
    """Test handling of unassigned.fasta."""

def test_detect_conflicts():
    """Test conflict detection across clusters."""

def test_detect_no_conflicts():
    """Test when no conflicts exist."""

def test_generate_cluster_mapping():
    """Test cluster ID mapping report generation."""

def test_generate_summary_report():
    """Test summary report formatting."""

def test_write_output_clusters_renumbered():
    """Test writing clusters with renumbering."""

def test_write_output_clusters_preserved():
    """Test writing clusters with preserved IDs."""
```

### Integration Tests

```python
def test_refine_gaphack_decompose_output():
    """Test refining output from gaphack-decompose."""

def test_refine_vsearch_output():
    """Test refining output from vsearch clustering."""

def test_refine_with_conflicts():
    """Test conflict resolution stage with known conflicts."""

def test_refine_close_clusters():
    """Test close cluster refinement stage."""

def test_refine_both_stages():
    """Test running both stages sequentially."""

def test_chained_refinement():
    """Test using output from one run as input to another."""

def test_preserve_ids():
    """Test --preserve-ids flag preserves cluster IDs."""

def test_no_timestamp_output():
    """Test --no-timestamp writes directly to output-dir."""
```

### End-to-End Tests

```python
def test_e2e_russula_dataset():
    """Test on real Russula dataset (1,429 sequences)."""
    # Initial clustering with decompose
    # Refine with both stages
    # Verify MECE property
    # Check cluster quality metrics

def test_e2e_vsearch_compatibility():
    """Test compatibility with vsearch output format."""

def test_e2e_iterative_refinement():
    """Test multiple rounds of chained refinement."""
```

## Performance Expectations

### Scaling Characteristics

- **Conflict resolution**: O(C × S) where C = conflicted clusters, S = sequences per component
  - Typical: 10-50 seconds for 100-200 clusters with conflicts

- **Proximity graph construction**: O(K × C × M) where K = knn_neighbors, C = clusters, M = medoid computation
  - Typical: 5-20 seconds for 100-200 clusters with K=20

- **Close cluster refinement**: O(P × S) where P = close pairs, S = sequences per component
  - Typical: 30-60 seconds for 100-200 clusters with 10-20 close pairs

### Total Expected Runtime

- **Small datasets** (10-50 clusters): < 10 seconds
- **Medium datasets** (50-200 clusters): 30-120 seconds
- **Large datasets** (200-500 clusters): 2-5 minutes

Fast enough that state management and checkpointing are unnecessary.

## Future Enhancements (Out of Scope for v1)

1. **JSON output format**: Machine-readable summary alongside text report
2. **Parallel processing**: Parallelize independent conflict/close cluster components
3. **Progressive refinement**: Stop when improvement metric plateaus
4. **Custom distance functions**: Allow user-provided distance calculation
5. **Incremental refinement**: Process only new/changed clusters
6. **Quality metrics**: Report ARI, homogeneity, completeness if ground truth provided
7. **Visualization**: Generate cluster dendrogram or distance heatmap

## Success Criteria

**v1 is successful if**:

1. ✓ Loads clusters from any FASTA-based clustering tool output
2. ✓ Automatically detects and resolves conflicts (MECE property guaranteed)
3. ✓ Optionally refines close clusters with iterative context expansion
4. ✓ Generates clear, actionable summary reports
5. ✓ Output format compatible with input format (enables chaining)
6. ✓ Runs fast enough to restart on failure (< 5 minutes for typical datasets)
7. ✓ Works seamlessly with gaphack-decompose output
8. ✓ Comprehensive test coverage (unit, integration, end-to-end)

---

**Implementation Status**: Ready to Begin
**Next Steps**: Implement `gaphack/refine_cli.py` and tests
