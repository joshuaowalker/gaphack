# gapHACk - Gap-Optimized Hierarchical Agglomerative Clustering

A Python package for DNA barcode clustering that optimizes for the barcode gap between intra-species and inter-species genetic distances.

## Overview

gapHACk implements a two-phase clustering algorithm designed specifically for DNA barcoding applications:

1. **Phase 1**: Fast greedy lumping below the min-split threshold (default 0.5% distance)
2. **Phase 2**: Gap-aware optimization using gap-based heuristic to find optimal clustering

The algorithm focuses on maximizing the "barcode gap" - the separation between intra-species and inter-species distances at a specified percentile (default P95) to handle outliers robustly.

**gapHACk includes three main tools**:
- **`gaphack`**: Core clustering tool for standard datasets
- **`gaphack-analyze`**: Analysis tool for pre-clustered FASTA files to assess distance distributions and barcode gap quality
- **`gaphack-decompose`**: Iterative BLAST-based clustering for large datasets (100K+ sequences) with post-processing refinement

## Features

### Core Clustering (`gaphack`)
- **Gap Optimization**: Uses gap-based heuristic to directly maximize the barcode gap
- **Target Mode**: Single-cluster focused clustering from seed sequences with `--target` parameter
- **Percentile-based Linkage**: Uses percentile-based complete linkage (default 95th percentile) for robust merge decisions
- **Two-phase Algorithm**: Fast initial clustering followed by gap-aware optimization to completion
- **Progress Tracking**: Unified progress bar with real-time gap and cluster information
- **Size-ordered Output**: Clusters numbered by size (largest first) for consistent results
- **Library-friendly**: Configurable progress bars and logging for integration into other applications

### Large-Scale Clustering (`gaphack-decompose`)
- **BLAST-based Neighborhoods**: Efficiently handles datasets with 100K+ sequences
- **Iterative Target Clustering**: Processes data in manageable chunks using target mode
- **Incremental Restart**: Checkpoint-based resumption with graceful interruption handling
- **Conflict Resolution**: Ensures mutually exclusive clustering (MECE) through full gapHACk refinement
- **Close Cluster Refinement**: Optimizes cluster boundaries for closely related groups
- **Flexible Stopping Criteria**: Control by cluster count, sequence coverage, or exhaustive processing
- **Memory Efficient**: Uses hash-based deduplication and medoid caching

### Analysis Tools (`gaphack-analyze`)
- **Pre-clustered Analysis**: Evaluate existing clustering results
- **Distance Distributions**: Calculate intra-cluster and inter-cluster distances
- **Barcode Gap Metrics**: Assess gap quality at P90/P95 levels
- **Visualization**: Generate histograms with percentile markers
- **Multiple Output Formats**: Text reports, JSON data, or TSV tables

## Performance

gapHACk includes multiprocessing capabilities for improved performance on multi-core systems:

- **Automatic Parallelization**: Uses all available CPU cores by default for gap-aware clustering
- **Single-process Mode**: Use `-t 0` for single-threaded operation (ideal for library usage)
- **Custom Thread Count**: Use `-t N` to specify number of worker processes
- **Scalable Performance**: Achieves ~4x speedup with 8 cores (200→800 steps/second)
- **Load Balancing**: Efficient workload distribution using offset-based coordinate mapping
- **Memory Efficient**: Persistent worker caches reduce initialization overhead

### Performance Examples

```bash
# Use all available cores (default)
gaphack input.fasta

# Use 4 worker processes
gaphack input.fasta -t 4

# Single-threaded mode (for library usage or debugging)
gaphack input.fasta -t 0

# Performance comparison with metrics export
gaphack large_dataset.fasta --export-metrics performance.json -v
```

### Performance Characteristics

- **Small datasets** (< 100 sequences): Single-process mode may be faster due to reduced overhead
- **Medium datasets** (100-1000 sequences): Multiprocessing provides significant speedup
- **Large datasets** (> 1000 sequences): Maximum benefit from multiprocessing parallelization

The gap-aware clustering phase scales with O(n³) complexity, making multiprocessing particularly beneficial for larger datasets.

## Installation

### From Source

```bash
git clone https://github.com/joshuaowalker/gaphack.git
cd gaphack
pip install -e .
```

### Using pip (once published)

```bash
pip install gaphack
```

## Quick Start

Try gapHACk with example data:

```bash
# Install the package
pip install git+https://github.com/joshuaowalker/gaphack.git

# Download example data
wget https://raw.githubusercontent.com/joshuaowalker/gaphack/main/examples/data/collybia_nuda_test.fasta

# Standard clustering (creates .cluster_001.fasta, .cluster_002.fasta, etc.)
gaphack collybia_nuda_test.fasta

# Large-scale decompose clustering (for bigger datasets)
gaphack-decompose collybia_nuda_test.fasta -o decompose_results

# Analyze pre-clustered results (from the latest output)
gaphack-analyze decompose_results/clusters/latest/*.fasta -o analysis_results

# With detailed output and custom parameters
gaphack collybia_nuda_test.fasta \
    -o output/clusters \
    --export-metrics metrics.json \
    --verbose
```

The example dataset contains 91 *Collybia nuda* ITS sequences from iNaturalist with known clustering structure. See [examples/README.md](examples/README.md) for more details.

## Usage

### Command Line Interface

Basic usage:
```bash
# Creates input.cluster_001.fasta, input.cluster_002.fasta, etc.
gaphack input.fasta

# Custom output base path
gaphack input.fasta -o results/myclusters

# Target mode clustering
gaphack input.fasta --target seeds.fasta -o target_cluster
```

With custom parameters:
```bash
# With adjusted identity (recommended, default)
gaphack input.fasta \
    --min-split 0.003 \
    --max-lump 0.03 \
    --target-percentile 90 \
    --alignment-method adjusted \
    --end-skip-distance 20 \
    --export-metrics gap_analysis.json \
    --threads 8 \
    --verbose

# Using traditional identity (more conservative)
gaphack input.fasta \
    --alignment-method traditional \
    --threads 0 \
    -o traditional_clusters
```

### Target Mode Clustering

Use `--target` to grow a single cluster from seed sequences:

```bash
# Basic target mode with seed sequences
gaphack input.fasta --target seeds.fasta -o target_results

# Target mode produces:
# - target_results.cluster_001.fasta (sequences in the target cluster)
# - target_results.unclustereds.fasta (sequences not processed for clustering)
```

Target mode focuses on growing one cluster from the provided seed sequences, making it suitable for cases where you want to extract sequences similar to specific targets without attempting to cluster all remaining sequences.

### Large-Scale Decompose Clustering (gaphack-decompose)

The `gaphack-decompose` tool handles datasets too large for standard clustering (100K+ sequences) using iterative BLAST-based neighborhoods:

```bash
# Basic decompose clustering
gaphack-decompose large_dataset.fasta -o results

# Directed mode with specific targets
gaphack-decompose large_dataset.fasta --targets priority_sequences.fasta -o results

# Stop after 50 clusters
gaphack-decompose large_dataset.fasta --max-clusters 50 -o results

# Enable conflict resolution and close cluster refinement
gaphack-decompose large_dataset.fasta \
    --resolve-conflicts \
    --refine-close-clusters 0.02 \
    -o results

# Custom BLAST parameters for specific datasets
gaphack-decompose large_dataset.fasta \
    --blast-max-hits 1500 \
    --min-identity 85.0 \
    -o results
```

**Output Organization:**

The tool creates a structured output directory with separate working and final results:

```
results/
├── work/                           # Working files (intermediate stages)
│   ├── initial/                    # Initial clustering stage
│   │   ├── cluster_00001I.fasta
│   │   ├── cluster_00002I.fasta
│   │   └── unassigned.fasta
│   ├── deconflicted/              # After conflict resolution
│   │   └── cluster_*C.fasta
│   └── refined_1/                  # After close cluster refinement
│       └── cluster_*R1.fasta
├── clusters/                       # Final numbered output
│   ├── 20251003_141530/           # Timestamp-based directory
│   │   ├── cluster_00001.fasta    # Final clusters (by size)
│   │   ├── cluster_00002.fasta
│   │   ├── decompose_assignments.tsv
│   │   └── decompose_report.txt
│   └── latest -> 20251003_141530/ # Symlink to most recent
└── state.json                      # Checkpoint state
```

**Key Features:**
- **Automatic finalization**: Creates numbered output in `clusters/latest/` after each stage
- **Organized working directory**: Stage-based subdirectories for intermediate files
- **Timestamp versioning**: Each finalization creates a new timestamped directory
- **Automatic mode detection**: Uses targets if provided, otherwise selects targets iteratively
- **BLAST neighborhoods**: Efficiently finds similar sequences for clustering
- **Conflict resolution**: `--resolve-conflicts` ensures each sequence appears in only one cluster
- **Close cluster refinement**: `--refine-close-clusters DISTANCE` optimizes boundaries between similar clusters
- **Incremental restart**: Gracefully handles interruptions with checkpoint-based resumption

#### Incremental Restart and Checkpointing

The `gaphack-decompose` tool supports graceful interruption handling and checkpoint-based resumption for long-running clustering jobs:

**Interruption Handling:**
```bash
# Start a long-running decompose job
gaphack-decompose large_dataset.fasta -o results

# Interrupt at any time with Ctrl+C - state is automatically saved
# The tool captures interruption signals and saves checkpoint before exiting
```

**Resume from Checkpoint:**
```bash
# Resume from the last checkpoint
gaphack-decompose --resume -o results

# Continue with modified stopping criteria
gaphack-decompose --resume -o results --max-clusters 100

# Resume and enable additional refinement
gaphack-decompose --resume -o results \
    --resolve-conflicts \
    --refine-close-clusters 0.02
```

**Checkpoint Management:**
- **Automatic checkpointing**: State saved every 10 iterations by default
- **Configurable interval**: Use `--checkpoint-interval N` to adjust frequency
- **Signal handling**: Ctrl+C (SIGINT) and SIGTERM trigger graceful shutdown with state save
- **State file**: All progress stored in `output_dir/state.json`
- **BLAST database caching**: Reuses BLAST database across resume operations

**Resume Workflow:**
1. Run `gaphack-decompose` with desired parameters and output directory
2. Interrupt at any time (Ctrl+C) or let it complete initial clustering
3. Use `--resume` to continue clustering from checkpoint
4. Final output is **automatically created** in `clusters/latest/` after each stage completes
5. Access results directly from `results/clusters/latest/*.fasta`

**Key Features:**
- **No data loss**: Checkpoints saved at regular intervals and on interruption
- **Flexible resumption**: Change stopping criteria or refinement settings when resuming
- **Efficient restart**: Reuses BLAST database and completed work
- **Safe by default**: Prevents accidental overwrites of existing state
- **Input validation**: Warns if input FASTA has changed since original run

### Analysis Tool (gaphack-analyze)

The `gaphack-analyze` tool evaluates pre-clustered FASTA files to assess distance distributions and barcode gap quality:

```bash
# Analyze pre-clustered files (each FASTA = one cluster)
gaphack-analyze cluster1.fasta cluster2.fasta cluster3.fasta

# Save results to custom directory with JSON format  
gaphack-analyze *.fasta -o analysis_results --format json

# Skip plots and use TSV output
gaphack-analyze clusters/*.fasta --no-plots --format tsv -o results.tsv

# Use traditional alignment method
gaphack-analyze cluster*.fasta --alignment-method traditional -v
```

**Analysis Output:**
- **Individual cluster histograms**: Distance distributions within each cluster
- **Global distance histogram**: Combined intra-cluster vs inter-cluster distances  
- **Percentile analysis**: P5, P25, P50, P75, P95 values for all distance sets
- **Barcode gap metrics**: Gap size and existence at P90/P95 levels
- **Multiple formats**: Text reports, JSON data, or TSV tables

#### Controlling Adjusted Identity Parameters

gapHACk provides fine-grained control over the adjusted identity algorithm:

```bash
# Disable specific adjustments (all enabled by default)
gaphack input.fasta \
    --no-homopolymer-normalization \
    --no-iupac-overlap \
    --no-indel-normalization

# Control repeat motif detection
gaphack input.fasta \
    --max-repeat-motif-length 1  # Only homopolymers
gaphack input.fasta \
    --max-repeat-motif-length 3  # Up to trinucleotides

# Combine different settings for experimentation
gaphack input.fasta \
    --no-homopolymer-normalization \
    --end-skip-distance 10 \
    --max-repeat-motif-length 1 \
    --export-metrics experiment.json
```

### Python API

#### Basic Usage with Default Alignment (adjusted identity)

```python
from gaphack import GapOptimizedClustering
from gaphack import load_sequences_from_fasta, calculate_distance_matrix

# Load sequences
sequences, headers = load_sequences_from_fasta("input.fasta")

# Calculate distance matrix using adjusted identity (default)
distance_matrix = calculate_distance_matrix(
    sequences, 
    alignment_method="adjusted",  # or "traditional"
    end_skip_distance=20
)

# Initialize clustering with custom parameters
clustering = GapOptimizedClustering(
    min_split=0.005,         # 0.5% minimum distance to split clusters
    max_lump=0.02,           # 2% maximum distance to lump clusters  
    target_percentile=95,    # Use P95 for gap optimization and linkage decisions
    num_threads=None,        # Auto-detect cores (default), 0 for single-process
    show_progress=True,      # Show progress bar (default True)
    logger=None              # Use default logger (default None)
)

# Perform clustering
clusters, singletons, metrics = clustering.cluster(distance_matrix)

# Process results
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {[headers[idx] for idx in cluster]}")

print(f"Singletons: {[headers[idx] for idx in singletons]}")
print(f"Best gap size: {metrics['best_config']['gap_size']:.4f}")
```

#### Using Pre-computed Distance Matrix (for Web Applications)

```python
import numpy as np
from gaphack import GapOptimizedClustering

# If you have a pre-computed distance matrix (e.g., from a caching layer)
distance_matrix = np.array([...])  # Your distance matrix

# Cluster directly without calculating distances
clustering = GapOptimizedClustering()
clusters, singletons, metrics = clustering.cluster(distance_matrix)

# The returned values use Python native types (no numpy) for JSON serialization
print(f"Gap size: {metrics['best_config']['gap_size']}")
```

#### Library Integration (Headless/Web Applications)

```python
import logging
from gaphack import GapOptimizedClustering

# Configure for library usage - no progress bars, single-process, custom logger
app_logger = logging.getLogger("my_app.clustering")
clustering = GapOptimizedClustering(
    num_threads=0,        # Single-process mode for library integration
    show_progress=False,  # Disable progress bars in headless environment
    logger=app_logger     # Use your application's logger
)

# Silent clustering for web APIs or batch processing
clusters, singletons, metrics = clustering.cluster(distance_matrix)

# Results include cluster sizes in descending order
print(f"Created {len(clusters)} clusters: {[len(c) for c in clusters]}")
```

#### Traditional Identity for Conservative Clustering

```python
from gaphack import GapOptimizedClustering
from gaphack import load_sequences_from_fasta, calculate_distance_matrix

# Use traditional identity (no adjustments) for more conservative clustering
sequences, headers = load_sequences_from_fasta("input.fasta")
distance_matrix = calculate_distance_matrix(
    sequences,
    alignment_method="traditional"  # No adjustments
)

# Note: Traditional identity often produces more clusters (over-splitting)
clustering = GapOptimizedClustering()
clusters, singletons, metrics = clustering.cluster(distance_matrix)
```


## Alignment Methods

gapHACk uses the `adjusted-identity` package for sequence alignment and supports two methods:

1. **Adjusted** (default): Uses adjusted-identity corrections
   - Handles terminal gaps and ambiguous bases properly
   - Optimized for fungal ITS sequences but suitable for any DNA barcoding
   - Configurable end-skip distance for primer regions (default: 20)
   - Recommended for most use cases

2. **Traditional**: Uses traditional (raw) identity calculation
   - No adjustments for terminal gaps or ambiguous bases
   - Provides BLAST-like identity scores
   - More conservative, may over-split clusters

### For Library Users

If you're integrating gapHACk into a web application or need custom distance calculations:

```python
import numpy as np
from gaphack import GapOptimizedClustering

# Calculate your own distance matrix
# (e.g., using cached alignments, custom algorithms, etc.)
distance_matrix = your_distance_calculation(sequences)

# Pass directly to the clustering algorithm
clustering = GapOptimizedClustering()
clusters, singletons, metrics = clustering.cluster(distance_matrix)
```

This approach is useful for:
- Web applications with caching layers
- Integration with existing alignment pipelines
- Using specialized distance metrics
- Avoiding redundant distance calculations

## Parameters

### Algorithm Parameters

- `min_split` (default: 0.005): Minimum distance to split clusters. Sequences closer than this are lumped together (assumed intraspecific).
- `max_lump` (default: 0.02): Maximum distance to lump clusters. Sequences farther than this are kept split (assumed interspecific).
- `target_percentile` (default: 95): Which percentile to use for gap optimization and linkage decisions (P95 provides robustness against outliers).

### Adjusted Identity Parameters

These parameters control the adjusted identity algorithm when `--alignment-method adjusted` is used (default):

- `end_skip_distance` (default: 20): Number of nucleotides to skip from each sequence end to avoid primer/editing artifacts
- `normalize_homopolymers` (default: enabled): Ignore homopolymer length differences (e.g., "AAA" vs "AAAA")
- `handle_iupac_overlap` (default: enabled): Allow IUPAC ambiguity codes to match via nucleotide intersection (e.g., R matches A or G)
- `normalize_indels` (default: enabled): Count contiguous indels as single evolutionary events rather than per-base penalties
- `max_repeat_motif_length` (default: 2): Maximum length of repeat motifs to detect (1=homopolymers, 2=dinucleotides, etc.)

**CLI Flags to Disable Defaults:**
- `--no-homopolymer-normalization`: Disable homopolymer normalization
- `--no-iupac-overlap`: Disable IUPAC overlap matching
- `--no-indel-normalization`: Disable indel normalization
- `--max-repeat-motif-length N`: Set motif length (use 1 to detect only homopolymers)

### Output Formats

- **FASTA format** (default): Creates separate FASTA files for each cluster
  - `basename.cluster_001.fasta`, `basename.cluster_002.fasta`, etc.
  - `basename.singletons.fasta` for unclustered sequences (or `basename.unclustereds.fasta` in target mode)
  - Clusters ordered by size (001 = largest, 002 = second largest, etc.)
  - Cluster numbers are zero-padded for proper sorting
  - Uses two-line format (header + sequence on single line)
- **TSV format**: Tab-separated values with columns `sequence_id` and `cluster_id`
- **Text format**: Human-readable clustering report

## Algorithm Details

**Note**: Throughout this documentation, "species" refers to Operational Taxonomic Units (OTUs) - clusters of sequences that are presumed to represent biological species based on genetic similarity, but have not been formally taxonomically validated.

### Core Algorithm (gaphack)

#### Distance Matrix Calculation

gapHACk begins by calculating a pairwise distance matrix from input sequences using the adjusted identity algorithm (default) or traditional BLAST identity:

- **Adjusted identity** (recommended): Applies corrections for sequencing artifacts, ambiguous bases, and terminal gaps to produce cleaner distance estimates that better reflect biological relationships
- **Traditional identity**: Uses raw BLAST-style identity scores without corrections

The distance matrix forms the foundation for all subsequent clustering decisions.

#### Barcode Gap

The barcode gap is the separation between the maximum intra-species distance and the minimum inter-species distance. A clear gap indicates good species delimitation, allowing confident assignment of sequences to species clusters.

#### Percentile Gaps

Instead of using absolute max/min values (which are sensitive to outliers), gapHACk uses percentile-based gaps for robustness:
- **P95 gap**: 95th percentile of intra-species distances vs 5th percentile of inter-species distances
- **P90 gap**: 90th percentile of intra-species distances vs 10th percentile of inter-species distances

This approach compares the "worst case" within species (upper percentile of intra-species distances) against the "best case" between species (lower percentile of inter-species distances), creating a conservative measure of separation that is robust to outliers.

The `target_percentile` parameter (default: 95) determines which percentile gap to optimize during clustering.

#### Two-Phase Clustering Strategy

**Phase 1: Fast Greedy Merging**
1. Start with each sequence as its own cluster
2. Merge all clusters with distances below `min_threshold` (default: 0.005 or 0.5%)
3. No gap calculation needed - assumes these are all intraspecific variation
4. Provides rapid initial clustering of clearly related sequences

**Phase 2: Gap-Optimized Merging**
Between `min_threshold` and `max_threshold` (default: 0.02 or 2%):
1. **Merge evaluation**: For each potential cluster merge, calculate the resulting gap metrics
2. **Percentile linkage**: Use the `merge_percentile` (default: 95th percentile) of pairwise distances between clusters for merge decisions - more conservative than average linkage
3. **Gap tracking**: Monitor the `target_percentile` gap size and record the configuration with the best gap
4. **Gap-based heuristic**: At each step, choose the merge that maximizes the barcode gap
5. **Termination condition**: Stop when all remaining merges exceed `max_threshold`
6. **Best tracking**: Track and return the clustering configuration that achieved the best gap

#### Parameter Roles

- **`min_split`**: Defines the boundary between Phase 1 (fast lumping) and Phase 2 (gap optimization). Sequences closer than this are assumed to represent intraspecific variation and are lumped together.

- **`max_lump`**: Upper limit for cluster lumping. Distances beyond this are assumed to represent interspecific divergence and clusters are kept split.

- **`target_percentile`**: Which percentile to use for gap optimization and linkage decisions (e.g., 95 = P95 gap). Higher percentiles are more robust to outliers but may be less sensitive to true gaps.

### Decompose Algorithm (gaphack-decompose)

For datasets too large for the O(n³) core algorithm, `gaphack-decompose` uses an iterative approach that carefully balances computational efficiency with clustering quality.

#### BLAST-based Neighborhood Discovery

1. **Target Selection**: Choose a target sequence (provided or selected from unassigned sequences)
2. **BLAST Search**: Find similar sequences using BLAST (default: top 1000 hits)
3. **Neighborhood Pruning**: Apply the N+N pruning strategy (see below)
4. **Target Clustering**: Apply target mode clustering to the pruned neighborhood
5. **Iteration**: Repeat until stopping criteria met

#### The N+N Neighborhood Pruning Strategy

A critical challenge in iterative clustering is ensuring the barcode gap can be properly evaluated. The gap calculation requires both intra-cluster distances (sequences within the same species) and inter-cluster distances (sequences from different species). Without sufficient inter-cluster context, the algorithm cannot determine whether a gap exists.

The N+N pruning strategy addresses this:

1. **Calculate distances** from each neighborhood sequence to the target(s)
2. **Sort sequences** by their minimum distance to any target
3. **Select N sequences within max_lump** (0.02 default): These are potential cluster members
4. **Add N additional sequences beyond max_lump**: These provide inter-cluster context

This balanced selection ensures:
- **Complete cluster coverage**: All sequences likely to belong to the target's species are included
- **Sufficient context**: An equal number of sequences from other species provide the inter-cluster distances needed for gap calculation
- **Computational efficiency**: The neighborhood is pruned to 2N sequences rather than processing all BLAST hits
- **Robust gap estimation**: The percentile-based gap calculation (e.g., P95) has sufficient samples from both intra- and inter-cluster distributions

#### Post-Processing Refinement

After initial clustering, two optional refinement stages ensure high-quality results:

##### Conflict Resolution (`--resolve-conflicts`)

Ensures mutually exclusive clustering (MECE property):
- Identifies sequences assigned to multiple clusters
- Groups conflicts into connected components
- Applies full gapHACk to each component using minimal scope
- Pure correctness focus with minimal computational overhead

##### Close Cluster Refinement (`--refine-close-clusters DISTANCE`)

Optimizes boundaries between closely related clusters:

**The Context Expansion Problem**: When clusters are very close (e.g., subspecies or recent divergences), applying full gapHACk to just those clusters often results in complete merging because there's no inter-cluster context to establish a gap. This is particularly problematic when refining clusters that are all within the max_lump distance.

**Iterative Context Expansion Solution**:
1. **Start with core clusters** to be refined
2. **Apply full gapHACk** and measure the resulting gap
3. **If gap is insufficient** (< 0.001):
   - Add one additional cluster beyond the expansion threshold as context
   - This provides inter-cluster distances needed for gap calculation
   - Reapply full gapHACk with expanded scope
4. **Iterate** up to 5 times or until a positive gap is achieved
5. **Return best result** across all iterations

This approach ensures that even closely related clusters can be properly evaluated by providing sufficient evolutionary context for the gap calculation.

#### Cluster Graph Architecture

The system uses a cluster graph to efficiently track relationships:
- **Medoid-based representation**: Each cluster represented by its medoid (most central sequence)
- **Proximity queries**: Efficient finding of nearby clusters for refinement
- **Dynamic updates**: Graph maintained as clusters merge or split
- **Scalable to 100K+ sequences**: Optimized for large-scale datasets

## Related Work

### Prior Art in Barcode Gap Discovery

gapHACk was developed independently but shares conceptual similarities with ABGD (Automatic Barcode Gap Discovery; Puillandre et al., 2011), which also automatically identifies the barcode gap for species delimitation. Both methods:
- Seek to identify the threshold between intraspecific and interspecific genetic variation
- Use pairwise distance matrices as input
- Apply recursive partitioning to handle heterogeneity across taxa

Key differences in gapHACk's approach:
- **Clustering algorithm**: gapHACk uses hierarchical agglomerative clustering with dynamic gap optimization, while ABGD uses graph-based partitioning with fixed thresholds
- **Gap-based optimization**: gapHACk uses a gap-based heuristic that directly optimizes for the barcode gap at each merge, rather than recursive application of fixed thresholds
- **Distance calculation**: gapHACk implements Stephen Russell's adjusted identity algorithm (Russell, 2025), which systematically corrects for sequencing artifacts and biological complexity that can obscure true genetic distances. This preprocessing approach can provide clearer barcode gaps than traditional BLAST identity scores
- **Percentile-based robustness**: gapHACk uses percentile gaps (e.g., P95) to handle outliers, rather than absolute min/max values

### Empirical Support for Dynamic Thresholds

Wilson et al. (2023) demonstrated substantial variation in barcode gaps across 11 macrofungal genera, with the middle of barcode gaps ranging from <2% to nearly 6%. Their findings that:
- ITS2 maintains barcode gaps better than ITS1 at higher quantiles
- Taxonomic "splitting" produces clearer gaps than "lumping"
- Many genera lack clear barcode gaps due to overlapping distance distributions

These observations validate gapHACk's approach of dynamically optimizing thresholds within user-specified ranges (default: 0.5-2% distance) rather than relying on fixed universal cutoffs, which would over- or under-estimate diversity depending on the taxonomic group. The configurable min_threshold and max_threshold parameters allow users to adjust these ranges based on their knowledge of the taxonomic group being studied.

### Adjusted Identity Algorithm

The adjusted identity algorithm (Russell, 2025) that gapHACk implements addresses critical issues with raw NCBI BLAST identity scores that can mislead species identification. The algorithm makes several key corrections:
- **Homopolymer normalization**: Differences in homopolymer run lengths (e.g., AAA vs AAAA) are not counted as mismatches, addressing common sequencing artifacts
- **Ambiguity code handling**: IUPAC codes are treated as matches when they overlap (e.g., R matches A or G)
- **End trimming**: Mismatches within the first and last 20 bases are excluded, as these regions often contain editing artifacts
- **Gap event counting**: Indels are counted as single evolutionary events rather than per-base penalties

These adjustments can substantially improve identity scores (e.g., from 97.4% to 99.8%), resulting in clearer barcode gaps and more reliable species delimitation. This is particularly important for fungal ITS sequences where sequencing artifacts are common.

For implementation details of the adjusted identity algorithm, see: https://github.com/joshuaowalker/adjusted-identity

### Theoretical Foundations

Randriamihamison et al. (2021) provided important theoretical support for using hierarchical clustering with general distance data. They showed that hierarchical clustering methods remain mathematically valid even when working with non-standard distance measures (like genetic distances that don't follow typical geometric rules). Their work also demonstrated that hierarchical clustering can sometimes find better solutions when exploring intermediate configurations rather than always choosing the locally optimal merger at each step. This finding supports gapHACk's approach of dynamically adjusting thresholds to find meaningful gaps, rather than using fixed cutoffs. While their analysis focused on Ward's linkage and we use average linkage, the general principles about the validity and behavior of hierarchical clustering with distance data remain applicable.

## Project Structure

```
gaphack/
├── gaphack/                   # Main package code
│   ├── __init__.py
│   ├── core.py               # Core clustering algorithm
│   ├── target_clustering.py  # Target mode clustering
│   ├── decompose.py          # Decompose clustering orchestrator
│   ├── decompose_cli.py      # Decompose command-line interface
│   ├── blast_neighborhood.py # BLAST neighborhood finder
│   ├── cluster_refinement.py # Conflict resolution and refinement
│   ├── cluster_graph.py      # Cluster proximity graph
│   ├── lazy_distances.py     # Efficient distance calculations
│   ├── scoped_distances.py   # Scoped distance providers
│   ├── utils.py              # Utility functions and alignment
│   ├── cli.py                # Main gaphack CLI
│   ├── analyze.py            # Analysis functions
│   └── analyze_cli.py        # Analysis tool CLI
├── tests/                     # Unit tests
├── examples/                  # Example datasets and documentation
│   ├── data/                 # Sample FASTA files
│   └── README.md             # Examples documentation
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=gaphack
```

### Code Style

```bash
# Format code
black gaphack tests

# Check style
flake8 gaphack tests

# Type checking
mypy gaphack
```

## Citation

If you use gapHACk in your research, please cite:

```
Walker, J. (2025). gapHACk: Gap-Optimized Hierarchical Agglomerative Clustering 
for DNA barcoding. https://github.com/joshuaowalker/gaphack
```

### Key References

- Puillandre, N., Lambert, A., Brouillet, S., & Achaz, G. (2012). ABGD, Automatic Barcode Gap Discovery for primary species delimitation. *Molecular Ecology*, 21(8), 1864-1877.
- Randriamihamison, N., Vialaneix, N., & Neuvial, P. (2021). Applicability and interpretability of Ward's hierarchical agglomerative clustering with or without contiguity constraints. *Journal of Classification*, 38, 363-389.
- Russell, S. (2025). Why NCBI BLAST Identity Scores Can Mislead Fungal Identifications — And How to Improve Them. *MycotaLab Substack*. https://mycotalab.substack.com/p/why-ncbi-blast-identity-scores-can
- Wilson, A.W., Eberhardt, U., Nguyen, N., et al. (2023). Does One Size Fit All? Variations in the DNA Barcode Gaps of Macrofungal Genera. *Journal of Fungi*, 9(8), 788. https://doi.org/10.3390/jof9080788

## License

BSD 2-Clause License - see LICENSE file for details.

## Acknowledgments

- This algorithm was developed as a standalone tool for the scientific community.
- Example dataset includes *Collybia nuda* ITS sequences from public iNaturalist observations, used for testing and demonstration purposes.