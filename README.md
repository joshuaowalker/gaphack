# gapHACk - Gap-Optimized Hierarchical Agglomerative Clustering

A Python package for DNA barcode clustering that optimizes for the barcode gap between intra-species and inter-species genetic distances.

## Overview

gapHACk implements a two-phase clustering algorithm designed for DNA barcoding applications:

1. **Phase 1**: Fast greedy lumping below the min-split threshold (default 0.5% distance)
2. **Phase 2**: Gap-aware optimization using gap-based heuristic to find optimal clustering

The algorithm focuses on maximizing the "barcode gap" - the separation between intra-species and inter-species distances at a specified percentile (default P95) to handle outliers robustly.

**gapHACk provides three complementary tools**:
- **`gaphack`**: Core clustering for medium-sized datasets (up to ~1,000 sequences)
- **`gaphack-refine`**: Iterative refinement to optimize cluster boundaries
- **`gaphack-analyze`**: Quality assessment for pre-clustered FASTA files

## Clustering Workflows

gapHACk supports two main workflows depending on your use case:

### Workflow 1: Large-Scale Clustering

For datasets with 1,000+ sequences, combine fast external clustering with gap-based refinement:

```bash
# Step 1: Fast initial clustering with vsearch (or CD-HIT, MMseqs2, etc.)
vsearch --cluster_fast input.fasta --id 0.97 --clusters cluster_

# Step 2: Refine cluster boundaries with gapHACk
gaphack-refine --input-dir clusters/ \
               --output-dir refined/ \
               --close-threshold 0.02

# Step 3: Assess quality
gaphack-analyze refined/latest/*.fasta -o analysis/
```

This workflow leverages fast external tools for initial approximate clustering, then applies gapHACk's gap optimization to refine boundaries between closely related groups.

### Workflow 2: Focused Expert Investigation

For targeted analysis or moderate-sized datasets:

```bash
# Analyze existing clusters to understand distance distributions
gaphack-analyze existing_clusters/*.fasta -o analysis/

# Extract and cluster sequences related to specific targets
gaphack full_dataset.fasta --target reference_seqs.fasta -o target_cluster

# Full gap-optimized clustering on focused dataset
gaphack focused_dataset.fasta -o clusters/
```

This workflow is useful for taxonomic investigations, quality control, or detailed analysis of specific groups.

## Features

### Core Clustering (`gaphack`)
- **Gap Optimization**: Uses gap-based heuristic to directly maximize the barcode gap
- **MSA-based Distances**: Uses SPOA multiple sequence alignment for consistent distance calculations across all sequence pairs
- **Target Mode**: Single-cluster focused clustering from seed sequences with `--target` parameter
- **Percentile-based Linkage**: Uses percentile-based complete linkage (default 95th percentile) for robust merge decisions
- **Two-phase Algorithm**: Fast initial clustering followed by gap-aware optimization to completion
- **Multiprocessing**: Automatic parallelization using all available CPU cores (configurable with `-t` flag)
- **Progress Tracking**: Real-time progress bar with gap and cluster information
- **Size-ordered Output**: Clusters numbered by size (largest first) for consistent results

### Iterative Refinement (`gaphack-refine`)
- **Neighborhood-based Refinement**: Iteratively refines cluster boundaries using proximity graphs
- **MSA-based Distances**: Consistent alignment for all sequences within each refinement scope
- **Convergence Detection**: AMI-based convergence tracking to detect stable clustering states
- **Iteration Checkpointing**: Saves state every N iterations for long-running jobs
- **Auto-resume**: Automatically detects and resumes from saved checkpoints
- **Global Gap Metrics**: Tracks barcode gap quality across iterations for progress monitoring
- **Deterministic Ordering**: Priority-based seed selection ensures reproducible refinement
- **Flexible Configuration**: Control scope size, iteration limits, and convergence thresholds

### Analysis Tools (`gaphack-analyze`)
- **Pre-clustered Analysis**: Evaluate existing clustering results
- **Distance Distributions**: Calculate intra-cluster and inter-cluster distances
- **Barcode Gap Metrics**: Assess gap quality at P90/P95 levels
- **Visualization**: Generate histograms with percentile markers
- **Multiple Output Formats**: Text reports, JSON data, or TSV tables

## Performance

### Multiprocessing Capabilities

gapHACk includes multiprocessing support for the gap-aware clustering phase:

- **Automatic Parallelization**: Uses all available CPU cores by default
- **Single-process Mode**: Use `-t 0` for single-threaded operation (ideal for library usage)
- **Custom Thread Count**: Use `-t N` to specify number of worker processes
- **Scalable Performance**: Achieves ~4x speedup with 8 cores (200→800 steps/second)
- **Memory Efficient**: Persistent worker caches reduce initialization overhead

```bash
# Use all available cores (default)
gaphack input.fasta

# Use 4 worker processes
gaphack input.fasta -t 4

# Single-threaded mode (for library usage or debugging)
gaphack input.fasta -t 0
```

### Performance Characteristics

- **Small datasets** (< 100 sequences): Single-process mode may be faster due to reduced overhead
- **Medium datasets** (100-1000 sequences): Multiprocessing provides significant speedup
- **Large datasets** (> 1000 sequences): Use vsearch + gaphack-refine workflow for best performance

The gap-aware clustering phase scales with O(n³) complexity, making the two-stage workflow (fast clustering + refinement) essential for large datasets.

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

# With detailed output and custom parameters
gaphack collybia_nuda_test.fasta \
    -o output/clusters \
    --export-metrics metrics.json \
    --verbose
```

The example dataset contains 91 *Collybia nuda* ITS sequences from iNaturalist with known clustering structure. See [examples/README.md](examples/README.md) for more details.

For larger datasets, use the two-stage workflow:

```bash
# Stage 1: Fast initial clustering with vsearch
vsearch --cluster_fast large_dataset.fasta --id 0.97 --clusters cluster_

# Stage 2: Gap-optimized refinement
gaphack-refine --input-dir clusters/ --output-dir refined/ --close-threshold 0.02

# Stage 3: Quality assessment
gaphack-analyze refined/latest/*.fasta -o analysis/
```

## Usage

### Core Clustering (`gaphack`)

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
# With custom thresholds and multi-threading
gaphack input.fasta \
    --min-split 0.003 \
    --max-lump 0.03 \
    --target-percentile 90 \
    --export-metrics gap_analysis.json \
    --threads 8 \
    --verbose
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

### Iterative Refinement (`gaphack-refine`)

The `gaphack-refine` tool optimizes cluster boundaries through iterative neighborhood-based refinement. It works with any input clustering (vsearch, CD-HIT, MMseqs2, or gaphack output).

#### Basic Usage

```bash
# Refine existing clusters
gaphack-refine --input-dir clusters/ \
               --output-dir refined/ \
               --close-threshold 0.02
```

**Input Format**: Directory containing one FASTA file per cluster (e.g., `cluster_001.fasta`, `cluster_002.fasta`, etc.)

**Output**: Timestamped directory with refined clusters and detailed summary report

#### Key Features

##### MSA-based Distance Calculation

gaphack-refine uses SPOA (Partial Order Alignment) to create a multiple sequence alignment for each refinement scope. This provides:
- **Consistent distances**: All sequences within a scope share the same alignment context
- **Biological accuracy**: Alignment-based distances better reflect evolutionary relationships
- **Computational efficiency**: One MSA + fast scoring vs. many pairwise alignments

##### Iteration Checkpointing and Auto-Resume

For long-running refinement jobs, checkpointing allows you to save progress and resume later:

```bash
# Enable checkpointing (save state every iteration)
gaphack-refine --input-dir clusters/ \
               --output-dir refined/ \
               --close-threshold 0.02 \
               --checkpoint-frequency 1

# Auto-resume from saved checkpoint
# (automatically detects state.json in input directory)
gaphack-refine --input-dir refined/latest/ \
               --output-dir refined_continued/ \
               --close-threshold 0.02
```

Checkpoints include:
- Current cluster assignments
- Iteration state and convergence tracking
- Refinement statistics and timing
- FASTA files for each iteration

##### Convergence Tracking

The tool uses multiple convergence indicators:
- **AMI (Adjusted Mutual Information)**: Measures clustering stability between iterations (0-1 scale)
- **Global gap metrics**: Tracks barcode gap quality across all clusters
- **Per-cluster convergence**: Detects when individual neighborhoods stabilize

Refinement continues until AMI ≈ 1.0 (no changes) or maximum iterations reached.

#### Advanced Options

```bash
# Custom convergence parameters
gaphack-refine --input-dir clusters/ \
               --output-dir refined/ \
               --close-threshold 0.02 \
               --max-iterations 20 \
               --max-scope-size 500

# Deterministic seed ordering for reproducibility
gaphack-refine --input-dir clusters/ \
               --output-dir refined/ \
               --close-threshold 0.02 \
               --random-seed 42

# Use vsearch instead of BLAST for proximity graph
gaphack-refine --input-dir clusters/ \
               --output-dir refined/ \
               --close-threshold 0.02 \
               --search-method vsearch
```

#### Output Organization

```
refined/
├── 20251013_143022/              # Timestamped results
│   ├── cluster_00001.fasta       # Refined clusters (by size)
│   ├── cluster_00002.fasta
│   ├── ...
│   ├── cluster_mapping.txt       # Original → final cluster ID mapping
│   └── refine_summary.txt        # Detailed summary report
├── latest -> 20251013_143022/    # Symlink to most recent
└── state.json                     # Checkpoint state (if checkpointing enabled)
```

The summary report includes:
- Iteration statistics and convergence metrics
- Global gap metrics at each iteration
- Cluster count changes and AMI scores
- Per-iteration timing breakdown

### Analysis Tool (`gaphack-analyze`)

The `gaphack-analyze` tool evaluates pre-clustered FASTA files to assess distance distributions and barcode gap quality:

```bash
# Analyze pre-clustered files (each FASTA = one cluster)
gaphack-analyze cluster1.fasta cluster2.fasta cluster3.fasta

# Save results to custom directory with JSON format
gaphack-analyze *.fasta -o analysis_results --format json

# Skip plots and use TSV output
gaphack-analyze clusters/*.fasta --no-plots --format tsv -o results.tsv
```

**Analysis Output:**
- **Individual cluster histograms**: Distance distributions within each cluster
- **Global distance histogram**: Combined intra-cluster vs inter-cluster distances
- **Percentile analysis**: P5, P25, P50, P75, P95 values for all distance sets
- **Barcode gap metrics**: Gap size and existence at P90/P95 levels
- **Multiple formats**: Text reports, JSON data, or TSV tables

### Python API

#### Basic Usage

```python
from gaphack import GapOptimizedClustering
from gaphack import load_sequences_from_fasta, calculate_distance_matrix

# Load sequences
sequences, headers, _ = load_sequences_from_fasta("input.fasta")

# Calculate distance matrix using MSA-based approach
# Uses SPOA for multiple sequence alignment with MycoBLAST-style adjustments
distance_matrix = calculate_distance_matrix(sequences)

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

## Distance Calculation

gapHACk uses the `adjusted-identity` package with standardized MycoBLAST-style adjustment parameters:

- **Homopolymer normalization**: Enabled - differences in homopolymer run lengths (e.g., AAA vs AAAA) are not counted as mismatches
- **IUPAC overlap handling**: Enabled - ambiguity codes treated as matches when they overlap (e.g., R matches A or G)
- **Indel normalization**: Enabled - contiguous indels counted as single evolutionary events
- **End skip distance**: 0 bases - no terminal region trimming
- **Repeat motif detection**: Disabled (max length 0) - only homopolymers are normalized

These parameters are hardcoded based on empirical validation with fungal ITS sequences and cannot be changed via CLI parameters

### MSA-based Distance Calculation

For multi-sequence operations (core gaphack, gaphack-refine), distances are calculated using SPOA multiple sequence alignment:

- **Consistency**: All sequence pairs within an alignment share the same gap placement
- **Biological relevance**: MSA-based distances better reflect evolutionary relationships than independent pairwise alignments
- **Performance**: One MSA (0.05s for 100 sequences) + fast scoring (0.1ms/pair) vs. many pairwise alignments (1ms/pair)
- **Graceful fallback**: If SPOA fails, automatically falls back to pairwise alignment

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

### Refinement Parameters (gaphack-refine)

- `close_threshold` (required): Distance threshold for finding nearby clusters during refinement
- `max_iterations` (default: 10): Maximum refinement iterations before stopping
- `max_scope_size` (default: 300): Maximum sequences for full gapHACk refinement within a single scope
- `checkpoint_frequency` (default: 0): Checkpoint every N iterations (0=disabled)
- `knn_neighbors` (default: 20): K for K-NN cluster proximity graph
- `search_method` (default: "blast"): Search method for proximity graph ("blast" or "vsearch")
- `random_seed` (default: None): Random seed for reproducibility (None = deterministic based on reclustering counts)

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

gapHACk begins by calculating a pairwise distance matrix from input sequences. The default approach uses SPOA multiple sequence alignment followed by pairwise distance scoring:

1. **Multiple Sequence Alignment**: SPOA creates a shared alignment space for all sequences
2. **Distance Scoring**: Pairwise distances calculated from the MSA using adjusted identity
3. **Fallback**: If MSA fails, falls back to independent pairwise alignments

This MSA-based approach provides more consistent and biologically relevant distances than independent pairwise alignments.

**Distance calculation:**
gapHACk uses MSA-based distance calculation with MycoBLAST-style adjusted identity parameters. This applies corrections for sequencing artifacts (homopolymer runs), ambiguous bases (IUPAC codes), and indel events to produce cleaner distance estimates that better reflect biological relationships.

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

### Iterative Refinement (gaphack-refine)

The gaphack-refine tool optimizes cluster boundaries through iterative neighborhood-based refinement. This approach is designed to work with any initial clustering, whether from vsearch, CD-HIT, MMseqs2, or gaphack itself.

#### Refinement Process

1. **Build Proximity Graph**: Create K-NN graph of cluster medoids using BLAST or vsearch
2. **Select Seed Clusters**: Process each cluster as a seed (deterministic priority order based on per-sequence reclustering counts)
3. **Build Refinement Scope**: For each seed:
   - Find neighbor clusters within `close_threshold`
   - Add context clusters beyond `close_threshold` (up to `max_scope_size` total sequences)
4. **Apply Full gapHACk**: Run complete gap-optimized clustering on the scope
5. **Update Clusters**: Replace input clusters with refined result if changed
6. **Check Convergence**: Repeat iterations until AMI ≈ 1.0 or maximum iterations reached

#### MSA-based Distance Calculation

Each refinement scope uses SPOA to create a multiple sequence alignment for all sequences in the scope (seed + neighbors + context). This provides:
- Consistent distances within the refinement scope
- Biologically meaningful gap calculations
- Efficient computation (one MSA per scope, not per sequence pair)

#### Convergence Criteria

- **AMI = 1.0**: Perfect agreement between input and output (no changes)
- **Global Gap Metrics**: Track barcode gap quality across all clusters using K-NN (K=3) approach
- **Per-scope Convergence**: Individual neighborhoods marked as converged when unchanged
- **Iteration Limit**: Default maximum of 10 iterations

#### Deterministic Seed Prioritization

Seeds are processed in priority order based on minimum per-sequence reclustering count:
- Clusters with sequences that have been reclustered fewer times are processed first
- Ensures fair coverage and prevents bias toward frequently-processed clusters
- Provides deterministic, reproducible refinement when no random seed is specified

## Related Work

### Prior Art in Barcode Gap Discovery

gapHACk shares conceptual similarities with ABGD (Automatic Barcode Gap Discovery; Puillandre et al., 2012), which also automatically identifies the barcode gap for species delimitation. Both methods:
- Seek to identify the threshold between intraspecific and interspecific genetic variation
- Use pairwise distance matrices as input
- Apply recursive partitioning to handle heterogeneity across taxa

Key differences in gapHACk's approach:
- **Clustering algorithm**: gapHACk uses hierarchical agglomerative clustering with dynamic gap optimization, while ABGD uses graph-based partitioning with fixed thresholds
- **Gap-based optimization**: gapHACk uses a gap-based heuristic that directly optimizes for the barcode gap at each merge, rather than recursive application of fixed thresholds
- **Distance calculation**: gapHACk implements the adjusted identity algorithm (Russell, 2025), which systematically corrects for sequencing artifacts and biological complexity that can obscure true genetic distances
- **Percentile-based robustness**: gapHACk uses percentile gaps (e.g., P95) to handle outliers, rather than absolute min/max values
- **MSA-based distances**: For refinement operations, gapHACk uses multiple sequence alignment to ensure consistent distance calculations

### Empirical Support for Dynamic Thresholds

Wilson et al. (2023) demonstrated substantial variation in barcode gaps across 11 macrofungal genera, with the middle of barcode gaps ranging from <2% to nearly 6%. Their findings validate gapHACk's approach of dynamically optimizing thresholds within user-specified ranges (default: 0.5-2% distance) rather than relying on fixed universal cutoffs. The configurable min_threshold and max_threshold parameters allow users to adjust these ranges based on their knowledge of the taxonomic group being studied.

### Adjusted Identity Algorithm

The adjusted identity algorithm (Russell, 2025) that gapHACk implements addresses critical issues with raw NCBI BLAST identity scores. The algorithm makes several key corrections:
- **Homopolymer normalization**: Differences in homopolymer run lengths (e.g., AAA vs AAAA) are not counted as mismatches, addressing common sequencing artifacts
- **Ambiguity code handling**: IUPAC codes are treated as matches when they overlap (e.g., R matches A or G)
- **End trimming**: Mismatches within the first and last 20 bases are excluded, as these regions often contain editing artifacts
- **Gap event counting**: Indels are counted as single evolutionary events rather than per-base penalties

These adjustments can substantially improve identity scores, resulting in clearer barcode gaps and more reliable species delimitation. This is particularly important for fungal ITS sequences where sequencing artifacts are common.

For implementation details, see: https://github.com/joshuaowalker/adjusted-identity

### Theoretical Foundations

Randriamihamison et al. (2021) provided theoretical support for using hierarchical clustering with general distance data. They showed that hierarchical clustering methods remain mathematically valid even when working with non-standard distance measures. This supports gapHACk's approach of dynamically adjusting thresholds to find meaningful gaps, rather than using fixed cutoffs.

## Project Structure

```
gaphack/
├── gaphack/                     # Main package code
│   ├── __init__.py
│   ├── core.py                 # Core clustering algorithm
│   ├── target_clustering.py    # Target mode clustering
│   ├── cluster_refinement.py   # Iterative refinement
│   ├── refinement_types.py     # Refinement tracking types
│   ├── cluster_graph.py        # Cluster proximity graph
│   ├── blast_neighborhood.py   # BLAST neighborhood finder
│   ├── distance_providers.py   # Distance calculation providers
│   ├── utils.py                # Utility functions and alignment
│   ├── cli.py                  # Main gaphack CLI
│   ├── refine_cli.py           # Refinement CLI
│   ├── analyze.py              # Analysis functions
│   └── analyze_cli.py          # Analysis tool CLI
├── tests/                       # Unit tests
├── examples/                    # Example datasets and documentation
│   ├── data/                   # Sample FASTA files
│   └── README.md               # Examples documentation
├── pyproject.toml              # Package configuration
└── README.md                   # This file
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

This tool was developed to provide practical, empirically validated approaches for DNA barcode clustering. The combination of gap-based optimization, MSA-based distances, and iterative refinement has proven effective for fungal ITS sequence datasets in our use cases. Example dataset includes *Collybia nuda* ITS sequences from public iNaturalist observations, used for testing and demonstration purposes.
