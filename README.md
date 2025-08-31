# gapHACk - Gap-Optimized Hierarchical Agglomerative Clustering

A Python package for DNA barcode clustering that optimizes for the barcode gap between intra-species and inter-species genetic distances.

## Overview

gapHACk implements a two-phase clustering algorithm designed specifically for DNA barcoding applications:

1. **Phase 1**: Fast greedy lumping below the min-split threshold (default 0.5% distance)  
2. **Phase 2**: Gap-aware optimization using gap-based heuristic to find optimal clustering

The algorithm focuses on maximizing the "barcode gap" - the separation between intra-species and inter-species distances at a specified percentile (default P95) to handle outliers robustly.

## Features

- **Gap Optimization**: Uses gap-based heuristic to directly maximize the barcode gap
- **Percentile-based Linkage**: Uses percentile-based complete linkage (default 95th percentile) for robust merge decisions
- **Two-phase Algorithm**: Fast initial clustering followed by gap-aware optimization to completion
- **Progress Tracking**: Unified progress bar with real-time gap and cluster information
- **Size-ordered Output**: Clusters numbered by size (largest first) for consistent results
- **Library-friendly**: Configurable progress bars and logging for integration into other applications

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

# Run on example data (creates .cluster_001.fasta, .cluster_002.fasta, etc.)
gaphack collybia_nuda_test.fasta

# With detailed output and custom base path
gaphack collybia_nuda_test.fasta \
    -o output/clusters \
    --export-metrics metrics.json \
    --verbose

# Output TSV format instead of FASTA
gaphack collybia_nuda_test.fasta \
    --format tsv \
    -o clusters.tsv
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
  - `basename.singletons.fasta` for unclustered sequences  
  - Clusters ordered by size (001 = largest, 002 = second largest, etc.)
  - Cluster numbers are zero-padded for proper sorting
- **TSV format**: Tab-separated values with columns `sequence_id` and `cluster_id`
- **Text format**: Human-readable clustering report

## Algorithm Details

**Note**: Throughout this documentation, "species" refers to Operational Taxonomic Units (OTUs) - clusters of sequences that are presumed to represent biological species based on genetic similarity, but have not been formally taxonomically validated.

### Distance Matrix Calculation

gapHACk begins by calculating a pairwise distance matrix from input sequences using the adjusted identity algorithm (default) or traditional BLAST identity:

- **Adjusted identity** (recommended): Applies corrections for sequencing artifacts, ambiguous bases, and terminal gaps to produce cleaner distance estimates that better reflect biological relationships
- **Traditional identity**: Uses raw BLAST-style identity scores without corrections

The distance matrix forms the foundation for all subsequent clustering decisions.

### Barcode Gap

The barcode gap is the separation between the maximum intra-species distance and the minimum inter-species distance. A clear gap indicates good species delimitation, allowing confident assignment of sequences to species clusters.

### Percentile Gaps

Instead of using absolute max/min values (which are sensitive to outliers), gapHACk uses percentile-based gaps for robustness:
- **P95 gap**: 95th percentile of intra-species distances vs 5th percentile of inter-species distances
- **P90 gap**: 90th percentile of intra-species distances vs 10th percentile of inter-species distances

This approach compares the "worst case" within species (upper percentile of intra-species distances) against the "best case" between species (lower percentile of inter-species distances), creating a conservative measure of separation that is robust to outliers.

The `target_percentile` parameter (default: 95) determines which percentile gap to optimize during clustering.

### Two-Phase Clustering Strategy

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

### Parameter Roles

- **`min_split`**: Defines the boundary between Phase 1 (fast lumping) and Phase 2 (gap optimization). Sequences closer than this are assumed to represent intraspecific variation and are lumped together.

- **`max_lump`**: Upper limit for cluster lumping. Distances beyond this are assumed to represent interspecific divergence and clusters are kept split.

- **`target_percentile`**: Which percentile to use for gap optimization and linkage decisions (e.g., 95 = P95 gap). Higher percentiles are more robust to outliers but may be less sensitive to true gaps.

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
├── gaphack/              # Main package code
│   ├── __init__.py
│   ├── core.py          # Core clustering algorithm
│   ├── utils.py         # Utility functions and alignment
│   └── cli.py           # Command-line interface
├── tests/               # Unit tests
├── examples/            # Example datasets and documentation
│   ├── data/           # Sample FASTA files
│   └── README.md       # Examples documentation
├── pyproject.toml       # Package configuration
└── README.md           # This file
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