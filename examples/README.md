# gapHACk Examples

This directory contains example datasets and results for testing and demonstrating the gapHACk clustering algorithm.

## Example Datasets

### collybia_nuda_test.fasta

A curated dataset of 91 ITS (Internal Transcribed Spacer) sequences from *Collybia nuda* (Wood Blewit) and related species, obtained from public iNaturalist observations.

**Dataset characteristics:**
- 91 sequences total
- Strong clustering structure with clear species boundaries
- Contains IUPAC ambiguous nucleotide codes (R, K, Y)
- Excellent test case for barcode gap optimization

**Expected results:**
- 3 main clusters
- 5 singletons
- Barcode gap ~1.58% at P95 percentile

## Running Examples

### Basic Usage

Run gapHACk on the example dataset using the default adjusted-identity alignment:

```bash
# Creates separate FASTA files for each cluster (default)
gaphack examples/data/collybia_nuda_test.fasta

# Or output to TSV format
gaphack examples/data/collybia_nuda_test.fasta --format tsv -o output.tsv
```

### With Detailed Metrics

Export detailed gap analysis metrics:

```bash
gaphack examples/data/collybia_nuda_test.fasta \
    -o results/clusters \
    --export-metrics metrics.json \
    --verbose
```

This will create:
- `results/clusters.cluster_001.fasta`
- `results/clusters.cluster_002.fasta`
- `results/clusters.cluster_003.fasta`
- `results/clusters.singletons.fasta`
- `metrics.json` with gap analysis

### Custom Parameters

Experiment with different clustering parameters:

```bash
# More conservative clustering (larger gap required)
gaphack examples/data/collybia_nuda_test.fasta \
    --min-split 0.01 \
    --max-lump 0.03 \
    -o conservative_clusters

# Target different percentile for gap optimization
gaphack examples/data/collybia_nuda_test.fasta \
    --target-percentile 90 \
    -o p90_clusters

# Output as TSV for analysis
gaphack examples/data/collybia_nuda_test.fasta \
    --target-percentile 90 \
    --format tsv \
    -o p90_results.tsv
```

## Output Files

When running gapHACk in FASTA format (default), the output files will be:

- `basename.cluster_001.fasta`: First cluster sequences
- `basename.cluster_002.fasta`: Second cluster sequences
- `basename.cluster_NNN.fasta`: Additional clusters (zero-padded)
- `basename.singletons.fasta`: Unclustered sequences

Cluster numbers are zero-padded (e.g., 001, 002, 003) to ensure proper sorting in file listings.

## Data Attribution

The *Collybia nuda* sequences are from public iNaturalist observations and are used here for educational and testing purposes under fair use principles. Each sequence ID includes the iNaturalist observation number for reference.

## Adding Your Own Examples

To test gapHACk with your own data:

1. Ensure your sequences are in FASTA format
2. DNA sequences should contain only standard nucleotides (ATCG) and IUPAC ambiguous codes
3. Sequence headers should be unique identifiers
4. For best results, include sequences from multiple related species to test gap detection