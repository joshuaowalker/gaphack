# BLAST Analysis Integration Guide

## Overview

`gaphack-blast` analyzes BLAST search results to identify **conspecific sequences** - sequences that belong to the same species as your query. It uses gap-optimized clustering to find a natural boundary between "same species" and "different species" hits.

> **Terminology note**: Terms like "species" and "conspecific" are used as shorthand throughout this document. In practice, gaphack operates on sequence clusters (OTUs - Operational Taxonomic Units), not formal species definitions. The barcode gap approach identifies natural clusters in sequence space, which often correspond to species boundaries but may represent finer or coarser taxonomic units depending on the marker and organism group.

**Primary use case**: You have a query sequence and ~100 BLAST hits. Which hits are truly the same species as your query?

## Quick Start

```bash
# Basic usage - query is first sequence in FASTA
cat query.fa blast_hits.fa | gaphack-blast > results.json

# From file
gaphack-blast combined_sequences.fasta -o results.json

# Human-readable output
gaphack-blast sequences.fasta --format text
```

## Input Format

**FASTA format, query sequence first.** The first sequence is treated as the query; all others are candidate hits to classify.

```fasta
>Query_sequence
ATCGATCG...
>Hit_1_from_BLAST
ATCGATCG...
>Hit_2_from_BLAST
ATCGATCC...
```

**Typical workflow**:
1. Run BLAST search with your query
2. Download top N hits as FASTA
3. Prepend your query sequence
4. Pipe to `gaphack-blast`

## Output Formats

### JSON (default)

```bash
gaphack-blast input.fasta --format json
```

```json
{
  "query": {
    "id": "Query_7113999",
    "length": 650
  },
  "summary": {
    "total_sequences": 101,
    "query_cluster_size": 14,
    "barcode_gap_found": true,
    "gap_size_percent": 0.82,
    "medoid_id": "ON561472",
    "medoid_index": 3,
    "intra_cluster_identity": {
      "min": 98.50,
      "p5": 98.80,
      "median": 99.20,
      "p95": 99.70,
      "max": 100.00
    },
    "nearest_non_member_identity": 97.68
  },
  "sequences": [
    {
      "index": 0,
      "id": "Query_7113999",
      "in_query_cluster": true,
      "identity_to_query": 100.0,
      "identity_to_query_normalized": 100.0,
      "identity_to_medoid_normalized": 99.85,
      "identity_to_nearest_non_member_normalized": 97.52
    }
  ],
  "diagnostics": {
    "method": "gap-optimized-target-clustering",
    "min_split": 0.005,
    "max_lump": 0.02,
    "normalization_length": 650,
    "identity_metric": "MycoBLAST-adjusted (homopolymer-normalized, indel-normalized)",
    "warnings": [],
    "histograms": {
      "intra_cluster": {
        "bin_width_percent": 0.5,
        "bin_starts": [98.5, 99.0, 99.5, 100.0],
        "counts": [2, 15, 42, 12],
        "frequencies": [0.0282, 0.2113, 0.5915, 0.1690]
      },
      "inter_cluster": {
        "bin_width_percent": 0.5,
        "bin_starts": [92.0, 92.5, 93.0, 94.5, 95.0, 97.5],
        "counts": [3, 8, 2, 1, 5, 1],
        "frequencies": [0.15, 0.4, 0.1, 0.05, 0.25, 0.05]
      }
    }
  }
}
```

### TSV

```bash
gaphack-blast input.fasta --format tsv
```

Tab-separated, one row per sequence. Suitable for spreadsheets or pandas.

### Text

```bash
gaphack-blast input.fasta --format text
```

Human-readable report for debugging.

## Field Reference

### Summary Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_sequences` | int | Total input sequences including query |
| `query_cluster_size` | int | Sequences classified as conspecific (including query) |
| `barcode_gap_found` | bool | Whether a clear species boundary was detected |
| `gap_size_percent` | float | Gap magnitude in percentage points (see below) |
| `medoid_id` | string | ID of the most representative sequence in query cluster |
| `medoid_index` | int | 0-based index of the medoid |
| `intra_cluster_identity` | object | Identity distribution within the query cluster |
| `nearest_non_member_identity` | float | Highest identity of any non-member to query |

### Per-Sequence Fields

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | 0-based position in input (0 = query) |
| `id` | string | FASTA header/ID |
| `in_query_cluster` | bool | **Primary result**: Is this sequence conspecific? |
| `identity_to_query` | float | Pairwise identity % to query (comparable to BLAST) |
| `identity_to_query_normalized` | float | Normalized identity % (used by clustering algorithm) |
| `identity_to_medoid_normalized` | float | Identity % to the cluster medoid |
| `identity_to_nearest_non_member_normalized` | float | For members only: identity % to nearest non-member |

### Diagnostic Fields

| Field | Type | Description |
|-------|------|-------------|
| `method` | string | Always "gap-optimized-target-clustering" |
| `min_split` | float | Distance threshold below which sequences always merge |
| `max_lump` | float | Distance threshold above which sequences never merge |
| `normalization_length` | int | Median sequence length used for normalization |
| `identity_metric` | string | Description of identity calculation method |
| `warnings` | array | Any issues encountered during analysis |
| `histograms` | object | Identity distribution histograms (see below) |

### Histogram Fields

The `histograms` object contains pre-binned identity distributions for visualization:

| Field | Type | Description |
|-------|------|-------------|
| `intra_cluster` | object | Distribution of identities within query cluster |
| `inter_cluster` | object | Distribution of identities from cluster members to non-members |

Each histogram object contains:

| Field | Type | Description |
|-------|------|-------------|
| `bin_width_percent` | float | Width of each bin in identity percentage points (0.5%) |
| `bin_starts` | array | Starting identity % for each non-empty bin |
| `counts` | array | Raw count of values in each bin |
| `frequencies` | array | Normalized frequencies (sum to 1.0) |

**Note**: Empty bins are omitted from the output for compactness.

## Understanding the Metrics

### Identity vs Distance

All metrics are reported as **identity percentages** (higher = more similar). Internally, the algorithm uses distances (`distance = 1 - identity/100`), but output uses identity for intuitive interpretation.

### Pairwise vs Normalized Identity

| Metric | Denominator | Use Case |
|--------|-------------|----------|
| `identity_to_query` | Pairwise overlap length | Comparable to BLAST percent identity |
| `identity_to_query_normalized` | Median sequence length | Used by clustering algorithm; consistent across comparisons |

**Why two metrics?** Pairwise identity varies with overlap length (short overlaps inflate identity). Normalized identity uses a fixed denominator (median length), making distances comparable across all sequence pairs in the analysis.

### The Barcode Gap

The **barcode gap** is the separation between intra-species and inter-species identity:

```
gap_size_percent = min(intra_cluster_identity) - max(inter_cluster_identity)
                 = intra_cluster_identity.min - nearest_non_member_identity
```

- **Positive gap**: Clear species boundary exists
- **Zero/negative gap**: No clear boundary; classification is uncertain

Example:
- Intra-cluster min: 98.5% (most distant conspecific)
- Nearest non-member: 97.68% (closest non-conspecific)
- Gap: 98.5 - 97.68 = **0.82%**

### Histograms for Visualization

The `diagnostics.histograms` field provides pre-binned identity distributions for rendering barcode gap visualizations:

- **`intra_cluster`**: Pairwise identities among sequences within the query cluster (O(N²) comparisons)
- **`inter_cluster`**: Identities from each cluster member to the nearest non-member (O(N) comparisons)

**Bin structure**: Fixed 0.5% bin width. Each bin is defined by its starting identity percentage. Empty bins are omitted for compactness.

**Counts vs Frequencies**: Both raw counts and normalized frequencies are provided:
- Use **counts** when you need the actual number of comparisons
- Use **frequencies** when overlaying intra and inter distributions on the same chart (since they have different sample sizes: N² vs N)

**Visualization example**: Render both distributions as overlapping histograms with different colors. A clear barcode gap appears as separation between the two distributions.

```
Identity %    intra_cluster (blue)    inter_cluster (red)
100.0         ████████
99.5          ████████████████████
99.0          ██████
98.5          ██
98.0
97.5                                  █
...
93.0                                  ██
92.5                                  ████████████████
92.0                                  ██████
              ↑ cluster members       ↑ nearest non-members
```

### Medoid

The **medoid** is the sequence with minimum total distance to all other cluster members - the most "central" or representative sequence. Useful for:
- Selecting a representative sequence for downstream analysis
- Quality checking: if medoid is very different from query, the cluster may be problematic

### Cluster Membership Decision

A sequence is classified as `in_query_cluster: true` if:
1. Its **maximum** distance to any cluster member is within threshold (complete linkage)
2. Adding it doesn't destroy the barcode gap

The algorithm explores all possible merges up to `max_lump` and returns the configuration with the best gap.

## Algorithm Parameters

| Parameter | Default | CLI Flag | Effect |
|-----------|---------|----------|--------|
| `min_split` | 0.005 (99.5%) | `--min-split` | Sequences within this distance always merge |
| `max_lump` | 0.02 (98.0%) | `--max-lump` | Sequences beyond this distance never merge |
| `target_percentile` | 100 | (not exposed) | Uses complete linkage (max distance) |

**Tuning guidance**:
- Decrease `min_split` for more conservative clustering (fewer false positives)
- Increase `max_lump` to consider more distant sequences as potential conspecifics

## Integration Patterns

### Web Service Integration

```python
import subprocess
import json

def analyze_blast_results(query_fasta: str, hits_fasta: str) -> dict:
    """Analyze BLAST results and return structured classification."""
    combined = query_fasta + "\n" + hits_fasta

    result = subprocess.run(
        ["gaphack-blast", "--format", "json"],
        input=combined,
        capture_output=True,
        text=True,
        timeout=60  # Typical analysis takes <10s for 100 sequences
    )

    if result.returncode != 0:
        raise RuntimeError(f"gaphack-blast failed: {result.stderr}")

    return json.loads(result.stdout)
```

### Extracting Conspecific Sequences

```python
def get_conspecific_ids(result: dict) -> list[str]:
    """Extract IDs of sequences in the query cluster."""
    return [
        seq["id"]
        for seq in result["sequences"]
        if seq["in_query_cluster"]
    ]
```

### Checking for Clear Species Boundary

```python
def has_clear_boundary(result: dict, min_gap: float = 0.5) -> bool:
    """Check if there's a clear barcode gap."""
    summary = result["summary"]
    return (
        summary["barcode_gap_found"] and
        summary["gap_size_percent"] is not None and
        summary["gap_size_percent"] >= min_gap
    )
```

### Rendering Barcode Gap Visualization

```python
def get_histogram_data(result: dict) -> tuple[dict, dict]:
    """Extract histogram data for visualization."""
    histograms = result["diagnostics"]["histograms"]
    return histograms["intra_cluster"], histograms["inter_cluster"]

def render_histogram_chart(result: dict):
    """Example: render overlapping histograms with matplotlib."""
    import matplotlib.pyplot as plt

    intra, inter = get_histogram_data(result)

    # Use frequencies for normalized comparison
    fig, ax = plt.subplots()

    # Intra-cluster (blue)
    ax.bar(intra["bin_starts"], intra["frequencies"],
           width=0.4, alpha=0.7, label="Intra-cluster", color="blue")

    # Inter-cluster (red)
    ax.bar(inter["bin_starts"], inter["frequencies"],
           width=0.4, alpha=0.7, label="Inter-cluster", color="red")

    ax.set_xlabel("Identity %")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Mark the gap region
    gap_start = result["summary"]["nearest_non_member_identity"]
    gap_end = result["summary"]["intra_cluster_identity"]["min"]
    if gap_start < gap_end:
        ax.axvspan(gap_start, gap_end, alpha=0.2, color="green", label="Gap")

    return fig
```

### Handling Edge Cases

```python
def interpret_result(result: dict) -> str:
    """Generate human-readable interpretation."""
    summary = result["summary"]
    warnings = result["diagnostics"]["warnings"]

    if warnings:
        return f"Analysis had issues: {'; '.join(warnings)}"

    if summary["query_cluster_size"] == 1:
        return "Query has no close matches in the database"

    if not summary["barcode_gap_found"]:
        return "No clear species boundary - all sequences may be conspecific or taxonomy is unclear"

    gap = summary["gap_size_percent"]
    n = summary["query_cluster_size"]
    return f"Found {n} conspecific sequences with {gap:.1f}% barcode gap"
```

## Concurrency

`gaphack-blast` is single-threaded by design. For web services, run multiple instances in parallel rather than parallelizing within a single call.

## Common Warnings

| Warning | Meaning | Action |
|---------|---------|--------|
| "Query has no close matches" | Query alone in cluster | Check query quality; may be novel species |
| "N sequences had insufficient overlap" | Short/divergent sequences | May need to filter input |
| "MSA alignment failed" | SPOA couldn't align sequences | Check for chimeras or non-homologous sequences |

## Comparison with BLAST Identity

`gaphack-blast` identity differs from raw BLAST percent identity:

| Aspect | BLAST | gaphack-blast |
|--------|-------|---------------|
| Alignment | Pairwise local | Multiple sequence (SPOA) |
| Gap handling | Affine gap penalty | Homopolymer normalization |
| Normalization | Alignment length | Median sequence length |

**Practical implication**: gaphack-blast identities are typically 0.5-2% lower than BLAST due to stricter gap handling. The clustering algorithm was trained on these adjusted identities.

## Logging

Logs go to stderr; output goes to stdout. Control verbosity:

```bash
# Quiet mode - only warnings/errors
gaphack-blast input.fasta -q > results.json

# Verbose mode - debug information
gaphack-blast input.fasta -v > results.json 2> debug.log
```
