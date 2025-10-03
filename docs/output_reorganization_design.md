# Output File Organization Redesign

## Overview

This document describes a simplification of the gaphack-decompose output file organization and CLI workflow. The redesign eliminates redundant files, simplifies user interaction, and provides clearer separation between working files and final output.

## Goals

1. **Clear separation** between working files (for resume/traceability) and user-facing output
2. **Automatic output** generation on completion (eliminate manual finalization step)
3. **Self-contained stages** - each stage directory contains complete cluster set
4. **Globally unique cluster IDs** for unambiguous tracking and reporting
5. **Simplified cleanup** - single directory removal instead of pattern-based file deletion
6. **Version preservation** - never overwrite previous results

## Current Problems

1. **Multiple output formats**: `initial.cluster_*`, `deconflicted.cluster_*`, `refined.cluster_*`, `decompose.cluster_*`, `cluster_*.fasta`
2. **Manual finalization**: Requires separate `--finalize` step to create final output
3. **Partial stage results**: Deconflicted/refined stages only contain modified clusters, requiring union with previous stages
4. **File obsolescence tracking**: Need tombstones or renaming to indicate superseded files
5. **Redundant cleanup**: `--cleanup` flag removes specific patterns
6. **Ambiguous cluster IDs**: Stage-local IDs require context to interpret in reports

## Proposed Directory Structure

```
output_dir/
  ├── state.json                      # Run state, parameters, stage tracking
  ├── work/                           # Working files (for resume/debugging)
  │   ├── initial/
  │   │   ├── cluster_00001I.fasta    # Globally unique IDs with stage suffix
  │   │   ├── cluster_00002I.fasta
  │   │   ├── cluster_00143I.fasta
  │   │   ├── unassigned.fasta
  │   │   └── .blast/                 # Search databases
  │   │       └── ...
  │   ├── deconflicted/               # Only created if --resolve-conflicts used
  │   │   ├── cluster_00001C.fasta    # 'C' suffix for conflict resolution
  │   │   ├── cluster_00002C.fasta
  │   │   ├── cluster_00138C.fasta
  │   │   └── unassigned.fasta
  │   └── refined_0.010/              # Threshold in directory name
  │       ├── cluster_00001R1.fasta   # 'R1' suffix for first refinement
  │       ├── cluster_00002R1.fasta
  │       ├── cluster_00135R1.fasta
  │       └── unassigned.fasta
  └── clusters/                       # Final output (user-facing)
      ├── 2025-01-15_143022/          # Timestamp: YYYY-MM-DD_HHMMSS
      │   ├── cluster_001.fasta       # Size-sorted (largest first)
      │   ├── cluster_002.fasta
      │   ├── cluster_138.fasta
      │   └── unassigned.fasta
      ├── 2025-01-15_150633/          # Second run with different parameters
      │   ├── cluster_001.fasta
      │   └── ...
      └── latest -> 2025-01-15_150633/  # Symlink to most recent
```

## Key Design Decisions

### 1. Complete Clusters Per Stage

**Each stage directory contains ALL clusters**, not just modified ones.

**Benefits:**
- Self-contained snapshots
- No union operations needed
- Easy to compare stages
- Simple to resume from any stage
- Clear audit trail

**Example:**
- `work/initial/`: 143 clusters (`cluster_00001I` through `cluster_00143I`)
- `work/deconflicted/`: 138 clusters (`cluster_00001C` through `cluster_00138C`)
- `work/refined_0.010/`: 135 clusters (`cluster_00001R1` through `cluster_00135R1`)

### 2. Globally Unique Cluster IDs with Stage Suffixes

Cluster IDs are unique across all stages using a compact suffix notation:

- **Initial clustering**: `cluster_00001I.fasta`, `cluster_00002I.fasta`, etc.
- **Conflict resolution**: `cluster_00001C.fasta`, `cluster_00002C.fasta`, etc.
- **Refinement (1st)**: `cluster_00001R1.fasta`, `cluster_00002R1.fasta`, etc.
- **Refinement (2nd)**: `cluster_00001R2.fasta`, `cluster_00002R2.fasta`, etc.
- **Refinement (3rd)**: `cluster_00001R3.fasta`, etc.

**Stage Suffix Codes:**
- `I` - Initial clustering
- `C` - Conflict resolution (deconflicted)
- `R1`, `R2`, `R3`, ... - Refinement iterations (numbered sequentially)

**ID Format:**
- Pattern: `cluster_{NNNNN}{SUFFIX}.fasta`
- Number: 5-digit zero-padded (supports up to 99,999 clusters)
- Suffix: Single letter or letter+digit for stage identification
- Example: `cluster_00042I.fasta`, `cluster_00023C.fasta`, `cluster_00087R1.fasta`

**ID Assignment:**
- On resume, scan existing `work/` directories to find max cluster number
- Track refinement count from existing `refined_*` directories
- Next cluster starts at max + 1 within each stage
- No counter in state.json needed

**Benefits:**
- Compact format: 5 digits handles 99,999 clusters (far exceeding typical 15K cluster datasets)
- Self-documenting: suffix indicates stage at a glance
- Globally unique: no ambiguity across stages
- Scalable: refinement counter allows unlimited refinement iterations
- Clean in reports: `cluster_00042R1` is concise and clear
- Easy to grep/search: `grep "cluster_00042" finds all related clusters across stages

### 3. Timestamped Output Directories

Final output uses timestamp-based directories with a `latest` symlink:

**Benefits:**
- Never overwrites previous results
- Can compare different refinement parameters
- Natural audit trail
- No confirmation prompts needed
- `latest/` provides convenient access to current output

**Timestamp format**: `YYYY-MM-DD_HHMMSS` (e.g., `2025-01-15_143022`)

### 4. Directory-Based Stage Organization

Use subdirectories instead of filename prefixes:

**Before:**
```
output_dir/
  ├── initial.cluster_0.fasta
  ├── initial.cluster_1.fasta
  ├── deconflicted.cluster_0.fasta
  └── refined.cluster_0.fasta
```

**After:**
```
output_dir/work/
  ├── initial/cluster_00001I.fasta
  ├── deconflicted/cluster_00001C.fasta
  └── refined_0.010/cluster_00001R1.fasta
```

**Benefits:**
- Clear visual separation
- Parameters in directory names
- Simple cleanup (remove directory)
- Consistent file naming
- No prefix redundancy

### 5. Automatic Output Generation

The `clusters/` directory is **automatically updated** whenever a run completes:

- Initial run completion → creates `clusters/{timestamp}/`
- Resume completion → creates new `clusters/{timestamp}/`
- `latest` symlink always points to most recent

**Eliminates:**
- `--finalize` flag (no longer needed)
- Manual finalization step
- User confusion about when output is ready

### 6. Simple Cleanup

**To remove working files:**
```bash
rm -rf output_dir/work/
```

**To keep only latest output:**
```bash
# Keep latest, remove older timestamped directories
cd output_dir/clusters/
ls -t | tail -n +2 | xargs rm -rf
```

**Optional CLI command:**
```bash
gaphack-decompose --clean-work output_dir/    # removes work/ directory
gaphack-decompose --clean-old output_dir/     # keeps only latest clusters/
```

**Eliminates:**
- `--cleanup` flag during finalization
- Pattern-based file removal
- Confusing cleanup semantics

## CLI Workflow Examples

### Initial Run

```bash
gaphack-decompose input.fasta -o out/ --resolve-conflicts --refine-close-clusters 0.01
```

**Creates:**
```
out/
  ├── state.json
  ├── work/
  │   ├── initial/
  │   │   ├── cluster_00001I.fasta (complete set)
  │   │   └── ...
  │   ├── deconflicted/
  │   │   ├── cluster_00001C.fasta (complete set)
  │   │   └── ...
  │   └── refined_0.010/
  │       ├── cluster_00001R1.fasta (complete set)
  │       └── ...
  └── clusters/
      ├── 2025-01-15_143022/
      │   ├── cluster_001.fasta (size-sorted)
      │   └── ...
      └── latest -> 2025-01-15_143022/
```

### Resume Initial Clustering

```bash
gaphack-decompose --resume out/
```

**Behavior:**
- Loads checkpoint from `work/initial/`
- Continues clustering
- Updates `work/initial/` with complete results
- Creates new `clusters/{timestamp}/` on completion
- Updates `latest` symlink

### Add Conflict Resolution

```bash
gaphack-decompose --resume out/ --resolve-conflicts
```

**Behavior:**
- Loads complete clusters from `work/initial/`
- Performs conflict resolution
- Creates `work/deconflicted/` with complete results
- Creates new `clusters/{timestamp}/`
- Updates `latest` symlink

### Apply Additional Refinement

```bash
gaphack-decompose --resume out/ --refine-close-clusters 0.02
```

**Behavior:**
- Loads from most recent complete stage (`work/deconflicted/` or `work/refined_0.010/`)
- Applies refinement at 0.02 threshold
- Creates `work/refined_0.020/` with complete results
- Creates new `clusters/{timestamp}/`
- Updates `latest` symlink

### Compare Refinement Thresholds

```bash
# First refinement
gaphack-decompose --resume out/ --refine-close-clusters 0.01

# Second refinement (from same source)
gaphack-decompose --resume out/ --refine-close-clusters 0.02
```

**Result:**
```
out/
  ├── work/
  │   ├── refined_0.010/
  │   │   └── cluster_00001R1.fasta
  │   └── refined_0.020/
  │       └── cluster_00001R2.fasta
  └── clusters/
      ├── 2025-01-15_143022/  # 0.01 threshold results
      ├── 2025-01-15_150633/  # 0.02 threshold results
      └── latest -> 2025-01-15_150633/
```

Both results preserved, can compare cluster counts and composition.

## Report Generation

The `decompose_report.txt` uses globally unique cluster IDs for unambiguous tracking:

```
gapHACk Decompose Report
========================
Run: 2025-01-15 14:30:22
Command: gaphack-decompose input.fasta -o out/ --resolve-conflicts --refine-close-clusters 0.01

Stage Progression
==================

Initial Clustering
------------------
Total clusters: 143
Total sequences: 1,429
Coverage: 100.0%

Cluster Details:
  cluster_00001I: 105 sequences
  cluster_00002I: 87 sequences
  cluster_00143I: 3 sequences

Conflict Resolution
-------------------
Conflicts detected: 12 sequences in 5 cluster pairs
Total clusters after resolution: 138 (5 merges)

Merge Details:
  cluster_00001C ← cluster_00001I + cluster_00042I (192 sequences)
  cluster_00002C ← cluster_00002I (unchanged, 87 sequences)
  cluster_00003C ← cluster_00003I + cluster_00015I (43 sequences)
  ...

Close Cluster Refinement (threshold=0.010)
-------------------------------------------
Total clusters after refinement: 135 (3 merges)

Merge Details:
  cluster_00001R1 ← cluster_00001C + cluster_00015C (247 sequences)
  cluster_00002R1 ← cluster_00002C (unchanged, 87 sequences)
  ...

Final Output
------------
Location: clusters/2025-01-15_143022/
Total clusters: 135 (size-sorted)

Top 10 Clusters:
  cluster_001: 247 sequences (source: cluster_00001R1)
  cluster_002: 192 sequences (source: cluster_00005R1)
  cluster_003: 156 sequences (source: cluster_00008R1)
  ...

File Mapping:
  work/initial/cluster_00001I.fasta → clusters/latest/cluster_001.fasta
  work/deconflicted/cluster_00001C.fasta → clusters/latest/cluster_001.fasta
  work/refined_0.010/cluster_00001R1.fasta → clusters/latest/cluster_001.fasta
```

**Scriptability:**
```bash
# Find all stages a cluster appeared in
grep "cluster_00042" decompose_report.txt

# Extract final cluster source mapping
grep "source: cluster_" decompose_report.txt

# Compare two refinement runs
diff clusters/2025-01-15_143022/cluster_001.fasta \
     clusters/2025-01-15_150633/cluster_001.fasta
```

## Implementation Changes

### Files Modified

1. **`decompose.py`**:
   - Modify cluster ID allocation to use global counter
   - Update checkpoint saving to use `work/{stage}/` directories
   - Auto-generate `clusters/{timestamp}/` on completion
   - Create/update `latest` symlink

2. **`decompose_cli.py`**:
   - Remove `--finalize` flag
   - Remove `--cleanup` flag (or repurpose as simple directory deletion)
   - Update `save_decompose_results()` to write to timestamped directory
   - Update paths throughout

3. **`resume.py`**:
   - Remove `finalize_decompose()` function
   - Update stage loading to use new directory structure
   - Scan `work/` directories to find max cluster ID on resume
   - Update `save_stage_results()` to write complete cluster sets

4. **`state.py`**:
   - Update `StateManager` paths for new directory structure
   - Remove finalization tracking (no longer needed)
   - Add stage directory name tracking
   - Update `save_stage_fasta()` to write to subdirectories

5. **`utils.py`**:
   - Add utility to scan directories for max cluster ID
   - Add utility to create timestamped directories
   - Add utility to update `latest` symlink

### New Utility Functions

```python
def get_stage_suffix(stage_name: str, refinement_count: int = 0) -> str:
    """Get stage suffix for cluster IDs.

    Args:
        stage_name: Name of stage ('initial', 'deconflicted', or 'refined_*')
        refinement_count: Number of refinement stages seen so far

    Returns:
        Stage suffix string ('I', 'C', 'R1', 'R2', etc.)
    """
    if stage_name == "initial":
        return "I"
    elif stage_name == "deconflicted":
        return "C"
    elif stage_name.startswith("refined"):
        return f"R{refinement_count + 1}"
    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def format_cluster_id(cluster_num: int, stage_suffix: str) -> str:
    """Format cluster ID with stage suffix.

    Args:
        cluster_num: Cluster number (1-based)
        stage_suffix: Stage suffix ('I', 'C', 'R1', etc.)

    Returns:
        Formatted cluster ID (e.g., 'cluster_00042I')
    """
    return f"cluster_{cluster_num:05d}{stage_suffix}"


def parse_cluster_id(cluster_id: str) -> tuple[int, str]:
    """Parse cluster ID into number and suffix.

    Args:
        cluster_id: Cluster ID string (e.g., 'cluster_00042I')

    Returns:
        Tuple of (cluster_number, stage_suffix)
    """
    import re
    match = re.match(r'cluster_(\d+)([ICR]\d*)', cluster_id)
    if not match:
        raise ValueError(f"Invalid cluster ID format: {cluster_id}")
    return int(match.group(1)), match.group(2)


def get_next_cluster_number(output_dir: Path) -> int:
    """Scan work/ directories to find max allocated cluster number.

    Returns the next available cluster number (max + 1).
    """
    max_num = 0
    work_dir = output_dir / "work"
    if work_dir.exists():
        for stage_dir in work_dir.iterdir():
            if stage_dir.is_dir():
                for cluster_file in stage_dir.glob("cluster_*.fasta"):
                    # Extract number from filename: cluster_00042I.fasta -> 42
                    try:
                        num, suffix = parse_cluster_id(cluster_file.stem)
                        max_num = max(max_num, num)
                    except ValueError:
                        continue

    return max_num + 1


def count_refinement_stages(output_dir: Path) -> int:
    """Count number of refinement stages already completed.

    Returns count of refined_* directories in work/.
    """
    work_dir = output_dir / "work"
    if not work_dir.exists():
        return 0

    return len([d for d in work_dir.iterdir()
                if d.is_dir() and d.name.startswith("refined_")])


def create_timestamped_output(output_dir: Path, clusters: Dict,
                              sequences: List, headers: List,
                              hash_to_headers: Dict) -> Path:
    """Create timestamped cluster output and update latest symlink.

    Returns path to created timestamp directory.
    """
    from datetime import datetime

    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = clusters_dir / timestamp
    output_path.mkdir(exist_ok=True)

    # Write size-sorted clusters
    write_clusters_size_sorted(output_path, clusters, sequences,
                               headers, hash_to_headers)

    # Update latest symlink
    latest_link = clusters_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(timestamp)

    return output_path


def write_complete_stage_clusters(stage_dir: Path, clusters: Dict,
                                  sequences: List, headers: List,
                                  stage_suffix: str) -> Dict[str, str]:
    """Write complete cluster set to stage directory with global IDs.

    Args:
        stage_dir: Directory to write clusters to
        clusters: Dict mapping internal cluster_id -> list of sequence headers
        sequences: Full sequence list
        headers: Full header list
        stage_suffix: Stage suffix for cluster IDs ('I', 'C', 'R1', etc.)

    Returns mapping from internal cluster_id to global cluster_id.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Sort clusters by size for consistent ID assignment
    sorted_clusters = sorted(clusters.items(),
                           key=lambda x: len(x[1]),
                           reverse=True)

    id_mapping = {}
    for i, (internal_id, cluster_headers) in enumerate(sorted_clusters, start=1):
        global_id = format_cluster_id(i, stage_suffix)
        id_mapping[internal_id] = global_id

        # Write cluster file
        cluster_file = stage_dir / f"{global_id}.fasta"
        write_cluster_fasta(cluster_file, cluster_headers,
                          sequences, headers)

    return id_mapping
```

### Backward Compatibility

**Not maintained.** This is a breaking change to the output structure.

Users with existing output directories should:
1. Complete any in-progress runs with the old version
2. Archive old results if needed
3. Start new runs with the new version

Migration script could be provided if needed, but given the experimental/development nature of the tool, clean break is preferred.

## Testing Plan

1. **Unit tests**:
   - Test `get_next_cluster_id_range()` with various directory structures
   - Test timestamped directory creation
   - Test symlink updates
   - Test complete cluster writing

2. **Integration tests**:
   - Full run with all stages
   - Resume at each stage
   - Multiple refinement thresholds
   - Verify directory structure
   - Verify cluster ID uniqueness
   - Verify complete cluster sets at each stage

3. **Validation**:
   - Compare cluster contents between old and new format
   - Verify no sequences lost or duplicated
   - Verify size-sorting in final output
   - Verify report accuracy

## Benefits Summary

1. ✅ **Simpler CLI** - automatic output, no manual finalization
2. ✅ **Clearer organization** - `work/` vs `clusters/` separation
3. ✅ **Self-contained stages** - complete clusters, no union operations
4. ✅ **Unambiguous tracking** - globally unique cluster IDs
5. ✅ **Version preservation** - timestamped outputs, never overwrite
6. ✅ **Easy cleanup** - delete `work/` directory
7. ✅ **Better reports** - clear cluster lineage tracking
8. ✅ **Easy comparison** - multiple parameter sets preserved
9. ✅ **Simplified code** - fewer special cases, clearer logic
10. ✅ **Better UX** - users only interact with `clusters/latest/`

## Timeline

1. **Phase 1**: Update data structures and ID allocation (1-2 hours)
2. **Phase 2**: Update file writing to new directory structure (2-3 hours)
3. **Phase 3**: Update resume logic and stage loading (2-3 hours)
4. **Phase 4**: Update reporting to use global IDs (1-2 hours)
5. **Phase 5**: Testing and validation (2-3 hours)

**Total estimate**: 8-13 hours of implementation work
