# gaphack-decompose Removal Plan

## Summary

This document outlines the complete removal of gaphack-decompose from the codebase, leaving only the core gap-optimized clustering tools (gaphack, gaphack-refine, gaphack-analyze).

## Files to Remove Completely

### Primary Source Files (7 files)
1. **`gaphack/decompose_cli.py`** (567 lines) - CLI entry point for decompose
2. **`gaphack/decompose.py`** (1017 lines) - Main decompose orchestration
   - **Exception**: Extract shared classes first (see Refactoring section)
3. **`gaphack/resume.py`** (251 lines) - Checkpoint/resume functionality
4. **`gaphack/state.py`** (424 lines) - State management and persistence
5. **`gaphack/target_selection.py`** (154 lines) - Target selection strategies
6. **`gaphack/cluster_id_utils.py`** - Cluster ID utilities (if exists)

### Files to Keep (used by gaphack-refine)
- **`gaphack/blast_neighborhood.py`** - Used by cluster_graph.py
- **`gaphack/vsearch_neighborhood.py`** - Used by cluster_graph.py
- **`gaphack/neighborhood_finder.py`** - Base class for neighborhood finders
- **`gaphack/cluster_graph.py`** - Used by refine_cli.py and cluster_refinement.py
- **`gaphack/target_clustering.py`** - Used by cli.py for target mode

### Test Files to Remove (11+ files)
1. **`tests/test_decompose.py`** - Main decompose tests
2. **`tests/test_resume.py`** - Resume functionality tests
3. **`tests/test_state.py`** - State management tests
4. **`tests/test_interruption.py`** - Interruption handling tests
5. **`tests/test_interruption_edge_cases.py`** - Edge case tests
6. **`tests/test_resume_hash_expansion.py`** - Hash expansion tests
7. **`tests/test_phase4_integration.py`** - Phase 4 integration tests (if decompose-specific)
8. **`tests/test_phase4_quality.py`** - Phase 4 quality tests (if decompose-specific)
9. **`tests/test_phase4_refinement.py`** - Phase 4 refinement tests (if decompose-specific)
10. **`tests/test_phase5_finalization.py`** - Phase 5 finalization tests
11. **`tests/test_separation_integration.py`** - Separation integration tests (if decompose-specific)
12. **`tests/test_report_integration.py`** - Report integration tests (if decompose-specific)
13. **`tests/test_principled_integration.py`** - Principled integration tests (if decompose-specific)
14. **`tests/test_blast_neighborhood.py`** - **KEEP** (used by cluster_graph)
15. **`tests/test_cluster_graph.py`** - **KEEP** (used by refine)
16. **`tests/test_vsearch.py`** - **KEEP** (used by cluster_graph)

## Refactoring Required

### 1. Extract Shared Classes from `decompose.py`

These classes are used by `cluster_refinement.py` and `refine_cli.py`:

**Create new file: `gaphack/refinement_types.py`**
```python
"""Shared types for cluster refinement and result tracking."""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ProcessingStageInfo:
    """Information about a processing stage."""
    stage_name: str
    clusters_before: Dict[str, List[str]] = field(default_factory=dict)
    clusters_after: Dict[str, List[str]] = field(default_factory=dict)
    components_processed: List[Dict] = field(default_factory=list)
    summary_stats: Dict = field(default_factory=dict)

@dataclass
class ClusterResults:
    """Results from cluster refinement."""
    clusters: Dict[str, List[str]] = field(default_factory=dict)
    unassigned: List[str] = field(default_factory=list)
    conflicts: Dict[str, List[str]] = field(default_factory=dict)

class ClusterIDGenerator:
    """Generates globally unique cluster IDs with stage suffixes."""
    # ... (copy full implementation from decompose.py)
```

### 2. Update Import Statements

Files requiring import updates:

**`gaphack/cluster_refinement.py`:**
```python
# OLD:
from .decompose import DecomposeResults, ProcessingStageInfo, ClusterIDGenerator

# NEW:
from .refinement_types import ClusterResults, ProcessingStageInfo, ClusterIDGenerator
```

**`gaphack/refine_cli.py`:**
```python
# OLD:
from .decompose import ClusterIDGenerator, ProcessingStageInfo

# NEW:
from .refinement_types import ClusterIDGenerator, ProcessingStageInfo
```

### 3. Update `gaphack/__init__.py`

**Remove:**
```python
from .decompose import DecomposeClustering
```

**Update `__all__`:**
```python
__all__ = [
    "GapOptimizedClustering",
    "TargetModeClustering",
    # "DecomposeClustering",  # REMOVE THIS LINE
    "DistanceProvider",
    # ... rest unchanged
]
```

### 4. Update `pyproject.toml`

**Remove CLI entry point:**
```toml
[project.scripts]
gaphack = "gaphack.cli:main"
gaphack-analyze = "gaphack.analyze_cli:main"
# gaphack-decompose = "gaphack.decompose_cli:main"  # REMOVE THIS LINE
gaphack-refine = "gaphack.refine_cli:main"
```

## Documentation Updates

### Files to Update
1. **`README.md`** - Remove all decompose references, add vsearch workflow
2. **`CLAUDE.md`** - Remove decompose sections
3. **`examples/README.md`** (if exists) - Update examples

### New Recommended Workflow Section

Add to README.md:

```markdown
## Large-Scale Clustering Workflow

For datasets with 100K+ sequences, use this two-step workflow:

### Step 1: Fast Initial Clustering with vsearch
```bash
# Create initial clusters at 97% identity
vsearch --cluster_fast input.fasta \
    --id 0.97 \
    --clusters cluster_ \
    --centroids centroids.fasta
```

### Step 2: Gap-Optimized Refinement
```bash
# Refine clusters with gaphack-refine
gaphack-refine --input-dir clusters/ \
    --output-dir refined/ \
    --refine-close-clusters 0.02
```

### Step 3: Quality Assessment
```bash
# Analyze final clustering quality
gaphack-analyze refined/latest/*.fasta -o analysis/
```

This approach combines:
- **vsearch**: Fast approximate clustering (C implementation, highly optimized)
- **gaphack-refine**: Gap optimization and boundary refinement
- **gaphack-analyze**: Quality metrics and validation

### Alternative: Expert Taxonomy Starting Point

If you have expert-assigned taxa, skip vsearch and start directly with refinement:

```bash
# Your taxa files: species_001.fasta, species_002.fasta, etc.
gaphack-refine --input-dir expert_taxa/ \
    --output-dir refined/ \
    --refine-close-clusters 0.02
```
```

## Migration Guide for Existing Users

### If You Used gaphack-decompose

**Old workflow:**
```bash
gaphack-decompose input.fasta -o results
gaphack-refine --input-dir results/clusters/latest/ --output-dir refined/
```

**New workflow:**
```bash
# Step 1: vsearch clustering (much faster)
vsearch --cluster_fast input.fasta --id 0.97 --clusters cluster_

# Step 2: gap-optimized refinement (same as before)
gaphack-refine --input-dir clusters/ --output-dir refined/ --refine-close-clusters 0.02
```

### Advantages of New Workflow
1. **Faster**: vsearch is highly optimized C code
2. **Simpler**: One less tool to learn
3. **More flexible**: Use any clustering tool (vsearch, MMseqs2, USEARCH, etc.)
4. **Focused**: gapHACk does what it does best - gap optimization

## Implementation Steps

### Phase 1: Extract Shared Code
- [ ] Create `gaphack/refinement_types.py` with shared classes
- [ ] Update imports in `cluster_refinement.py`
- [ ] Update imports in `refine_cli.py`
- [ ] Run tests to verify refactoring

### Phase 2: Remove Files
- [ ] Remove decompose_cli.py
- [ ] Remove decompose.py
- [ ] Remove resume.py
- [ ] Remove state.py
- [ ] Remove target_selection.py
- [ ] Remove cluster_id_utils.py (if exists)
- [ ] Remove test files (listed above)

### Phase 3: Update Configuration
- [ ] Update `__init__.py` (remove DecomposeClustering export)
- [ ] Update `pyproject.toml` (remove CLI entry point)
- [ ] Run: `pip install -e .` to rebuild

### Phase 4: Update Documentation
- [ ] Update README.md with new workflow
- [ ] Update CLAUDE.md (remove decompose sections)
- [ ] Add migration guide
- [ ] Update examples

### Phase 5: Verification
- [ ] Run full test suite: `pytest`
- [ ] Verify CLI commands work: `gaphack --help`, `gaphack-refine --help`, `gaphack-analyze --help`
- [ ] Verify gaphack-decompose is gone: `gaphack-decompose --help` should fail
- [ ] Test example workflow with vsearch + gaphack-refine

## Lines of Code Reduction

**Estimated removal:**
- Source files: ~2,400 lines
- Test files: ~3,000+ lines
- **Total: ~5,400+ lines removed**

**Maintenance burden reduction:**
- Fewer CLI tools to maintain (4 → 3)
- Simpler architecture (no checkpoint/resume complexity)
- Clearer value proposition (gap optimization, not clustering)
- Less testing surface area

## Risk Assessment

### Low Risk
- Shared classes (ClusterIDGenerator, ProcessingStageInfo) are simple dataclasses
- gaphack-refine is well-tested independently
- No known external users yet

### Testing Requirements
- Verify cluster_refinement.py still works after import changes
- Verify refine_cli.py still works after import changes
- Run full test suite after file removal
- Manual testing of example workflows

## Questions to Resolve

1. Does `cluster_id_utils.py` exist? (need to verify)
2. Are there any example data files specific to decompose? (check examples/)
3. Are there any GitHub workflows or CI configs that reference decompose?
4. Are there any benchmark scripts that use decompose?
