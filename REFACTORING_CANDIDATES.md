# Refactoring Candidates: Unused and Low-Value Code

This document identifies CLI options, functions, and code that appears to be unused or of low value for potential removal.

## Priority 1: Bugs/Issues (Fix Required)

### 1. **Missing `conflicted_clusters` in initial_state**
- **Location**: `gaphack/refine_cli.py:267`
- **Issue**: Referenced but never set in `initial_state` dict (line 651-659)
- **Impact**: Will cause KeyError if conflicts are detected
- **Fix**: Either:
  - Add `'conflicted_clusters': len(set(cid for clusters in conflicts.values() for cid in clusters))` to initial_state
  - OR remove the line that references it (since we show conflicts_count already)

## Priority 2: Unused CLI Arguments

### 1. **`--blast-evalue` (refine_cli.py:586)**
- **Status**: Defined but never used
- **Reason**: The argument is parsed but `args.blast_evalue` is never referenced in the code
- **Impact**: Currently uses hardcoded default (1e-5) in `ClusterGraph`
- **Recommendation**: Remove argument OR wire it up to `ClusterGraph` initialization

### 2. **`--min-identity` (refine_cli.py:588)**
- **Status**: Defined but never used
- **Reason**: The argument is parsed but `args.min_identity` is never referenced in the code
- **Impact**: Currently uses hardcoded default (90.0%) in `ClusterGraph`
- **Recommendation**: Remove argument OR wire it up to `ClusterGraph` initialization

### 3. **`--expansion-threshold` (refine_cli.py:578)**
- **Status**: Marked as "(Legacy)" and only used in parameters dict
- **Reason**: Set in parameters dict (line 789) but never displayed or actually used
- **Impact**: None - completely unused
- **Recommendation**: Remove argument and the line that sets it in parameters dict

## Priority 3: Unused Function Parameters

### 1. **`stage1_clusters` and `stage2_clusters` in `generate_cluster_mapping_report()`**
- **Location**: `gaphack/refine_cli.py:153-154`
- **Status**: Always passed as `None` (lines 757-758)
- **Reason**: Remnants from two-pass architecture (Pass 1/Pass 2)
- **Impact**: Parameters exist but serve no purpose
- **Recommendation**: Remove parameters and simplify function signature:
  ```python
  def generate_cluster_mapping_report(
      original_clusters: Dict[str, List[str]],
      final_clusters: Dict[str, List[str]],
      output_path: Path
  ) -> None:
  ```
- **Also update**: Docstring comment about "Deconflicted_ID" and "Refined_ID" (line 172)

### 2. **`stage1_info` parameter in `generate_refinement_summary()`**
- **Location**: `gaphack/refine_cli.py:229`
- **Status**: Always passed as `None` (line 801)
- **Reason**: Remnant from two-pass architecture
- **Impact**: Parameter exists but never used
- **Recommendation**: Remove parameter and simplify function

## Priority 4: Unused Imports

### 1. **`ClusterGraph` import in refine_cli.py**
- **Location**: `gaphack/refine_cli.py:23`
- **Status**: Imported but never used
- **Reason**: `ClusterGraph` is created internally by `refine_clusters()`, not in CLI
- **Recommendation**: Remove import

## Priority 5: Outdated Comments/Docstrings

### 1. **"conflict resolution and close cluster refinement" in parser description**
- **Location**: `gaphack/refine_cli.py:546`
- **Current**: "Refine existing clusters using conflict resolution and close cluster refinement"
- **Issue**: Conflicts are now resolved during iterative refinement, not as separate step
- **Recommendation**: Change to "Refine existing clusters using iterative neighborhood-based refinement"

### 2. **"two-pass refinement" in docstring**
- **Location**: `gaphack/refine_cli.py:234`
- **Current**: "Generate detailed summary report for two-pass refinement"
- **Issue**: No longer two-pass
- **Recommendation**: Change to "Generate detailed summary report for iterative refinement"

### 3. **Comment about Pass 1/Pass 2 in cluster_mapping.txt**
- **Location**: `gaphack/refine_cli.py:172`
- **Current**: "# Original_ID → [Deconflicted_ID] → [Refined_ID] → Final_ID"
- **Issue**: No longer have separate deconflicted and refined stages
- **Recommendation**: Change to "# Original_ID → Final_ID"

## Summary

### Can Remove Immediately (Low Risk)
1. `--expansion-threshold` argument + line 789
2. `ClusterGraph` import
3. Update docstrings and comments

### Should Fix (Bug)
1. Add `conflicted_clusters` to `initial_state` OR remove the reference

### Should Remove (Simplification)
1. `stage1_clusters` and `stage2_clusters` parameters (2 functions affected)
2. `stage1_info` parameter (1 function affected)

### Decide: Remove or Wire Up
1. `--blast-evalue` - Either remove OR connect to ClusterGraph
2. `--min-identity` - Either remove OR connect to ClusterGraph

**Estimated Line Reduction**: ~50-100 lines (arguments, parameters, docstrings)
**Risk Level**: Low (mostly removing already-unused code)
