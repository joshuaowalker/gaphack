# gaphack-decompose Refactoring Plan

## Status: ✅ COMPLETE (100%)

### ✅ Completed Steps (All 10)
- [x] **Step 1**: Removed CLI arguments (`--resolve-conflicts`, `--refine-close-clusters`)
- [x] **Step 2-3**: Removed refinement code from `DecomposeClustering` (parameters, methods)
- [x] **Step 4-5**: Updated `decompose()` main loop and conflict reporting
- [x] **Step 6**: Simplified state management (removed refinement stages)
- [x] **Step 7**: Simplified resume logic (removed refinement parameters and functions)
- [x] **Step 8**: Simplified `DecomposeResults` dataclass (removed refinement fields)
- [x] **Step 9**: Updated output behavior (removed final renumbering, kept internal IDs)
- [x] **Step 10**: Updated tests to reflect new behavior (all tests passing)

### 📊 Final Code Changes Summary
- **Lines removed**: ~600 lines
- **Files modified**: 5 core files (`state.py`, `decompose.py`, `decompose_cli.py`, `resume.py`, test files)
- **Methods deleted**: 5 (`_resolve_conflicts`, `_refine_close_clusters_via_refinement`, `_create_proximity_graph`, `_apply_conflict_resolution_stage`, `_apply_close_cluster_refinement_stage`)
- **State classes removed**: 2 (`ConflictResolutionStage`, `CloseClusterRefinementStage`)
- **DataClass fields removed**: 3 from `DecomposeResults` (`verification_results`, `processing_stages`, `active_to_final_mapping`)
- **Test results**: All 46 core tests passing

---

## Objective
Reduce gaphack-decompose scope to **initial clustering only**, removing all post-processing refinement stages. Refinement will be handled exclusively by gaphack-refine.

## Rationale
- **Clearer separation of concerns**: Clustering vs. refinement
- **Simpler resume logic**: Only need to resume initial clustering
- **Better modularity**: Each tool has focused responsibility
- **Conflicts are expected**: Not errors, but natural output of overlapping neighborhoods

---

## Changes Overview

### What Stays in gaphack-decompose
✅ Initial neighborhood-based clustering
✅ Conflict detection and reporting (as informational, not error)
✅ Resume initial clustering
✅ State management for initial stage
✅ Output to `work/initial/` directory

### What Moves to gaphack-refine (already implemented)
➡️ Conflict resolution
➡️ Close cluster refinement
➡️ Finalization (final numbered output)

---

## Implementation Steps

### Step 1: Remove CLI Arguments
**File**: `gaphack/decompose_cli.py`

**Remove**:
- `--resolve-conflicts` argument
- `--refine-close-clusters` argument

**Update help text** to clarify scope and point to gaphack-refine.

---

### Step 2: Remove Class Parameters
**File**: `gaphack/decompose.py`

**In `DecomposeClustering.__init__()`**, remove:
- `resolve_conflicts: bool = False`
- `refine_close_clusters: bool = False`
- `close_cluster_threshold: float = 0.0`

**Remove imports**:
- Imports from `cluster_refinement` (only used for refinement)
- `ClusterGraph` import (only used for close cluster refinement)

---

### Step 3: Remove Refinement Methods
**File**: `gaphack/decompose.py`

**Delete these methods**:
- `_resolve_conflicts()` (lines ~1089-1154)
- `_refine_close_clusters_via_refinement()` (lines ~1156-1223)
- `_create_proximity_graph()` (lines ~1226-1250)

---

### Step 4: Simplify Main decompose() Method
**File**: `gaphack/decompose.py`

**Remove from `decompose()` method** (lines ~761-791):
- Conflict resolution stage
- Close cluster refinement stage
- Intermediate saves after refinement
- All verification calls except initial conflict detection

**Keep**:
- Initial clustering loop (lines 225-753)
- Conflict detection (but change messaging)
- Hash ID expansion
- Final output to `work/initial/`

**Update final section** (lines 796-851):
- Remove "CRITICAL: Always perform final comprehensive conflict verification"
- Simplify to just report conflicts as informational
- Remove final renumbering (keeps internal cluster IDs: `cluster_00001I`, etc.)
- Remove verification_results tracking

---

### Step 5: Update Conflict Reporting
**File**: `gaphack/decompose.py`

**Change from ERROR to INFO**:

```python
# OLD (line ~827):
self.logger.error(f"❌ Final verification reveals {len(final_verification['conflicts'])} conflicts in output")

# NEW:
if results.conflicts:
    self.logger.info(f"ℹ️  {len(results.conflicts)} conflicts detected")
    self.logger.info(f"   Use gaphack-refine to resolve: gaphack-refine {output_dir} --resolve-conflicts")
else:
    self.logger.info("✓ No conflicts detected")
```

**Add conflict report file** (optional helper):

```python
def _write_conflict_report(self, results: DecomposeResults, output_dir: Path) -> None:
    """Write human-readable conflict summary."""
    if not results.conflicts:
        return

    report_file = output_dir / "conflicts_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"Conflicts: {len(results.conflicts)} sequences\n\n")
        for seq_id, cluster_ids in sorted(results.conflicts.items()):
            f.write(f"{seq_id}: {', '.join(cluster_ids)}\n")
```

---

### Step 6: Simplify State Management
**File**: `gaphack/state.py`

**Remove stage classes**:
- `ConflictResolutionStage` (no longer needed)
- `CloseClusterRefinementStage` (no longer needed)

**Update `DecomposeState`**:
- Remove `conflict_resolution` field
- Remove `close_cluster_refinement` field
- Update `stage` values: only `"initial_clustering"` or `"finalized"`

**Update state transitions**:
- `initial_clustering` (in progress) → `initial_clustering` (completed)
- No intermediate stages

---

### Step 7: Simplify Resume Logic
**File**: `gaphack/resume.py`

**In `resume_decompose()`**, remove:
- `resolve_conflicts` parameter
- `refine_close_clusters` parameter
- `_apply_conflict_resolution_stage()` function
- `_apply_close_cluster_refinement_stage()` function

**Simplify to**:
```python
def resume_decompose(output_dir: Path,
                     max_clusters: Optional[int] = None,
                     max_sequences: Optional[int] = None,
                     **kwargs) -> DecomposeResults:
    """Resume initial clustering only."""

    state = DecomposeState.load(output_dir)

    if not state.initial_clustering.completed:
        # Continue initial clustering
        return _continue_initial_clustering(...)
    else:
        # Already complete
        logger.info("Initial clustering already complete")
        logger.info(f"For refinement: gaphack-refine {output_dir}")
        return _load_current_results(...)
```

---

### Step 8: Update DecomposeResults
**File**: `gaphack/decompose.py`

**Simplify `DecomposeResults` dataclass**:
- Keep `clusters` (non-conflicted sequences)
- Keep `all_clusters` (all sequences including conflicts)
- Keep `conflicts` (sequences assigned to multiple clusters)
- Keep `unassigned`
- Remove `verification_results` (not needed without refinement)
- Remove `processing_stages` (no refinement stages)
- Remove `active_to_final_mapping` (no renumbering)

---

### Step 9: Update Output Behavior
**File**: `gaphack/decompose_cli.py`

**Update `save_decompose_results()`**:
- Remove final renumbering
- Keep internal cluster IDs (`cluster_00001I`, etc.)
- Output to timestamp directory as before
- Add conflicts_report.txt if conflicts exist

---

### Step 10: Update Tests
**Files**: `tests/test_decompose*.py`, `tests/test_resume.py`

**Remove**:
- All tests for `--resolve-conflicts` flag in decompose
- All tests for `--refine-close-clusters` flag in decompose
- Tests for refinement stages in decompose

**Update**:
- Tests should expect conflicts in output
- Tests should verify conflicts are reported informationally
- Resume tests simplified to only test initial clustering resume

**Keep**:
- All initial clustering tests
- Conflict detection tests (but expect INFO not ERROR)
- State management tests (simplified)

---

## Key Discoveries During Implementation

### State Management Insights
- `DecomposeState` was heavily intertwined with refinement stages
- Removed 2 stage classes: `ConflictResolutionStage`, `CloseClusterRefinementStage`
- State transitions now simplified: `initial_clustering` → `finalized` (no intermediate stages)
- `get_current_stage_directory()` simplified from 4 branches to 2
- Removed 3 helper methods: `can_apply_conflict_resolution()`, `can_apply_close_cluster_refinement()`, and refinement-specific `update_stage_completion()` logic

### DecomposeClustering Cleanup
- Three major methods deleted (~160 lines):
  - `_resolve_conflicts()` - called cluster_refinement.resolve_conflicts()
  - `_refine_close_clusters_via_refinement()` - called cluster_refinement.refine_close_clusters()
  - `_create_proximity_graph()` - created ClusterGraph for refinement
- Removed parameters: `resolve_conflicts`, `refine_close_clusters`, `close_cluster_threshold`
- Removed attribute: `self.knn_neighbors` (only used for refinement)
- State persistence parameters cleaned up in `decompose()` method

### Conflict Reporting Changes
- **Before**: Used `self.logger.error()` with ❌ emoji - treated conflicts as failures
- **After**: Uses `self.logger.info()` with ℹ️ emoji - treats conflicts as expected
- Added helpful message pointing to `gaphack-refine` for resolution
- Removed comprehensive verification system (was only needed to validate refinement)
- Removed final renumbering logic (kept internal cluster IDs like `cluster_00001I`)

### All Steps Completed

**Step 7 (Resume)** - ✅ DONE:
- ✅ Removed `resolve_conflicts` and `refine_close_clusters` parameters from `resume_decompose()`
- ✅ Deleted helper functions: `_apply_conflict_resolution_stage()`, `_apply_close_cluster_refinement_stage()` (~235 lines)
- ✅ Simplified resume logic to only handle initial clustering continuation
- ✅ Updated docstrings and parameter passing

**Step 8 (DecomposeResults)** - ✅ DONE:
- ✅ Removed fields: `verification_results`, `processing_stages`, `active_to_final_mapping`
- ✅ Updated `_expand_hash_ids_to_headers()` to not copy removed fields
- ✅ Kept essential fields: `clusters`, `all_clusters`, `conflicts`, `unassigned`, `iteration_summaries`, `command_line`, `start_time`

**Step 9 (Output)** - ✅ DONE:
- ✅ Updated `save_decompose_results()` to use original cluster IDs (no renumbering)
- ✅ Outputs clusters with internal IDs (e.g., `cluster_00001I.fasta`)
- ✅ Removed processing stages and active_to_final_mapping sections from report
- ✅ Updated docstring to reflect new output format

**Step 10 (Tests)** - ✅ DONE:
- ✅ Fixed import errors in `test_state.py` (removed deleted stage classes)
- ✅ Updated test assertions to match new state structure (removed refinement stage checks)
- ✅ Fixed all references to removed DecomposeResults fields
- ✅ All 46 core tests passing (test_decompose.py, test_state.py, test_resume.py)

---

## Testing Strategy

### Manual Testing Checklist
1. ✅ Run decompose on Russula dataset with conflicts expected
2. ✅ Verify conflicts reported as INFO, not ERROR
3. ✅ Verify output in `work/initial/` with cluster IDs like `cluster_00001I`
4. ✅ Interrupt decompose mid-run, verify resume works
5. ✅ Run gaphack-refine on decompose output, verify it works end-to-end
6. ✅ Check conflicts_report.txt is generated

### Automated Testing
```bash
# Should pass after refactoring:
pytest tests/test_decompose.py -v
pytest tests/test_resume.py -v
pytest tests/test_integration.py -v
```

---

## Expected Outcomes

### Before Refactoring
```bash
$ gaphack-decompose input.fasta --resolve-conflicts --refine-close-clusters 0.02 -o out/
# Outputs: clusters/latest/cluster_00001.fasta (final numbered, refined)
```

### After Refactoring
```bash
$ gaphack-decompose input.fasta -o out/
# Outputs: work/initial/cluster_00001I.fasta (initial clustering with conflicts)

$ gaphack-refine out/ --resolve-conflicts --refine-close-clusters 0.02
# Outputs: clusters/latest/cluster_00001.fasta (final numbered, refined)
```

---

## Files Modified

### Core Implementation
- `gaphack/decompose.py` (~300 lines removed)
- `gaphack/decompose_cli.py` (~20 lines removed)
- `gaphack/state.py` (~50 lines removed)
- `gaphack/resume.py` (~200 lines removed)

### Tests
- `tests/test_decompose.py` (updated)
- `tests/test_resume.py` (simplified)
- `tests/test_integration.py` (updated workflow)

### Documentation
- `README.md` (updated workflow examples)
- `CLAUDE.md` (updated architecture notes)

---

## Implementation Order (Completed)

1. ✅ **Step 6** (State) - Foundation changes - **COMPLETED**
2. ✅ **Step 2-3** (Remove refinement code) - Core simplification - **COMPLETED**
3. ✅ **Step 1** (CLI args) - Interface changes - **COMPLETED**
4. ✅ **Step 4-5** (Main loop & reporting) - Behavior changes - **COMPLETED**
5. ✅ **Step 7** (Resume) - Simplified resume logic - **COMPLETED**
6. ✅ **Step 8-9** (Results & output) - Output format - **COMPLETED**
7. ✅ **Step 10** (Tests) - Verification - **COMPLETED**

**Total time**: ~3 hours
**Complexity**: Medium (as expected with no backwards compatibility needed)

---

## Notes for Resuming Implementation

### Files with Partial Changes (need cleanup)
1. **`decompose.py`**:
   - ✅ Removed refinement methods
   - ✅ Updated main loop
   - ❌ Still has `_verify_no_conflicts()` method (may not be needed)
   - ❌ Still has `_renumber_clusters_sequentially()` method (may not be needed)
   - ❌ `DecomposeResults` dataclass still has refinement fields

2. **`resume.py`** - NOT YET MODIFIED:
   - ❌ Still accepts `resolve_conflicts` and `refine_close_clusters` parameters
   - ❌ Has `_apply_conflict_resolution_stage()` function
   - ❌ Has `_apply_close_cluster_refinement_stage()` function
   - ❌ Complex resume logic checks for refinement stages

3. **`decompose_cli.py`**:
   - ✅ Removed refinement arguments
   - ✅ Removed parameter passing
   - ❌ May need to update help text to clarify new workflow

## Verification

### Test Results
```bash
# Core tests passing:
$ python -m pytest tests/test_decompose.py tests/test_state.py tests/test_resume.py -v
# Result: 46 passed in 32.31s ✓
```

### Suggested Follow-up Cleanup (Optional)
These items could be cleaned up but are not blocking:

1. **Unused methods in `decompose.py`** (investigate if still used):
   - `_verify_no_conflicts()` - may be redundant now
   - `_renumber_clusters_sequentially()` - likely unused after removing renumbering

2. **`ProcessingStageInfo` dataclass**:
   - Still defined in `decompose.py` but no longer used by decompose
   - Kept because it's still used by `gaphack-refine` CLI

3. **Unused imports**:
   - May have leftover imports from `cluster_refinement` module in `decompose.py`
   - Can run a cleanup pass to remove unused imports

4. **Help text updates**:
   - Could enhance CLI help text to better explain new workflow
   - Add examples showing gaphack-decompose → gaphack-refine workflow
