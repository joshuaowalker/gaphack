# Phase 5 Complete: Finalization

## Status: ✅ COMPLETE

All Phase 5 work is done, including implementation and testing of finalization capability.

## What Was Implemented

### Phase 5: Finalization
Implemented capability to create final numbered cluster output and mark runs as complete.

**Key Features**:
1. Create final numbered output (cluster_001.fasta, cluster_002.fasta, etc.)
2. Clusters numbered by size (largest first)
3. Mark state as "finalized"
4. Optional cleanup of intermediate stage files
5. Handle already-finalized directories gracefully
6. Track finalization statistics in state

**Implementation Details**:

#### Core Function Added (`decompose.py`)

1. **`finalize_decompose()`** (lines 1897-2046)
   - Creates final numbered cluster output from most recent stage
   - Determines source stage (initial, deconflicted, or refined)
   - Loads sequences and reconstructs from hash IDs to original headers
   - Sorts clusters by size (largest first)
   - Writes cluster_001.fasta, cluster_002.fasta, etc.
   - Writes unassigned.fasta if applicable
   - Updates state with finalization statistics
   - Optionally removes intermediate files

**Key Logic**:
```python
# Determine most recent stage
if state.close_cluster_refinement.completed:
    source_stage = "refined"
elif state.conflict_resolution.completed:
    source_stage = "deconflicted"
elif state.initial_clustering.completed:
    source_stage = "initial"

# Sort clusters by size
cluster_sizes.sort(key=lambda x: x[1], reverse=True)

# Write final files with sequential numbering
for final_num, (old_cluster_id, size) in enumerate(cluster_sizes, start=1):
    final_filename = output_path / f"cluster_{final_num:03d}.fasta"
```

#### State Management Enhancement (`state.py`)

1. **`FinalizedStage` class** (lines 88-94)
   - Added statistics tracking fields:
     - `total_clusters`: Number of final clusters
     - `total_sequences`: Total sequences in clusters
     - `unassigned_sequences`: Sequences not assigned
     - `source_stage`: Stage that was finalized from

#### CLI Integration (`decompose_cli.py`)

1. **New Arguments** (lines 388-392)
   - `--finalize`: Create final output and mark complete
   - `--cleanup`: Remove intermediate files (requires --finalize)

2. **Validation Logic** (lines 406-413)
   - --cleanup requires --finalize
   - --finalize requires --resume
   - Clear error messages for invalid usage

3. **Finalization Handler** (lines 425-437)
   - Calls finalize_decompose() when --finalize is used
   - Handles errors gracefully
   - Logs completion

## Test Summary

**Total: 6 tests passing** (`tests/test_phase5_finalization.py`)
- `test_finalize_from_initial_clustering` - PASSED
- `test_finalize_after_refinement` - PASSED
- `test_finalize_with_cleanup` - PASSED
- `test_finalize_already_finalized` - PASSED
- `test_finalize_incomplete_clustering_fails` - PASSED
- `test_finalize_preserves_sequence_content` - PASSED

All tests use realistic biological sequences and verify:
- Final numbering is correct and sorted by size
- State transitions and completion flags
- Statistics tracking
- Cleanup behavior (removes earlier stages, keeps source)
- Error handling for incomplete runs
- Sequence content preservation from input to final output

## Files Modified

### Core Implementation
- `gaphack/decompose.py`:
  - Added `finalize_decompose()` function (150 lines)
  - Handles cluster renumbering and file creation
  - Implements optional cleanup logic

- `gaphack/state.py`:
  - Extended `FinalizedStage` with statistics fields
  - Tracks source stage for traceability

- `gaphack/decompose_cli.py`:
  - Added --finalize and --cleanup arguments
  - Added validation logic
  - Integrated finalization into resume flow

### Tests
- `tests/test_phase5_finalization.py` (new file, 403 lines)
  - Test finalization from initial stage
  - Test finalization after refinement stages
  - Test cleanup functionality
  - Test already-finalized handling
  - Test error cases
  - Test content preservation

### Documentation
- `docs/PHASE5_COMPLETE.md` (this file)

## Key Design Decisions

### 1. Largest-First Numbering
Final clusters are numbered by size in descending order:
- cluster_001.fasta = largest cluster
- cluster_002.fasta = second largest
- etc.

This makes it easy to identify major clusters at a glance.

### 2. Three-Digit Numbering Format
Using `cluster_001` instead of `cluster_1` ensures:
- Alphabetical sort = size sort
- Consistent filename width
- Professional appearance

### 3. Source Stage Retention
When cleanup is enabled:
- Remove earlier stages (initial.*, deconflicted.* if finalizing from refined)
- Keep source stage files for traceability
- Never remove the stage being finalized from

Example: Finalizing from refined stage with cleanup:
- ❌ Remove: initial.cluster_*.fasta
- ❌ Remove: deconflicted.cluster_*.fasta
- ✅ Keep: refined.cluster_*.fasta (source)
- ✅ Create: cluster_001.fasta, cluster_002.fasta, etc.

### 4. Hash ID Expansion
Finalization expands internal hash IDs back to original headers:
- Initial stage files contain hash IDs (internal representation)
- Final files contain original headers from input
- Handles duplicate sequences correctly (multiple headers per hash)

### 5. Idempotency
Running finalization twice on the same directory:
- Issues warning "Output directory already finalized"
- Does not error
- Does not create duplicate files
- Returns gracefully

### 6. Error Prevention
- Cannot finalize incomplete runs (error raised)
- Cannot use --cleanup without --finalize (error raised)
- Cannot use --finalize without --resume (error raised)
- Clear error messages guide users to correct usage

## Integration with Existing Phases

**Phase 1** (State Management): Extended with FinalizedStage tracking
**Phase 2** (Initial Clustering Continuation): Provides data to finalize
**Phase 3** (Graceful Interruption): Can finalize after interruption recovery
**Phase 4** (Staged Refinement): Can finalize from any refinement stage
**Phase 5** (Finalization): NEW - Completes the incremental workflow

## Usage Examples

### Basic Finalization

```bash
# Run clustering
gaphack-decompose input.fasta -o results/

# Finalize (creates cluster_001.fasta, etc.)
gaphack-decompose --resume -o results/ --finalize
```

### Finalization with Cleanup

```bash
# After refinement stages
gaphack-decompose --resume -o results/ --resolve-conflicts
gaphack-decompose --resume -o results/ --refine-close-clusters 0.02

# Finalize and remove intermediate files
gaphack-decompose --resume -o results/ --finalize --cleanup
```

### Complete Workflow

```bash
# Initial clustering
gaphack-decompose input.fasta -o results/

# Apply refinements
gaphack-decompose --resume -o results/ --resolve-conflicts
gaphack-decompose --resume -o results/ --refine-close-clusters 0.02

# Finalize
gaphack-decompose --resume -o results/ --finalize --cleanup
```

Result structure after finalization with cleanup:
```
results/
├── state.json                      # Status: completed
├── refined.cluster_*.fasta         # Source stage (kept)
├── cluster_001.fasta               # Final output (largest)
├── cluster_002.fasta               # Final output (second largest)
├── cluster_003.fasta               # etc.
└── unassigned.fasta                # Unassigned sequences (if any)
```

## Output Structure

### Before Finalization
```
decompose_output/
├── state.json
├── initial.cluster_0001.fasta
├── initial.cluster_0002.fasta
├── initial.unassigned.fasta
├── deconflicted.cluster_0001.fasta
├── deconflicted.cluster_0002.fasta
├── refined.cluster_0001.fasta
└── refined.cluster_0002.fasta
```

### After Finalization (without cleanup)
```
decompose_output/
├── state.json                      # stage="finalized"
├── initial.cluster_0001.fasta      # Kept
├── initial.cluster_0002.fasta
├── initial.unassigned.fasta
├── deconflicted.cluster_0001.fasta # Kept
├── deconflicted.cluster_0002.fasta
├── refined.cluster_0001.fasta      # Kept (source)
├── refined.cluster_0002.fasta
├── cluster_001.fasta               # NEW: Final output
├── cluster_002.fasta               # NEW: Final output
└── unassigned.fasta                # NEW: Final unassigned
```

### After Finalization (with cleanup)
```
decompose_output/
├── state.json                      # stage="finalized"
├── refined.cluster_0001.fasta      # Source retained
├── refined.cluster_0002.fasta
├── cluster_001.fasta               # Final output
├── cluster_002.fasta               # Final output
└── unassigned.fasta                # Final unassigned
```

## State Tracking

The state.json file tracks finalization:

```json
{
  "status": "completed",
  "stage": "finalized",
  "stages": {
    "finalized": {
      "completed": true,
      "total_clusters": 3,
      "total_sequences": 1429,
      "unassigned_sequences": 15,
      "source_stage": "refined"
    }
  }
}
```

## Known Limitations

**Addressed in This Phase**:
- ✅ Cannot create final numbered output
- ✅ Cannot mark run as "completed"
- ✅ Cannot cleanup intermediate stage files
- ✅ Intermediate files clutter output directory

**No Remaining Limitations for Core Workflow**

The incremental restart implementation is now complete with all planned phases implemented!

## Future Enhancements (Not Planned)

Potential future work (outside current scope):
1. **Incremental sequence addition**: Add new sequences to existing clusters
2. **Rollback capability**: Discard refinement and return to earlier stage
3. **Parallel resumption**: Run multiple resume operations in parallel
4. **Automatic finalization**: Option to auto-finalize when all stages complete

These are noted for potential future development but are not required for the core incremental workflow.

## Success Criteria Met ✅

- ✅ User can create final numbered cluster output
- ✅ Clusters are numbered by size (largest first)
- ✅ State is marked as "completed" and "finalized"
- ✅ User can optionally cleanup intermediate files
- ✅ Source stage files are retained for traceability
- ✅ Already-finalized directories handled gracefully
- ✅ Clear error messages for invalid operations
- ✅ Hash IDs properly expanded to original headers
- ✅ Tests verify all functionality
- ✅ Documentation complete

Phase 5 is complete! All incremental restart phases (1-5) are now implemented and tested.

## Complete Incremental Workflow

The full workflow is now available:

1. **Phase 1**: State management and checkpointing ✅
2. **Phase 2**: Resume and continue initial clustering ✅
3. **Phase 3**: Graceful interruption handling ✅
4. **Phase 4**: Apply refinement stages after clustering ✅
5. **Phase 5**: Create final numbered output ✅

Users can now:
- Start clustering runs
- Interrupt and resume at any time
- Add more clusters incrementally
- Apply refinement stages independently
- Create clean final output
- All with persistent state tracking

The incremental restart design is fully realized!