# Incremental Restart Design for gaphack-decompose

## Overview

This document describes the design for incremental restart capability in `gaphack-decompose`, allowing users to:
- Save and resume interrupted clustering runs
- Add more clusters to existing results incrementally
- Apply refinement stages (conflict resolution, close cluster refinement) after initial clustering
- Chain multiple refinement passes with different parameters
- Inspect and manually edit intermediate results between stages

## Design Principles

### 1. FASTA Files as Source of Truth

**Core Principle**: The `.fasta` files are the authoritative source for cluster membership. The JSON state file contains only metadata, parameters, and file paths - never cluster membership data.

**Implications**:
- On resume, cluster state is rebuilt by reading FASTA files
- Users can manually edit FASTA files between runs
- No risk of hidden state inconsistencies between JSON and FASTA
- AssignmentTracker, conflict detection, etc. are reconstructed from FASTA

**Example**: User can manually merge two cluster files, delete sequences, or move sequences between clusters before resuming, and the system will respect those changes.

### 2. Flat File Structure

All output files live in a single directory with stage prefixes to indicate processing history. This makes it clear that all FASTA files represent "active" state, without ambiguity about superseded stages.

```
decompose_output/
├── state.json                          # Metadata and parameters only
├── .blast/                             # BLAST database (avoids input dir clutter)
│   ├── blast_db.nhr
│   ├── blast_db.nin
│   └── blast_db.nsq
├── initial.cluster_0001.fasta          # Initial clustering results
├── initial.cluster_0002.fasta
├── initial.unassigned.fasta
├── deconflicted.cluster_0001.fasta     # After conflict resolution
├── deconflicted.cluster_0002.fasta
├── refined.cluster_0001.fasta          # After close cluster refinement
├── refined.cluster_0002.fasta
├── cluster_001.fasta                   # Final output (largest cluster)
├── cluster_002.fasta                   # Final output (second largest)
└── unassigned.fasta                    # Final unassigned sequences
```

### 3. Checkpointing by Default

Since we're on major version 0 with no user base, checkpointing is the default behavior. State is saved:
- Every N iterations (configurable, default: 5)
- After each major stage (initial clustering complete, conflict resolution complete, etc.)
- On clean interruption (KeyboardInterrupt)

### 4. Output Directory Instead of Prefix

The `-o/--output` parameter now specifies a directory rather than a filename prefix:

```bash
# New behavior
gaphack-decompose input.fasta -o decompose_output/

# Default if not specified
gaphack-decompose input.fasta  # Uses "decompose_output/" by default
```

## State File Format

### state.json Structure

```json
{
  "version": "0.5.0",
  "status": "in_progress|completed",
  "stage": "initial_clustering|conflict_resolution|close_cluster_refinement|finalized",

  "input": {
    "fasta_path": "/absolute/path/to/input.fasta",
    "fasta_hash": "sha256_hash_of_input",
    "total_sequences": 10000,
    "deduplicated_sequences": 9850
  },

  "parameters": {
    "min_split": 0.005,
    "max_lump": 0.02,
    "target_percentile": 95,
    "blast_max_hits": 1000,
    "blast_evalue": 1e-5,
    "blast_threads": null,
    "min_identity": null
  },

  "stages": {
    "initial_clustering": {
      "completed": true,
      "total_iterations": 42,
      "total_sequences_processed": 8532,
      "coverage_percentage": 87.3,
      "total_clusters": 45,
      "max_clusters_limit": 50,
      "max_sequences_limit": null,
      "cluster_file_pattern": "initial.cluster_*.fasta",
      "unassigned_file": "initial.unassigned.fasta"
    },
    "conflict_resolution": {
      "completed": true,
      "conflicts_before": 23,
      "conflicts_after": 0,
      "clusters_before": 45,
      "clusters_after": 43,
      "cluster_file_pattern": "deconflicted.cluster_*.fasta"
    },
    "close_cluster_refinement": {
      "completed": true,
      "threshold": 0.02,
      "clusters_before": 43,
      "clusters_after": 38,
      "cluster_file_pattern": "refined.cluster_*.fasta",
      "refinement_history": [
        {"threshold": 0.02, "timestamp": "2025-01-29T10:30:00"},
        {"threshold": 0.01, "timestamp": "2025-01-29T11:45:00"}
      ]
    },
    "finalized": {
      "completed": false
    }
  },

  "metadata": {
    "command_history": [
      "gaphack-decompose input.fasta -o output/ --max-clusters 50",
      "gaphack-decompose --resume output/ --resolve-conflicts",
      "gaphack-decompose --resume output/ --refine-close-clusters 0.02"
    ],
    "start_time": "2025-01-29T10:00:00",
    "last_checkpoint": "2025-01-29T12:15:23",
    "gaphack_version": "0.5.0"
  }
}
```

**Key points:**
- No cluster membership data in JSON
- File patterns (not lists) point to FASTA files
- Statistics are for reporting only
- Command history tracks all operations
- Refinement history allows tracking multiple passes

## CLI Interface

### Primary Command

```bash
gaphack-decompose [input.fasta] [options]
```

**New Options:**
- `-o/--output DIR` - Output directory (default: `decompose_output/`)
- `--resume` - Resume from existing output directory (no input.fasta needed)
- `--checkpoint-interval N` - Save state every N iterations (default: 5)
- `--force-input-change` - Allow resume with modified input FASTA
- `--keep-blast-db` - Keep BLAST database after completion (default: keep)
- `--finalize` - Create final numbered output and mark as complete

**Existing Options** (unchanged):
- `--targets FASTA` - Target sequences for directed mode
- `--max-clusters N` - Maximum clusters to create
- `--max-sequences N` - Maximum sequences to process
- `--resolve-conflicts` - Enable conflict resolution
- `--refine-close-clusters DIST` - Enable close cluster refinement
- `--blast-max-hits N` - BLAST parameters
- `--min-split DIST`, `--max-lump DIST`, etc. - Clustering parameters

### Use Cases and Examples

#### Use Case 1: Basic Run with Automatic Checkpointing

```bash
# Start clustering (checkpointing enabled by default)
gaphack-decompose input.fasta -o results/

# Interrupted? Resume automatically
gaphack-decompose --resume results/

# Finalize when complete
gaphack-decompose --resume results/ --finalize
```

#### Use Case 2: Incremental Cluster Addition

```bash
# Initial clustering with limit
gaphack-decompose input.fasta -o results/ --max-clusters 50

# Review results, decide to add more (absolute count)
gaphack-decompose --resume results/ --max-clusters 100

# Add even more
gaphack-decompose --resume results/ --max-clusters 150

# Final pass - cluster all remaining sequences
gaphack-decompose --resume results/

# Finalize
gaphack-decompose --resume results/ --finalize
```

**Note**: `--max-clusters` is **absolute** - specifying 100 means "process until 100 total clusters", not "add 100 more clusters".

#### Use Case 3: Staged Refinement

```bash
# Initial clustering only (no refinement)
gaphack-decompose input.fasta -o results/

# Later: resolve conflicts
gaphack-decompose --resume results/ --resolve-conflicts

# Later: refine close clusters
gaphack-decompose --resume results/ --refine-close-clusters 0.02

# Later: even tighter refinement (chains on previous)
gaphack-decompose --resume results/ --refine-close-clusters 0.01

# Finalize
gaphack-decompose --resume results/ --finalize
```

**Note**: Refinement stages **chain** - each refinement applies to the most recent cluster state. To start fresh, copy the directory before applying new refinement.

#### Use Case 4: One-Shot with All Refinements

```bash
# Run everything at once
gaphack-decompose input.fasta -o results/ \
  --resolve-conflicts \
  --refine-close-clusters 0.02

# System reboots or user interrupts - resume automatically
gaphack-decompose --resume results/

# Finalize when complete
gaphack-decompose --resume results/ --finalize
```

#### Use Case 5: Exploratory Analysis with Manual Edits

```bash
# Initial clustering
gaphack-decompose input.fasta -o explore/

# Manually inspect and edit FASTA files
# - Merge similar clusters by concatenating files
# - Remove outlier sequences
# - Split clusters by creating new files

# Resume with manual edits incorporated
gaphack-decompose --resume explore/ --refine-close-clusters 0.02

# Finalize
gaphack-decompose --resume explore/ --finalize
```

## Implementation Architecture

### New Classes

```python
@dataclass
class DecomposeState:
    """Persistent state for incremental decompose runs.

    Contains only metadata, parameters, and file paths.
    Never contains cluster membership data.
    """
    version: str
    status: str  # "in_progress", "completed"
    stage: str   # current stage

    input: Dict[str, Any]  # path, hash, sequence counts
    parameters: Dict[str, Any]  # clustering parameters
    stages: Dict[str, Dict[str, Any]]  # stage completion info
    metadata: Dict[str, Any]  # command history, timestamps

    def save(self, output_dir: Path) -> None:
        """Save state to output_dir/state.json."""

    @classmethod
    def load(cls, output_dir: Path) -> 'DecomposeState':
        """Load state from output_dir/state.json."""

    def validate_input_hash(self, input_fasta: str, force: bool = False) -> bool:
        """Verify input FASTA matches recorded hash."""

    def get_current_stage_files(self) -> List[Path]:
        """Get FASTA files for current stage."""

    def can_continue_clustering(self) -> bool:
        """Check if more initial clustering can be added."""

    def can_apply_refinement(self, refinement_type: str) -> bool:
        """Check if refinement stage can be applied."""

    def update_stage_completion(self, stage: str, stats: Dict) -> None:
        """Mark stage complete and record statistics."""
```

```python
class StateManager:
    """Manages state persistence and reconstruction."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_file = output_dir / "state.json"

    def checkpoint(self, state: DecomposeState) -> None:
        """Save checkpoint (called periodically during clustering)."""

    def load_clusters_from_stage(self, stage: str) -> Dict[str, List[str]]:
        """Rebuild cluster dict by reading stage FASTA files."""

    def rebuild_assignment_tracker(self, clusters: Dict[str, List[str]],
                                   all_headers: List[str]) -> AssignmentTracker:
        """Reconstruct AssignmentTracker from FASTA files."""

    def detect_conflicts(self, stage_files: List[Path]) -> Dict[str, List[str]]:
        """Find sequences appearing in multiple cluster files."""

    def save_stage_results(self, results: DecomposeResults, stage: str,
                          sequences: List[str], headers: List[str]) -> None:
        """Save FASTA files for a stage with appropriate prefixes."""
```

### Modified DecomposeClustering

```python
class DecomposeClustering:
    """Main decompose clustering with incremental restart support."""

    def __init__(self, ...,
                 output_dir: Optional[Path] = None,
                 checkpoint_interval: int = 5,
                 state_manager: Optional[StateManager] = None):
        """Initialize with output directory and checkpoint settings."""
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.state_manager = state_manager

    def decompose(self, input_fasta: str,
                  state: Optional[DecomposeState] = None,
                  **kwargs) -> DecomposeResults:
        """Run decompose clustering with checkpointing.

        If state is provided, continues from that state.
        Otherwise starts fresh.
        """
        # Set up signal handler for clean interruption
        # Save checkpoint every N iterations
        # Update state after each major stage

    def _checkpoint_iteration(self, iteration: int,
                             results: DecomposeResults) -> None:
        """Save checkpoint if interval reached."""
        if iteration % self.checkpoint_interval == 0:
            self.state_manager.checkpoint(self.state)

    def _handle_interruption(self, signum, frame) -> None:
        """Handle KeyboardInterrupt gracefully."""
        # Finish current iteration
        # Save state
        # Exit cleanly
```

### Key Functions

```python
def resume_decompose(output_dir: Path,
                     max_clusters: Optional[int] = None,
                     max_sequences: Optional[int] = None,
                     resolve_conflicts: bool = False,
                     refine_close_clusters: float = 0.0,
                     force_input_change: bool = False,
                     **kwargs) -> DecomposeResults:
    """Resume decompose clustering from saved state.

    Determines current stage and continues appropriately:
    - If in initial_clustering: continue adding clusters
    - If refinement requested: apply refinement stage
    - If all complete and --finalize: create final output
    """
    # Load state
    state = DecomposeState.load(output_dir)

    # Validate input
    if not state.validate_input_hash(state.input['fasta_path'], force_input_change):
        raise ValueError("Input FASTA has changed. Use --force-input-change to override.")

    # Load current clusters from FASTA files
    state_manager = StateManager(output_dir)
    current_clusters = state_manager.load_clusters_from_stage(state.stage)

    # Rebuild assignment tracker
    all_headers = load_sequences_from_fasta(state.input['fasta_path'])[1]
    assignment_tracker = state_manager.rebuild_assignment_tracker(
        current_clusters, all_headers
    )

    # Determine action based on stage and parameters
    if not state.stages['initial_clustering']['completed']:
        # Continue initial clustering
        return continue_initial_clustering(state, max_clusters, max_sequences, **kwargs)
    elif resolve_conflicts and not state.stages['conflict_resolution']['completed']:
        # Apply conflict resolution
        return apply_conflict_resolution(state, **kwargs)
    elif refine_close_clusters > 0:
        # Apply close cluster refinement (chains on current state)
        return apply_close_cluster_refinement(state, refine_close_clusters, **kwargs)
    else:
        # Nothing to do
        print("Clustering already complete. Use --finalize to create final output.")
        return load_results_from_state(state)
```

```python
def finalize_decompose(output_dir: Path, cleanup: bool = False) -> None:
    """Create final numbered cluster output.

    - Reads most recent stage FASTA files
    - Renumbers clusters by size (cluster_001 = largest)
    - Writes final cluster_*.fasta files
    - Marks state as finalized
    - Optionally removes intermediate stage files
    """
    state = DecomposeState.load(output_dir)

    # Determine most recent stage
    if state.stages['close_cluster_refinement']['completed']:
        source_stage = 'refined'
    elif state.stages['conflict_resolution']['completed']:
        source_stage = 'deconflicted'
    else:
        source_stage = 'initial'

    # Load clusters from source stage
    clusters = StateManager(output_dir).load_clusters_from_stage(source_stage)

    # Renumber by size
    final_clusters = renumber_clusters_by_size(clusters)

    # Write final FASTA files
    for cluster_id, headers in final_clusters.items():
        write_cluster_fasta(output_dir / f"cluster_{cluster_id}.fasta", headers, ...)

    # Update state
    state.stages['finalized']['completed'] = True
    state.save(output_dir)

    # Optional cleanup
    if cleanup:
        remove_stage_files(output_dir, ['initial', 'deconflicted', 'refined'])
```

## BLAST Database Management

### Location and Persistence

BLAST databases are created in `{output_dir}/.blast/` to:
- Avoid cluttering input directory
- Prevent collisions when multiple runs use same input
- Persist across resumes for faster restart
- Allow cleanup by removing output directory

```python
def create_blast_database(input_fasta: str, output_dir: Path) -> Path:
    """Create BLAST database in output directory.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Output directory for decompose run

    Returns:
        Path to BLAST database (without extension)
    """
    blast_dir = output_dir / ".blast"
    blast_dir.mkdir(exist_ok=True)

    db_path = blast_dir / "blast_db"

    # Create database using makeblastdb
    subprocess.run([
        "makeblastdb",
        "-in", input_fasta,
        "-dbtype", "nucl",
        "-out", str(db_path)
    ])

    return db_path
```

### Cleanup Policy

By default, BLAST databases are kept after completion to allow resumption. Users can:
- Delete output directory to remove everything
- Use `--cleanup` with `--finalize` to remove intermediate files including BLAST DB
- Manually delete `.blast/` directory if needed

## Validation and Error Handling

### Input Validation on Resume

When resuming, validate:
1. **Output directory exists** and contains `state.json`
2. **Input FASTA exists** at recorded path
3. **Input hash matches** recorded hash (unless `--force-input-change`)
4. **Stage FASTA files exist** and are readable
5. **Parameter compatibility** for new operations

### Parameter Compatibility Rules

**Cannot change** (require fresh start):
- `min_split`, `max_lump`, `target_percentile` during initial clustering continuation
- Input FASTA (without `--force-input-change`)

**Can change** (allowed for new stages):
- `--resolve-conflicts` (add if not yet applied)
- `--refine-close-clusters` (add or change threshold, chains on current)
- `--max-clusters`, `--max-sequences` (new limits for continuation)
- BLAST parameters (for new clustering work)

### Error Recovery

**Corrupted state.json**:
- Cannot resume automatically
- User must manually reconstruct or start fresh
- Document JSON schema for manual recovery

**Missing FASTA files**:
- Error and refuse to resume
- List missing files in error message
- User must restore files or start fresh

**Hash mismatch on input**:
- Error by default
- Allow with `--force-input-change` flag
- Warn about potential inconsistencies

**Interrupted during checkpoint**:
- State file writes are atomic (write to temp, rename)
- If corrupted, fall back to previous checkpoint
- Log checkpoint history for recovery

## Future Enhancements

### Incremental Sequence Addition

Design space for future work: adding new sequences to existing clusters with minimal disruption.

Potential approach:
```bash
gaphack-decompose --add-sequences new_seqs.fasta --resume results/
```

Would need:
- Strategy for assigning new sequences to existing clusters vs creating new clusters
- Heuristics for minimal disruption (e.g., only create new clusters if necessary)
- Conflict detection between new and existing sequences
- Decision on whether to re-run refinement after addition

This is noted as future work and not part of the current implementation.

### Parallel Resumption

For very large datasets, might want to run multiple resume operations in parallel:
- Partition unassigned sequences
- Run multiple decompose instances
- Merge results

Would require:
- Shared state coordination
- Lock files or atomic operations
- Merge strategy for results

Also noted as future work.

### Rollback to Previous Stage

Ability to discard refinement and return to earlier stage:
```bash
gaphack-decompose --rollback conflict_resolution --resume results/
```

Could be implemented by:
- Tracking stage dependencies in state
- Removing superseded stage files
- Updating state to previous stage

Low priority - users can copy directory before refinement instead.

## Migration Plan

### Version 0.4.0 → 0.5.0

**Breaking Changes**:
- `-o/--output` now specifies directory instead of prefix
- Default output directory is `decompose_output/` instead of using input filename
- BLAST database location moved to output directory

**Migration for existing workflows**:
```bash
# Old (0.4.0):
gaphack-decompose input.fasta -o results

# Would create: results.cluster_001.fasta, results.state.json

# New (0.5.0):
gaphack-decompose input.fasta -o results/

# Creates: results/cluster_001.fasta, results/state.json
```

**Compatibility**:
- No automatic migration from 0.4.0 results (fresh start required)
- Document migration as breaking change in CHANGELOG
- Since no user base, breaking change is acceptable

## Testing Strategy

### Unit Tests

- `DecomposeState` serialization/deserialization
- Input hash validation
- Stage file discovery and loading
- AssignmentTracker reconstruction
- Conflict detection from FASTA files
- Parameter validation logic

### Integration Tests

- Full workflow: initial → resume → refine → finalize
- Interruption recovery (simulate KeyboardInterrupt)
- Manual FASTA edits between stages
- Chain refinement with different thresholds
- Error cases: missing files, hash mismatch, corrupted state

### Regression Tests

- Ensure final output matches non-incremental runs
- Verify checkpoint/resume produces identical results to non-interrupted
- Validate that manual FASTA edits are respected

## Documentation Updates

### README.md

Add section on incremental restart:
- Quick examples of resume workflow
- Explanation of checkpoint-by-default behavior
- Note about FASTA as source of truth

### CLI Help Text

Update `gaphack-decompose --help` with:
- New `-o` directory behavior
- `--resume` usage
- `--finalize` explanation
- Examples of incremental workflows

### CLAUDE.md

Add implementation notes:
- State management architecture
- FASTA reconstruction approach
- Design decisions and rationale
- Future work: incremental sequence addition

## Implementation Phases

### Phase 1: State Persistence Foundation (Priority: High)

**Goal**: Basic state saving and loading infrastructure

**Tasks**:
- Implement `DecomposeState` class with JSON serialization
- Implement `StateManager` for file operations
- Add output directory structure creation
- Move BLAST database to `{output_dir}/.blast/`
- Add basic checkpoint saving (without interruption handling yet)

**Testing**: Unit tests for state serialization and file management

**Deliverable**: State files are created but not yet used for resumption

### Phase 2: Initial Clustering Continuation (Priority: High)

**Goal**: Resume interrupted or limited initial clustering

**Tasks**:
- Implement FASTA file reading to reconstruct cluster state
- Rebuild `AssignmentTracker` from FASTA files
- Add `--resume` CLI flag with validation
- Support `--max-clusters` and `--max-sequences` continuation
- Implement input hash validation

**Testing**: Integration test for interrupted and limited clustering resumption

**Deliverable**: Can resume initial clustering after interruption or limits

### Phase 3: Graceful Interruption Handling (Priority: Medium)

**Goal**: Clean shutdown on Ctrl+C

**Tasks**:
- Add signal handler for KeyboardInterrupt
- Implement atomic state file writes (temp + rename)
- Finish current iteration before saving
- Add checkpoint interval configuration

**Testing**: Simulate interruptions at various points in workflow

**Deliverable**: Ctrl+C cleanly saves state and can resume

### Phase 4: Staged Refinement (Priority: Medium)

**Goal**: Apply refinement stages to existing clustering

**Tasks**:
- Refactor `_resolve_conflicts()` to work with loaded state
- Refactor `_refine_close_clusters()` to work with loaded state
- Implement chained refinement (apply to current FASTA state)
- Track refinement history in state
- Add stage completion flags

**Testing**: Integration tests for adding refinements after initial clustering

**Deliverable**: Can apply and chain refinement stages independently

### Phase 5: Finalization (Priority: Low)

**Goal**: Create clean final output

**Tasks**:
- Implement `--finalize` command
- Renumber clusters from most recent stage
- Optional cleanup of intermediate files
- Mark state as finalized

**Testing**: Verify finalized output matches non-incremental runs

**Deliverable**: Clean final output with optional intermediate cleanup

### Phase 6: Documentation and Polish (Priority: Low)

**Goal**: Complete user-facing documentation

**Tasks**:
- Update README with incremental examples
- Update CLI help text
- Add troubleshooting guide
- Document manual FASTA editing workflows
- Add CLAUDE.md implementation notes

**Testing**: User acceptance testing with example workflows

**Deliverable**: Complete documentation for incremental features

## Open Questions

1. **Checkpoint storage size**: Should we implement checkpoint rotation (keep last N checkpoints) or just one checkpoint?
   - **Recommendation**: Single checkpoint is sufficient. State file is small (few KB).

2. **Conflict detection performance**: For large datasets with many clusters, scanning all FASTA files for conflicts could be slow. Should we cache conflict detection results?
   - **Recommendation**: Start simple (scan on resume). Optimize later if needed.

3. **Stage file naming**: Should internal cluster IDs be sequential within stage (`initial.cluster_0001`) or globally unique (`initial.cluster_1847`)?
   - **Recommendation**: Sequential within stage for readability. Mapping tracked in state.

4. **BLAST database updates**: If using `--force-input-change` with added sequences, should we update BLAST DB or recreate?
   - **Recommendation**: Recreate (simpler, safer). Document that `--force-input-change` recreates BLAST DB.

5. **Resume without input specification**: Should `--resume` auto-detect input FASTA from state, or require explicit path?
   - **Recommendation**: Auto-detect from state. Makes resume simpler: `gaphack-decompose --resume output/`

## Success Criteria

Incremental restart implementation is successful when:

1. ✅ User can resume interrupted runs without data loss
2. ✅ User can add clusters incrementally with `--max-clusters` limits
3. ✅ User can apply refinement stages after initial clustering
4. ✅ User can chain multiple refinement passes
5. ✅ Manual FASTA edits between stages are respected
6. ✅ Final output matches non-incremental equivalent runs
7. ✅ State file is human-readable for debugging
8. ✅ Clear error messages for invalid resume attempts
9. ✅ Documentation covers all incremental workflows
10. ✅ Backwards compatibility: old CLI usage still works with directory interpretation

## Conclusion

This incremental restart design provides:
- **Flexibility**: Stop and continue at natural breakpoints
- **Safety**: Long runs won't lose progress on interruption
- **Exploration**: Try different refinement strategies on same initial clustering
- **Transparency**: FASTA files as source of truth, editable by users
- **Simplicity**: Flat file structure, clear state management

The design prioritizes simplicity and correctness over optimization, with clear paths for future enhancement.