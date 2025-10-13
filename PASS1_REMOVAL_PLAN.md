# Plan: Remove Pass 1 from gaphack-refine

## Executive Summary

Remove Pass 1 (conflict resolution + isolated individual refinement) from gaphack-refine based on empirical observation that Pass 2's first iteration with neighborhood context provides equivalent or better results. This simplifies the codebase by ~800-1000 lines while maintaining clustering quality.

**Key Insight**: Pass 1 refines clusters in isolation without inter-cluster context, which is suboptimal for gap calculation. Pass 2's first iteration with proper neighborhood context subsumes all Pass 1 functionality while providing better results.

## Rationale

1. **Empirical Evidence**: Running `--pass2-only` produces similar results to full two-pass refinement
2. **Redundancy**: Pass 1's isolated refinement is strictly worse than Pass 2's neighborhood-based refinement
3. **Conflict Rarity**: Modern clustering tools (vsearch, CD-HIT) produce MECE output; conflicts are rare
4. **Natural Resolution**: Iterative refinement naturally resolves any conflicts during first iteration

## 1. Functions to Remove

### `gaphack/cluster_refinement.py`

**`pass1_resolve_and_split()` (lines 702-871, ~170 lines)**
- Orchestrates Pass 1: conflict resolution + individual cluster refinement
- Calls `resolve_conflicts()` and then refines remaining clusters
- Builds ProcessingStageInfo for tracking
- **Why remove**: Entire Pass 1 logic being eliminated

**`resolve_conflicts()` (lines 1196-1373, ~180 lines)**
- Graph-based conflict resolution using DFS component analysis
- Groups conflicted clusters into components
- Runs full gapHACk on each component
- **Why remove**: Conflicts will be handled naturally by iterative refinement

**`find_conflict_components()` (lines 555-594, ~40 lines)**
- Uses DFS to find connected components in conflict graph
- Builds cluster adjacency graph where edges represent shared sequences
- **Why remove**: Part of conflict resolution infrastructure

**`get_all_conflicted_cluster_ids()` (lines 413-429, ~17 lines)**
- Extracts set of all cluster IDs that have conflicts
- Helper for conflict resolution
- **Why remove**: Part of conflict resolution infrastructure

**Total lines removed**: ~407 lines from cluster_refinement.py

## 2. Functions to Rename

### `pass2_iterative_merge()` → `iterative_refinement()`

**Location**: `gaphack/cluster_refinement.py` (lines 929-1077, ~150 lines)

**Current signature**:
```python
def pass2_iterative_merge(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: RefinementConfig,
    cluster_id_generator: ClusterIDGenerator,
    show_progress: bool = True
) -> Tuple[Dict[str, List[str]], ProcessingStageInfo]:
```

**New signature** (unchanged except name):
```python
def iterative_refinement(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: RefinementConfig,
    cluster_id_generator: ClusterIDGenerator,
    show_progress: bool = True
) -> Tuple[Dict[str, List[str]], ProcessingStageInfo]:
```

**Changes**:
- Rename function
- Update docstring to remove "Pass 2" references
- Update stage_name from "Pass 2: Iterative Merge" to "Iterative Refinement"

## 3. Functions to Simplify

### `two_pass_refinement()` → `refine_clusters()`

**Location**: `gaphack/cluster_refinement.py` (lines 1080-1193, ~115 lines)

**Current signature**:
```python
def two_pass_refinement(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    conflicts: Dict[str, List[str]],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: RefinementConfig,
    cluster_id_generator: ClusterIDGenerator,
    run_pass1: bool = True,
    run_pass2: bool = True,
    show_progress: bool = True
) -> Tuple[Dict[str, List[str]], List[ProcessingStageInfo]]:
```

**New signature**:
```python
def refine_clusters(
    all_clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    min_split: float,
    max_lump: float,
    target_percentile: int,
    config: RefinementConfig,
    cluster_id_generator: ClusterIDGenerator,
    show_progress: bool = True
) -> Tuple[Dict[str, List[str]], ProcessingStageInfo]:
```

**Key changes**:
- Remove `conflicts` parameter (no longer needed)
- Remove `run_pass1` and `run_pass2` parameters (always run iterative refinement)
- Return single `ProcessingStageInfo` instead of `List[ProcessingStageInfo]` (only one stage now)
- Simplify logic: just call `iterative_refinement()` directly
- Remove all conditional logic for running Pass 1 vs Pass 2
- Update docstring to reflect single-stage refinement

## 4. CLI Changes (`gaphack/refine_cli.py`)

### Arguments to Remove

**`--pass1-only`** (line 598-599):
```python
parser.add_argument('--pass1-only', action='store_true',
                   help='Run only Pass 1 (resolve conflicts and split)')
```

**`--pass2-only`** (line 600-601):
```python
parser.add_argument('--pass2-only', action='store_true',
                   help='Run only Pass 2 (iterative merge)')
```

**`--refine-close-clusters`** (lines 596-597):
```python
parser.add_argument('--refine-close-clusters', action='store_true',
                   help='Refine clusters within close distance threshold after main refinement')
```

### Arguments to Modify

**`--close-threshold`** (lines 593-595):
- Current: Optional parameter with default None
- New: **Required parameter** (no default)
- Rationale: Close cluster refinement is now core functionality, not optional

```python
# OLD
parser.add_argument('--close-threshold', type=float, default=None,
                   help='Distance threshold for close cluster refinement (default: max_lump)')

# NEW
parser.add_argument('--close-threshold', type=float, required=True,
                   help='Distance threshold for finding nearby clusters during refinement')
```

### Main Function Updates

**Remove conflict detection** (lines ~130-145):
```python
# REMOVE THIS BLOCK
conflicts = detect_conflicts(clusters)
if conflicts:
    logger.info(f"Detected {len(conflicts)} sequences in multiple clusters")
    if not args.pass1_only and args.pass2_only:
        logger.error("Cannot run Pass 2 only with conflicts present")
        sys.exit(1)
else:
    logger.info("No conflicts detected - clusters are MECE")
```

**Simplify refinement call** (lines ~150-170):
```python
# OLD
final_clusters, tracking_stages = two_pass_refinement(
    all_clusters=clusters,
    sequences=sequences,
    headers=headers,
    conflicts=conflicts,
    min_split=args.min_split,
    max_lump=args.max_lump,
    target_percentile=args.target_percentile,
    config=config,
    cluster_id_generator=cluster_id_generator,
    run_pass1=not args.pass2_only,
    run_pass2=not args.pass1_only,
    show_progress=not args.quiet
)

# NEW
final_clusters, stage_info = refine_clusters(
    all_clusters=clusters,
    sequences=sequences,
    headers=headers,
    min_split=args.min_split,
    max_lump=args.max_lump,
    target_percentile=args.target_percentile,
    config=config,
    cluster_id_generator=cluster_id_generator,
    show_progress=not args.quiet
)
```

## 5. Summary Report Changes

### `generate_refinement_summary()` (lines 224-413)

**Remove Pass 1 section** (lines ~260-310):
- Remove all Pass 1 statistics and reporting
- Remove conflict resolution statistics
- Simplify to single "Iterative Refinement" section

**Before** (pseudo-structure):
```
=== Pass 1: Resolve and Split ===
- Initial clusters: X
- Conflicts: Y sequences
- Components resolved: Z
- After conflict resolution: A clusters
- Individual refinement: B→C clusters
- Final Pass 1: C clusters

=== Pass 2: Iterative Merge ===
- Initial: C clusters
- Iterations: N
- Final: D clusters
```

**After**:
```
=== Iterative Refinement ===
- Initial clusters: X
- Iterations: N
- Final clusters: D
- Converged scopes: M/N
- Duration: T seconds
```

### `generate_cluster_mapping_report()` (lines 151-221)

**Simplify to original→final mapping**:
- Remove intermediate stage tracking
- Single mapping: original cluster ID → final cluster ID(s)
- No need for Pass 1 → Pass 2 transitions

## 6. RefinementConfig Changes

**Location**: `gaphack/cluster_refinement.py` (lines 74-123)

**Remove or update**:
- `refine_close_clusters` field (line 90) - now always True (remove field)
- Documentation referring to Pass 1/Pass 2 distinction

**Updated docstring**:
```python
@dataclass
class RefinementConfig:
    """Configuration for iterative cluster refinement.

    Args:
        max_full_gaphack_size: Maximum size for full gapHACk refinement
        close_threshold: Distance threshold for finding nearby clusters
        max_iterations: Maximum number of refinement iterations
        k_neighbors: Number of nearest neighbors in proximity graph
        search_method: Method for proximity search ("blast" or "vsearch")
        random_seed: Seed for randomizing seed order (None = no randomization)
    """
```

## 7. Test File Updates

### `tests/test_two_pass_russula.py`

**Rename file**: `test_two_pass_russula.py` → `test_iterative_russula.py`

**Update class name**: `TestTwoPassRussulaIntegration` → `TestIterativeRussulaIntegration`

**Remove Pass 1 specific tests**:
- `test_pass1_conflict_resolution()` (lines 154-212) - REMOVE entirely
- `test_two_pass_convergence_behavior()` - UPDATE to call `refine_clusters()`
- `test_pass2_merge_behavior()` - RENAME to `test_iterative_merge_behavior()`

**Update all function calls**:
```python
# OLD
final_clusters, tracking_stages = two_pass_refinement(
    all_clusters=initial_clusters,
    sequences=sequences,
    headers=headers,
    conflicts={},
    min_split=0.005,
    max_lump=0.02,
    target_percentile=95,
    config=config,
    cluster_id_generator=cluster_id_generator,
    run_pass1=True,
    run_pass2=True,
    show_progress=False
)

# NEW
final_clusters, stage_info = refine_clusters(
    all_clusters=initial_clusters,
    sequences=sequences,
    headers=headers,
    min_split=0.005,
    max_lump=0.02,
    target_percentile=95,
    config=config,
    cluster_id_generator=cluster_id_generator,
    show_progress=False
)
```

**Update assertions**:
```python
# OLD
if len(tracking_stages) >= 2:
    pass2_info = tracking_stages[1]

# NEW
iterations = stage_info.summary_stats.get('iterations', 0)
convergence_reason = stage_info.summary_stats.get('convergence_reason', 'unknown')
```

### `tests/test_cluster_refinement.py`

**Update all tests**:
- Remove tests for `pass1_resolve_and_split()`, `resolve_conflicts()`, `find_conflict_components()`
- Update `test_two_pass_refinement_*` tests to call `refine_clusters()`
- Remove conflict-related test cases
- Update assertions to work with single ProcessingStageInfo return

### `tests/test_refine_integration.py`

**Update CLI integration tests**:
- Remove `--pass1-only` and `--pass2-only` flag tests
- Add `--close-threshold` to all test commands
- Update expected output format (no Pass 1/Pass 2 distinction)

### `tests/test_cli.py`

**Update if needed**:
- Check for any references to two-pass refinement
- Update to single-stage refinement model

## 8. Documentation Updates

### `CLAUDE.md`

**Update "Refinement Implementation" section** (lines ~120-160):

**Remove**:
- Conflict Resolution Architecture subsection
- Pass 1/Pass 2 distinction

**Update to**:
```markdown
### Refinement Implementation
**Purpose**: Iterative neighborhood-based refinement to optimize clustering quality

**Key Components**:
- `iterative_refinement()`: Main refinement loop using proximity graph
- `refine_scope_with_gaphack()`: Runs full gapHACk on seed + neighborhood
- `build_refinement_scope()`: Builds seed cluster + neighbors within max_lump + context

**Refinement Process**:
1. Build proximity graph (BLAST/vsearch K-NN)
2. For each seed cluster (deterministic priority order):
   - Find nearby clusters within max_lump
   - Expand context beyond max_lump (gradient area)
   - Run full gapHACk on combined scope
   - Update clusters if changed
3. Iterate until convergence (AMI change < threshold)

**Convergence Criteria**:
- AMI change < 0.001 between iterations (stable clustering)
- All scopes converged (no changes in last 2 iterations)
- Maximum iterations reached (default: 10)
```

### `docs/REFINEMENT_DESIGN.md`

**If this file exists**, update to remove Pass 1 references and update architecture diagrams.

## 9. Expected Impact

### Lines of Code
- **cluster_refinement.py**: -407 lines (removed functions) + simplified `refine_clusters()`
- **refine_cli.py**: -50 lines (removed conflict detection, simplified main)
- **tests/test_two_pass_russula.py**: -100 lines (removed Pass 1 tests)
- **tests/test_cluster_refinement.py**: -150 lines (removed conflict resolution tests)
- **Documentation**: -100 lines (simplified explanations)

**Total**: ~800-1000 lines removed

### Functionality
- **No loss in clustering quality**: Pass 2 first iteration ≥ Pass 1 quality
- **Faster for MECE input**: No conflict detection overhead
- **Simpler mental model**: Single iterative refinement process
- **Clearer convergence**: Track single refinement process, not two stages

### Performance
- **MECE input (common case)**: Slightly faster (no conflict detection)
- **Conflicted input (rare case)**: Same or slightly slower (resolved in first iteration vs dedicated pass)

## 10. Migration Path

### For Users

**Old command** (with conflicts):
```bash
gaphack-refine --input-dir clusters/ --output-dir refined/
```

**New command** (explicit close-threshold):
```bash
gaphack-refine --input-dir clusters/ --output-dir refined/ --close-threshold 0.02
```

**Old command** (Pass 2 only):
```bash
gaphack-refine --input-dir clusters/ --output-dir refined/ --pass2-only
```

**New command** (same behavior):
```bash
gaphack-refine --input-dir clusters/ --output-dir refined/ --close-threshold 0.02
```

### For Developers

**Old API**:
```python
from gaphack.cluster_refinement import two_pass_refinement

final_clusters, stages = two_pass_refinement(
    all_clusters=clusters,
    sequences=sequences,
    headers=headers,
    conflicts=conflicts,
    min_split=0.005,
    max_lump=0.02,
    target_percentile=95,
    config=config,
    cluster_id_generator=id_gen,
    run_pass1=True,
    run_pass2=True
)
```

**New API**:
```python
from gaphack.cluster_refinement import refine_clusters

final_clusters, stage_info = refine_clusters(
    all_clusters=clusters,
    sequences=sequences,
    headers=headers,
    min_split=0.005,
    max_lump=0.02,
    target_percentile=95,
    config=config,
    cluster_id_generator=id_gen
)
```

## 11. Implementation Order

1. **`gaphack/cluster_refinement.py`**:
   - Rename `pass2_iterative_merge()` → `iterative_refinement()`
   - Simplify `two_pass_refinement()` → `refine_clusters()`
   - Remove `pass1_resolve_and_split()`, `resolve_conflicts()`, `find_conflict_components()`, `get_all_conflicted_cluster_ids()`
   - Update RefinementConfig

2. **`gaphack/refine_cli.py`**:
   - Remove `--pass1-only`, `--pass2-only`, `--refine-close-clusters` arguments
   - Make `--close-threshold` required
   - Remove conflict detection
   - Update call to `refine_clusters()`
   - Simplify summary report generation

3. **Tests**:
   - Update `test_two_pass_russula.py` → `test_iterative_russula.py`
   - Update `test_cluster_refinement.py`
   - Update `test_refine_integration.py`
   - Run full test suite

4. **Documentation**:
   - Update `CLAUDE.md`
   - Update any other docs

5. **Final verification**:
   - Run test suite
   - Test CLI with real data
   - Verify summary reports look correct

## 12. Risk Assessment

### Low Risk
- **Well-defined scope**: Clear set of functions to remove
- **Empirical validation**: User already tested that Pass 2 alone gives equivalent results
- **Comprehensive tests**: Extensive test suite will catch regressions

### Moderate Risk
- **API breaking change**: Developers using `two_pass_refinement()` directly will need updates
- **CLI breaking change**: Users must add `--close-threshold` parameter

### Mitigation
- Update all internal uses in same commit
- Clear migration documentation
- Version bump to indicate breaking change
