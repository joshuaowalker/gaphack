# Refactoring decompose.py for Maintainability

## Overview

The `decompose.py` file has grown to 2,049 lines during the incremental restart implementation, with the `DecomposeClustering` class alone being 1,150 lines (56% of the file). This document outlines a refactoring plan to improve maintainability by extracting logically cohesive modules.

## Current State

### File Structure Analysis
```
Total lines: 2,049

class    ClusterIDGenerator                  Lines   19-  37 (  19 lines,   0.9%)
class    ProcessingStageInfo                 Lines   38-  47 (  10 lines,   0.5%)
class    DecomposeResults                    Lines   48-  66 (  19 lines,   0.9%)
class    AssignmentTracker                   Lines   67- 117 (  51 lines,   2.5%)
class    TargetSelector                      Lines  118- 152 (  35 lines,   1.7%)
class    BlastResultMemory                   Lines  153- 183 (  31 lines,   1.5%)
class    NearbyTargetSelector                Lines  184- 258 (  75 lines,   3.7%)
class    DecomposeClustering                 Lines  259-1408 (1150 lines,  56.1%)
function resume_decompose                    Lines 1409-1576 ( 168 lines,   8.2%)
function _continue_initial_clustering        Lines 1577-1650 (  74 lines,   3.6%)
function _apply_conflict_resolution_stage    Lines 1651-1776 ( 126 lines,   6.1%)
function _apply_close_cluster_refinement_stage Lines 1777-1896 ( 120 lines,   5.9%)
function finalize_decompose                  Lines 1897-2050 ( 154 lines,   7.5%)
```

### Problem Areas
1. **DecomposeClustering class**: 1,150 lines is too large for a single class
2. **Mixed concerns**: Target selection, core clustering, resume logic all in one file
3. **Difficult navigation**: Finding specific functionality requires scrolling through 2,000+ lines
4. **Testing complexity**: Hard to test individual concerns in isolation

## Refactoring Plan

### Phase 1: Extract Target Selection Logic

**New file**: `gaphack/target_selection.py` (~150 lines)

**Classes to move**:
- `TargetSelector` (35 lines) - Base interface for target selection strategies
- `BlastResultMemory` (31 lines) - Memory pool for BLAST neighborhoods
- `NearbyTargetSelector` (75 lines) - Spiral target selector for systematic coverage

**Rationale**: These three classes form a cohesive unit responsible for target selection during iterative clustering. They are independent of the core decompose logic and resume functionality.

**Dependencies**:
- Import `AssignmentTracker` from `decompose` (will remain there)
- No circular dependencies expected

### Phase 2: Extract Resume/Checkpoint Logic

**New file**: `gaphack/resume.py` (~440 lines)

**Functions to move**:
- `resume_decompose()` (168 lines) - Main resume entry point
- `_continue_initial_clustering()` (74 lines) - Resume initial clustering phase
- `_apply_conflict_resolution_stage()` (126 lines) - Apply/resume conflict resolution
- `_apply_close_cluster_refinement_stage()` (120 lines) - Apply/resume close cluster refinement
- `finalize_decompose()` (154 lines) - Create final numbered output

**Rationale**: All resume and finalization logic is distinct from the core iterative clustering. This separation makes the checkpoint/resume system easier to understand and modify.

**Dependencies**:
- Import `DecomposeClustering`, `DecomposeResults` from `decompose`
- Import `DecomposeState` from `state`
- Import refinement functions from `cluster_refinement`
- No circular dependencies expected

### Phase 3: Keep Core Decompose Logic

**Remaining in**: `gaphack/decompose.py` (~1,460 lines)

**Classes to keep**:
- `ClusterIDGenerator` (19 lines) - Generates cluster IDs
- `ProcessingStageInfo` (10 lines) - Tracks processing stage metadata
- `DecomposeResults` (19 lines) - Result container
- `AssignmentTracker` (51 lines) - Tracks sequence assignments
- `DecomposeClustering` (1,150 lines) - Main orchestrator

**Rationale**: These classes are tightly coupled to the core decompose algorithm and represent the primary decompose functionality. Keeping them together maintains cohesion.

## Refactoring Principles

### 1. Single Responsibility
- **Target Selection**: Handles all target selection strategies
- **Resume Logic**: Handles all checkpoint/resume/finalization operations
- **Core Decompose**: Handles iterative BLAST-based clustering

### 2. Minimize Coupling
- Extract modules should have minimal dependencies on each other
- Avoid circular imports by careful dependency ordering
- Use explicit imports rather than `from module import *`

### 3. Preserve Test Coverage
- All 355 existing tests must continue to pass
- No changes to public APIs or behavior
- Only internal reorganization

### 4. Maintain Documentation
- Update import statements in all affected files
- Update CLAUDE.md with new file structure
- Keep docstrings intact during moves

## Implementation Steps

### Step 1: Create target_selection.py
1. Create new file `gaphack/target_selection.py`
2. Copy classes: `TargetSelector`, `BlastResultMemory`, `NearbyTargetSelector`
3. Add necessary imports (typing, logging, AssignmentTracker)
4. Add module docstring explaining purpose

### Step 2: Update decompose.py for target_selection
1. Remove moved classes from `decompose.py`
2. Add import: `from .target_selection import TargetSelector, BlastResultMemory, NearbyTargetSelector`
3. Verify no broken references

### Step 3: Create resume.py
1. Create new file `gaphack/resume.py`
2. Copy functions: `resume_decompose`, `_continue_initial_clustering`, `_apply_conflict_resolution_stage`, `_apply_close_cluster_refinement_stage`, `finalize_decompose`
3. Add necessary imports (Path, typing, DecomposeClustering, DecomposeResults, DecomposeState, refinement functions)
4. Add module docstring explaining purpose

### Step 4: Update decompose.py for resume
1. Remove moved functions from `decompose.py`
2. Keep internal resume support in `DecomposeClustering.decompose()` method (signal handlers, checkpoint saving)
3. No import needed in decompose.py (resume.py imports from decompose.py, not vice versa)

### Step 5: Update decompose_cli.py
1. Update imports to: `from .resume import resume_decompose, finalize_decompose`
2. Verify CLI still works correctly

### Step 6: Update __init__.py
1. Add exports for new modules if needed for public API
2. Verify backward compatibility

### Step 7: Run Tests
1. Run full test suite: `pytest`
2. Verify all 355 tests pass
3. Check for any import errors or missing references

### Step 8: Update Documentation
1. Update `CLAUDE.md` project structure section
2. Update any references to file organization
3. Update module-level docstrings

## Verification Checklist

### Functional Verification
- [ ] All 355 tests pass without modification
- [ ] CLI commands work: `gaphack-decompose`, `--resume`, `--finalize`
- [ ] No behavioral changes in clustering results
- [ ] No performance regressions

### Code Quality Verification
- [ ] No circular imports
- [ ] All imports resolve correctly
- [ ] No broken references to moved code
- [ ] Docstrings preserved and accurate
- [ ] Type hints intact

### File Organization Verification
- [ ] `decompose.py`: ~1,460 lines
- [ ] `target_selection.py`: ~150 lines (new)
- [ ] `resume.py`: ~440 lines (new)
- [ ] Total lines unchanged: ~2,050 lines

### Documentation Verification
- [ ] CLAUDE.md updated with new file structure
- [ ] Module docstrings explain purpose
- [ ] No references to old file organization

## Assumptions

1. **No API changes**: All public functions maintain same signatures
2. **Import changes only**: Only import statements change in calling code
3. **Test stability**: Tests don't depend on internal file organization
4. **Single-pass refactoring**: Can complete in one session without breaking intermediate states

## Potential Issues and Mitigations

### Issue 1: Circular Imports
**Risk**: `resume.py` imports from `decompose.py` and vice versa

**Mitigation**:
- `resume.py` imports from `decompose.py` (DecomposeClustering, DecomposeResults)
- `decompose.py` does NOT import from `resume.py`
- Resume functionality called from CLI, not from decompose internals

### Issue 2: Missing Dependencies
**Risk**: Moved code might have implicit dependencies not caught until runtime

**Mitigation**:
- Run tests after each extraction phase
- Use static analysis (mypy) to catch import issues
- Verify all imports explicitly

### Issue 3: Test Failures
**Risk**: Tests might import from old locations

**Mitigation**:
- Check test imports after refactoring
- Most tests import from high-level modules (__init__.py) which remain stable
- Run incremental test suites during refactoring

## Post-Refactoring Maintenance

### File Size Targets
- Keep individual files under 1,500 lines when possible
- Extract modules when logical boundaries exist
- Maintain single responsibility per module

### Future Refactoring Opportunities
If `DecomposeClustering` grows beyond 1,200 lines again, consider extracting:
- **Distance provider management**: Methods related to distance calculation
- **BLAST operations**: Methods related to BLAST database and neighborhood discovery
- **Result assembly**: Methods related to building DecomposeResults

## Success Criteria

✅ All 355 tests pass
✅ No circular imports
✅ File sizes reduced to manageable levels
✅ Logical separation of concerns achieved
✅ Documentation updated
✅ No behavioral changes
✅ Import statements updated in all affected files

## Timeline Estimate

- **Step 1-2** (target_selection): 15 minutes
- **Step 3-4** (resume): 20 minutes
- **Step 5-6** (CLI updates): 10 minutes
- **Step 7** (testing): 15 minutes
- **Step 8** (documentation): 10 minutes

**Total**: ~70 minutes for complete refactoring with verification

## Rollback Plan

If issues arise:
1. Git revert to commit before refactoring started
2. Create a branch for refactoring work
3. Test thoroughly in branch before merging to main

The refactoring can be done atomically in a single commit, or split into two commits (target_selection, then resume) for easier rollback if needed.