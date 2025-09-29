# Phase 3 Follow-up Issues

## Critical Issues to Address

### Issue 1: Post-processing runs after interruption
**Problem**: When user presses Ctrl+C during initial clustering, the code breaks out of the loop but then continues to run conflict resolution and close cluster refinement stages before exiting.

**Expected behavior**: Save checkpoint and exit immediately without running refinement stages.

**Fix location**: `decompose.py` around line 838-851
- Need to check `interruption_requested` flag before running `_resolve_conflicts()` and `_refine_close_clusters_via_refinement()`
- Add early return after checkpoint save if interrupted

**Code change needed**:
```python
# After main clustering loop, before refinement
if interruption_requested['flag']:
    self.logger.info("Skipping refinement stages due to interruption")
    # Return early with current results
    return results

# Apply cluster refinement for conflict resolution if enabled
if getattr(self, 'resolve_conflicts', False) and results.conflicts:
    ...
```

---

### Issue 2: No detection of partial state on fresh run
**Problem**: Running `gaphack-decompose input.fasta -o output/` twice overwrites existing state without warning, even if previous run was interrupted.

**Expected behavior**:
- Detect if `output/state.json` exists
- If state shows `status="in_progress"`, error with message: "Output directory contains partial state. Use --resume to continue or --force to overwrite."
- If state shows `status="completed"`, error with: "Output directory contains completed run. Use different output directory or --force to overwrite."

**Fix location**: `decompose_cli.py` around line 483-500, before initializing decomposer

**Code change needed**:
```python
# Before creating decomposer (line ~483)
if not args.resume:
    state_file = output_dir / "state.json"
    if state_file.exists():
        # Load existing state
        from .state import DecomposeState
        try:
            existing_state = DecomposeState.load(output_dir)
            if existing_state.status == "in_progress":
                logger.error(f"Output directory contains partial state from interrupted run")
                logger.error(f"Use --resume to continue, or delete {output_dir} to start fresh")
                sys.exit(1)
            elif existing_state.status == "completed":
                logger.error(f"Output directory contains completed run")
                logger.error(f"Use different output directory or delete {output_dir}")
                sys.exit(1)
        except Exception as e:
            logger.warning(f"Could not load existing state: {e}")
            logger.error(f"Output directory {output_dir} exists. Delete it or use different path.")
            sys.exit(1)
```

---

### Issue 3: Multiprocessing signal handling
**Problem**: The signal handler is only installed in the main process. During conflict resolution and close cluster refinement, core gapHACk uses multiprocessing workers which may not handle SIGINT properly.

**Investigation needed**:
1. How does Python's `signal.signal()` interact with `multiprocessing.Pool`?
2. Are SIGINT signals delivered to worker processes or only parent?
3. Do workers need their own signal handlers?
4. Should we use `Pool.terminate()` on interruption?

**OS behavior (Unix/POSIX)**:
- Ctrl+C sends SIGINT to foreground process group (all processes)
- Python's multiprocessing may mask signals in workers
- Workers may need explicit signal handling or parent needs to terminate them

**Fix location**:
- `core.py` multiprocessing setup (if workers need handlers)
- `decompose.py` signal handler (if parent needs to terminate pool)

**Possible approach**:
```python
# In decompose.py signal handler
def handle_interruption(signum, frame):
    if not interruption_requested['flag']:
        interruption_requested['flag'] = True
        self.logger.info("\nInterruption received...")

        # If multiprocessing pool is active, terminate it
        if hasattr(self, '_active_pool') and self._active_pool:
            self.logger.debug("Terminating multiprocessing workers...")
            self._active_pool.terminate()
            self._active_pool.join(timeout=5)
    else:
        # Force exit
        signal.signal(signal.SIGINT, original_handler)
        raise KeyboardInterrupt
```

**Testing needed**:
1. Interrupt during initial clustering (single-threaded BLAST) - WORKS ✅
2. Interrupt during conflict resolution (multiprocessing gapHACk) - NEEDS TESTING ⚠️
3. Interrupt during close cluster refinement (multiprocessing gapHACk) - NEEDS TESTING ⚠️
4. Check for zombie processes after interruption
5. Check that temp files are cleaned up

---

## Testing Plan

Create `tests/test_interruption_edge_cases.py`:

```python
def test_interrupt_skips_refinement():
    """Test that interruption during clustering skips refinement stages."""
    # Set resolve_conflicts=True but interrupt during clustering
    # Verify refinement didn't run

def test_detect_partial_state_on_restart():
    """Test that rerunning same command detects partial state."""
    # Run with max_clusters=3 to create partial state
    # Try running same command again without --resume
    # Should error with helpful message

def test_multiprocessing_cleanup_on_interrupt():
    """Test that worker processes are cleaned up on interrupt."""
    # Run large enough dataset to trigger multiprocessing in refinement
    # Interrupt during refinement
    # Check for zombie processes
```

---

## Priority

1. **CRITICAL**: Issue 1 (post-processing on interrupt) - breaks user expectation
2. **HIGH**: Issue 2 (partial state detection) - prevents accidental data loss
3. **MEDIUM**: Issue 3 (multiprocessing) - works in most cases but edge case exists

## Estimated Effort

- Issue 1: 30 minutes (straightforward flag check)
- Issue 2: 1 hour (need validation logic and good error messages)
- Issue 3: 2-4 hours (requires investigation, testing, and possibly core.py changes)

**Total**: ~4-6 hours for Phase 3.5 cleanup