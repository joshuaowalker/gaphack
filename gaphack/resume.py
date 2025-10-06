"""
Resume and finalization logic for decompose clustering.

This module contains functions for:
- Resuming decompose clustering from checkpoints
- Continuing initial clustering after interruption
- Applying refinement stages (conflict resolution, close cluster refinement)
- Finalizing numbered output from intermediate stages

These functions enable incremental restart capabilities in gaphack-decompose mode.
"""

import logging
import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .decompose import (
    DecomposeResults,
    AssignmentTracker,
    ClusterIDGenerator,
    DecomposeClustering
)
from .state import DecomposeState, StateManager
from .utils import load_sequences_with_deduplication

logger = logging.getLogger(__name__)


def resume_decompose(output_dir: Path,
                     max_clusters: Optional[int] = None,
                     max_sequences: Optional[int] = None,
                     resolve_conflicts: bool = False,
                     refine_close_clusters: float = 0.0,
                     force_input_change: bool = False,
                     checkpoint_interval: int = 10,
                     **kwargs) -> DecomposeResults:
    """Resume decompose clustering from saved state.

    Determines current stage and continues appropriately:
    - If in initial_clustering: continue adding clusters
    - If refinement requested: apply refinement stage
    - If all complete: report completion status

    Args:
        output_dir: Output directory containing state.json
        max_clusters: New maximum cluster count (absolute, not incremental)
        max_sequences: New maximum sequence count (absolute, not incremental)
        resolve_conflicts: Whether to apply conflict resolution
        refine_close_clusters: Distance threshold for close cluster refinement (0.0 = disabled)
        force_input_change: Allow resuming with modified input FASTA
        checkpoint_interval: Save checkpoint every N iterations (default: 10)
        **kwargs: Additional parameters passed to DecomposeClustering

    Returns:
        DecomposeResults with updated clustering

    Raises:
        FileNotFoundError: If output directory or state file doesn't exist
        ValueError: If input FASTA has changed without force flag
    """
    logger.info(f"Resuming decompose from: {output_dir}")

    # Load state
    state = DecomposeState.load(output_dir)
    logger.info(f"Loaded state: stage={state.stage}, status={state.status}")

    # Validate input
    input_fasta = state.input.fasta_path
    if not Path(input_fasta).exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    state.validate_input_hash(input_fasta, force=force_input_change)

    # Load current clusters from FASTA files
    state_manager = StateManager(output_dir)
    current_stage_dir = state.get_current_stage_directory(output_dir)
    current_clusters = state_manager.load_clusters_from_stage_directory(current_stage_dir)
    logger.info(f"Loaded {len(current_clusters)} clusters from directory '{current_stage_dir}'")

    # Load unassigned sequences from current stage
    unassigned_file = current_stage_dir / "unassigned.fasta"
    if unassigned_file.exists():
        unassigned_headers = state_manager.load_unassigned_sequences(str(unassigned_file.relative_to(output_dir)))
    else:
        unassigned_headers = []
        logger.debug(f"No unassigned file found in {current_stage_dir}")

    # Load all sequences and headers
    sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)
    all_headers = []
    for headers_list in hash_to_headers.values():
        all_headers.extend(headers_list)

    # Rebuild assignment tracker
    assignment_tracker = state_manager.rebuild_assignment_tracker(current_clusters, all_headers)
    logger.info(f"Rebuilt assignment tracker: {len(assignment_tracker.assigned_sequences)} assigned")

    # Determine action based on stage and parameters
    if not state.initial_clustering.completed:
        # Continue initial clustering
        logger.info("Continuing initial clustering from checkpoint")
        return _continue_initial_clustering(
            state=state,
            state_manager=state_manager,
            input_fasta=input_fasta,
            assignment_tracker=assignment_tracker,
            current_clusters=current_clusters,
            max_clusters=max_clusters,
            max_sequences=max_sequences,
            resolve_conflicts=resolve_conflicts,
            refine_close_clusters=refine_close_clusters,
            checkpoint_interval=checkpoint_interval,
            **kwargs
        )
    elif resolve_conflicts and not state.conflict_resolution.completed:
        # Apply conflict resolution
        logger.info("Applying conflict resolution stage")
        return _apply_conflict_resolution_stage(
            state=state,
            state_manager=state_manager,
            input_fasta=input_fasta,
            current_clusters=current_clusters,
            assignment_tracker=assignment_tracker,
            **kwargs
        )
    elif refine_close_clusters > 0:
        # Apply close cluster refinement
        logger.info(f"Applying close cluster refinement with threshold={refine_close_clusters}")
        return _apply_close_cluster_refinement_stage(
            state=state,
            state_manager=state_manager,
            input_fasta=input_fasta,
            current_clusters=current_clusters,
            close_threshold=refine_close_clusters,
            **kwargs
        )
    else:
        # Nothing to do
        logger.info("Clustering already complete. No action requested.")
        logger.info(f"  Total clusters: {len(current_clusters)}")
        logger.info(f"  Unassigned sequences: {len(unassigned_headers)}")
        logger.info("Use --resolve-conflicts or --refine-close-clusters to apply refinement.")

        # Return results object from current state
        results = DecomposeResults()
        results.clusters = current_clusters
        results.all_clusters = current_clusters.copy()
        results.unassigned = unassigned_headers
        results.conflicts = assignment_tracker.get_conflicts()
        results.total_iterations = state.initial_clustering.total_iterations
        results.total_sequences_processed = state.initial_clustering.total_sequences
        results.coverage_percentage = state.initial_clustering.coverage_percentage

        # Expand hash IDs back to original headers for output compatibility
        # The clusters and unassigned contain hash IDs, but CLI expects expanded headers
        # Note: hash_to_headers was already loaded at line 1448

        # Expand clusters
        expanded_clusters = {}
        for cluster_id, hash_ids in results.clusters.items():
            expanded_clusters[cluster_id] = []
            for hash_id in hash_ids:
                if hash_id in hash_to_headers:
                    expanded_clusters[cluster_id].extend(hash_to_headers[hash_id])
                else:
                    expanded_clusters[cluster_id].append(hash_id)
        results.clusters = expanded_clusters

        expanded_all_clusters = {}
        for cluster_id, hash_ids in results.all_clusters.items():
            expanded_all_clusters[cluster_id] = []
            for hash_id in hash_ids:
                if hash_id in hash_to_headers:
                    expanded_all_clusters[cluster_id].extend(hash_to_headers[hash_id])
                else:
                    expanded_all_clusters[cluster_id].append(hash_id)
        results.all_clusters = expanded_all_clusters

        # Expand unassigned
        expanded_unassigned = []
        for hash_id in results.unassigned:
            if hash_id in hash_to_headers:
                expanded_unassigned.extend(hash_to_headers[hash_id])
            else:
                expanded_unassigned.append(hash_id)
        results.unassigned = expanded_unassigned

        # Expand conflicts
        expanded_conflicts = {}
        for hash_id, cluster_ids in results.conflicts.items():
            if hash_id in hash_to_headers:
                for original_header in hash_to_headers[hash_id]:
                    expanded_conflicts[original_header] = cluster_ids
            else:
                expanded_conflicts[hash_id] = cluster_ids
        results.conflicts = expanded_conflicts

        # Automatically create final output if not already finalized
        if not state.finalized.completed:
            from .decompose_cli import save_decompose_results
            save_decompose_results(results, state_manager.output_dir, input_fasta)
            logger.info("Final output created in clusters/latest/")

        return results


def _continue_initial_clustering(state: DecomposeState,
                                 state_manager: StateManager,
                                 input_fasta: str,
                                 assignment_tracker: AssignmentTracker,
                                 current_clusters: Dict[str, List[str]],
                                 max_clusters: Optional[int] = None,
                                 max_sequences: Optional[int] = None,
                                 resolve_conflicts: bool = False,
                                 refine_close_clusters: float = 0.0,
                                 checkpoint_interval: int = 10,
                                 **kwargs) -> DecomposeResults:
    """Continue initial clustering from checkpoint.

    Args:
        state: Current decompose state
        state_manager: State manager for saving checkpoints
        input_fasta: Input FASTA path
        assignment_tracker: Reconstructed assignment tracker
        current_clusters: Currently assigned clusters
        max_clusters: New maximum cluster count (absolute)
        max_sequences: New maximum sequence count (absolute)
        resolve_conflicts: Whether to apply conflict resolution after clustering
        refine_close_clusters: Distance threshold for close cluster refinement
        checkpoint_interval: Save checkpoint every N iterations (default: 10)
        **kwargs: Additional parameters

    Returns:
        DecomposeResults with continued clustering
    """
    logger.info("Continuing initial clustering from checkpoint")

    # Update limits if provided
    if max_clusters is not None:
        state.initial_clustering.max_clusters_limit = max_clusters
        logger.info(f"Updated max_clusters limit to: {max_clusters} (absolute)")

    if max_sequences is not None:
        state.initial_clustering.max_sequences_limit = max_sequences
        logger.info(f"Updated max_sequences limit to: {max_sequences} (absolute)")

    # Extract clustering parameters from state
    params = state.parameters

    # Initialize decomposition clustering with same parameters
    decomposer = DecomposeClustering(
        min_split=params['min_split'],
        max_lump=params['max_lump'],
        target_percentile=params['target_percentile'],
        blast_max_hits=params['blast_max_hits'],
        blast_threads=params.get('blast_threads'),
        blast_evalue=params['blast_evalue'],
        min_identity=params.get('min_identity'),
        resolve_conflicts=resolve_conflicts,
        refine_close_clusters=refine_close_clusters > 0.0,
        close_cluster_threshold=refine_close_clusters,
        show_progress=kwargs.get('show_progress', True),
        logger=logger
    )

    # Run decompose with continuation state
    # The decomposer will check output_dir for existing state and continue from there
    results = decomposer.decompose(
        input_fasta=input_fasta,
        targets_fasta=kwargs.get('targets_fasta'),
        max_clusters=max_clusters,
        max_sequences=max_sequences,
        output_dir=str(state_manager.output_dir),
        resume_from_state=state,  # Pass state to indicate continuation
        checkpoint_interval=checkpoint_interval
    )

    # Automatically create final output if clustering completed
    state_after = DecomposeState.load(state_manager.output_dir)
    if state_after.initial_clustering.completed and not state_after.finalized.completed:
        from .decompose_cli import save_decompose_results
        save_decompose_results(results, state_manager.output_dir, input_fasta)
        logger.info("Final output created in clusters/latest/")

    return results


def _apply_conflict_resolution_stage(state: DecomposeState,
                                      state_manager: StateManager,
                                      input_fasta: str,
                                      current_clusters: Dict[str, List[str]],
                                      assignment_tracker: AssignmentTracker,
                                      **kwargs) -> DecomposeResults:
    """Apply conflict resolution stage to loaded clusters.

    Args:
        state: Current decompose state
        state_manager: State manager for saving results
        input_fasta: Input FASTA path
        current_clusters: Current cluster assignments (from FASTA files)
        assignment_tracker: Reconstructed assignment tracker
        **kwargs: Additional parameters

    Returns:
        DecomposeResults with conflicts resolved
    """
    from .cluster_refinement import resolve_conflicts, RefinementConfig

    logger.info(f"Applying conflict resolution to {len(current_clusters)} clusters")

    # Get conflicts from assignment tracker
    conflicts = assignment_tracker.get_conflicts()
    logger.info(f"Found {len(conflicts)} sequences with conflicts")

    if len(conflicts) == 0:
        logger.info("No conflicts to resolve - marking stage complete")
        # Update state to mark conflict resolution as complete even if no work needed
        state.conflict_resolution.completed = True
        state.conflict_resolution.conflicts_before = 0
        state.conflict_resolution.conflicts_after = 0
        state.conflict_resolution.clusters_before = len(current_clusters)
        state.conflict_resolution.clusters_after = len(current_clusters)
        state.conflict_resolution.total_clusters = len(current_clusters)
        # Keep using initial clustering directory since no new files created
        state.conflict_resolution.stage_directory = state.initial_clustering.stage_directory
        state.stage = "conflict_resolution"

        # Don't create new files - just update state
        state_manager.checkpoint(state)

        # Return current state as results
        results = DecomposeResults()
        results.clusters = current_clusters
        results.all_clusters = current_clusters.copy()
        results.conflicts = {}
        results.unassigned = []  # Load from state if needed
        return results

    # Load sequences and headers
    sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)

    # Create refinement configuration
    config = RefinementConfig(
        max_full_gaphack_size=300  # Conservative limit for performance
    )

    # Apply conflict resolution
    params = state.parameters
    conflict_id_generator = ClusterIDGenerator(stage_name="deconflicted")
    resolved_clusters, conflict_tracking = resolve_conflicts(
        conflicts=conflicts,
        all_clusters=current_clusters,
        sequences=sequences,
        headers=hash_ids,
        config=config,
        min_split=params['min_split'],
        max_lump=params['max_lump'],
        target_percentile=params['target_percentile'],
        cluster_id_generator=conflict_id_generator
    )

    logger.info(f"Conflict resolution complete: {len(current_clusters)} -> {len(resolved_clusters)} clusters")

    # Build results object
    results = DecomposeResults()
    results.clusters = resolved_clusters
    results.all_clusters = resolved_clusters.copy()
    results.conflicts = {}  # All conflicts resolved
    results.unassigned = []  # Preserve from state if needed
    results.total_iterations = state.initial_clustering.total_iterations
    results.total_sequences_processed = state.initial_clustering.total_sequences
    results.coverage_percentage = state.initial_clustering.coverage_percentage
    results.processing_stages = [conflict_tracking]

    # Save results to FASTA files in deconflicted stage directory
    from .cluster_id_utils import get_stage_directory
    stage_dir = get_stage_directory(state_manager.output_dir, "deconflicted")
    state_manager.save_stage_results(
        results=results,
        stage_dir=stage_dir,
        sequences=sequences,
        headers=hash_ids,
        hash_to_headers=hash_to_headers
    )

    # Update state
    state.conflict_resolution.completed = True
    state.conflict_resolution.conflicts_before = len(conflicts)
    state.conflict_resolution.conflicts_after = 0
    state.conflict_resolution.clusters_before = len(current_clusters)
    state.conflict_resolution.clusters_after = len(resolved_clusters)
    state.conflict_resolution.total_clusters = len(resolved_clusters)
    state.conflict_resolution.stage_directory = "work/deconflicted"
    state.stage = "conflict_resolution"
    state_manager.checkpoint(state)

    logger.info(f"Conflict resolution stage complete and saved")

    # Automatically create final output
    from .decompose_cli import save_decompose_results
    save_decompose_results(results, state_manager.output_dir, input_fasta)
    logger.info("Final output created in clusters/latest/")

    return results


def _apply_close_cluster_refinement_stage(state: DecomposeState,
                                          state_manager: StateManager,
                                          input_fasta: str,
                                          current_clusters: Dict[str, List[str]],
                                          close_threshold: float,
                                          **kwargs) -> DecomposeResults:
    """Apply close cluster refinement stage to loaded clusters.

    Args:
        state: Current decompose state
        state_manager: State manager for saving results
        input_fasta: Input FASTA path
        current_clusters: Current cluster assignments (from FASTA files)
        close_threshold: Distance threshold for close cluster refinement
        **kwargs: Additional parameters

    Returns:
        DecomposeResults with close clusters refined
    """
    from .cluster_refinement import refine_close_clusters, RefinementConfig
    from .cluster_graph import ClusterGraph

    logger.info(f"Applying close cluster refinement to {len(current_clusters)} clusters")
    logger.info(f"Close threshold: {close_threshold}")

    # Load sequences and headers
    sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)

    # Create proximity graph for cluster proximity queries
    # Note: ClusterGraph uses MSA internally for all distance calculations
    proximity_graph = ClusterGraph(
        clusters=current_clusters,
        sequences=sequences,
        headers=hash_ids,
        k_neighbors=20,
        show_progress=False  # Don't show progress in resume context
    )

    # Create refinement configuration
    config = RefinementConfig(
        max_full_gaphack_size=300,
        close_cluster_expansion_threshold=close_threshold
    )

    # Determine refinement count (how many refinement stages already exist)
    from .cluster_id_utils import count_refinement_stages
    refinement_count = count_refinement_stages(state_manager.output_dir)

    # Apply close cluster refinement
    params = state.parameters
    refinement_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=refinement_count)
    refined_clusters, refinement_tracking = refine_close_clusters(
        all_clusters=current_clusters,
        sequences=sequences,
        headers=hash_ids,
        proximity_graph=proximity_graph,
        config=config,
        min_split=params['min_split'],
        max_lump=params['max_lump'],
        target_percentile=params['target_percentile'],
        close_threshold=params['max_lump'],  # Use max_lump as base threshold
        cluster_id_generator=refinement_id_generator
    )

    logger.info(f"Close cluster refinement complete: {len(current_clusters)} -> {len(refined_clusters)} clusters")

    # Build results object
    results = DecomposeResults()
    results.clusters = refined_clusters
    results.all_clusters = refined_clusters.copy()
    results.conflicts = {}
    results.unassigned = []  # Preserve from state if needed
    results.total_iterations = state.initial_clustering.total_iterations
    results.total_sequences_processed = state.initial_clustering.total_sequences
    results.coverage_percentage = state.initial_clustering.coverage_percentage
    results.processing_stages = [refinement_tracking]

    # Save results to FASTA files in refined stage directory
    from .cluster_id_utils import get_stage_directory
    stage_dir = get_stage_directory(state_manager.output_dir, "refined", refinement_count=refinement_count)
    state_manager.save_stage_results(
        results=results,
        stage_dir=stage_dir,
        sequences=sequences,
        headers=hash_ids,
        hash_to_headers=hash_to_headers
    )

    # Update state - note that close cluster refinement can be chained
    state.close_cluster_refinement.completed = True
    state.close_cluster_refinement.threshold = close_threshold
    state.close_cluster_refinement.clusters_before = len(current_clusters)
    state.close_cluster_refinement.clusters_after = len(refined_clusters)
    state.close_cluster_refinement.total_clusters = len(refined_clusters)
    state.close_cluster_refinement.stage_directory = f"work/refined_{refinement_count + 1}"
    state.stage = "close_cluster_refinement"

    # Add to refinement history
    history_entry = {
        'threshold': close_threshold,
        'timestamp': datetime.datetime.now().isoformat(),
        'clusters_before': len(current_clusters),
        'clusters_after': len(refined_clusters)
    }
    state.close_cluster_refinement.refinement_history.append(history_entry)

    state_manager.checkpoint(state)

    logger.info(f"Close cluster refinement stage complete and saved")

    # Automatically create final output
    from .decompose_cli import save_decompose_results
    save_decompose_results(results, state_manager.output_dir, input_fasta)
    logger.info("Final output created in clusters/latest/")

    return results


def finalize_decompose(output_dir: str, cleanup: bool = False) -> None:
    """Create final numbered cluster output and optionally cleanup intermediate files.

    Args:
        output_dir: Output directory containing decompose state and cluster files
        cleanup: If True, remove intermediate stage files (initial.*, deconflicted.*, refined.*)

    This function:
    1. Determines the most recent refinement stage
    2. Loads clusters from that stage
    3. Renumbers clusters by size (cluster_001.fasta = largest)
    4. Writes final cluster_*.fasta and unassigned.fasta files
    5. Marks state as finalized
    6. Optionally removes intermediate stage files
    """
    from pathlib import Path
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    output_path = Path(output_dir)

    # Load state
    state = DecomposeState.load(output_path)
    state_manager = StateManager(output_path)

    # Check if already finalized
    if state.finalized.completed:
        logger.warning("Output directory already finalized")
        return

    # Verify initial clustering is complete
    if not state.initial_clustering.completed:
        raise ValueError("Cannot finalize: initial clustering is not complete. Run decompose or resume first.")

    # Determine the most recent stage to finalize from
    from .cluster_id_utils import get_latest_stage_directory
    work_dir = output_path / "work"

    try:
        latest_stage_dir = get_latest_stage_directory(work_dir)
        source_stage = latest_stage_dir.name
        logger.info(f"Finalizing from {source_stage} stage (directory: {latest_stage_dir})")
    except FileNotFoundError as e:
        raise ValueError(f"Cannot finalize: no stage directories found in {work_dir}")

    # Determine clean source stage name for state tracking
    if source_stage.startswith("refined_"):
        source_stage_name = "refined"
    elif source_stage == "deconflicted":
        source_stage_name = "deconflicted"
    elif source_stage == "initial":
        source_stage_name = "initial"
    else:
        raise ValueError(f"Unknown source stage: {source_stage}")

    # Load input sequences and setup
    input_fasta = state.input.fasta_path
    sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)

    # Load clusters from most recent stage directory
    current_clusters = state_manager.load_clusters_from_stage_directory(latest_stage_dir)

    # Also check for unassigned file from source stage
    unassigned_headers = []
    unassigned_file = latest_stage_dir / "unassigned.fasta"
    if unassigned_file.exists():
        for record in SeqIO.parse(unassigned_file, "fasta"):
            unassigned_headers.append(record.id)
        logger.info(f"Loaded {len(unassigned_headers)} unassigned sequences from {unassigned_file.name}")

    # Sort clusters by size (largest first)
    cluster_sizes = [(cluster_id, len(seqs)) for cluster_id, seqs in current_clusters.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Renumbering {len(cluster_sizes)} clusters by size")

    # Create timestamp-based output directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clusters_dir = output_path / "clusters"
    timestamp_dir = clusters_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Create 'latest' symlink
    latest_symlink = clusters_dir / "latest"
    if latest_symlink.exists() or latest_symlink.is_symlink():
        latest_symlink.unlink()
    latest_symlink.symlink_to(timestamp, target_is_directory=True)

    # Create header-to-index mapping for sequence lookup (hash_id -> index)
    header_to_idx = {hash_id: i for i, hash_id in enumerate(hash_ids)}

    # Write final numbered cluster files
    for final_num, (old_cluster_id, size) in enumerate(cluster_sizes, start=1):
        cluster_headers = current_clusters[old_cluster_id]
        final_filename = timestamp_dir / f"cluster_{final_num:05d}.fasta"

        records = []
        for header in cluster_headers:
            # Expand hash IDs to original headers (handles duplicates)
            if header in hash_to_headers:
                original_headers = hash_to_headers[header]
            else:
                original_headers = [header]

            for orig_header in original_headers:
                seq_idx = header_to_idx[header if header in header_to_idx else orig_header]
                record = SeqRecord(Seq(sequences[seq_idx]), id=orig_header, description="")
                records.append(record)

        with open(final_filename, 'w') as f:
            SeqIO.write(records, f, "fasta-2line")

        logger.info(f"Wrote {final_filename.name} with {len(records)} sequences")

    # Write final unassigned file if any unassigned sequences exist
    if unassigned_headers:
        final_unassigned = timestamp_dir / "unassigned.fasta"
        records = []
        for header in unassigned_headers:
            # Expand hash IDs
            if header in hash_to_headers:
                original_headers = hash_to_headers[header]
            else:
                original_headers = [header]

            for orig_header in original_headers:
                seq_idx = header_to_idx[header if header in header_to_idx else orig_header]
                record = SeqRecord(Seq(sequences[seq_idx]), id=orig_header, description="")
                records.append(record)

        with open(final_unassigned, 'w') as f:
            SeqIO.write(records, f, "fasta-2line")

        logger.info(f"Wrote {final_unassigned.name} with {len(records)} sequences")

    # Update state to mark as finalized
    state.finalized.completed = True
    state.finalized.source_stage = source_stage_name
    state.finalized.total_clusters = len(cluster_sizes)
    state.finalized.total_sequences = sum(size for _, size in cluster_sizes)
    state.finalized.unassigned_sequences = len(unassigned_headers)
    state.stage = "finalized"
    state_manager.checkpoint(state)

    logger.info(f"Finalization complete: {len(cluster_sizes)} clusters, {state.finalized.total_sequences} sequences, {len(unassigned_headers)} unassigned")
    logger.info(f"Results saved to: {timestamp_dir}")
    logger.info(f"Symlink created: {latest_symlink} -> {timestamp}/")

    # Optional cleanup of intermediate files
    if cleanup:
        logger.info("Cleaning up intermediate stage files")

        stage_prefixes = []
        if source_stage != "initial":
            stage_prefixes.append("initial")
        if source_stage == "refined":
            stage_prefixes.extend(["deconflicted"])

        import shutil
        removed_count = 0
        work_dir = output_path / "work"

        # Map stage names to directory names
        for prefix in stage_prefixes:
            if prefix == "initial":
                stage_cleanup_dir = work_dir / "initial"
            elif prefix == "deconflicted":
                stage_cleanup_dir = work_dir / "deconflicted"
            else:
                continue  # Skip unknown prefixes

            if stage_cleanup_dir.exists():
                # Count files before removal
                file_count = len(list(stage_cleanup_dir.glob("*")))
                shutil.rmtree(stage_cleanup_dir)
                removed_count += file_count
                logger.debug(f"Removed directory {stage_cleanup_dir} with {file_count} files")

        logger.info(f"Removed {removed_count} intermediate files")
        logger.info("Note: Source stage files retained for traceability")