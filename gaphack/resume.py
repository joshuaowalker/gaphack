"""
Resume and finalization logic for decompose clustering.

This module contains functions for:
- Resuming decompose clustering from checkpoints
- Continuing initial clustering after interruption
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
                     force_input_change: bool = False,
                     checkpoint_interval: int = 10,
                     **kwargs) -> DecomposeResults:
    """Resume decompose clustering from saved state.

    Determines current stage and continues appropriately:
    - If in initial_clustering: continue adding clusters
    - If all complete: report completion status

    Args:
        output_dir: Output directory containing state.json
        max_clusters: New maximum cluster count (absolute, not incremental)
        max_sequences: New maximum sequence count (absolute, not incremental)
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

    # Determine action based on stage
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
            checkpoint_interval=checkpoint_interval,
            **kwargs
        )
    else:
        # Initial clustering complete
        logger.info("Initial clustering already complete.")
        logger.info(f"  Total clusters: {len(current_clusters)}")
        logger.info(f"  Unassigned sequences: {len(unassigned_headers)}")
        logger.info(f"  Conflicts: {len(assignment_tracker.get_conflicts())}")
        logger.info("")
        logger.info("For refinement: gaphack-refine {output_dir}")

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