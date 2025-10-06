"""Decomposition clustering for gaphack using BLAST neighborhoods and target clustering."""

import logging
import copy
import datetime
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .blast_neighborhood import BlastNeighborhoodFinder
from .vsearch_neighborhood import VsearchNeighborhoodFinder
from .neighborhood_finder import NeighborhoodFinder
from .target_clustering import TargetModeClustering
from .utils import load_sequences_from_fasta, load_sequences_with_deduplication, calculate_distance_matrix
from .distance_providers import DistanceProvider
from .state import DecomposeState, StateManager, create_initial_state
from .target_selection import TargetSelector, BlastResultMemory, NearbyTargetSelector
from .cluster_id_utils import (
    get_stage_suffix, format_cluster_id, parse_cluster_id,
    get_next_cluster_number, get_stage_directory
)

logger = logging.getLogger(__name__)


class ClusterIDGenerator:
    """Generates globally unique cluster IDs with stage suffixes.

    Format: cluster_{NNNNN}{SUFFIX} where:
    - NNNNN is a 5-digit number (01-99999)
    - SUFFIX is I (initial), C (conflict resolution), or R1/R2/R3... (refinements)
    """

    def __init__(self, stage_name: str = "initial", refinement_count: int = 0,
                 starting_number: Optional[int] = None):
        """Initialize cluster ID generator.

        Args:
            stage_name: Stage name ("initial", "deconflicted", "refined")
            refinement_count: For refined stages, the refinement number (0-based)
            starting_number: Starting cluster number (if resuming from existing clusters)
        """
        self.stage_name = stage_name
        self.refinement_count = refinement_count
        self.stage_suffix = get_stage_suffix(stage_name, refinement_count)
        self.counter = starting_number if starting_number is not None else 1

    def next_id(self) -> str:
        """Generate next sequential cluster ID with stage suffix."""
        cluster_id = format_cluster_id(self.counter, self.stage_suffix)
        self.counter += 1
        return cluster_id

    def get_current_count(self) -> int:
        """Get current counter value."""
        return self.counter - 1


@dataclass
class ProcessingStageInfo:
    """Information about a processing stage (conflict resolution or close cluster refinement)."""
    stage_name: str
    clusters_before: Dict[str, List[str]] = field(default_factory=dict)  # active_id -> sequence headers
    clusters_after: Dict[str, List[str]] = field(default_factory=dict)   # active_id -> sequence headers
    components_processed: List[Dict] = field(default_factory=list)       # details about each component processed
    summary_stats: Dict = field(default_factory=dict)                    # before/after counts, etc.


@dataclass
class DecomposeResults:
    """Results from decomposition clustering."""
    clusters: Dict[str, List[str]] = field(default_factory=dict)  # cluster_id -> sequence headers (non-conflicted only)
    all_clusters: Dict[str, List[str]] = field(default_factory=dict)  # cluster_id -> sequence headers (including conflicts)
    unassigned: List[str] = field(default_factory=list)  # sequence headers never assigned
    conflicts: Dict[str, List[str]] = field(default_factory=dict)  # seq_id -> cluster_ids (multi-assigned)
    iteration_summaries: List[Dict] = field(default_factory=list)  # per-iteration statistics
    total_iterations: int = 0
    total_sequences_processed: int = 0
    coverage_percentage: float = 0.0
    command_line: str = ""  # command used to run decompose
    start_time: str = ""  # ISO timestamp of run start


class AssignmentTracker:
    """Tracks sequence assignments and detects conflicts."""
    
    def __init__(self):
        self.assignments: Dict[str, List[Tuple[str, int]]] = {}  # seq_id -> [(cluster_id, iteration), ...]
        self.assigned_sequences: Set[str] = set()
    
    def assign_sequence(self, seq_id: str, cluster_id: str, iteration: int) -> None:
        """Assign a sequence to a cluster."""
        if seq_id not in self.assignments:
            self.assignments[seq_id] = []
        
        self.assignments[seq_id].append((cluster_id, iteration))
        self.assigned_sequences.add(seq_id)
    
    def assign_sequences(self, seq_ids: List[str], cluster_id: str, iteration: int) -> None:
        """Assign multiple sequences to a cluster."""
        for seq_id in seq_ids:
            self.assign_sequence(seq_id, cluster_id, iteration)
    
    def is_assigned(self, seq_id: str) -> bool:
        """Check if sequence has been assigned to any cluster."""
        return seq_id in self.assigned_sequences
    
    def get_conflicts(self) -> Dict[str, List[str]]:
        """Get sequences assigned to multiple clusters."""
        conflicts = {}
        for seq_id, assignments in self.assignments.items():
            if len(assignments) > 1:
                cluster_ids = [cluster_id for cluster_id, _ in assignments]
                conflicts[seq_id] = cluster_ids
        return conflicts
    
    def get_single_assignments(self) -> Dict[str, str]:
        """Get sequences assigned to exactly one cluster."""
        single_assignments = {}
        for seq_id, assignments in self.assignments.items():
            if len(assignments) == 1:
                cluster_id, _ = assignments[0]
                single_assignments[seq_id] = cluster_id
        return single_assignments
    
    def get_all_assignments(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get all assignments including conflicts."""
        return self.assignments.copy()
    
    def get_unassigned(self, all_sequence_ids: List[str]) -> List[str]:
        """Get sequences that were never assigned."""
        return [seq_id for seq_id in all_sequence_ids if not self.is_assigned(seq_id)]




class DecomposeClustering:
    """Main orchestrator for decomposition clustering."""
    
    def __init__(self,
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 95,
                 blast_max_hits: int = 1000,
                 blast_threads: Optional[int] = None,
                 blast_evalue: float = 1e-5,
                 min_identity: Optional[float] = None,
                 search_method: str = "blast",
                 show_progress: bool = True,
                 logger: Optional[logging.Logger] = None):
        """Initialize decomposition clustering.

        Args:
            min_split: Minimum distance to split clusters in target clustering
            max_lump: Maximum distance to lump clusters in target clustering
            target_percentile: Percentile for gap optimization
            blast_max_hits: Maximum BLAST/vsearch hits per query (default: 1000)
            blast_threads: BLAST/vsearch thread count (auto if None)
            blast_evalue: BLAST e-value threshold (vsearch uses min_identity only)
            min_identity: BLAST/vsearch identity threshold (auto if None)
            search_method: Search method for neighborhood discovery: 'blast' or 'vsearch' (default: 'blast')
            show_progress: Show progress bars
            logger: Logger instance
        """
        self.min_split = min_split
        self.max_lump = max_lump
        self.target_percentile = target_percentile
        self.blast_max_hits = blast_max_hits
        self.blast_threads = blast_threads
        self.blast_evalue = blast_evalue
        self.min_identity = min_identity
        self.search_method = search_method
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)

        # Initialize persistent distance provider (will be created on first use)
        self._global_distance_provider = None

        # Initialize target clustering (disable individual progress bars)
        self.target_clustering = TargetModeClustering(
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile,
            show_progress=False,  # Disable individual progress bars
            logger=self.logger
        )

    def _create_neighborhood_finder(self, sequences: List[str], headers: List[str],
                                     output_dir: Optional[Path]) -> NeighborhoodFinder:
        """Factory method to create neighborhood finder based on search method.

        Args:
            sequences: List of DNA sequences
            headers: List of sequence headers
            output_dir: Output directory for database files

        Returns:
            NeighborhoodFinder instance (BLAST or vsearch)

        Raises:
            ValueError: If search_method is not recognized
        """
        if self.search_method == "blast":
            return BlastNeighborhoodFinder(sequences, headers, output_dir=output_dir)
        elif self.search_method == "vsearch":
            return VsearchNeighborhoodFinder(sequences, headers, output_dir=output_dir)
        else:
            raise ValueError(f"Unknown search method: {self.search_method}. Choose 'blast' or 'vsearch'.")
    
    def decompose(self, input_fasta: str,
                 targets_fasta: Optional[str] = None,
                 max_clusters: Optional[int] = None,
                 max_sequences: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 resume_from_state: Optional[DecomposeState] = None,
                 checkpoint_interval: int = 10) -> DecomposeResults:
        """Perform decomposition clustering.

        Args:
            input_fasta: Path to input FASTA file with all sequences
            targets_fasta: Path to FASTA file with target sequences (directed mode)
            max_clusters: Maximum clusters to create (undirected mode)
            max_sequences: Maximum sequences to assign (undirected mode)
            output_dir: Output directory for results and BLAST database (optional)
            resume_from_state: DecomposeState to resume from (internal use for continuation)
            checkpoint_interval: Save checkpoint every N iterations (default: 10)

        Returns:
            DecomposeResults with clustering results
        """
        import signal

        # Set up signal handler for graceful interruption
        interruption_requested = {'flag': False}
        original_handler = None

        def handle_interruption(signum, frame):
            """Handle KeyboardInterrupt gracefully."""
            if not interruption_requested['flag']:
                interruption_requested['flag'] = True
                self.logger.info("\nInterruption received (Ctrl+C). Finishing current iteration and saving checkpoint...")
                self.logger.info("Press Ctrl+C again to force exit (may lose progress)")
            else:
                self.logger.warning("Force exit requested. Progress may be lost.")
                # Restore original handler and re-raise
                signal.signal(signal.SIGINT, original_handler)
                raise KeyboardInterrupt

        # Install signal handler
        original_handler = signal.signal(signal.SIGINT, handle_interruption)

        try:
            # Auto-detect mode based on targets
            if targets_fasta:
                mode = "directed"
                self.logger.info("Starting decomposition clustering in directed mode (targets provided)")
            else:
                mode = "undirected"
                self.logger.info("Starting decomposition clustering in undirected mode")

            # Load input sequences with deduplication
            sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)
            self.logger.info(f"Loaded {len(sequences)} unique sequences from {input_fasta}")

            # For backward compatibility, create headers list using hash_ids
            headers = hash_ids

            # Sequence overlaps are always allowed
            self.logger.info("Sequence overlaps: allowed")

            # Convert output_dir to Path if provided
            output_dir_path = Path(output_dir) if output_dir else None

            # Initialize state management if output_dir provided
            state_manager = None
            state = None
            if output_dir_path:
                import sys
                state_manager = StateManager(output_dir_path)

                if resume_from_state:
                    # Resume from existing state
                    state = resume_from_state
                    self.logger.info(f"Resuming from checkpoint: iteration {state.initial_clustering.total_iterations}")
                else:
                    # Build parameters dict
                    parameters = {
                        "min_split": self.min_split,
                        "max_lump": self.max_lump,
                        "target_percentile": self.target_percentile,
                        "blast_max_hits": self.blast_max_hits,
                        "blast_evalue": self.blast_evalue,
                        "min_identity": self.min_identity,
                        "max_clusters": max_clusters,
                        "max_sequences": max_sequences
                    }

                    # Create initial state
                    command = ' '.join(sys.argv) if hasattr(sys, 'argv') else "gaphack-decompose"
                    from . import __version__
                    state = create_initial_state(
                        input_fasta=input_fasta,
                        parameters=parameters,
                        command=command,
                        version=__version__
                    )
                    self.logger.info(f"State management initialized in {output_dir_path}")

            # Initialize neighborhood finder (BLAST or vsearch) with output directory
            neighborhood_finder = self._create_neighborhood_finder(sequences, headers, output_dir=output_dir_path)

            # Initialize or load assignment tracker and clusters
            assignment_tracker = AssignmentTracker()
            existing_clusters = {}

            if resume_from_state and state_manager:
                # Load existing clusters from FASTA files
                stage_dir = state.get_current_stage_directory(output_dir_path)
                existing_clusters = state_manager.load_clusters_from_stage_directory(stage_dir)
                self.logger.info(f"Loaded {len(existing_clusters)} existing clusters from checkpoint")

                # Rebuild assignment tracker from existing clusters
                assignment_tracker = state_manager.rebuild_assignment_tracker(existing_clusters, headers)

                # Initialize cluster ID generator to continue from max existing number
                starting_number = get_next_cluster_number(existing_clusters)
                cluster_id_generator = ClusterIDGenerator(stage_name="initial", starting_number=starting_number)
                self.logger.info(f"Resuming cluster generation from cluster #{starting_number}")
            else:
                # Start fresh with number 1
                cluster_id_generator = ClusterIDGenerator(stage_name="initial")

            # Initialize target selector based on mode
            if mode == "directed":
                target_sequences, target_headers, _ = load_sequences_from_fasta(targets_fasta)
                self.logger.info(f"Loaded {len(target_headers)} target sequences from targets file")

                # Find matching sequences in input based on sequence content
                self.logger.info(f"Attempting to match {len(target_sequences)} target sequences against {len(sequences)} input sequences")
                matched_hash_ids = self._find_matching_targets_by_content(target_sequences, target_headers, sequences, hash_ids)

                if not matched_hash_ids:
                    self.logger.error("No target sequences found matching sequences in input file")
                    self.logger.error("Target headers preview:")
                    for i, header in enumerate(target_headers[:3]):
                        self.logger.error(f"  Target {i+1}: {header}")
                    self.logger.error("Input hash_ids preview:")
                    for i, hash_id in enumerate(hash_ids[:3]):
                        self.logger.error(f"  Input {i+1}: {hash_id}")
                    raise ValueError("No target sequences found matching sequences in input file")

                self.logger.info(f"Found {len(matched_hash_ids)} target sequences matching input sequences")

                # Target selector now works with hash_ids directly
                target_selector = TargetSelector(matched_hash_ids)
            elif mode == "undirected":
                # Create nearby target selector with all sequence headers
                target_selector = NearbyTargetSelector(
                    all_headers=headers,
                    max_clusters=max_clusters,
                    max_sequences=max_sequences
                )
                self.logger.info(f"Initialized undirected mode with nearby target selection: "
                               f"max_clusters={max_clusters}, max_sequences={max_sequences}")
            else:
                raise ValueError(f"Unknown mode '{mode}'")

            # Initialize progress tracking based on mode and stopping criteria
            if mode == "directed":
                # Directed mode: track target completion
                progress_mode = "targets"
                progress_total = len(matched_hash_ids)
                progress_unit = " targets"
                progress_desc = "Processing targets"
            elif mode == "undirected":
                if max_clusters:
                    # Cluster count mode: track cluster creation
                    progress_mode = "clusters"
                    progress_total = max_clusters
                    progress_unit = " clusters"
                    progress_desc = "Creating clusters"
                elif max_sequences:
                    # Sequence count mode: track sequence assignment
                    progress_mode = "sequences"
                    progress_total = max_sequences
                    progress_unit = " sequences"
                    progress_desc = "Assigning sequences"
                else:
                    # Exhaustive mode: track sequences until all assigned
                    progress_mode = "sequences_exhaustive"
                    progress_total = len(headers)
                    progress_unit = " sequences"
                    progress_desc = "Assigning sequences"
            else:
                # Fallback
                progress_mode = "targets"
                progress_total = float('inf')
                progress_unit = " targets"
                progress_desc = "Processing"

            # Statistics tracking for progress bar
            cluster_sizes = []
            gap_sizes = []

            # Calculate initial progress value when resuming
            initial_progress = 0
            if resume_from_state:
                if progress_mode == "clusters":
                    initial_progress = state.initial_clustering.total_clusters
                elif progress_mode in ("sequences", "sequences_exhaustive"):
                    initial_progress = state.initial_clustering.total_sequences
                elif progress_mode == "targets":
                    initial_progress = len(existing_clusters)  # Number of clusters created

            # Create overall progress bar
            from tqdm import tqdm
            pbar = tqdm(total=progress_total, desc=progress_desc, unit=progress_unit, initial=initial_progress,
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

            # Main decomposition loop
            if resume_from_state:
                iteration = state.initial_clustering.total_iterations
                self.logger.info(f"Resuming from iteration {iteration}")
            else:
                iteration = 0
            results = DecomposeResults()

            # Load existing clusters into results if resuming
            if existing_clusters:
                results.all_clusters = existing_clusters.copy()

            try:
                while target_selector.has_more_targets(assignment_tracker):
                    iteration += 1
                    # All iteration start messages now at debug level - progress bar shows the info
                    self.logger.debug(f"Starting iteration {iteration}")
                    # Get next target(s)
                    target_headers_for_iteration = target_selector.get_next_target(assignment_tracker)
                    if not target_headers_for_iteration:
                        break
                    
                    self.logger.debug(f"Iteration {iteration}: targeting sequences {target_headers_for_iteration}")
                    
                    # Find BLAST neighborhood
                    neighborhood_headers = neighborhood_finder.find_neighborhood(target_headers_for_iteration,
                                                                          max_hits=self.blast_max_hits,
                                                                          e_value_threshold=self.blast_evalue,
                                                                          min_identity=self.min_identity)
    
                    
                    self.logger.debug(f"Found neighborhood of {len(neighborhood_headers)} sequences")
                    
                    # Store BLAST neighborhood in memory for nearby target selection (before pruning)
                    if target_headers_for_iteration:
                        target_selector.add_blast_neighborhood(target_headers_for_iteration[0], neighborhood_headers)
                    
                    # Use all sequences in neighborhood (overlaps allowed)
                    sequences_for_clustering = neighborhood_headers
                    self.logger.debug(f"Using all {len(sequences_for_clustering)} sequences in neighborhood")
                    
                    # Get target indices in the neighborhood subset
                    neighborhood_to_full_idx = {h: headers.index(h) for h in sequences_for_clustering}
                    target_indices_in_neighborhood = []
                    
                    for target_header in target_headers_for_iteration:
                        if target_header in sequences_for_clustering:
                            neighborhood_idx = sequences_for_clustering.index(target_header)
                            target_indices_in_neighborhood.append(neighborhood_idx)
                    
                    if not target_indices_in_neighborhood:
                        self.logger.debug("No targets found in neighborhood subset - skipping iteration")
                        continue
                    
                    # Extract sequences for neighborhood
                    neighborhood_sequences = [sequences[headers.index(h)] for h in sequences_for_clustering]
    
                    short_count = 0
                    long_count = 0
                    for sequence in neighborhood_sequences:
                        if len(sequence) < 400:
                            short_count += 1
                        else:
                            long_count += 1
    
                    self.logger.debug(f"Found {short_count} short and {long_count} long sequences")
    
                    # OPTIMIZATION: Prune neighborhood based on distance to targets
                    # This reduces computational cost of target clustering by focusing on closest candidates
                    pruned_sequences, pruned_headers, pruned_target_indices = self._prune_neighborhood_by_distance(
                        neighborhood_sequences, sequences_for_clustering, target_indices_in_neighborhood, sequences, headers
                    )

                    # Create MSA-based distance provider for pruned neighborhood
                    # This provides consistent alignment across all sequences in the neighborhood
                    self.logger.debug(f"Creating MSA-based distance provider for {len(pruned_sequences)} pruned neighborhood sequences")
                    from .distance_providers import MSACachedDistanceProvider
                    pruned_distance_provider = MSACachedDistanceProvider(
                        pruned_sequences,
                        pruned_headers
                    )

                    # Perform target clustering on pruned neighborhood
                    target_cluster_indices, remaining_indices, clustering_metrics = self.target_clustering.cluster(
                        pruned_distance_provider, pruned_target_indices, pruned_sequences
                    )
                    
                    # Convert indices back to sequence headers
                    target_cluster_headers = [pruned_headers[i] for i in target_cluster_indices]
                    remaining_headers = [pruned_headers[i] for i in remaining_indices]

                    # Generate unique cluster ID using the configured generator
                    cluster_id = cluster_id_generator.next_id()

                    # Assign sequences to cluster
                    assignment_tracker.assign_sequences(target_cluster_headers, cluster_id, iteration)
                    
                    # Update BLAST memory for nearby target selection
                    target_selector.mark_sequences_processed(target_cluster_headers)
                    
                    # Record iteration summary
                    iteration_summary = {
                        'iteration': iteration,
                        'target_headers': target_headers_for_iteration,
                        'neighborhood_size': len(neighborhood_headers),
                        'sequences_for_clustering': len(sequences_for_clustering),
                        'pruned_size': len(pruned_sequences),
                        'cluster_size': len(target_cluster_headers),
                        'remaining_size': len(remaining_headers),
                        'cluster_id': cluster_id,
                        'gap_size': clustering_metrics['best_config'].get('gap_size', 0.0)
                    }
                    
                    results.iteration_summaries.append(iteration_summary)
                    
                    # Update statistics for progress bar
                    cluster_sizes.append(len(target_cluster_headers))
                    gap_sizes.append(clustering_metrics['best_config'].get('gap_size', 0.0))
                    
                    # Calculate statistics for progress bar
                    median_cluster_size = sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0
                    median_gap_size = sorted(gap_sizes)[len(gap_sizes)//2] if gap_sizes else 0
                    assigned_total = len(assignment_tracker.assigned_sequences)
                    clusters_created = iteration
                    
                    # Update progress based on mode
                    if progress_mode == "targets":
                        # Directed mode: increment by 1 target processed
                        progress_increment = 1
                        postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                    elif progress_mode == "clusters":
                        # Cluster count mode: increment by 1 cluster created
                        progress_increment = 1
                        postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                    elif progress_mode in ["sequences", "sequences_exhaustive"]:
                        # Sequence count mode: count unique sequences assigned (deduplicate)
                        unique_assigned_count = len(assignment_tracker.assigned_sequences)
                        progress_increment = unique_assigned_count - pbar.n  # Only increment by new unique assignments
    
                        postfix = f"clusters={clusters_created}, med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}"
                    else:
                        # Fallback
                        progress_increment = 1
                        postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                    
                    # Update progress bar
                    pbar.set_postfix_str(postfix)
                    pbar.update(progress_increment)
                    
                    # All iteration completion messages at debug level - progress bar shows the main info
                    gap_size = clustering_metrics['best_config'].get('gap_size', 0.0)
                    self.logger.debug(f"Iteration {iteration} complete: "
                                    f"clustered {len(target_cluster_headers)} sequences, "
                                    f"gap size: {gap_size:.4f}")
    
                    # Save checkpoint every N iterations if state management enabled
                    if state_manager and state and iteration % checkpoint_interval == 0:
                        try:
                            # Get current clusters from assignment tracker
                            current_clusters = {}
                            all_assignments = assignment_tracker.get_all_assignments()
                            for seq_id, assignments in all_assignments.items():
                                for cluster_id, _ in assignments:
                                    if cluster_id not in current_clusters:
                                        current_clusters[cluster_id] = []
                                    current_clusters[cluster_id].append(seq_id)
    
                            # Save checkpoint
                            stage_dir = state.get_current_stage_directory(output_dir_path)
                            state_manager.save_stage_fasta(current_clusters, sequences, headers, stage_dir)
                            state.initial_clustering.total_clusters = len(current_clusters)
                            state.initial_clustering.total_sequences = len(assignment_tracker.assigned_sequences)
                            state.initial_clustering.total_iterations = iteration
                            state.save(output_dir_path)
                            self.logger.debug(f"Checkpoint saved at iteration {iteration}")
                        except Exception as e:
                            self.logger.warning(f"Failed to save checkpoint at iteration {iteration}: {e}")
    
                    # Check for interruption request
                    if interruption_requested['flag']:
                        self.logger.info(f"Interruption requested. Saving checkpoint at iteration {iteration}...")
                        # Save final checkpoint before exit
                        if state_manager and state:
                            try:
                                # Get current clusters from assignment tracker
                                current_clusters = {}
                                all_assignments = assignment_tracker.get_all_assignments()
                                for seq_id, assignments in all_assignments.items():
                                    for cluster_id, _ in assignments:
                                        if cluster_id not in current_clusters:
                                            current_clusters[cluster_id] = []
                                        current_clusters[cluster_id].append(seq_id)
    
                                # Save checkpoint
                                stage_dir = state.get_current_stage_directory(output_dir_path)
                                state_manager.save_stage_fasta(current_clusters, sequences, headers, stage_dir)
                                state.initial_clustering.total_clusters = len(current_clusters)
                                state.initial_clustering.total_sequences = len(assignment_tracker.assigned_sequences)
                                state.initial_clustering.total_iterations = iteration
                                state.save(output_dir_path)
                                self.logger.info(f"Checkpoint saved successfully. You can resume with --resume {output_dir_path}")
                            except Exception as e:
                                self.logger.error(f"Failed to save checkpoint on interruption: {e}")
                        break
    
                    # Check stopping criteria for undirected mode
                    if max_clusters and iteration >= max_clusters:
                        self.logger.info(f"Reached maximum clusters ({max_clusters}) - stopping")
                        break
    
                    if max_sequences:
                        assigned_count = len(assignment_tracker.assigned_sequences)
                        if assigned_count >= max_sequences:
                            self.logger.info(f"Reached maximum sequences ({max_sequences}) - stopping")
                            break

            finally:
                # Close progress bar
                pbar.close()
                # Clean up neighborhood finder database
                neighborhood_finder.cleanup()

                # Process final results
                results.total_iterations = iteration
                results.total_sequences_processed = len(assignment_tracker.assigned_sequences)
                results.coverage_percentage = (len(assignment_tracker.assigned_sequences) / len(headers)) * 100.0
            
                # Save final checkpoint after initial clustering if state management enabled
                if state_manager and state:
                    try:
                        # Get current clusters from assignment tracker
                        current_clusters = {}
                        all_assignments = assignment_tracker.get_all_assignments()
                        for seq_id, assignments in all_assignments.items():
                            for cluster_id, _ in assignments:
                                if cluster_id not in current_clusters:
                                    current_clusters[cluster_id] = []
                                current_clusters[cluster_id].append(seq_id)
            
                        # Save final initial clustering state
                        stage_dir = state.get_current_stage_directory(output_dir_path)
                        state_manager.save_stage_fasta(current_clusters, sequences, headers, stage_dir)
                        state.initial_clustering.total_clusters = len(current_clusters)
                        state.initial_clustering.total_sequences = len(assignment_tracker.assigned_sequences)
                        state.initial_clustering.total_iterations = iteration
                        state.initial_clustering.coverage_percentage = (len(assignment_tracker.assigned_sequences) / len(headers)) * 100.0
                        state.stage = "initial_clustering"
            
                        # Only mark as completed if we reached natural exhaustion (not stopped by limits or interruption)
                        stopped_by_limit = False
                        if max_clusters and iteration >= max_clusters:
                            stopped_by_limit = True
                        if max_sequences and len(assignment_tracker.assigned_sequences) >= max_sequences:
                            stopped_by_limit = True

                        # Don't mark as completed if interrupted
                        interrupted = interruption_requested['flag']
                        state.initial_clustering.completed = not stopped_by_limit and not interrupted

                        if interrupted:
                            self.logger.info(f"Initial clustering interrupted (can be resumed)")
                        elif stopped_by_limit:
                            self.logger.info(f"Initial clustering paused at limit (can be resumed)")
                        else:
                            self.logger.info(f"Initial clustering completed (exhausted all sequences)")
            
                        state.save(output_dir_path)
                        self.logger.info(f"Final checkpoint saved after {iteration} iterations")
                    except Exception as e:
                        self.logger.warning(f"Failed to save final checkpoint: {e}")
                
                # Get single assignments (no conflicts) for summary reporting
                single_assignments = assignment_tracker.get_single_assignments()
                
                # Group non-conflicted sequences by cluster
                results.clusters = {}
                for seq_id, cluster_id in single_assignments.items():
                    if cluster_id not in results.clusters:
                        results.clusters[cluster_id] = []
                    results.clusters[cluster_id].append(seq_id)
                
                # Group ALL sequences by cluster (including conflicts) for FASTA generation
                results.all_clusters = {}
                all_assignments = assignment_tracker.get_all_assignments()
                for seq_id, assignments in all_assignments.items():
                    for cluster_id, iteration in assignments:
                        if cluster_id not in results.all_clusters:
                            results.all_clusters[cluster_id] = []
                        results.all_clusters[cluster_id].append(seq_id)
                
                # Get unassigned sequences
                results.unassigned = assignment_tracker.get_unassigned(headers)
                
                # Get conflicts
                results.conflicts = assignment_tracker.get_conflicts()
                
                # Enhanced final summary with aggregate statistics
                total_clustered_seqs = sum(len(cluster) for cluster in results.clusters.values())
                avg_cluster_size = total_clustered_seqs / len(results.clusters) if results.clusters else 0
                large_clusters = sum(1 for cluster in results.clusters.values() if len(cluster) > 10)
                
                self.logger.info(f"Decomposition complete: {len(results.clusters)} clusters created "
                                f"({large_clusters} with >10 sequences), "
                                f"{results.total_sequences_processed} sequences assigned ({results.coverage_percentage:.1f}% coverage), "
                                f"average cluster size: {avg_cluster_size:.1f}")
                
                if results.conflicts:
                    self.logger.info(f"Conflicts detected: {len(results.conflicts)}")

                # Check for interruption before post-processing
                if interruption_requested['flag']:
                    self.logger.info("Skipping post-processing stages due to interruption")
                    # Expand hash IDs before returning (otherwise CLI gets hash IDs not original headers)
                    results = self._expand_hash_ids_to_headers(results, hash_to_headers)
                    # Return results immediately without refinement
                    return results

                # Expand hash IDs back to original headers
                results = self._expand_hash_ids_to_headers(results, hash_to_headers)

                # Report conflict status (informational - not an error)
                if results.conflicts:
                    self.logger.info(f"ℹ️  {len(results.conflicts)} conflicts detected (sequences assigned to multiple clusters)")
                    self.logger.info(f"   These can be resolved using: gaphack-refine {output_dir_path} --resolve-conflicts")
                else:
                    self.logger.info("✓ No conflicts detected - clustering is clean")

                # Initial clustering complete
                self.logger.info(f"\nInitial clustering complete:")
                self.logger.info(f"  Clusters: {len(results.all_clusters)}")
                self.logger.info(f"  Sequences assigned: {results.total_sequences_processed}")
                self.logger.info(f"  Unassigned: {len(results.unassigned)}")
                self.logger.info(f"  Conflicts: {len(results.conflicts)}")
                self.logger.info(f"  Output: {output_dir_path}/work/initial/")

                return results

        finally:
            # Restore original signal handler
            if original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)
    
    def _prune_neighborhood_by_distance(self, neighborhood_sequences: List[str],
                                       neighborhood_headers: List[str],
                                       target_indices: List[int],
                                       global_sequences: List[str],
                                       global_headers: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """Prune neighborhood using principled N+N approach for balanced cluster/context representation.

        Strategy:
        - Take all N sequences within max_lump distance (potential cluster members)
        - Take N additional closest sequences beyond max_lump (context for gap estimation)
        - This ensures complete cluster coverage while providing balanced intra/inter-cluster
          context for robust barcode gap evaluation at target percentiles

        Args:
            neighborhood_sequences: All sequences in neighborhood
            neighborhood_headers: Headers for neighborhood sequences
            target_indices: Indices of target sequences within neighborhood
            global_sequences: Full global sequence list
            global_headers: Full global header list

        Returns:
            Tuple of (pruned_sequences, pruned_headers, new_target_indices)
        """
        self.logger.debug(f"Pruning neighborhood from {len(neighborhood_sequences)} sequences")

        # Calculate distances from all sequences to target sequences
        target_sequences = [neighborhood_sequences[i] for i in target_indices]

        # Create MSA-based distance provider for neighborhood
        # This provides consistent alignment across all sequences in the neighborhood
        from .distance_providers import MSACachedDistanceProvider
        neighborhood_distance_provider = MSACachedDistanceProvider(
            neighborhood_sequences,
            neighborhood_headers
        )

        # Calculate minimum distance from each sequence to any target
        sequence_min_distances = []
        for seq_idx in range(len(neighborhood_sequences)):
            if seq_idx in target_indices:
                # Target sequences have distance 0 to themselves
                min_dist = 0.0
            else:
                # Find minimum distance to any target
                distances_to_targets = neighborhood_distance_provider.get_distances_from_sequence(
                    seq_idx, set(target_indices)
                )
                min_dist = min(distances_to_targets.values())

            sequence_min_distances.append((seq_idx, min_dist))

        # Sort by distance for principled selection
        sequence_min_distances.sort(key=lambda x: x[1])

        # N+N selection: N sequences within max_lump + N closest sequences beyond max_lump
        within_max_lump = [(idx, dist) for idx, dist in sequence_min_distances if dist <= self.max_lump]
        beyond_max_lump = [(idx, dist) for idx, dist in sequence_min_distances if dist > self.max_lump]

        N = len(within_max_lump)  # Number of potential cluster members

        # Take all sequences within max_lump (complete cluster coverage)
        selected_indices = [idx for idx, _ in within_max_lump]

        # Take N additional closest sequences beyond max_lump for gap estimation context
        # (or all available if fewer than N sequences beyond max_lump)
        context_count = min(N, len(beyond_max_lump))
        context_sequences = beyond_max_lump[:context_count]
        selected_indices.extend([idx for idx, _ in context_sequences])

        self.logger.debug(f"N+N pruning: {N} sequences within max_lump distance {self.max_lump:.4f}")
        self.logger.debug(f"Added {context_count} context sequences beyond max_lump for gap estimation")
        self.logger.debug(f"Total selected: {len(selected_indices)} sequences (N+N = {N}+{context_count})")

        # Build pruned data structures
        pruned_sequences = [neighborhood_sequences[i] for i in selected_indices]
        pruned_headers = [neighborhood_headers[i] for i in selected_indices]

        # Map old target indices to new indices in pruned set
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        pruned_target_indices = [old_to_new_idx[old_idx] for old_idx in target_indices
                               if old_idx in old_to_new_idx]

        if not pruned_target_indices:
            self.logger.error("All target sequences were pruned - this should not happen!")
            # Fallback to original data
            return neighborhood_sequences, neighborhood_headers, target_indices

        self.logger.debug(f"Pruned neighborhood: {len(pruned_sequences)} sequences retained")

        return pruned_sequences, pruned_headers, pruned_target_indices
    
    def _find_matching_targets(self, target_sequences: List[str], target_headers: List[str], 
                              input_sequences: List[str], input_headers: List[str]) -> List[Dict[str, str]]:
        """Find target sequences that match input sequences by sequence content.
        
        Args:
            target_sequences: List of target sequences
            target_headers: List of target headers
            input_sequences: List of input sequences
            input_headers: List of input headers
            
        Returns:
            List of dicts with keys: 'target_header', 'input_header', 'target_sequence'
        """
        matched_targets = []
        
        # Create a mapping of normalized input sequences to headers for fast lookup
        input_seq_to_header = {}
        for seq, header in zip(input_sequences, input_headers):
            normalized_seq = seq.upper().strip()
            if normalized_seq in input_seq_to_header:
                # Handle duplicate sequences - keep first occurrence
                self.logger.debug(f"Duplicate sequence found in input, keeping first occurrence: {input_seq_to_header[normalized_seq]}")
            else:
                input_seq_to_header[normalized_seq] = header
        
        # Find matches
        for i, (target_seq, target_header) in enumerate(zip(target_sequences, target_headers)):
            normalized_target = target_seq.upper().strip()

            if normalized_target in input_seq_to_header:
                matched_targets.append({
                    'target_header': target_header,
                    'input_header': input_seq_to_header[normalized_target],
                    'target_sequence': target_seq
                })
                self.logger.info(f"MATCH {i+1}: Target '{target_header}' -> Input '{input_seq_to_header[normalized_target]}'")
            else:
                self.logger.warning(f"NO MATCH {i+1}: Target '{target_header}' (length: {len(target_seq)}) not found in input sequences")
        
        return matched_targets

    def _find_matching_targets_by_content(self, target_sequences: List[str], target_headers: List[str],
                                        input_sequences: List[str], input_hash_ids: List[str]) -> List[str]:
        """Find target sequences that match input sequences by content, return matching hash_ids.

        Args:
            target_sequences: List of target sequences
            target_headers: List of target headers
            input_sequences: List of unique input sequences
            input_hash_ids: List of hash_ids corresponding to input_sequences

        Returns:
            List of matching hash_ids from input
        """
        matched_hash_ids = []

        # Create a mapping of normalized input sequences to hash_ids for fast lookup
        input_seq_to_hash_id = {}
        for seq, hash_id in zip(input_sequences, input_hash_ids):
            normalized_seq = seq.upper().strip()
            input_seq_to_hash_id[normalized_seq] = hash_id

        # Find matches
        for i, (target_seq, target_header) in enumerate(zip(target_sequences, target_headers)):
            normalized_target = target_seq.upper().strip()

            if normalized_target in input_seq_to_hash_id:
                hash_id = input_seq_to_hash_id[normalized_target]
                if hash_id not in matched_hash_ids:  # Avoid duplicates
                    matched_hash_ids.append(hash_id)
                self.logger.info(f"MATCH {i+1}: Target '{target_header}' -> Hash '{hash_id}'")
            else:
                self.logger.warning(f"NO MATCH {i+1}: Target '{target_header}' (length: {len(target_seq)}) not found in input sequences")

        return matched_hash_ids

    def _expand_hash_ids_to_headers(self, results: DecomposeResults, hash_to_headers: Dict[str, List[str]]) -> DecomposeResults:
        """Expand hash IDs back to original headers in the results.

        Args:
            results: Results with hash IDs
            hash_to_headers: Mapping from hash_id to list of original headers

        Returns:
            Results with original headers expanded
        """
        expanded_results = DecomposeResults()
        expanded_results.total_iterations = results.total_iterations
        expanded_results.total_sequences_processed = results.total_sequences_processed
        expanded_results.coverage_percentage = results.coverage_percentage
        expanded_results.iteration_summaries = results.iteration_summaries

        # Copy tracking fields
        expanded_results.command_line = results.command_line
        expanded_results.start_time = results.start_time

        # Expand clusters
        for cluster_id, hash_ids in results.clusters.items():
            expanded_headers = []
            for hash_id in hash_ids:
                if hash_id in hash_to_headers:
                    expanded_headers.extend(hash_to_headers[hash_id])
                else:
                    # Fallback - keep hash_id if no mapping found
                    expanded_headers.append(hash_id)
            expanded_results.clusters[cluster_id] = expanded_headers

        # Expand all_clusters
        for cluster_id, hash_ids in results.all_clusters.items():
            expanded_headers = []
            for hash_id in hash_ids:
                if hash_id in hash_to_headers:
                    expanded_headers.extend(hash_to_headers[hash_id])
                else:
                    expanded_headers.append(hash_id)
            expanded_results.all_clusters[cluster_id] = expanded_headers

        # Expand unassigned
        expanded_unassigned = []
        for hash_id in results.unassigned:
            if hash_id in hash_to_headers:
                expanded_unassigned.extend(hash_to_headers[hash_id])
            else:
                expanded_unassigned.append(hash_id)
        expanded_results.unassigned = expanded_unassigned

        # Expand conflicts
        for hash_id, cluster_ids in results.conflicts.items():
            if hash_id in hash_to_headers:
                # Create conflict entries for each original header
                for original_header in hash_to_headers[hash_id]:
                    expanded_results.conflicts[original_header] = cluster_ids
            else:
                expanded_results.conflicts[hash_id] = cluster_ids

        return expanded_results

    def _renumber_clusters_sequentially(self, clusters: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Renumber clusters with sequential cluster_XXX naming for consistency.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers

        Returns:
            Tuple of (renumbered_clusters, active_to_final_mapping)
        """
        if not clusters:
            return {}, {}

        # Sort clusters by size (largest first) for consistent ordering
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Create new sequential mapping and track active→final mapping
        renumbered = {}
        active_to_final_mapping = {}
        for i, (old_cluster_id, cluster_headers) in enumerate(sorted_clusters, 1):
            new_cluster_id = f"cluster_{i:03d}"
            renumbered[new_cluster_id] = cluster_headers
            active_to_final_mapping[old_cluster_id] = new_cluster_id

        self.logger.debug(f"Renumbered {len(clusters)} clusters with sequential IDs: "
                         f"{list(clusters.keys())[:3]}... → {list(renumbered.keys())[:3]}...")

        return renumbered, active_to_final_mapping

    def _verify_no_conflicts(self, clusters: Dict[str, List[str]],
                             original_conflicts: Optional[Dict[str, List[str]]] = None,
                             context: str = "verification") -> Dict[str, any]:
        """Verify no conflicts of cluster assignments using comprehensive verification.

        Args:
            clusters: Dictionary mapping cluster_id -> list of sequence headers
            original_conflicts: Optional original conflicts for comparison
            context: Context string for logging

        Returns:
            Dictionary with verification results from comprehensive scan
        """
        from .cluster_refinement import verify_no_conflicts

        return verify_no_conflicts(
            clusters=clusters,
            original_conflicts=original_conflicts,
            context=context
        )


