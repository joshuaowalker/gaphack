"""Decomposition clustering for gaphack using BLAST neighborhoods and target clustering."""

import logging
import copy
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .blast_neighborhood import BlastNeighborhoodFinder
from .target_clustering import TargetModeClustering
from .utils import load_sequences_from_fasta, calculate_distance_matrix
from .lazy_distances import DistanceProviderFactory

logger = logging.getLogger(__name__)


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


class SupervisedTargetSelector:
    """Target selection strategy for supervised mode using provided target sequences."""
    
    def __init__(self, target_headers: List[str]):
        self.target_headers = target_headers
        self.used_targets: Set[str] = set()
    
    def get_next_target(self, assignment_tracker: AssignmentTracker) -> Optional[List[str]]:
        """Get next target sequence(s) for clustering."""
        # Find first unused target that hasn't been assigned yet
        for target_header in self.target_headers:
            if (target_header not in self.used_targets and 
                not assignment_tracker.is_assigned(target_header)):
                self.used_targets.add(target_header)
                return [target_header]
        
        return None  # No more unassigned targets
    
    def has_more_targets(self, assignment_tracker: AssignmentTracker) -> bool:
        """Check if there are more targets to process."""
        for target_header in self.target_headers:
            if (target_header not in self.used_targets and 
                not assignment_tracker.is_assigned(target_header)):
                return True
        return False
    
    def add_blast_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """No-op for supervised mode - doesn't use BLAST memory."""
        pass
    
    def mark_sequences_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """No-op for supervised mode - doesn't need memory management."""
        pass


class BlastResultMemory:
    """Memory pool for storing BLAST neighborhoods for spiral target selection."""
    
    def __init__(self):
        self.unprocessed_neighborhoods: Dict[str, Set[str]] = {}  # target_header -> neighborhood_headers
        self.candidate_pool: Set[str] = set()  # All unassigned sequences from previous neighborhoods
        self.fully_processed_targets: Set[str] = set()
    
    def add_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """Add a BLAST neighborhood to memory."""
        neighborhood_set = set(neighborhood_headers)
        self.unprocessed_neighborhoods[target_header] = neighborhood_set
        self.candidate_pool.update(neighborhood_set)
        logger.debug(f"Added neighborhood for {target_header}: {len(neighborhood_headers)} sequences, "
                    f"total pool: {len(self.candidate_pool)}")
    
    def get_spiral_candidates(self, assignment_tracker: AssignmentTracker) -> List[str]:
        """Get unassigned sequences from BLAST neighborhoods for spiral selection."""
        candidates = []
        for seq_header in self.candidate_pool:
            if not assignment_tracker.is_assigned(seq_header):
                candidates.append(seq_header)
        return candidates
    
    def mark_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """Mark sequences as processed and clean up empty neighborhoods."""
        processed_set = set(processed_headers)
        
        if not allow_overlaps:
            # No overlaps mode: remove processed sequences from candidate pool
            self.candidate_pool -= processed_set
            
            # Clean up neighborhoods that are now fully processed
            targets_to_remove = []
            for target_header, neighborhood in self.unprocessed_neighborhoods.items():
                neighborhood -= processed_set
                if not neighborhood:  # Neighborhood is empty
                    targets_to_remove.append(target_header)
            
            # Remove empty neighborhoods
            for target_header in targets_to_remove:
                del self.unprocessed_neighborhoods[target_header]
                self.fully_processed_targets.add(target_header)
            
            logger.debug(f"Processed {len(processed_headers)} sequences (no overlaps), "
                        f"pool now: {len(self.candidate_pool)}, "
                        f"active neighborhoods: {len(self.unprocessed_neighborhoods)}")
        else:
            # Overlap mode: keep sequences in candidate pool for future clusters
            logger.debug(f"Processed {len(processed_headers)} sequences (overlaps allowed), "
                        f"pool unchanged: {len(self.candidate_pool)}, "
                        f"active neighborhoods: {len(self.unprocessed_neighborhoods)}")


class SpiralTargetSelector:
    """Target selection strategy using spiral exploration with random fallback."""
    
    def __init__(self, all_headers: List[str], max_clusters: Optional[int] = None, 
                 max_sequences: Optional[int] = None):
        self.all_headers = all_headers
        self.max_clusters = max_clusters
        self.max_sequences = max_sequences
        self.iteration_count = 0
        self.blast_memory = BlastResultMemory()
        self.used_targets: Set[str] = set()
        
        # Initialize with random seed if no BLAST history available
        import random
        self.random_state = random.Random(42)  # Deterministic for reproducibility
    
    def get_next_target(self, assignment_tracker: AssignmentTracker) -> Optional[List[str]]:
        """Get next target using spiral logic with random fallback."""
        self.iteration_count += 1
        
        # Try spiral selection first: pick from previous BLAST neighborhoods
        spiral_candidates = self.blast_memory.get_spiral_candidates(assignment_tracker)
        spiral_candidates = [h for h in spiral_candidates if h not in self.used_targets]
        
        target_header = None
        selection_method = ""
        
        if spiral_candidates:
            # Spiral selection: choose from BLAST neighborhood candidates
            target_header = self.random_state.choice(spiral_candidates)
            selection_method = "spiral"
        else:
            # Random fallback: choose any unassigned sequence
            unassigned_candidates = [h for h in self.all_headers 
                                   if (not assignment_tracker.is_assigned(h) and 
                                       h not in self.used_targets)]
            if unassigned_candidates:
                target_header = self.random_state.choice(unassigned_candidates)
                selection_method = "random"
        
        if target_header:
            self.used_targets.add(target_header)
            logger.debug(f"Iteration {self.iteration_count}: selected '{target_header}' via {selection_method} "
                        f"(spiral_pool: {len(spiral_candidates)}, total_unassigned: {len([h for h in self.all_headers if not assignment_tracker.is_assigned(h)])})")
            return [target_header]
        
        return None  # No more targets available
    
    def has_more_targets(self, assignment_tracker: AssignmentTracker) -> bool:
        """Check if there are more targets to process based on stopping criteria."""
        # Check cluster count limit
        if self.max_clusters and self.iteration_count >= self.max_clusters:
            return False
        
        # Check sequence assignment limit
        if self.max_sequences:
            assigned_count = len(assignment_tracker.assigned_sequences)
            if assigned_count >= self.max_sequences:
                return False
        
        # Check if any unassigned sequences remain
        unassigned_candidates = [h for h in self.all_headers 
                               if (not assignment_tracker.is_assigned(h) and 
                                   h not in self.used_targets)]
        return len(unassigned_candidates) > 0
    
    def add_blast_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """Store BLAST neighborhood before pruning for future spiral selection."""
        self.blast_memory.add_neighborhood(target_header, neighborhood_headers)
    
    def mark_sequences_processed(self, processed_headers: List[str], allow_overlaps: bool = True) -> None:
        """Update memory after clustering iteration."""
        self.blast_memory.mark_processed(processed_headers, allow_overlaps)


class DecomposeClustering:
    """Main orchestrator for decomposition clustering."""
    
    def __init__(self, 
                 min_split: float = 0.005,
                 max_lump: float = 0.02,
                 target_percentile: int = 95,
                 blast_max_hits: int = 500,
                 blast_threads: Optional[int] = None,
                 blast_evalue: float = 1e-5,
                 min_identity: Optional[float] = None,
                 allow_overlaps: bool = True,
                 show_progress: bool = True,
                 logger: Optional[logging.Logger] = None):
        """Initialize decomposition clustering.
        
        Args:
            min_split: Minimum distance to split clusters in target clustering
            max_lump: Maximum distance to lump clusters in target clustering
            target_percentile: Percentile for gap optimization
            blast_max_hits: Maximum BLAST hits per query
            blast_threads: BLAST thread count (auto if None)
            blast_evalue: BLAST e-value threshold
            min_identity: BLAST identity threshold (auto if None)
            allow_overlaps: Allow sequences to appear in multiple clusters (default: True)
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
        self.allow_overlaps = allow_overlaps
        self.show_progress = show_progress
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize target clustering (disable individual progress bars)
        self.target_clustering = TargetModeClustering(
            min_split=min_split,
            max_lump=max_lump,
            target_percentile=target_percentile,
            show_progress=False,  # Disable individual progress bars
            logger=self.logger
        )
    
    def decompose(self, input_fasta: str, 
                 targets_fasta: Optional[str] = None,
                 strategy: str = "supervised",
                 max_clusters: Optional[int] = None,
                 max_sequences: Optional[int] = None) -> DecomposeResults:
        """Perform decomposition clustering.
        
        Args:
            input_fasta: Path to input FASTA file with all sequences
            targets_fasta: Path to FASTA file with target sequences (supervised mode)
            strategy: Target selection strategy ("supervised", "random", "spiral")
            max_clusters: Maximum clusters to create (unsupervised modes)
            max_sequences: Maximum sequences to assign (unsupervised modes)
            
        Returns:
            DecomposeResults with clustering results
        """
        self.logger.info(f"Starting decomposition clustering with strategy '{strategy}'")
        
        # Validate inputs early
        if strategy == "supervised":
            if not targets_fasta:
                raise ValueError("targets_fasta is required for supervised mode")
        elif strategy == "unsupervised":
            # No validation needed - will run until input exhausted if no limits specified
            pass
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Valid strategies: 'supervised', 'unsupervised'")
        
        # Load input sequences
        sequences, headers, header_mapping = load_sequences_from_fasta(input_fasta)
        self.logger.info(f"Loaded {len(sequences)} sequences from {input_fasta}")
        
        # Log overlap policy
        overlap_mode = "allowed" if self.allow_overlaps else "disabled"
        self.logger.info(f"Sequence overlaps: {overlap_mode}")
        
        # Initialize BLAST neighborhood finder
        blast_finder = BlastNeighborhoodFinder(sequences, headers)
        
        # Initialize assignment tracker
        assignment_tracker = AssignmentTracker()
        
        # Initialize target selector based on strategy
        if strategy == "supervised":
            target_sequences, target_headers, _ = load_sequences_from_fasta(targets_fasta)
            self.logger.info(f"Loaded {len(target_headers)} target sequences from targets file")
            
            # Find matching sequences in input based on sequence content, not headers
            matched_target_info = self._find_matching_targets(target_sequences, target_headers, sequences, headers)
            
            if not matched_target_info:
                self.logger.error("No target sequences found matching sequences in input file")
                raise ValueError("No target sequences found matching sequences in input file")
            
            self.logger.info(f"Found {len(matched_target_info)} target sequences matching input sequences")
            
            # Create target selector with input file headers (not target file headers)
            matched_input_headers = [info['input_header'] for info in matched_target_info]
            target_selector = SupervisedTargetSelector(matched_input_headers)
        elif strategy == "unsupervised":
            # Create spiral target selector with all sequence headers
            target_selector = SpiralTargetSelector(
                all_headers=headers,
                max_clusters=max_clusters, 
                max_sequences=max_sequences
            )
            self.logger.info(f"Initialized unsupervised mode with spiral target selection: "
                           f"max_clusters={max_clusters}, max_sequences={max_sequences}")
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
        
        # Initialize progress tracking based on strategy and stopping criteria
        if strategy == "supervised":
            # Supervised mode: track target completion
            progress_mode = "targets"
            progress_total = len(matched_input_headers)
            progress_unit = " targets"
            progress_desc = "Processing targets"
        elif strategy == "unsupervised":
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
        
        # Create overall progress bar
        from tqdm import tqdm
        pbar = tqdm(total=progress_total, desc=progress_desc, unit=progress_unit,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        
        # Main decomposition loop
        iteration = 0
        results = DecomposeResults()
        
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
                neighborhood_headers = blast_finder.find_neighborhood(
                    target_headers_for_iteration,
                    max_hits=self.blast_max_hits,
                    e_value_threshold=self.blast_evalue,
                    min_identity=self.min_identity
                )
                
                self.logger.debug(f"Found neighborhood of {len(neighborhood_headers)} sequences")
                
                # Store BLAST neighborhood in memory for spiral target selection (before pruning)
                if target_headers_for_iteration:
                    target_selector.add_blast_neighborhood(target_headers_for_iteration[0], neighborhood_headers)
                
                # Filter neighborhood based on overlap policy
                if self.allow_overlaps:
                    # Allow overlaps: use all sequences in neighborhood
                    sequences_for_clustering = neighborhood_headers
                    self.logger.debug(f"Using all {len(sequences_for_clustering)} sequences in neighborhood (overlaps allowed)")
                else:
                    # No overlaps: filter to unassigned sequences only  
                    sequences_for_clustering = [h for h in neighborhood_headers 
                                              if not assignment_tracker.is_assigned(h)]
                    
                    if len(sequences_for_clustering) < len(target_headers_for_iteration):
                        self.logger.debug(f"Some targets already assigned - using {len(sequences_for_clustering)} unassigned sequences")
                    
                    if not sequences_for_clustering:
                        self.logger.debug(f"No unassigned sequences in neighborhood - skipping iteration")
                        continue
                
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
                
                # OPTIMIZATION: Prune neighborhood based on distance to targets
                # This reduces computational cost of target clustering by focusing on closest candidates
                pruned_sequences, pruned_headers, pruned_target_indices = self._prune_neighborhood_by_distance(
                    neighborhood_sequences, sequences_for_clustering, target_indices_in_neighborhood
                )
                
                # Create lazy distance provider for pruned neighborhood
                self.logger.debug(f"Creating distance provider for {len(pruned_sequences)} pruned neighborhood sequences")
                distance_provider = DistanceProviderFactory.create_lazy_provider(
                    pruned_sequences,
                    alignment_method="adjusted",  # Use consistent parameters
                    end_skip_distance=20,
                    normalize_homopolymers=True,
                    handle_iupac_overlap=True,
                    normalize_indels=True,
                    max_repeat_motif_length=2
                )
                
                # Perform target clustering on pruned neighborhood
                target_cluster_indices, remaining_indices, clustering_metrics = self.target_clustering.cluster(
                    distance_provider, pruned_target_indices, pruned_sequences
                )
                
                # Log distance computation statistics
                if hasattr(distance_provider, 'get_cache_stats'):
                    stats = distance_provider.get_cache_stats()
                    if stats['theoretical_max'] > 0:
                        coverage_pct = 100.0 * stats['cached_distances'] / stats['theoretical_max']
                        self.logger.debug(f"Distance computation stats: {stats['cached_distances']} computed "
                                       f"out of {stats['theoretical_max']} possible "
                                       f"({coverage_pct:.1f}% coverage)")
                    else:
                        self.logger.debug(f"Distance computation stats: {stats['cached_distances']} computed "
                                       f"(no pairwise distances needed for single sequence)")
                
                # Convert indices back to sequence headers
                target_cluster_headers = [pruned_headers[i] for i in target_cluster_indices]
                remaining_headers = [pruned_headers[i] for i in remaining_indices]
                
                # Generate unique cluster ID
                cluster_id = f"cluster_{iteration:03d}"
                
                # Assign sequences to cluster
                assignment_tracker.assign_sequences(target_cluster_headers, cluster_id, iteration)
                
                # Update BLAST memory for spiral target selection
                target_selector.mark_sequences_processed(target_cluster_headers, self.allow_overlaps)
                
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
                    # Supervised mode: increment by 1 target processed
                    progress_increment = 1
                    postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                elif progress_mode == "clusters":
                    # Cluster count mode: increment by 1 cluster created
                    progress_increment = 1
                    postfix = f"med_clust={median_cluster_size}, med_gap={median_gap_size:.3f}, assigned={assigned_total}"
                elif progress_mode in ["sequences", "sequences_exhaustive"]:
                    # Sequence count mode: track unique sequences assigned
                    if self.allow_overlaps:
                        # In overlap mode, count unique sequences assigned (deduplicate)
                        unique_assigned_count = len(assignment_tracker.assigned_sequences)
                        progress_increment = unique_assigned_count - pbar.n  # Only increment by new unique assignments
                    else:
                        # In no-overlap mode, all assignments are unique
                        progress_increment = len(target_cluster_headers)
                    
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
                
                # Check stopping criteria for unsupervised modes
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
            # Clean up BLAST database
            blast_finder.cleanup()
        
        # Process final results
        results.total_iterations = iteration
        results.total_sequences_processed = len(assignment_tracker.assigned_sequences)
        results.coverage_percentage = (len(assignment_tracker.assigned_sequences) / len(headers)) * 100.0
        
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
        
        return results
    
    def _prune_neighborhood_by_distance(self, neighborhood_sequences: List[str], 
                                       neighborhood_headers: List[str], 
                                       target_indices: List[int]) -> Tuple[List[str], List[str], List[int]]:
        """Prune neighborhood by distance to targets to reduce computational cost.
        
        Args:
            neighborhood_sequences: All sequences in neighborhood
            neighborhood_headers: Headers for neighborhood sequences
            target_indices: Indices of target sequences within neighborhood
            
        Returns:
            Tuple of (pruned_sequences, pruned_headers, new_target_indices)
        """
        if len(neighborhood_sequences) <= 100:  # Skip pruning for small neighborhoods
            return neighborhood_sequences, neighborhood_headers, target_indices
        
        self.logger.debug(f"Pruning neighborhood from {len(neighborhood_sequences)} sequences")
        
        # Calculate distances from all sequences to target sequences
        target_sequences = [neighborhood_sequences[i] for i in target_indices]
        
        # Create a temporary distance provider for pruning calculation
        temp_distance_provider = DistanceProviderFactory.create_lazy_provider(
            neighborhood_sequences,
            alignment_method="adjusted",
            end_skip_distance=20,
            normalize_homopolymers=True,
            handle_iupac_overlap=True,
            normalize_indels=True,
            max_repeat_motif_length=2
        )
        
        # Calculate minimum distance from each sequence to any target
        sequence_min_distances = []
        for seq_idx in range(len(neighborhood_sequences)):
            if seq_idx in target_indices:
                # Target sequences have distance 0 to themselves
                min_dist = 0.0
            else:
                # Find minimum distance to any target
                distances_to_targets = temp_distance_provider.get_distances_from_sequence(
                    seq_idx, set(target_indices)
                )
                min_dist = min(distances_to_targets.values())
            
            sequence_min_distances.append((seq_idx, min_dist))
        
        # Count sequences within max_lump distance
        within_max_lump = sum(1 for _, dist in sequence_min_distances if dist <= self.max_lump)
        
        # Calculate target size for pruning (2x sequences within max_lump, minimum 50)
        target_pruned_size = max(50, min(len(neighborhood_sequences), 2 * within_max_lump))
        
        self.logger.debug(f"Found {within_max_lump} sequences within max_lump distance {self.max_lump:.4f}")
        self.logger.debug(f"Pruning to {target_pruned_size} closest sequences (2x estimate)")
        
        # Sort by distance and take the closest sequences
        sequence_min_distances.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in sequence_min_distances[:target_pruned_size]]
        
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
        for target_seq, target_header in zip(target_sequences, target_headers):
            normalized_target = target_seq.upper().strip()
            
            if normalized_target in input_seq_to_header:
                matched_targets.append({
                    'target_header': target_header,
                    'input_header': input_seq_to_header[normalized_target],
                    'target_sequence': target_seq
                })
                self.logger.debug(f"Matched target '{target_header}' to input '{input_seq_to_header[normalized_target]}'")
            else:
                self.logger.warning(f"Target sequence '{target_header}' not found in input sequences")
        
        return matched_targets