"""
Target selection strategies for iterative decompose clustering.

This module provides different strategies for selecting target sequences during
iterative clustering in gaphack-decompose mode:
- TargetSelector: Uses explicitly provided target sequences
- NearbyTargetSelector: Uses BLAST neighborhoods for proximity-based selection
- BlastResultMemory: Memory pool for storing BLAST neighborhoods

These classes enable systematic coverage of sequence space during decompose mode.
"""

import logging
from typing import Dict, List, Optional, Set
import random

logger = logging.getLogger(__name__)


class TargetSelector:
    """Target selection strategy using provided target sequences."""

    def __init__(self, target_headers: List[str]):
        self.target_headers = target_headers
        self.used_targets: Set[str] = set()

    def get_next_target(self, assignment_tracker) -> Optional[List[str]]:
        """Get next target sequence(s) for clustering."""
        # Find first unused target that hasn't been assigned yet
        for target_header in self.target_headers:
            if (target_header not in self.used_targets and
                not assignment_tracker.is_assigned(target_header)):
                self.used_targets.add(target_header)
                return [target_header]

        return None  # No more unassigned targets

    def has_more_targets(self, assignment_tracker) -> bool:
        """Check if there are more targets to process."""
        for target_header in self.target_headers:
            if (target_header not in self.used_targets and
                not assignment_tracker.is_assigned(target_header)):
                return True
        return False

    def add_blast_neighborhood(self, target_header: str, neighborhood_headers: List[str]) -> None:
        """No-op for directed mode - doesn't use BLAST memory."""
        pass

    def mark_sequences_processed(self, processed_headers: List[str]) -> None:
        """No-op for directed mode - doesn't need memory management."""
        pass


class BlastResultMemory:
    """Memory pool for storing BLAST neighborhoods for nearby target selection."""

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

    def get_nearby_candidates(self, assignment_tracker) -> List[str]:
        """Get unassigned sequences from BLAST neighborhoods for nearby selection."""
        candidates = []
        for seq_header in self.candidate_pool:
            if not assignment_tracker.is_assigned(seq_header):
                candidates.append(seq_header)
        return candidates

    def mark_processed(self, processed_headers: List[str]) -> None:
        """Mark sequences as processed. Sequences remain in candidate pool for future clusters."""
        logger.debug(f"Processed {len(processed_headers)} sequences, "
                    f"pool unchanged: {len(self.candidate_pool)}, "
                    f"active neighborhoods: {len(self.unprocessed_neighborhoods)}")


class NearbyTargetSelector:
    """Target selection strategy using nearby sequence exploration with random fallback."""

    def __init__(self, all_headers: List[str], max_clusters: Optional[int] = None,
                 max_sequences: Optional[int] = None):
        self.all_headers = all_headers
        self.max_clusters = max_clusters
        self.max_sequences = max_sequences
        self.iteration_count = 0
        self.blast_memory = BlastResultMemory()
        self.used_targets: Set[str] = set()

        # Initialize with random seed if no BLAST history available
        self.random_state = random.Random(42)  # Deterministic for reproducibility

    def get_next_target(self, assignment_tracker) -> Optional[List[str]]:
        """Get next target using nearby sequence logic with random fallback."""
        self.iteration_count += 1

        # Try nearby selection first: pick from previous BLAST neighborhoods
        nearby_candidates = self.blast_memory.get_nearby_candidates(assignment_tracker)
        nearby_candidates = [h for h in nearby_candidates if h not in self.used_targets]

        target_header = None
        selection_method = ""

        if nearby_candidates:
            # Nearby selection: choose from BLAST neighborhood candidates
            target_header = self.random_state.choice(nearby_candidates)
            selection_method = "nearby"
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
                        f"(nearby_pool: {len(nearby_candidates)}, total_unassigned: {len([h for h in self.all_headers if not assignment_tracker.is_assigned(h)])})")
            return [target_header]

        return None  # No more targets available

    def has_more_targets(self, assignment_tracker) -> bool:
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
        """Store BLAST neighborhood before pruning for future nearby selection."""
        self.blast_memory.add_neighborhood(target_header, neighborhood_headers)

    def mark_sequences_processed(self, processed_headers: List[str]) -> None:
        """Update memory after clustering iteration."""
        self.blast_memory.mark_processed(processed_headers)