"""Shared types for cluster refinement and result tracking.

This module contains dataclasses and utilities used by gaphack-refine
for tracking cluster refinement stages and generating cluster IDs.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


def get_stage_suffix(stage_name: str, refinement_count: int = 0) -> str:
    """Get stage suffix for cluster IDs.

    Args:
        stage_name: Stage name ("initial", "deconflicted", "refined")
        refinement_count: For refined stages, the refinement number (0-based)

    Returns:
        Stage suffix character(s)

    Examples:
        >>> get_stage_suffix("initial")
        'I'
        >>> get_stage_suffix("deconflicted")
        'C'
        >>> get_stage_suffix("refined", refinement_count=0)
        'R1'
    """
    if stage_name == "initial":
        return "I"
    elif stage_name == "deconflicted":
        return "C"
    elif stage_name == "refined":
        return f"R{refinement_count + 1}"
    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def format_cluster_id(cluster_num: int, stage_suffix: str) -> str:
    """Format cluster ID with stage suffix.

    Args:
        cluster_num: Cluster number (1-based)
        stage_suffix: Stage suffix from get_stage_suffix()

    Returns:
        Formatted cluster ID

    Examples:
        >>> format_cluster_id(1, "I")
        'cluster_00001I'
        >>> format_cluster_id(113, "C")
        'cluster_00113C'
    """
    return f"cluster_{cluster_num:05d}{stage_suffix}"


def parse_cluster_id(cluster_id: str) -> Tuple[int, str]:
    """Parse cluster ID into number and suffix.

    Args:
        cluster_id: Cluster ID string (e.g., "cluster_00001I", "cluster_00023R1")

    Returns:
        Tuple of (cluster_number, stage_suffix)

    Raises:
        ValueError: If cluster ID format is invalid
    """
    match = re.match(r'cluster_(\d+)([ICR]\d*)', cluster_id)
    if not match:
        raise ValueError(f"Invalid cluster ID format: {cluster_id}")
    return int(match.group(1)), match.group(2)


def get_next_cluster_number(existing_clusters: Dict[str, List[str]]) -> int:
    """Get the next available cluster number from existing clusters.

    Scans all cluster IDs to find the maximum number used, regardless of stage suffix.

    Args:
        existing_clusters: Dict mapping cluster_id -> sequence headers

    Returns:
        Next available cluster number (max + 1, or 1 if no clusters exist)
    """
    if not existing_clusters:
        return 1

    max_num = 0
    for cluster_id in existing_clusters.keys():
        try:
            num, _ = parse_cluster_id(cluster_id)
            max_num = max(max_num, num)
        except ValueError:
            # Skip invalid cluster IDs
            continue

    return max_num + 1


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
