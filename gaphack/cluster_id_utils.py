"""Utility functions for cluster ID management in decompose output."""

import re
from pathlib import Path
from typing import Tuple, List, Dict


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
        >>> get_stage_suffix("refined", refinement_count=2)
        'R3'
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
        >>> format_cluster_id(23, "R1")
        'cluster_00023R1'
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

    Examples:
        >>> parse_cluster_id("cluster_00001I")
        (1, 'I')
        >>> parse_cluster_id("cluster_00113C")
        (113, 'C')
        >>> parse_cluster_id("cluster_00023R1")
        (23, 'R1')
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

    Examples:
        >>> get_next_cluster_number({"cluster_00001I": ["seq1"], "cluster_00002I": ["seq2"]})
        3
        >>> get_next_cluster_number({"cluster_00005C": ["seq1"], "cluster_00002I": ["seq2"]})
        6
        >>> get_next_cluster_number({})
        1
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


def count_refinement_stages(output_dir: Path) -> int:
    """Count how many refinement stages exist in work/ directory.

    Args:
        output_dir: Output directory containing work/ subdirectory

    Returns:
        Number of refinement stage directories (refined_1/, refined_2/, etc.)

    Examples:
        If work/ contains: initial/, deconflicted/, refined_1/, refined_2/
        Returns: 2
    """
    work_dir = output_dir / "work"
    if not work_dir.exists():
        return 0

    count = 0
    i = 1
    while (work_dir / f"refined_{i}").exists():
        count += 1
        i += 1

    return count


def get_stage_directory(output_dir: Path, stage_name: str, refinement_count: int = 0) -> Path:
    """Get the directory path for a given stage.

    Args:
        output_dir: Output directory
        stage_name: Stage name ("initial", "deconflicted", "refined")
        refinement_count: For refined stages, the refinement number (0-based)

    Returns:
        Path to stage directory

    Examples:
        >>> get_stage_directory(Path("/out"), "initial")
        Path('/out/work/initial')
        >>> get_stage_directory(Path("/out"), "refined", refinement_count=1)
        Path('/out/work/refined_2')
    """
    work_dir = output_dir / "work"

    if stage_name == "initial":
        return work_dir / "initial"
    elif stage_name == "deconflicted":
        return work_dir / "deconflicted"
    elif stage_name == "refined":
        return work_dir / f"refined_{refinement_count + 1}"
    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def format_cluster_filename(cluster_num: int, stage_suffix: str) -> str:
    """Format cluster filename with stage suffix.

    Args:
        cluster_num: Cluster number (1-based)
        stage_suffix: Stage suffix from get_stage_suffix()

    Returns:
        Cluster filename

    Examples:
        >>> format_cluster_filename(1, "I")
        'cluster_00001I.fasta'
        >>> format_cluster_filename(113, "C")
        'cluster_00113C.fasta'
    """
    return f"{format_cluster_id(cluster_num, stage_suffix)}.fasta"


def load_all_stage_clusters(work_dir: Path) -> Dict[str, List[str]]:
    """Load all clusters from all stages in work/ directory.

    Scans all stage subdirectories and loads cluster membership from FASTA files.

    Args:
        work_dir: Path to work/ directory

    Returns:
        Dict mapping cluster_id -> list of sequence headers (all stages combined)
    """
    from Bio import SeqIO

    clusters = {}

    # Scan all stage directories
    if not work_dir.exists():
        return clusters

    for stage_dir in sorted(work_dir.iterdir()):
        if not stage_dir.is_dir():
            continue

        # Load all cluster files from this stage
        for cluster_file in sorted(stage_dir.glob("cluster_*.fasta")):
            # Extract cluster ID from filename (without .fasta)
            cluster_id = cluster_file.stem

            # Read sequence headers
            headers = []
            for record in SeqIO.parse(cluster_file, "fasta"):
                headers.append(record.id)

            if headers:
                clusters[cluster_id] = headers

    return clusters


def get_latest_stage_directory(work_dir: Path) -> Path:
    """Get the most recent stage directory in work/.

    Returns the latest stage in processing order:
    1. refined_N (highest N)
    2. deconflicted/
    3. initial/

    Args:
        work_dir: Path to work/ directory

    Returns:
        Path to latest stage directory

    Raises:
        FileNotFoundError: If no stage directories exist
    """
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory does not exist: {work_dir}")

    # Check for refined stages (highest number first)
    i = 100  # Upper bound for search
    while i >= 1:
        refined_dir = work_dir / f"refined_{i}"
        if refined_dir.exists():
            return refined_dir
        i -= 1

    # Check for deconflicted
    deconflicted_dir = work_dir / "deconflicted"
    if deconflicted_dir.exists():
        return deconflicted_dir

    # Check for initial
    initial_dir = work_dir / "initial"
    if initial_dir.exists():
        return initial_dir

    raise FileNotFoundError(f"No stage directories found in {work_dir}")
