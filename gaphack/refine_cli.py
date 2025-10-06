"""Command-line interface for gaphack-refine.

Standalone tool for applying conflict resolution and close cluster refinement
to existing cluster assignments from any clustering tool.
"""

import argparse
import logging
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .cluster_refinement import (
    resolve_conflicts, refine_close_clusters,
    RefinementConfig, verify_no_conflicts
)
from .cluster_graph import ClusterGraph
from .decompose import ClusterIDGenerator, ProcessingStageInfo
from . import __version__

logger = logging.getLogger(__name__)


def load_clusters_from_directory(input_dir: Path) -> Tuple[
    Dict[str, List[str]],  # clusters: cluster_id → sequence_headers
    List[str],              # all_sequences
    List[str],              # all_headers
    List[str],              # unassigned_headers
    Dict[str, str]          # header_mapping: header_id → full_header
]:
    """Load all cluster FASTA files from directory.

    Args:
        input_dir: Directory containing cluster FASTA files

    Returns:
        Tuple of:
        - clusters: Dict mapping cluster_id to list of sequence headers
        - all_sequences: Unified list of all sequences from all clusters
        - all_headers: Unified list of all headers from all clusters
        - unassigned_headers: Headers from unassigned.fasta (if present)
        - header_mapping: Dict mapping sequence ID to full header (ID + description)

    Raises:
        FileNotFoundError: If input directory doesn't exist
        ValueError: If no cluster files found
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Find all FASTA files
    fasta_files = list(input_dir.glob("*.fasta")) + list(input_dir.glob("*.fa"))

    if not fasta_files:
        raise ValueError(f"No cluster files (*.fasta or *.fa) found in {input_dir}")

    clusters = {}
    all_sequences = []
    all_headers = []
    unassigned_headers = []
    header_mapping = {}  # header_id → full_header (preserves metadata)

    # Track sequence index for unified lists
    seq_idx = 0

    for fasta_file in sorted(fasta_files):
        # Extract cluster ID from filename
        cluster_id = fasta_file.stem  # Filename without extension

        # Special handling for unassigned.fasta
        if cluster_id.lower() == "unassigned":
            logger.info(f"Loading unassigned sequences from {fasta_file.name}")
            for record in SeqIO.parse(fasta_file, "fasta"):
                unassigned_headers.append(record.id)
                # Preserve full header for unassigned sequences too
                full_header = record.description if record.description else record.id
                header_mapping[record.id] = full_header
            logger.info(f"Loaded {len(unassigned_headers)} unassigned sequences")
            continue

        # Load cluster sequences
        cluster_headers = []
        cluster_sequences = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            cluster_headers.append(record.id)
            cluster_sequences.append(str(record.seq))

            # Preserve full header (ID + description) for metadata preservation
            full_header = record.description if record.description else record.id
            header_mapping[record.id] = full_header

        if not cluster_headers:
            logger.warning(f"Empty cluster file ignored: {fasta_file.name}")
            continue

        # Add to unified lists
        all_sequences.extend(cluster_sequences)
        all_headers.extend(cluster_headers)

        # Store cluster
        clusters[cluster_id] = cluster_headers
        seq_idx += len(cluster_headers)

        logger.debug(f"Loaded cluster '{cluster_id}': {len(cluster_headers)} sequences")

    if not clusters:
        raise ValueError(f"No valid cluster files found in {input_dir}")

    logger.info(f"Loaded {len(clusters)} clusters with {len(all_headers)} total sequences")

    return clusters, all_sequences, all_headers, unassigned_headers, header_mapping


def detect_conflicts(clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Detect sequences assigned to multiple clusters.

    Args:
        clusters: Dict mapping cluster_id to list of sequence headers

    Returns:
        Dict mapping sequence_header → list of cluster_ids containing it
        Only includes sequences in multiple clusters (conflicts)
    """
    # Build sequence -> cluster mapping
    sequence_assignments = defaultdict(list)

    for cluster_id, sequence_headers in clusters.items():
        for seq_header in sequence_headers:
            sequence_assignments[seq_header].append(cluster_id)

    # Filter to only conflicts (sequences in multiple clusters)
    conflicts = {
        seq_header: cluster_ids
        for seq_header, cluster_ids in sequence_assignments.items()
        if len(cluster_ids) > 1
    }

    return conflicts


def generate_cluster_mapping_report(
    original_clusters: Dict[str, List[str]],
    stage1_clusters: Optional[Dict[str, List[str]]],
    stage2_clusters: Optional[Dict[str, List[str]]],
    final_clusters: Dict[str, List[str]],
    output_path: Path
) -> None:
    """Generate cluster_mapping.txt showing ID transformations.

    Args:
        original_clusters: Initial input clusters
        stage1_clusters: After conflict resolution (None if skipped)
        stage2_clusters: After close cluster refinement (None if skipped)
        final_clusters: Final renumbered clusters
        output_path: Path to write mapping file
    """
    lines = []
    lines.append("# gaphack-refine cluster ID mapping")
    lines.append(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# ")
    lines.append("# Shows cluster ID transformations through refinement stages:")
    lines.append("# Original_ID → [Deconflicted_ID] → [Refined_ID] → Final_ID")
    lines.append("")

    # Build reverse mapping: final_id -> original sequences
    final_to_sequences = {
        final_id: set(headers)
        for final_id, headers in final_clusters.items()
    }

    # Track which original clusters went where
    original_to_final = defaultdict(list)

    for original_id, original_headers in original_clusters.items():
        original_set = set(original_headers)

        # Find which final cluster(s) contain these sequences
        for final_id, final_sequences in final_to_sequences.items():
            overlap = original_set & final_sequences
            if overlap:
                overlap_pct = (len(overlap) / len(original_set)) * 100
                original_to_final[original_id].append((final_id, len(overlap), overlap_pct))

    # Sort by original cluster ID for readability
    for original_id in sorted(original_clusters.keys()):
        destinations = original_to_final.get(original_id, [])

        if not destinations:
            # Cluster was completely removed (merged or filtered)
            lines.append(f"{original_id} → [removed/merged]")
        elif len(destinations) == 1:
            # Simple 1:1 or 1:many mapping
            final_id, count, pct = destinations[0]
            if pct >= 99.0:
                lines.append(f"{original_id} → {final_id}")
            else:
                lines.append(f"{original_id} → {final_id} ({count} sequences, {pct:.1f}%)")
        else:
            # Cluster was split across multiple final clusters
            lines.append(f"{original_id} → [split across {len(destinations)} clusters]")
            for final_id, count, pct in sorted(destinations, key=lambda x: x[1], reverse=True):
                lines.append(f"  → {final_id} ({count} sequences, {pct:.1f}%)")

    lines.append("")
    lines.append(f"# Total original clusters: {len(original_clusters)}")
    lines.append(f"# Total final clusters: {len(final_clusters)}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Cluster mapping written to {output_path.name}")


def generate_refinement_summary(
    input_dir: Path,
    output_dir: Path,
    parameters: Dict,
    initial_state: Dict,
    stage1_info: Optional[ProcessingStageInfo],
    stage2_info: Optional[ProcessingStageInfo],
    final_state: Dict,
    timing: Dict
) -> str:
    """Generate detailed summary report.

    Returns formatted summary text for refine_summary.txt
    """
    lines = []

    # Header
    lines.append("gaphack-refine Summary Report")
    lines.append("=" * 60)
    lines.append(f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Version: {__version__}")
    lines.append(f"Command: {' '.join(sys.argv)}")
    lines.append("")

    # Input section
    lines.append("Input")
    lines.append("-" * 60)
    lines.append(f"Input directory: {input_dir}")
    lines.append(f"Total input files: {initial_state['cluster_files']} cluster FASTA files")
    if initial_state['unassigned_count'] > 0:
        lines.append(f"                  + 1 unassigned.fasta ({initial_state['unassigned_count']} sequences)")
    lines.append(f"Total sequences: {initial_state['total_sequences']}")
    lines.append(f"Unique sequence headers: {initial_state['unique_headers']}")
    if initial_state['duplicate_headers'] > 0:
        lines.append(f"  ({initial_state['duplicate_headers']} duplicate headers detected)")
    lines.append(f"Input clusters: {initial_state['input_clusters']}")
    lines.append("")

    # Initial conflicts
    if initial_state['conflicts_count'] > 0:
        lines.append("Initial Conflicts Detected")
        lines.append("-" * 60)
        lines.append(f"Conflicted sequences: {initial_state['conflicts_count']}")
        lines.append(f"Conflicted clusters: {initial_state['conflicted_clusters']}")
        lines.append("")

    # Algorithm parameters
    lines.append("Algorithm Parameters")
    lines.append("-" * 60)
    lines.append(f"min_split: {parameters['min_split']}")
    lines.append(f"max_lump: {parameters['max_lump']}")
    lines.append(f"target_percentile: {parameters['target_percentile']}")
    lines.append(f"max_scope_size: {parameters['max_scope_size']}")
    if parameters.get('search_method'):
        lines.append(f"search_method: {parameters['search_method']}")
        lines.append(f"knn_neighbors: {parameters.get('knn_neighbors', 20)}")
    lines.append("")

    # Stage 1: Conflict Resolution
    lines.append("Stage 1: Conflict Resolution")
    lines.append("-" * 60)
    if stage1_info and stage1_info.summary_stats.get('conflicts_count', 0) > 0:
        lines.append("Status: APPLIED (conflicts detected)")
        lines.append("Method: Full gapHACk with minimal scope (no expansion)")
        lines.append("")

        components = stage1_info.components_processed
        lines.append(f"Conflict Components Processed: {len(components)}")
        for comp in components:
            if comp.get('processed', False):
                lines.append(f"  Component {comp['component_index'] + 1}: "
                           f"{comp['clusters_before_count']} clusters "
                           f"({comp['sequences_count']} sequences) → "
                           f"{comp['clusters_after_count']} clusters")

        lines.append("")
        lines.append(f"Clusters before: {stage1_info.summary_stats['clusters_before_count']}")
        lines.append(f"Clusters after: {stage1_info.summary_stats['clusters_after_count']}")

        conflicts_before = stage1_info.summary_stats.get('conflicts_count', 0)
        conflicts_after = stage1_info.summary_stats.get('remaining_conflicts_count', 0)
        if conflicts_before > 0:
            resolution_rate = ((conflicts_before - conflicts_after) / conflicts_before) * 100
            lines.append(f"Conflicts resolved: {conflicts_before} → {conflicts_after} ({resolution_rate:.1f}% resolution)")

        lines.append(f"Duration: {timing.get('stage1', 0):.1f} seconds")
    else:
        lines.append("Status: SKIPPED (no conflicts detected)")
        lines.append(f"Duration: {timing.get('stage1', 0):.1f} seconds")
    lines.append("")

    # Stage 2: Close Cluster Refinement
    lines.append("Stage 2: Close Cluster Refinement")
    lines.append("-" * 60)
    if stage2_info:
        close_threshold = stage2_info.summary_stats.get('close_threshold', 0.0)
        lines.append(f"Status: APPLIED (--refine-close-clusters {close_threshold})")
        lines.append(f"Threshold: {close_threshold:.4f}")

        expansion_threshold = parameters.get('expansion_threshold')
        if expansion_threshold:
            lines.append(f"Expansion threshold: {expansion_threshold:.4f}")

        lines.append("")
        lines.append(f"Proximity graph construction:")
        lines.append(f"  K-NN neighbors per cluster: {parameters.get('knn_neighbors', 20)}")
        lines.append(f"  Search method: {parameters.get('search_method', 'blast').upper()}")
        lines.append(f"  Graph construction time: {timing.get('proximity_graph', 0):.1f} seconds")
        lines.append("")

        close_pairs = stage2_info.summary_stats.get('close_pairs_found', 0)
        lines.append(f"Close pairs found: {close_pairs} pairs")

        components = stage2_info.components_processed
        lines.append(f"Connected components: {len(components)}")
        lines.append("")

        if components:
            lines.append(f"Close Cluster Components Processed: {len(components)}")
            for comp in components:
                if comp.get('processed', False):
                    lines.append(f"  Component {comp['component_index'] + 1}: "
                               f"{comp['clusters_before_count']} clusters "
                               f"({comp.get('sequences_count', 0)} sequences) → "
                               f"{comp['clusters_after_count']} clusters")

        lines.append("")
        lines.append(f"Clusters before: {stage2_info.summary_stats['clusters_before_count']}")
        lines.append(f"Clusters after: {stage2_info.summary_stats['clusters_after_count']}")

        cluster_change = stage2_info.summary_stats.get('cluster_count_change', 0)
        if cluster_change < 0:
            lines.append(f"Clusters merged: {abs(cluster_change)}")

        lines.append(f"Duration: {timing.get('stage2', 0):.1f} seconds")
    else:
        lines.append("Status: SKIPPED (--refine-close-clusters not specified or 0.0)")
    lines.append("")

    # Final Verification
    lines.append("Final Verification")
    lines.append("-" * 60)
    lines.append(f"Final clusters: {final_state['final_clusters']}")
    lines.append(f"Total sequences: {final_state['total_sequences']}")
    lines.append(f"Conflicts remaining: {final_state['conflicts_remaining']}")

    if final_state['mece_satisfied']:
        lines.append("MECE property: SATISFIED ✓")
    else:
        lines.append("MECE property: VIOLATED ✗")
    lines.append("")

    # Cluster Size Distribution
    lines.append("Cluster Size Distribution")
    lines.append("-" * 60)
    dist = final_state['size_distribution']
    lines.append(f"  1-10 sequences: {dist['1-10']} clusters ({dist['1-10_pct']:.1f}%)")
    lines.append(f"  11-50 sequences: {dist['11-50']} clusters ({dist['11-50_pct']:.1f}%)")
    lines.append(f"  51-100 sequences: {dist['51-100']} clusters ({dist['51-100_pct']:.1f}%)")
    lines.append(f"  100+ sequences: {dist['100+']} clusters ({dist['100+_pct']:.1f}%)")
    lines.append("")
    lines.append(f"Largest cluster: {final_state['largest_cluster_size']} sequences")
    lines.append(f"Smallest cluster: {final_state['smallest_cluster_size']} sequences")
    lines.append(f"Median cluster size: {final_state['median_cluster_size']} sequences")
    lines.append(f"Mean cluster size: {final_state['mean_cluster_size']:.1f} sequences")
    lines.append("")

    # Output
    lines.append("Output")
    lines.append("-" * 60)
    lines.append(f"Output directory: {output_dir}")
    lines.append(f"Cluster files: {final_state['final_clusters']} FASTA files")
    if final_state.get('unassigned_file'):
        lines.append(f"Unassigned file: Yes ({final_state.get('unassigned_count', 0)} sequences)")
    lines.append(f"Cluster mapping: cluster_mapping.txt")
    lines.append(f"Renumbering: {'Enabled (by size, largest first)' if parameters.get('renumber', True) else 'Disabled (IDs preserved)'}")
    lines.append("")

    # Processing Summary
    lines.append("Processing Summary")
    lines.append("-" * 60)
    lines.append(f"Total processing time: {timing['total']:.1f} seconds")
    lines.append(f"  - Input loading: {timing.get('loading', 0):.1f} seconds")
    if timing.get('stage1', 0) > 0:
        lines.append(f"  - Conflict resolution: {timing['stage1']:.1f} seconds")
    if timing.get('proximity_graph', 0) > 0:
        lines.append(f"  - Proximity graph: {timing['proximity_graph']:.1f} seconds")
    if timing.get('stage2', 0) > 0:
        lines.append(f"  - Close cluster refinement: {timing['stage2']:.1f} seconds")
    lines.append(f"  - Output writing: {timing.get('writing', 0):.1f} seconds")
    lines.append("")

    # Status
    cluster_change = final_state['final_clusters'] - initial_state['input_clusters']
    cluster_change_pct = (cluster_change / initial_state['input_clusters']) * 100 if initial_state['input_clusters'] > 0 else 0

    lines.append("Status: SUCCESS")
    lines.append(f"Final cluster count: {final_state['final_clusters']} "
                f"(from {initial_state['input_clusters']}, {cluster_change:+d} / {cluster_change_pct:+.1f}%)")

    if initial_state['conflicts_count'] > 0:
        lines.append(f"Conflicts resolved: {initial_state['conflicts_count']} → {final_state['conflicts_remaining']}")

    return '\n'.join(lines)


def write_output_clusters(
    clusters: Dict[str, List[str]],
    sequences: List[str],
    headers: List[str],
    unassigned_headers: List[str],
    output_dir: Path,
    header_mapping: Dict[str, str],
    renumber: bool = True
) -> Dict[str, str]:
    """Write cluster FASTA files to output directory.

    Args:
        clusters: Dict mapping cluster_id to list of sequence headers
        sequences: List of all sequences
        headers: List of all headers (indices match sequences)
        unassigned_headers: List of unassigned sequence headers
        output_dir: Output directory path
        header_mapping: Dict mapping sequence ID to full header (ID + description)
        renumber: If True, renumber clusters by size (largest = cluster_00001)

    Returns:
        Dict mapping original_cluster_id → final_cluster_id
    """
    # Create header to sequence mapping
    header_to_seq = {header: seq for header, seq in zip(headers, sequences)}

    # Optionally renumber by size
    if renumber:
        # Sort by size (largest first)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Create mapping and renumbered clusters
        id_mapping = {}
        renumbered_clusters = {}

        for idx, (old_id, cluster_headers) in enumerate(sorted_clusters, start=1):
            new_id = f"cluster_{idx:05d}"
            id_mapping[old_id] = new_id
            renumbered_clusters[new_id] = cluster_headers

        final_clusters = renumbered_clusters
    else:
        # Preserve original IDs
        id_mapping = {cluster_id: cluster_id for cluster_id in clusters.keys()}
        final_clusters = clusters

    # Write cluster FASTA files
    for cluster_id, cluster_headers in final_clusters.items():
        cluster_file = output_dir / f"{cluster_id}.fasta"

        records = []
        for header in cluster_headers:
            if header in header_to_seq:
                seq = header_to_seq[header]

                # Get full header from mapping and preserve original metadata
                full_header = header_mapping.get(header, header)
                original_desc = full_header.split(' ', 1)[1] if ' ' in full_header else ""

                # Combine original description with cluster info
                if original_desc:
                    full_description = f"{original_desc} {cluster_id}"
                else:
                    full_description = cluster_id

                record = SeqRecord(
                    Seq(seq),
                    id=header,
                    description=full_description
                )
                records.append(record)
            else:
                logger.warning(f"Header '{header}' not found in sequence list")

        with open(cluster_file, 'w') as f:
            SeqIO.write(records, f, "fasta-2line")

        logger.debug(f"Wrote {cluster_file.name} with {len(records)} sequences")

    # Write unassigned.fasta if present
    if unassigned_headers:
        unassigned_file = output_dir / "unassigned.fasta"

        records = []
        for header in unassigned_headers:
            if header in header_to_seq:
                seq = header_to_seq[header]

                # Get full header from mapping and preserve original metadata
                full_header = header_mapping.get(header, header)
                original_desc = full_header.split(' ', 1)[1] if ' ' in full_header else ""

                # Combine original description with unassigned info
                if original_desc:
                    full_description = f"{original_desc} unassigned"
                else:
                    full_description = "unassigned"

                record = SeqRecord(
                    Seq(seq),
                    id=header,
                    description=full_description
                )
                records.append(record)

        with open(unassigned_file, 'w') as f:
            SeqIO.write(records, f, "fasta-2line")

        logger.info(f"Wrote {unassigned_file.name} with {len(records)} sequences")

    logger.info(f"Wrote {len(final_clusters)} cluster FASTA files to {output_dir}")

    return id_mapping


def calculate_cluster_statistics(clusters: Dict[str, List[str]]) -> Dict:
    """Calculate cluster size distribution and statistics."""
    sizes = [len(cluster) for cluster in clusters.values()]

    if not sizes:
        return {
            '1-10': 0, '1-10_pct': 0.0,
            '11-50': 0, '11-50_pct': 0.0,
            '51-100': 0, '51-100_pct': 0.0,
            '100+': 0, '100+_pct': 0.0,
        }

    total_clusters = len(sizes)

    dist = {
        '1-10': sum(1 for s in sizes if 1 <= s <= 10),
        '11-50': sum(1 for s in sizes if 11 <= s <= 50),
        '51-100': sum(1 for s in sizes if 51 <= s <= 100),
        '100+': sum(1 for s in sizes if s > 100),
    }

    # Calculate percentages
    dist['1-10_pct'] = (dist['1-10'] / total_clusters) * 100
    dist['11-50_pct'] = (dist['11-50'] / total_clusters) * 100
    dist['51-100_pct'] = (dist['51-100'] / total_clusters) * 100
    dist['100+_pct'] = (dist['100+'] / total_clusters) * 100

    return dist


USAGE_EXAMPLES = """
Examples:
  # Refine gaphack-decompose output (conflict resolution only)
  gaphack-refine --input-dir initial_run/clusters/latest/ --output-dir refined/

  # Refine with both conflict resolution and close cluster refinement
  gaphack-refine --input-dir clusters/ --output-dir refined/ --refine-close-clusters 0.02

  # Refine vsearch output with custom parameters
  gaphack-refine --input-dir vsearch_clusters/ --output-dir refined/ \\
                 --refine-close-clusters 0.025 --max-lump 0.03 --min-split 0.01

  # Preserve original cluster IDs (no renumbering)
  gaphack-refine --input-dir clusters/ --output-dir refined/ --preserve-ids

  # Chained refinement (iterative)
  gaphack-refine --input-dir round1/ --output-dir round2/ --refine-close-clusters 0.025
"""


def main():
    """Main entry point for gaphack-refine CLI."""
    parser = argparse.ArgumentParser(
        description="Refine existing clusters using conflict resolution and close cluster refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES
    )

    # Required arguments
    parser.add_argument('--input-dir', required=True, type=Path,
                       help='Directory containing cluster FASTA files (one per cluster)')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for refined clusters')

    # Refinement stage controls
    parser.add_argument('--refine-close-clusters', type=float, default=0.0,
                       help='Enable close cluster refinement with distance threshold (default: 0.0, disabled)')

    # Algorithm parameters
    parser.add_argument('--min-split', type=float, default=0.005,
                       help='Minimum distance to split clusters (default: 0.005)')
    parser.add_argument('--max-lump', type=float, default=0.02,
                       help='Maximum distance to lump clusters (default: 0.02)')
    parser.add_argument('--target-percentile', type=int, default=95,
                       help='Percentile for gap optimization (default: 95)')

    # Advanced refinement parameters
    parser.add_argument('--max-scope-size', type=int, default=300,
                       help='Maximum sequences for full gapHACk refinement (default: 300)')
    parser.add_argument('--expansion-threshold', type=float, default=None,
                       help='Distance threshold for scope expansion (default: 1.2 × close_threshold)')

    # Proximity graph parameters
    parser.add_argument('--search-method', choices=['blast', 'vsearch'], default='blast',
                       help='Search method for proximity graph (default: blast)')
    parser.add_argument('--knn-neighbors', type=int, default=20,
                       help='K for K-NN cluster graph (default: 20)')
    parser.add_argument('--blast-evalue', type=float, default=1e-5,
                       help='BLAST e-value threshold (default: 1e-5)')
    parser.add_argument('--min-identity', type=float, default=None,
                       help='Minimum sequence identity %% (default: auto)')

    # Output options
    parser.add_argument('--renumber', action='store_true', default=True,
                       help='Renumber clusters by size (default: True)')
    parser.add_argument('--preserve-ids', action='store_true',
                       help='Preserve input cluster IDs (don\'t renumber)')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Write directly to output-dir (no timestamped subdirectory)')

    # Other options
    parser.add_argument('--show-progress', action='store_true', default=True,
                       help='Show progress bars (default: True)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging verbosity (default: INFO)')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Track timing
    timing = {}
    total_start = time.time()

    try:
        # Validate preserve-ids vs renumber
        renumber = args.renumber and not args.preserve_ids

        # Stage 0: Load and validate
        logger.info(f"gaphack-refine v{__version__}")
        logger.info(f"Loading clusters from {args.input_dir}")

        load_start = time.time()
        clusters, sequences, headers, unassigned_headers, header_mapping = load_clusters_from_directory(args.input_dir)
        timing['loading'] = time.time() - load_start

        if len(clusters) < 2:
            logger.error(f"At least 2 clusters required for refinement (found {len(clusters)})")
            sys.exit(1)

        # Detect conflicts
        conflicts = detect_conflicts(clusters)

        # Count unique vs duplicate headers
        unique_headers = len(set(headers))
        duplicate_headers = len(headers) - unique_headers

        # Count conflicted clusters
        conflicted_clusters_set = set()
        for cluster_ids in conflicts.values():
            conflicted_clusters_set.update(cluster_ids)

        # Report initial state
        initial_state = {
            'cluster_files': len(clusters),
            'total_sequences': len(headers),
            'unique_headers': unique_headers,
            'duplicate_headers': duplicate_headers,
            'input_clusters': len(clusters),
            'unassigned_count': len(unassigned_headers),
            'conflicts_count': len(conflicts),
            'conflicted_clusters': len(conflicted_clusters_set)
        }

        logger.info(f"Loaded {len(clusters)} clusters with {len(headers)} sequences ({unique_headers} unique headers)")
        if duplicate_headers > 0:
            logger.info(f"  {duplicate_headers} duplicate headers detected")
        if conflicts:
            logger.warning(f"Detected {len(conflicts)} conflicted sequences across {len(conflicted_clusters_set)} clusters")
        else:
            logger.info("No conflicts detected")

        # Track clusters through stages
        current_clusters = clusters.copy()
        stage1_clusters = None
        stage2_clusters = None

        # Stage 1: Conflict Resolution (always applied if conflicts exist)
        stage1_info = None
        if conflicts:
            logger.info(f"Stage 1: Resolving {len(conflicts)} conflicts")
            stage1_start = time.time()

            config = RefinementConfig(max_full_gaphack_size=args.max_scope_size)
            conflict_id_generator = ClusterIDGenerator(stage_name="deconflicted")

            current_clusters, stage1_info = resolve_conflicts(
                conflicts=conflicts,
                all_clusters=current_clusters,
                sequences=sequences,
                headers=headers,
                config=config,
                min_split=args.min_split,
                max_lump=args.max_lump,
                target_percentile=args.target_percentile,
                cluster_id_generator=conflict_id_generator
            )

            stage1_clusters = current_clusters.copy()
            timing['stage1'] = time.time() - stage1_start

            remaining_conflicts = stage1_info.summary_stats.get('remaining_conflicts_count', 0)
            if remaining_conflicts > 0:
                logger.warning(f"Stage 1 complete: {remaining_conflicts} conflicts remain unresolved")
            else:
                logger.info(f"Stage 1 complete: All conflicts resolved")
            logger.info(f"  {len(clusters)} → {len(current_clusters)} clusters ({timing['stage1']:.1f}s)")
        else:
            logger.info("Stage 1: Skipped (no conflicts detected)")
            timing['stage1'] = 0.0

        # Stage 2: Close Cluster Refinement (optional)
        stage2_info = None
        if args.refine_close_clusters > 0.0:
            logger.info(f"Stage 2: Refining close clusters (threshold={args.refine_close_clusters:.4f})")

            # Build proximity graph
            proximity_start = time.time()
            logger.info(f"Building {args.search_method.upper()} K-NN proximity graph (K={args.knn_neighbors})")

            proximity_graph = ClusterGraph(
                clusters=current_clusters,
                sequences=sequences,
                headers=headers,
                k_neighbors=args.knn_neighbors,
                blast_evalue=args.blast_evalue,
                blast_identity=args.min_identity or 90.0,
                search_method=args.search_method,
                show_progress=args.show_progress
            )
            timing['proximity_graph'] = time.time() - proximity_start
            logger.info(f"  Proximity graph built ({timing['proximity_graph']:.1f}s)")

            # Apply close cluster refinement
            stage2_start = time.time()

            expansion_threshold = args.expansion_threshold
            if expansion_threshold is None:
                expansion_threshold = args.refine_close_clusters * 1.2

            config = RefinementConfig(
                max_full_gaphack_size=args.max_scope_size,
                close_cluster_expansion_threshold=expansion_threshold
            )

            refinement_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

            current_clusters, stage2_info = refine_close_clusters(
                all_clusters=current_clusters,
                sequences=sequences,
                headers=headers,
                proximity_graph=proximity_graph,
                config=config,
                min_split=args.min_split,
                max_lump=args.max_lump,
                target_percentile=args.target_percentile,
                close_threshold=args.refine_close_clusters,
                cluster_id_generator=refinement_id_generator
            )

            stage2_clusters = current_clusters.copy()
            timing['stage2'] = time.time() - stage2_start

            clusters_before = stage2_info.summary_stats['clusters_before_count']
            clusters_after = stage2_info.summary_stats['clusters_after_count']
            logger.info(f"Stage 2 complete: {clusters_before} → {clusters_after} clusters ({timing['stage2']:.1f}s)")
        else:
            logger.info("Stage 2: Skipped (--refine-close-clusters not specified or 0.0)")
            timing['stage2'] = 0.0
            timing['proximity_graph'] = 0.0

        # Final verification
        logger.info("Performing final conflict verification")
        final_verification = verify_no_conflicts(current_clusters, conflicts, context="final")

        if not final_verification['no_conflicts']:
            logger.error(f"CRITICAL: Final verification failed - {final_verification['conflict_count']} conflicts remain")
            sys.exit(1)

        # Create output directory
        if args.no_timestamp:
            final_output_dir = args.output_dir
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_dir = args.output_dir / timestamp

        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Write output clusters
        write_start = time.time()
        logger.info(f"Writing refined clusters to {final_output_dir}")

        id_mapping = write_output_clusters(
            clusters=current_clusters,
            sequences=sequences,
            headers=headers,
            unassigned_headers=unassigned_headers,
            output_dir=final_output_dir,
            header_mapping=header_mapping,
            renumber=renumber
        )

        timing['writing'] = time.time() - write_start

        # Create 'latest' symlink (if using timestamp)
        if not args.no_timestamp:
            latest_symlink = args.output_dir / "latest"
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()
            latest_symlink.symlink_to(timestamp, target_is_directory=True)
            logger.info(f"Created symlink: latest -> {timestamp}/")

        # Generate cluster mapping report
        final_clusters = {id_mapping.get(cid, cid): headers for cid, headers in current_clusters.items()}

        generate_cluster_mapping_report(
            original_clusters=clusters,
            stage1_clusters=stage1_clusters,
            stage2_clusters=stage2_clusters,
            final_clusters=final_clusters,
            output_path=final_output_dir / "cluster_mapping.txt"
        )

        # Calculate final statistics
        cluster_sizes = [len(cluster) for cluster in current_clusters.values()]
        size_distribution = calculate_cluster_statistics(current_clusters)

        final_state = {
            'final_clusters': len(current_clusters),
            'total_sequences': sum(cluster_sizes),
            'conflicts_remaining': final_verification['conflict_count'],
            'mece_satisfied': final_verification['no_conflicts'],
            'size_distribution': size_distribution,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0,
            'mean_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            'unassigned_file': len(unassigned_headers) > 0,
            'unassigned_count': len(unassigned_headers)
        }

        # Generate summary report
        timing['total'] = time.time() - total_start

        parameters = {
            'min_split': args.min_split,
            'max_lump': args.max_lump,
            'target_percentile': args.target_percentile,
            'max_scope_size': args.max_scope_size,
            'expansion_threshold': args.expansion_threshold or (args.refine_close_clusters * 1.2 if args.refine_close_clusters > 0 else None),
            'search_method': args.search_method,
            'knn_neighbors': args.knn_neighbors,
            'renumber': renumber
        }

        summary_text = generate_refinement_summary(
            input_dir=args.input_dir,
            output_dir=final_output_dir,
            parameters=parameters,
            initial_state=initial_state,
            stage1_info=stage1_info,
            stage2_info=stage2_info,
            final_state=final_state,
            timing=timing
        )

        summary_path = final_output_dir / "refine_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)

        logger.info(f"Summary report written to {summary_path.name}")

        # Final summary
        logger.info("=" * 60)
        logger.info("Refinement complete!")
        logger.info(f"  Input clusters: {len(clusters)}")
        logger.info(f"  Final clusters: {len(current_clusters)} ({len(current_clusters) - len(clusters):+d})")
        logger.info(f"  Conflicts resolved: {len(conflicts)} → {final_verification['conflict_count']}")
        logger.info(f"  Total time: {timing['total']:.1f}s")
        logger.info(f"  Output: {final_output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Refinement failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
