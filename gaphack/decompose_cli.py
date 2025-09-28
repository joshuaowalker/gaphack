"""CLI for gaphack-decompose command."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .decompose import DecomposeClustering, DecomposeResults
from .utils import save_clusters_to_file
from typing import Dict, List


def _save_decompose_fasta_files(all_clusters: Dict[str, List[str]], unassigned: List[str],
                               output_base: str, headers: List[str], sequences: List[str],
                               header_mapping: Dict[str, str]) -> None:
    """Save decompose clusters to FASTA files with preserved cluster IDs."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO
    
    # Create header to index mapping - headers contain sequence IDs
    header_to_idx = {header: i for i, header in enumerate(headers)}
    
    # Save each cluster to a separate FASTA file with original cluster ID
    for cluster_id, cluster_headers in all_clusters.items():
        # Convert cluster_id to valid filename (replace problematic characters)
        safe_cluster_id = cluster_id.replace('_', '_')  # Keep underscores as-is
        cluster_file = f"{output_base}.{safe_cluster_id}.fasta"
        
        records = []
        for header in cluster_headers:
            if header in header_to_idx:
                seq_idx = header_to_idx[header]
                # Get full header from mapping and preserve original metadata
                full_header = header_mapping.get(header, header)
                original_desc = full_header.split(' ', 1)[1] if ' ' in full_header else ""
                
                # Combine original description with cluster info
                if original_desc:
                    full_description = f"{original_desc} {cluster_id}"
                else:
                    full_description = cluster_id
                
                record = SeqRecord(
                    Seq(sequences[seq_idx]),
                    id=header,  # Use sequence ID as BioPython ID
                    description=full_description
                )
                records.append(record)
        
        if records:  # Only create file if it has sequences
            with open(cluster_file, 'w') as f:
                SeqIO.write(records, f, "fasta-2line")
            logging.debug(f"Wrote {len(records)} sequences to {cluster_file}")
    
    # Save unassigned sequences
    if unassigned:
        unassigned_file = f"{output_base}.unassigneds.fasta"
        records = []
        for header in unassigned:
            if header in header_to_idx:
                seq_idx = header_to_idx[header]
                # Get full header from mapping and preserve original metadata
                full_header = header_mapping.get(header, header)
                original_desc = full_header.split(' ', 1)[1] if ' ' in full_header else ""
                
                # Combine original description with unassigned info
                if original_desc:
                    full_description = f"{original_desc} unassigned"
                else:
                    full_description = "unassigned"
                
                record = SeqRecord(
                    Seq(sequences[seq_idx]),
                    id=header,  # Use sequence ID as BioPython ID
                    description=full_description
                )
                records.append(record)
        
        if records:
            with open(unassigned_file, 'w') as f:
                SeqIO.write(records, f, "fasta-2line")
            logging.debug(f"Wrote {len(records)} unassigned sequences to {unassigned_file}")


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def save_decompose_results(results: DecomposeResults, output_base: str, 
                          input_fasta: str = None) -> None:
    """Save decomposition clustering results to files."""
    output_base_path = Path(output_base)
    
    # Load sequences from input FASTA if provided (for FASTA output)
    sequences = None
    all_headers = None
    header_mapping = None
    hash_to_headers = None
    if input_fasta:
        from .utils import load_sequences_with_deduplication
        sequences, hash_ids, hash_to_headers = load_sequences_with_deduplication(input_fasta)

        # Create all_headers list containing all original headers for backward compatibility
        all_headers = []
        for headers_list in hash_to_headers.values():
            all_headers.extend(headers_list)

        # Create sequences and headers lists that match each other for FASTA output
        # We need to expand the deduplicated sequences back to match all original headers
        expanded_sequences = []
        expanded_headers = []
        for hash_id, headers_list in hash_to_headers.items():
            # Find the sequence for this hash_id
            hash_index = hash_ids.index(hash_id)
            sequence = sequences[hash_index]

            # Add this sequence for each original header
            for header in headers_list:
                expanded_sequences.append(sequence)
                expanded_headers.append(header)

        # Update sequences and all_headers to be the expanded versions
        sequences = expanded_sequences
        all_headers = expanded_headers

        # Create header_mapping for backward compatibility (use header as both key and value)
        header_mapping = {header: header for header in all_headers}

    # Create a mapping of headers to indices for the loaded sequences
    if all_headers:
        header_to_idx = {header: i for i, header in enumerate(all_headers)}
        
        # Convert clusters to index-based format (use all_clusters to include conflicts in FASTA files)
        cluster_list = []
        for cluster_id, cluster_headers in results.all_clusters.items():
            cluster_indices = []
            for header in cluster_headers:
                if header in header_to_idx:
                    cluster_indices.append(header_to_idx[header])
                else:
                    logging.warning(f"Header '{header}' not found in input sequences")
            if cluster_indices:  # Only add non-empty clusters
                cluster_list.append(cluster_indices)
        
        # Convert unassigned to index-based format
        unassigned_indices = []
        for header in results.unassigned:
            if header in header_to_idx:
                unassigned_indices.append(header_to_idx[header])
            else:
                logging.warning(f"Unassigned header '{header}' not found in input sequences")
        
        # Save FASTA files with preserved cluster IDs
        _save_decompose_fasta_files(results.all_clusters, results.unassigned, output_base, 
                                   all_headers, sequences, header_mapping)
    
    # Save main TSV assignment file
    tsv_file = f"{output_base}.decompose_assignments.tsv"
    with open(tsv_file, 'w') as f:
        f.write("sequence_id\tcluster_id\n")
        
        # Write cluster assignments
        for cluster_id, cluster_headers in results.clusters.items():
            for header in cluster_headers:
                f.write(f"{header}\t{cluster_id}\n")
        
        # Write unassigned
        for header in results.unassigned:
            f.write(f"{header}\tunassigned\n")
    
    # Save conflicts if any
    if results.conflicts:
        conflicts_file = f"{output_base}.decompose_conflicts.tsv"
        with open(conflicts_file, 'w') as f:
            f.write("sequence_id\tcluster_ids\n")
            for seq_id, cluster_ids in results.conflicts.items():
                f.write(f"{seq_id}\t{','.join(cluster_ids)}\n")
    
    # Save enhanced summary report
    report_file = f"{output_base}.decompose_report.txt"
    with open(report_file, 'w') as f:
        f.write("Gaphack-Decompose Clustering Report\n")
        f.write("=" * 40 + "\n\n")

        # Header with run metadata
        if results.start_time:
            f.write(f"Run timestamp: {results.start_time}\n")
        if results.command_line:
            f.write(f"Command line: {results.command_line}\n")
        f.write("\n")

        # Basic summary statistics
        f.write("Summary Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total iterations: {results.total_iterations}\n")
        f.write(f"Total sequences processed: {results.total_sequences_processed}\n")
        f.write(f"Coverage percentage: {results.coverage_percentage:.1f}%\n")
        f.write(f"Clusters created: {len(results.clusters)}\n")
        f.write(f"Unassigned sequences: {len(results.unassigned)}\n")
        f.write(f"Conflicts detected: {len(results.conflicts)}\n\n")

        # Verification results
        if results.verification_results:
            f.write("Verification Summary:\n")
            f.write("-" * 20 + "\n")
            for stage, verification in results.verification_results.items():
                f.write(f"{stage.title()}: {verification['conflict_count']} conflicts, "
                       f"Conflict-free: {verification['no_conflicts']}\n")
            f.write("\n")

        # Iteration summary with active cluster IDs (moved after verification)
        if results.iteration_summaries:
            f.write("Iteration Summary:\n")
            f.write("-" * 20 + "\n")
            for summary in results.iteration_summaries:
                # Show the cluster name created during this iteration
                cluster_name = summary.get('cluster_id', 'unknown')

                f.write(f"Iteration {summary['iteration']}: "
                       f"cluster_size={summary['cluster_size']}, "
                       f"gap_size={summary['gap_size']:.4f}, "
                       f"cluster_name={cluster_name}\n")
            f.write("\n")

        # Processing stages (conflict resolution and close cluster refinement)
        if results.processing_stages:
            for stage_info in results.processing_stages:
                f.write(f"{stage_info.stage_name}:\n")
                f.write("-" * (len(stage_info.stage_name) + 1) + "\n")

                # Stage summary
                stats = stage_info.summary_stats
                before_count = stats.get('clusters_before_count', 0)
                after_count = stats.get('clusters_after_count', 0)
                change = after_count - before_count

                f.write(f"Clusters before: {before_count}\n")
                f.write(f"Clusters after: {after_count}\n")
                f.write(f"Net change: {change:+d}\n")

                if 'conflicts_count' in stats:  # Conflict resolution
                    f.write(f"Conflicts resolved: {stats['conflicts_count']}\n")
                    f.write(f"Components processed: {stats.get('components_processed_count', 0)}\n")
                    f.write(f"Remaining conflicts: {stats.get('remaining_conflicts_count', 0)}\n")

                if 'close_pairs_found' in stats:  # Close cluster refinement
                    f.write(f"Close pairs found: {stats['close_pairs_found']}\n")
                    f.write(f"Close threshold: {stats.get('close_threshold', 'N/A')}\n")
                    f.write(f"Components processed: {stats.get('components_processed_count', 0)}\n")

                # Component-by-component transformations will be shown in Component Details section below

                # Component details
                if stage_info.components_processed:
                    f.write(f"\nComponent Details:\n")
                    for comp in stage_info.components_processed:
                        status = "✓ processed" if comp.get('processed', False) else "✗ skipped"
                        f.write(f"  Component {comp['component_index']}: "
                               f"{comp['clusters_before_count']} → {comp['clusters_after_count']} clusters "
                               f"({status})\n")
                        if 'skipped_reason' in comp:
                            f.write(f"    Reason: {comp['skipped_reason']}\n")
                        # Show source → destination cluster mapping
                        if 'clusters_before' in comp:
                            f.write(f"    Source: {', '.join(sorted(comp['clusters_before']))}\n")
                        if 'clusters_after' in comp and comp.get('processed', False):
                            f.write(f"    Result: {', '.join(sorted(comp['clusters_after']))}\n")
                f.write("\n")

        # Active to final cluster mapping
        if results.active_to_final_mapping:
            f.write("Active to Final Cluster Mapping:\n")
            f.write("-" * 35 + "\n")
            # Group by final cluster for better readability
            final_to_active = {}
            for active_id, final_id in results.active_to_final_mapping.items():
                if final_id not in final_to_active:
                    final_to_active[final_id] = []
                final_to_active[final_id].append(active_id)

            for final_id in sorted(final_to_active.keys()):
                active_list = final_to_active[final_id]
                f.write(f"{final_id}: {', '.join(sorted(active_list))}\n")
            f.write("\n")



        # Conflicts (if any)
        if results.conflicts:
            f.write("Conflicts:\n")
            f.write("-" * 10 + "\n")
            for seq_id, cluster_ids in results.conflicts.items():
                f.write(f"{seq_id}: assigned to {', '.join(cluster_ids)}\n")
            f.write("\n")
    
    print(f"Results saved:")
    if sequences:
        print(f"  FASTA clusters: {output_base}.cluster_*.fasta")
        if results.unassigned:
            print(f"  FASTA unassigned: {output_base}.unassigneds.fasta")
    print(f"  Assignments: {tsv_file}")
    print(f"  Report: {report_file}")
    if results.conflicts:
        print(f"  Conflicts: {conflicts_file}")


def main():
    """Main entry point for gaphack-decompose CLI."""
    parser = argparse.ArgumentParser(
        description="Iterative BLAST-based clustering for large datasets using target mode clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Directed mode with target sequences
  gaphack-decompose input.fasta --targets targets.fasta -o results

  # Undirected mode with cluster count limit
  gaphack-decompose input.fasta --max-clusters 50 -o results

  # Undirected mode with sequence coverage limit
  gaphack-decompose input.fasta --max-sequences 1000 -o results

  # Undirected mode until input exhausted (cluster all sequences)
  gaphack-decompose input.fasta -o results

  # Disable overlaps (each sequence in at most one cluster)
  gaphack-decompose input.fasta --no-overlaps -o results

  # Custom BLAST parameters
  gaphack-decompose input.fasta --targets targets.fasta \\
    --blast-max-hits 1000 --min-identity 85.0 -o results
        """
    )
    
    # Input/output arguments
    parser.add_argument('input_fasta', 
                       help='Input FASTA file containing all sequences to cluster')
    parser.add_argument('-o', '--output', required=True,
                       help='Output base path for result files')
    
    # Target selection arguments
    parser.add_argument('--targets',
                       help='FASTA file containing target sequences for directed mode')
    parser.add_argument('--max-clusters', type=int,
                       help='Maximum clusters to create (undirected mode, optional)')
    parser.add_argument('--max-sequences', type=int,
                       help='Maximum sequences to assign (undirected mode, optional)')
    parser.add_argument('--no-overlaps', action='store_true',
                       help='Disable sequence overlaps - each sequence assigned to at most one cluster')
    
    # BLAST parameters
    parser.add_argument('--blast-max-hits', type=int, default=1000,
                       help='Maximum BLAST hits per query (default: 1000)')
    parser.add_argument('--blast-threads', type=int,
                       help='Number of BLAST threads (default: auto)')
    parser.add_argument('--blast-evalue', type=float, default=1e-5,
                       help='BLAST e-value threshold (default: 1e-5)')
    parser.add_argument('--min-identity', type=float,
                       help='Minimum BLAST identity percentage (default: auto-calculated)')
    
    # Clustering parameters (from existing gapHACk)
    parser.add_argument('--min-split', type=float, default=0.005,
                       help='Minimum distance to split clusters (default: 0.005)')
    parser.add_argument('--max-lump', type=float, default=0.02,
                       help='Maximum distance to lump clusters (default: 0.02)')
    parser.add_argument('--target-percentile', type=int, default=95,
                       help='Percentile for gap optimization (default: 95)')
    
    # Conflict resolution arguments
    parser.add_argument('--resolve-conflicts', action='store_true',
                       help='Enable cluster refinement for conflict resolution using full gapHACk (minimal scope)')
    parser.add_argument('--refine-close-clusters', type=float, default=0.0, metavar='DISTANCE',
                       help='Enable close cluster refinement with distance threshold (0.0 = disabled, e.g. 0.02 for 2%% distance)')
    parser.add_argument('--proximity-graph', choices=['brute-force', 'blast-knn'], default='brute-force',
                       help='Proximity graph implementation for cluster refinement (default: brute-force)')
    parser.add_argument('--knn-neighbors', type=int, default=20,
                       help='Number of K-nearest neighbors for BLAST K-NN graph (default: 20)')

    # Control arguments
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not Path(args.input_fasta).exists():
        logger.error(f"Input FASTA file not found: {args.input_fasta}")
        sys.exit(1)
    
    # Auto-detect mode based on targets
    if args.targets:
        # Directed mode - validate targets file exists
        if not Path(args.targets).exists():
            logger.error(f"Targets FASTA file not found: {args.targets}")
            sys.exit(1)
        logger.info("Running in directed mode (targets provided)")
    else:
        # Undirected mode
        if not args.max_clusters and not args.max_sequences:
            logger.info("Running in undirected mode - will cluster until input is exhausted")
        else:
            logger.info("Running in undirected mode with stopping criteria")
    
    # Initialize decomposition clustering
    logger.info("Initializing gaphack-decompose")
    
    decomposer = DecomposeClustering(
        min_split=args.min_split,
        max_lump=args.max_lump,
        target_percentile=args.target_percentile,
        blast_max_hits=args.blast_max_hits,
        blast_threads=args.blast_threads,
        blast_evalue=args.blast_evalue,
        min_identity=args.min_identity,
        allow_overlaps=not args.no_overlaps,
        resolve_conflicts=args.resolve_conflicts,
        refine_close_clusters=args.refine_close_clusters > 0.0,
        close_cluster_threshold=args.refine_close_clusters,
        proximity_graph=args.proximity_graph,
        knn_neighbors=args.knn_neighbors,
        show_progress=not args.no_progress,
        logger=logger
    )
    
    try:
        # Capture command line and start time for reporting
        import sys
        import datetime
        command_line = ' '.join(sys.argv)
        start_time = datetime.datetime.now().isoformat()

        # Run decomposition clustering
        results = decomposer.decompose(
            input_fasta=args.input_fasta,
            targets_fasta=args.targets,
            max_clusters=args.max_clusters,
            max_sequences=args.max_sequences
        )

        # Add metadata to results for reporting
        results.command_line = command_line
        results.start_time = start_time
        
        # Save results
        save_decompose_results(results, args.output, args.input_fasta)
        
        logger.info("Decomposition clustering completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Decomposition clustering failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()