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
    if input_fasta:
        from .utils import load_sequences_from_fasta
        sequences, all_headers, header_mapping = load_sequences_from_fasta(input_fasta)
    
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
    
    # Save summary report
    report_file = f"{output_base}.decompose_report.txt"
    with open(report_file, 'w') as f:
        f.write("Gaphack-Decompose Clustering Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total iterations: {results.total_iterations}\n")
        f.write(f"Total sequences processed: {results.total_sequences_processed}\n")
        f.write(f"Coverage percentage: {results.coverage_percentage:.1f}%\n")
        f.write(f"Clusters created: {len(results.clusters)}\n")
        f.write(f"Unassigned sequences: {len(results.unassigned)}\n")
        f.write(f"Conflicts detected: {len(results.conflicts)}\n\n")
        
        f.write("Cluster Summary:\n")
        f.write("-" * 20 + "\n")
        for cluster_id, cluster_headers in results.clusters.items():
            f.write(f"{cluster_id}: {len(cluster_headers)} sequences\n")
        
        if results.iteration_summaries:
            f.write("\nIteration Summary:\n")
            f.write("-" * 20 + "\n")
            for summary in results.iteration_summaries:
                f.write(f"Iteration {summary['iteration']}: "
                       f"targets={summary['target_headers']}, "
                       f"neighborhood_size={summary['neighborhood_size']}, "
                       f"pruned_size={summary.get('pruned_size', 'N/A')}, "
                       f"cluster_size={summary['cluster_size']}, "
                       f"gap_size={summary['gap_size']:.4f}\n")
        
        if results.conflicts:
            f.write("\nConflicts:\n")
            f.write("-" * 20 + "\n")
            for seq_id, cluster_ids in results.conflicts.items():
                f.write(f"{seq_id}: assigned to {', '.join(cluster_ids)}\n")
    
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
  # Supervised mode with target sequences
  gaphack-decompose input.fasta --targets targets.fasta -o results

  # Unsupervised mode with cluster count limit
  gaphack-decompose input.fasta --strategy unsupervised --max-clusters 50 -o results

  # Unsupervised mode with sequence coverage limit
  gaphack-decompose input.fasta --strategy unsupervised --max-sequences 1000 -o results

  # Unsupervised mode until input exhausted (cluster all sequences)
  gaphack-decompose input.fasta --strategy unsupervised -o results

  # Disable overlaps (each sequence in at most one cluster)
  gaphack-decompose input.fasta --strategy unsupervised --no-overlaps -o results

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
                       help='FASTA file containing target sequences for supervised mode')
    parser.add_argument('--strategy', choices=['supervised', 'unsupervised'], default='supervised',
                       help='Target selection strategy (default: supervised)')
    parser.add_argument('--max-clusters', type=int,
                       help='Maximum clusters to create (unsupervised mode, optional)')
    parser.add_argument('--max-sequences', type=int,
                       help='Maximum sequences to assign (unsupervised mode, optional)')
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
    
    # Cluster merging arguments
    parser.add_argument('--merge-overlaps', action='store_true',
                       help='Enable post-processing to merge overlapping clusters')
    parser.add_argument('--containment-threshold', type=float, default=0.8,
                       help='Containment coefficient threshold for merging clusters (default: 0.8)')

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
    
    # Strategy-specific validation
    if args.strategy == "supervised":
        if not args.targets:
            logger.error("--targets is required for supervised mode")
            sys.exit(1)
        if not Path(args.targets).exists():
            logger.error(f"Targets FASTA file not found: {args.targets}")
            sys.exit(1)
    elif args.strategy == "unsupervised":
        if args.targets:
            logger.warning("--targets specified but ignored in unsupervised mode")
        if not args.max_clusters and not args.max_sequences:
            logger.info("No stopping criteria specified - will cluster until input is exhausted")
    
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
        merge_overlaps=args.merge_overlaps,
        containment_threshold=args.containment_threshold,
        show_progress=not args.no_progress,
        logger=logger
    )
    
    try:
        # Run decomposition clustering
        results = decomposer.decompose(
            input_fasta=args.input_fasta,
            targets_fasta=args.targets,
            strategy=args.strategy,
            max_clusters=args.max_clusters,
            max_sequences=args.max_sequences
        )
        
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