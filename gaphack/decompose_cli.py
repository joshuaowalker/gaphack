"""CLI for gaphack-decompose command."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .decompose import DecomposeClustering, DecomposeResults
from .utils import save_clusters_to_file


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
    if input_fasta:
        from .utils import load_sequences_from_fasta
        sequences, all_headers = load_sequences_from_fasta(input_fasta)
    
    # Create a mapping of headers to indices for the loaded sequences
    if all_headers:
        header_to_idx = {header: i for i, header in enumerate(all_headers)}
        
        # Convert clusters to index-based format
        cluster_list = []
        for cluster_id, cluster_headers in results.clusters.items():
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
        
        # Save FASTA files using existing utility
        from .utils import save_clusters_to_file
        save_clusters_to_file(
            clusters=cluster_list,
            singletons=unassigned_indices,
            output_path=output_base,
            headers=all_headers,
            sequences=sequences,
            format='fasta',
            singleton_label='unassigned'
        )
    
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
                       help='FASTA file containing target sequences for supervised mode (required)')
    
    # BLAST parameters
    parser.add_argument('--blast-max-hits', type=int, default=500,
                       help='Maximum BLAST hits per query (default: 500)')
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
    
    if not args.targets:
        logger.error("--targets is required for supervised mode")
        sys.exit(1)
    
    if not Path(args.targets).exists():
        logger.error(f"Targets FASTA file not found: {args.targets}")
        sys.exit(1)
    
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
        show_progress=not args.no_progress,
        logger=logger
    )
    
    try:
        # Run decomposition clustering
        results = decomposer.decompose(
            input_fasta=args.input_fasta,
            targets_fasta=args.targets,
            strategy="supervised"
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