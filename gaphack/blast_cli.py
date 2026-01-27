"""
Command-line interface for gaphack-blast.

Analyzes BLAST results to identify conspecific sequences (same species as query).
"""

import argparse
import json
import logging
import sys
from typing import List, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging based on verbosity level."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stderr  # Log to stderr so stdout is clean for output
    )


def load_sequences_from_stdin() -> Tuple[List[str], List[str]]:
    """Load sequences from stdin in FASTA format."""
    sequences = []
    headers = []

    try:
        for record in SeqIO.parse(sys.stdin, "fasta"):
            sequences.append(str(record.seq).upper())
            headers.append(record.id)
    except Exception as e:
        logging.error(f"Error reading FASTA from stdin: {e}")
        raise

    return sequences, headers


def load_sequences_from_file(path: str) -> Tuple[List[str], List[str]]:
    """Load sequences from a FASTA file."""
    sequences = []
    headers = []

    try:
        for record in SeqIO.parse(path, "fasta"):
            sequences.append(str(record.seq).upper())
            headers.append(record.id)
    except Exception as e:
        logging.error(f"Error reading FASTA file: {e}")
        raise

    return sequences, headers


def main():
    """Main entry point for gaphack-blast CLI."""
    parser = argparse.ArgumentParser(
        description='Analyze BLAST results to identify conspecific sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - read from stdin, first sequence is query
  cat blast_results.fasta | gaphack-blast > results.json

  # From file (first sequence is query)
  gaphack-blast blast_results.fasta -o results.json

  # Human-readable output
  gaphack-blast blast_results.fasta --format text

  # Tab-separated output for spreadsheets
  gaphack-blast blast_results.fasta --format tsv

  # Adjust clustering thresholds
  gaphack-blast blast_results.fasta --min-split 0.003 --max-lump 0.03

Output fields:
  - in_query_cluster: Whether the sequence clusters with the query
  - distance_to_query: MycoBLAST-adjusted distance (0.01 = 99% effective identity)
  - gap_size: Barcode gap magnitude (inter_min - intra_max)
        """
    )

    # Input
    parser.add_argument(
        'input',
        nargs='?',
        help='Input FASTA file (default: stdin). First sequence is the query.'
    )

    # Output
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'text', 'tsv'],
        default='json',
        help='Output format (default: json)'
    )

    # Algorithm parameters
    parser.add_argument(
        '--min-split',
        type=float,
        default=0.005,
        help='Minimum distance to split clusters (default: 0.005 = 99.5%% identity)'
    )
    parser.add_argument(
        '--max-lump',
        type=float,
        default=0.02,
        help='Maximum distance to lump clusters (default: 0.02 = 98%% identity)'
    )

    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (to stderr)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    try:
        # Load sequences
        if args.input:
            logging.info(f"Loading sequences from {args.input}")
            sequences, headers = load_sequences_from_file(args.input)
        else:
            logging.info("Reading sequences from stdin")
            sequences, headers = load_sequences_from_stdin()

        if not sequences:
            logging.error("No sequences found in input")
            sys.exit(1)

        logging.info(f"Loaded {len(sequences)} sequences")
        logging.info(f"Query: {headers[0]}")

        # Import analysis module (deferred to speed up --help)
        from .blast_analysis import BlastAnalyzer, format_text_output, format_tsv_output

        # Run analysis
        analyzer = BlastAnalyzer(
            min_split=args.min_split,
            max_lump=args.max_lump,
            show_progress=not args.quiet
        )

        result = analyzer.analyze(sequences, headers)

        # Format output
        if args.format == 'json':
            output = json.dumps(result.to_dict(), indent=2)
        elif args.format == 'text':
            output = format_text_output(result)
        elif args.format == 'tsv':
            output = format_tsv_output(result)

        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
                if not output.endswith('\n'):
                    f.write('\n')
            logging.info(f"Results written to {args.output}")
        else:
            print(output)

        # Report summary to stderr
        if not args.quiet:
            logging.info(f"Analysis complete: {result.query_cluster_size}/{result.total_sequences} sequences in query cluster")
            if result.barcode_gap_found:
                logging.info(f"Barcode gap found: {result.gap_size_percent:.2f}%")
            else:
                logging.info("No barcode gap found")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
