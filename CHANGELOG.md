# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] - 2026-11-19

**Results can change:** the target-mode fix below can make the cluster that `gaphack --target`
and `gaphack-blast` report larger (never smaller), so that it includes every sequence merged
within `min_split`.

### Fixed
- **Target mode could leave out sequences within `min_split`**: `gaphack --target` (and
  `gaphack-blast`) report the cluster size with the widest gap, but the choice also considered
  sizes before a mandatory merge (distance ≤ `min_split`), so a sequence a fraction of a
  percent from the cluster could be left out. The result is now the widest-gap size among
  those that include every merge within `min_split`, as full gapHACk does. Merge history is
  unchanged. On a 200-sequence Russula test set the query's cluster grows from 18 to 40
  members with `gaphack --target` (10 to 34 with `gaphack-blast`), all of the query's species
  (47 in the set); on seven other test sets results are unchanged or gain one member.

- **`gaphack.__version__` was wrong in 0.7.1**: it still read 0.7.0 (the CLIs' `--version` was
  right). It now comes from `gaphack/version.py`, and a test checks it against `pyproject.toml`.

### Added
- **`gaphack-blast --nearest-cluster`** (`BlastAnalyzer(find_nearest_cluster=True)`): grows the
  cluster nearest to the query's from the closest outside sequence (target mode, same
  thresholds and alignment) and reports its seed, size, medoid and within-cluster identities,
  the identity distributions within the query cluster and between the two, and the barcode
  gap between them at P90, P95 and P100. Output appears only when requested.
- **`gaphack-blast --target-percentile`**: the linkage/gap percentile (default 100, complete
  linkage; `gaphack --target` uses 95). Reported as `diagnostics.target_percentile`.
- `SubsetDistanceProvider`: a view of a distance provider restricted to some sequences.

## [0.7.1] - 2026-11-19

### Added
- **Published on PyPI** (`pip install gaphack`): a GitHub Actions workflow
  (`.github/workflows/publish.yml`) runs the tests on Python 3.8, 3.10 and 3.12, checks that the
  release tag matches the package version, and uploads to PyPI with trusted publishing when a
  GitHub release is published.
- README lists the external programs gapHACk runs (SPOA, BLAST+, vsearch), which pip does not
  install.

### Fixed
- **Refinement ran to the iteration limit after converging**: the loop tests `AMI == 1.0`, but
  scikit-learn can return `1.0000000000000002` for identical clusterings, depending on label
  order (which varies with `PYTHONHASHSEED`). `gaphack-refine` then repeated the final
  iteration until `--max-iterations` and reported "Reached iteration limit without convergence".
  Identical clusterings are now reported as exactly 1.0. Final clusters are unchanged.

- **adjusted-identity declared as a git URL**: the dependency pointed at the repository's
  default branch, so installs were unpinned and package indexes and `pip install --no-deps`
  setups could not resolve it. It is now the PyPI requirement `adjusted-identity>=0.2.10`
  (for the scoring API below). Scores are unchanged from 0.2.6: 0.2.6 and 0.2.9 agree on all
  194,276 aligned pairs from seven datasets, and 0.2.10 adds no scoring change.

- **scikit-learn missing from the package dependencies**: `gaphack-refine` imports
  `sklearn.metrics`, but `pip install` did not install scikit-learn, so on a clean environment
  `gaphack-refine` failed at start-up with `ModuleNotFoundError: No module named 'sklearn'`.

### Performance
Pairwise scoring of aligned sequences was ~90% of `gaphack-refine` and `gaphack-analyze` runtime.
None of these changes alters a distance or a clustering result.
- **Score without display strings**: distances only need the adjusted mismatch count, so they
  use adjusted-identity's `score_alignment_metrics()` (new in 0.2.10), which returns the same
  metrics as `score_alignment()` without building the annotated alignment strings.
- **Vectorized column filter**: `filter_msa_positions()` uses a numpy mask instead of a
  per-character loop.
- **Pair score cache**: results are remembered per aligned pair, so alignments that are rebuilt
  with the same sequences (every refinement iteration, and again in analysis) are not rescored.
  `GAPHACK_PAIR_CACHE_SIZE` sets the maximum number of pairs kept (default 200000, 0 disables).
- **Medoids reused across refinement iterations**: only clusters whose membership changed are
  realigned when the proximity graph is rebuilt.

Measured on `gaphack --target` → `gaphack-refine --search-method vsearch` → `gaphack-analyze`
(median of 2 runs, 8-CPU Mac, adjusted-identity 0.2.10), with byte-identical output files:
91 sequences 17.4 s → 9.6 s, 150 sequences 39.2 s → 23.6 s, 200 sequences 62.3 s → 31.3 s,
300 sequences 171.0 s → 75.8 s (1.7–2.3×). Part of the gain on the 91- and 300-sequence sets is
the convergence fix: 0.7.0 ran to the iteration limit in 3 of those 4 runs.

### Tests
- `tests/test_pair_scoring.py`: the fast scoring path and the cache are each compared against
  the `score_alignment()` path on every pair of a real 50-sequence alignment
- Convergence tests for `compute_ami_for_refinement` and an end-to-end refinement run
- The `dev` extra installs `hypothesis`, which `tests/test_pair_scoring.py` imports

## [0.7.0] - 2026-01-27

### Added
- **New CLI tool**: `gaphack-blast` for analyzing BLAST search results to identify conspecific sequences
  - Gap-optimized clustering to find natural species boundaries in BLAST hits
  - JSON, TSV, and text output formats
  - Pre-binned identity histograms for barcode gap visualization (intra-cluster and inter-cluster)
  - Medoid identification for representative sequence selection
  - Comprehensive per-sequence classification with identity metrics
- **Version introspection**: All CLI tools now support `--version` flag
  - `gaphack --version`
  - `gaphack-analyze --version`
  - `gaphack-refine --version`
  - `gaphack-blast --version`
- **Centralized version management**: New `gaphack/version.py` module for consistent version across all tools
- **Histogram output**: Pre-binned identity distributions in `gaphack-blast` diagnostics
  - 0.5% bin width with empty bins omitted
  - Both raw counts and normalized frequencies for overlay visualization
  - Supports rendering barcode gap charts in integrating applications

### Documentation
- New integration guide: `docs/blast_analysis_integration.md` with:
  - Complete field reference for JSON output
  - Python integration examples
  - Histogram visualization code samples
  - Algorithm parameter tuning guidance

### Tests
- Comprehensive test suite for `gaphack-blast` (26 tests)
  - Unit tests with mocked MSA for fast execution
  - Integration tests with real biological sequences
  - Histogram functionality tests

### Changed
- **Default output location**: `gaphack` now writes output files to the current working directory instead of the input file's directory
  - Before: `gaphack /data/seqs.fasta` → `/data/seqs.cluster_001.fasta`
  - After: `gaphack /data/seqs.fasta` → `./seqs.cluster_001.fasta`
  - Use `-o /path/to/output` to specify a different location
- Output path now logged at INFO level (previously DEBUG) so users can see where files are written

### Fixed
- `--target-percentile` now correctly affects linkage distance calculation in target mode (previously hardcoded to P95 regardless of setting)

## [0.6.0] - 2025-10-17

### Changed
- **Local gap method is now the default**: Changed from global gap to local gap for more robust clustering with high-variance datasets
- **Alpha parameter defaults to 0.0**: Maximize total gap with no parsimony penalty (favors splitting over lumping)
- **Global gap retained for literature comparison**: Available via `--gap-method global` for validation against published studies

### Added
- **Parameter validation**: Strict validation prevents incompatible parameter combinations
  - `--alpha` can only be used with `--gap-method local`
  - `--target-percentile` can only be used with `--gap-method global`
- **Comprehensive gap methods documentation**: New "Gap Calculation Methods" section in README explaining:
  - Local gap method with multi-percentile calculation [90, 95, 100]
  - Theoretical grounding in silhouette coefficients (Rousseeuw, 1987)
  - Global gap method and its limitations (Wilson et al. 2023)
  - When to use each method with comparison table
  - Alpha parameter tuning examples
- **Silhouette coefficient comparison**: Documented conceptual foundations linking local gap to established clustering literature

### Documentation
- Updated all usage examples to reflect new defaults
- Added Rousseeuw (1987) to Key References
- Enhanced Features section to highlight local gap optimization
- Provided detailed guidance on gap method selection

### Rationale
Local gap is more robust for real-world datasets with high variance, as demonstrated by Wilson et al. (2023) showing global gaps fail in >50% of macrofungal genera. The connection to silhouette coefficients establishes theoretical foundations for the local nearest-neighbor evaluation approach.

## [0.5.1] - 2025-10-13

### Fixed
- Bug fixes merged from main branch during feature development

## [0.5.0] - 2025-10-13

### Removed
- **gaphack-decompose CLI tool**: Simplified to two-tool architecture (gaphack, gaphack-refine, gaphack-analyze)
- **Pass 1 from gaphack-refine**: Streamlined to single-pass iterative refinement architecture
- **Obsolete alignment-method parameter**: Removed non-functional parameter from CLI

### Added
- **Iteration checkpointing**: Save and auto-resume refinement state for long-running jobs
- **MSA coverage-based alignment trimming**: Improved distance quality by filtering low-coverage alignment regions
- **Deterministic seed prioritization**: Reproducible refinement with complete neighborhood consistency
- **Medoid and gap metric caching**: Significant performance improvements in iterative refinement
- **Enhanced iteration logging**: Detailed sequence and cluster statistics for progress monitoring
- **Density-adaptive K-NN graph**: Improved proximity graph construction for refinement

### Changed
- **Refinement architecture**: Simplified from two-pass to single iterative approach with neighborhood-based refinement
- **Convergence detection**: Enhanced AMI-based approach with improved convergence tracking
- **Distance calculation**: Enhanced MSA-based distances with overlap filtering and median normalization
- **Refinement types**: Extracted shared types to dedicated `refinement_types.py` module

### Fixed
- **Refine CLI bugs**: Removed unused code and fixed checkpoint-related bug
- **Convergence detection**: Fixed bug in AMI-based convergence logic

### Improved
- **Code organization**: Better separation of concerns with dedicated refinement types module
- **Documentation**: Updated CLAUDE.md to reflect simplified architecture
- **API clarity**: Cleaner refinement API with reduced complexity

## [0.4.0] - 2025-01-29

### Added
- **Comprehensive testing infrastructure**: 317 passing tests with 18,958 lines of test code
- **Real biological validation**: 1,429 Russula fungal ITS sequences with ground truth annotations
- **Test subsets for development**: Multiple Russula subsets (50, 100, 200, 300, 500 sequences)
- **Performance baselines**: Established thresholds for regression detection across dataset sizes
- **Quality metrics validation**: Empirically validated ARI (>0.85), homogeneity (>0.90), completeness (>0.85)
- **Custom pytest markers**: integration, performance, memory, scalability, quality test categorization
- **Phase 4 testing documentation**: Comprehensive `tests/PHASE4_SUMMARY.md` with testing strategy
- **Contributor documentation**: Enhanced `CLAUDE.md` with architectural details and implementation notes

### Changed
- **Gap-based greedy algorithm**: Removed beam search in favor of simpler gap-based heuristic with completion tracking
- **Neighborhood pruning**: Replaced arbitrary thresholds with principled N+N approach (N within max_lump + N beyond for context)
- **Triangle inequality filtering**: Unified implementation across all clustering modes with 5% tolerance
- **Cluster naming**: Eliminated redundant intermediate renumbering for clean 1:1 lineage traceability
- **Overlap handling**: Simplified by removing `--no-overlaps` mode and `jaccard_overlap_threshold` parameter
- **Close cluster refinement**: Removed significant difference check, added iterative context expansion
- **Conflict resolution**: Separated from close cluster refinement for clearer responsibilities
- **BLAST parameters**: Optimized for biological accuracy (removed aggressive pruning)
- **Post-processing verification**: Added comprehensive MECE (Mutually Exclusive, Collectively Exhaustive) verification

### Fixed
- **CLI bugs**: Fixed multiple command-line interface issues discovered during testing
- **Decompose report bugs**:
  - Fixed missing component details for close cluster refinement (incorrect indentation)
  - Fixed incomplete active-to-final cluster mapping (eliminated redundant renumbering)
- **Distance calculation consistency**: Fixed discrepancy between clustering loop and K-NN proximity graph
- **Cluster naming collisions**: Fixed conflicts in processing stage cluster IDs
- **Test failures**: Integrated demo tests into main test suite and fixed failures

### Improved
- **Decompose orchestration**: Enhanced processing stage tracking with `ProcessingStageInfo`
- **Cluster refinement**: Iterative context expansion for proper gap calculation in close clusters
- **Distance providers**: Three-tier architecture (global cache, scoped providers, precomputed)
- **Cluster graph**: BLAST K-NN graph for O(K) proximity queries instead of O(C)
- **Memory efficiency**: Hash-based sequence deduplication system
- **Medoid caching**: Optimized cluster medoid calculations with persistent caching
- **Conflict synchronization**: Improved tracking through iterative cluster merging
- **Code organization**: Major terminology cleanup and consistent naming throughout

### Removed
- **Beam search algorithm**: Replaced with simpler gap-based greedy approach
- **`--no-overlaps` mode**: Simplified overlap handling
- **`jaccard_overlap_threshold` parameter**: No longer needed with new approach
- **Significant difference check**: Removed from cluster refinement
- **Legacy overlap merging**: Replaced with principled reclustering
- **Arbitrary neighborhood thresholds**: Replaced with N+N approach
- **Proximity graph dependency**: Removed from conflict resolution (unnecessary)

### Documentation
- Enhanced `CLAUDE.md` with:
  - Quick reference section with architecture overview
  - Three-tier distance provider architecture details
  - Post-processing and refinement implementation details
  - Cluster graph infrastructure documentation
  - Testing infrastructure and performance baselines
  - Future work and experimental features
- Added comprehensive design documents for gapHACk improvements
- Documented N+N neighborhood pruning strategy
- Documented iterative context expansion for close cluster refinement
- Added algorithm development history and evolution notes

## [0.3.0] - 2025-01-06

### Added
- **New CLI tool**: `gaphack-analyze` for analyzing pre-clustered FASTA files
- **Target mode clustering**: `--target` parameter for single-cluster focused clustering from seed sequences
- **Distance analysis functions**: Calculate intra-cluster and inter-cluster distance distributions
- **Visualization capabilities**: Generate histograms of distance distributions with percentile markers
- **Multiple output formats**: Text reports, JSON data, and TSV tables
- **Barcode gap assessment**: Evaluate gap quality and existence for pre-clustered data
- **Comprehensive analysis**: Individual cluster analysis plus global cross-cluster comparisons
- **Matplotlib integration**: Professional-quality histogram generation with statistics
- **Full parameter support**: All alignment method options available for analysis

### Changed
- **FASTA output format**: Now uses two-line format (header + sequence on single line) instead of wrapped sequences
- **Target mode labeling**: Sequences not processed in target mode are labeled "unclustered" instead of "singleton"

### Dependencies
- Added `matplotlib>=3.5.0` for histogram generation and visualization

## [0.2.0] - 2025-01-06

### Added
- Multiprocessing support for gap-aware clustering phase with automatic CPU core detection
- `--threads` / `-t` CLI option to control parallelization (default: auto-detect, 0: single-process)
- `num_threads` parameter in `GapOptimizedClustering` constructor for API users
- Single-process mode (`-t 0`) for library integration and debugging
- Persistent worker caches to reduce initialization overhead in multiprocessing
- Offset-based workload distribution for optimal load balancing across worker processes

### Improved
- **Performance**: ~4x speedup on 8-core systems (200→800 steps/second) 
- **Progress tracking**: More frequent progress bar updates in single-process mode (every ~1 second)
- **Memory efficiency**: Eliminated large data structure serialization through mathematical coordinate mapping
- **Code organization**: Clean separation between single-process and multi-process implementations

### Fixed
- Cache refresh bug in single-process mode that caused early termination
- Progress bar update frequency optimized for better user feedback

## [0.1.1] - 2025-01-04

### Added
- Gap-optimized hierarchical agglomerative clustering algorithm
- Two-phase clustering strategy (fast greedy + gap-aware optimization)
- Support for adjusted identity alignment method with configurable parameters
- CLI interface with comprehensive parameter controls
- Python API for library integration
- Multiple output formats (FASTA, TSV, text)
- Percentile-based gap optimization for robustness against outliers
- Comprehensive test suite and example datasets

### Features
- Automatic barcode gap discovery and optimization
- Configurable min-split and max-lump thresholds
- Progress tracking with real-time metrics
- Size-ordered cluster output
- Metrics export for analysis and debugging