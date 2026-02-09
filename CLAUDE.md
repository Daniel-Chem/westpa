# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WESTPA (Weighted Ensemble Simulation Toolkit with Parallelization and Analysis) is a Python package for constructing and running stochastic simulations using the weighted ensemble method. It is used for long-timescale molecular dynamics simulations. The `python-api` branch (current) is developing a new, cleaner public Python API on top of the existing `westpa2` codebase.

## Build and Development

```bash
# Install in development mode (includes test and pre-commit dependencies)
pip install -e .[dev]

# Install for testing only
pip install -e .[tests]

# Build requires Cython - there are 6 Cython extension modules:
#   westpa.fasthist._fasthist, westpa.trajtree._trajtree, westpa.mclib._mclib,
#   westpa.core.binning._assign, westpa.core.kinetics._kinetics, westpa.core.reweight._reweight
```

## Testing

```bash
# Run full test suite
pytest -v --cov=westpa --cov-report=xml tests

# Run a single test file
pytest tests/test_binning.py

# Run a single test
pytest tests/test_binning.py::TestRectilinearBinMapper::test_simple
```

Tests use fixtures defined in `tests/conftest.py` that set up temporary directories with reference HDF5 files and config from `tests/refs/`. Most test fixtures set `WEST_SIM_ROOT` env var and `chdir` to a temp directory. Tests support both numpy 1.x and 2.x.

## Linting and Formatting

```bash
# Run all pre-commit hooks (black + flake8 + trailing whitespace)
pre-commit run --all-files
```

- **Black**: line length 132, skip string normalization
- **Flake8**: line length 132, ignores E203, E266, E501, W503, W504, E402, E731
- Both exclude `versioneer.py`, `_version.py`, and `doc/`

## Architecture

### Source Layout

All source code lives under `src/westpa/`. The package uses `setuptools` with `find_packages(where="src")`.

### New Python API (`python-api` branch)

The new public API is centered on these core classes, all importable from `westpa`:

- **`Simulation`** (`core/simulation.py`) - Main entry point. Configured with a propagator, pcoord_calculator, bin_mapper, resampler, and optional source/sink. Call `initialize()` then `run()`.
- **`State`** (`core/state.py`) - Microstate definition (coord, file, or id).
- **`Segment`** (`core/segment.py`) - Trajectory segment with weight, pcoord, status, and auxiliary data.
- **`Source`** / **`Sink`** (`core/source_sink.py`) - Source distribution and sink indicator for walker recycling.
- **`Propagator`** / **`BatchedPropagator`** (`core/propagators/`) - Interface for dynamics engines.
- **BinMapper** subclasses (`core/binning/`) - `RectilinearBinMapper`, `VoronoiBinMapper`, `FuncBinMapper`, `RecursiveBinMapper`, `PiecewiseBinMapper`, `MABBinMapper`.
- **Resampler** subclasses (`core/resamplers/`) - `HuberKimResampler` (default), `MultinomialResampler`, `ResidualResampler`.

### Simulation Loop

`Simulation.run()` iterates: `_prepare_iteration()` -> `_propagate()` -> `_run_we()` -> `_prepare_next_iteration()`. Propagation is dispatched through a `WorkManager`. Resampling applies per-bin with configurable target counts.

### Legacy/Existing Systems

- **`WESTSystem`** (`core/systems.py`) and **`BasisState`/`TargetState`** (`core/states.py`) - Legacy system configuration interface (YAML-driven via `westpa.rc`).
- **`we_driver.py`** / **`sim_manager.py`** / **`data_manager.py`** - Original simulation engine, still used by CLI tools.
- **`cli/core/`** - Core CLI commands (`w_init`, `w_run`, `w_fork`, `w_states`, `w_succ`, `w_truncate`).
- **`cli/tools/`** - 20+ analysis CLI tools (`w_assign`, `w_kinetics`, `w_pdist`, `w_reweight`, etc.).
- **`work_managers/`** - Parallel execution backends: `SerialWorkManager`, `ThreadsWorkManager`, `ProcessWorkManager`, MPI, ZeroMQ.
- **`westext/`** - Extensions: adaptive Voronoi binning, string method, WESS, WEED, HAMSM restarting.

### Data Storage

All simulation data is stored in HDF5 files (`west.h5`), managed by `core/data_manager.py` and `core/h5io.py`. The new API uses `core/_data_manager.py` (`DataManager`).

## CI

GitHub Actions runs on push/PR to `westpa2`/`develop`:
- Lint job: `pre-commit run --all-files`
- Test matrix: Python 3.11-3.14, numpy 1 and 2, across Ubuntu (x86+ARM) and macOS (Intel+ARM)
- 45-minute timeout per test run