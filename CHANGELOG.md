# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 1: Python Package Foundation - Initial Implementation

##### Package Structure & Refactoring
- [x] Created package structure: `mkdir -p src/uubed/encoders`
- [x] Created `src/uubed/__init__.py` with version and exports
- [x] Created `src/uubed/encoders/__init__.py`
- [x] Updated `src/uubed/__version__.py` with version "0.1.0"

##### Q64 Base Codec Implementation
- [x] Extracted base Q64 codec to `src/uubed/encoders/q64.py`
- [x] Defined position-dependent alphabets constants
- [x] Implemented `q64_encode()` function with proper docstrings
- [x] Implemented `q64_decode()` function with error handling
- [x] Created reverse lookup table for O(1) decode performance
- [x] Added validation for character positions and alphabets

##### Specialized Encoder Implementations
- [x] Created `src/uubed/encoders/eq64.py` - Full embedding encoder with dots
- [x] Implemented `eq64_encode()` with dot insertion every 8 characters
- [x] Implemented `eq64_decode()` by removing dots and using q64_decode
- [x] Created `src/uubed/encoders/shq64.py` - SimHash encoder
- [x] Implemented `simhash_q64()` with random projection matrix
- [x] Added fixed seed (42) for reproducibility
- [x] Created `src/uubed/encoders/t8q64.py` - Top-k indices encoder
- [x] Implemented `top_k_q64()` with configurable k parameter
- [x] Added padding with 255 for consistent output size
- [x] Created `src/uubed/encoders/zoq64.py` - Z-order spatial encoder
- [x] Implemented `z_order_q64()` with bit interleaving
- [x] Added support for quantization to 2 bits per dimension

##### High-Level API
- [x] Created `src/uubed/api.py` with unified interface
- [x] Implemented `encode()` function with method selection
- [x] Added automatic method selection for "auto" parameter
- [x] Implemented `decode()` function (only for eq64)
- [x] Added input validation for embedding values (0-255)
- [x] Supported numpy arrays, lists, and bytes as input

##### Testing Suite
- [x] Created `tests/test_encoders.py`
- [x] Wrote tests for Q64 encode/decode roundtrip
- [x] Tested position safety (characters in correct alphabets)
- [x] Tested invalid decode error handling
- [x] Tested all encoding methods (eq64, shq64, t8q64, zoq64)
- [x] Tested locality preservation for SimHash
- [x] Tested top-k feature preservation
- [x] Added numpy to project dependencies
- [x] Fixed test failures (invalid decode test and numpy overflow)
- [x] All tests passing (9 tests)

##### Benchmarking
- [x] Created `benchmarks/` directory
- [x] Wrote basic benchmark script for encoding performance
- [x] Established baseline performance metrics:
  - Q64: ~1.0-1.4 MB/s throughput
  - Eq64: ~0.6-0.8 MB/s (slower due to dot insertion)
  - Shq64: ~0.1-0.4 MB/s (slower due to matrix operations)
  - T8q64: ~1.3-5.5 MB/s (fastest for large embeddings)
  - Zoq64: ~1.5-7.0 MB/s (fastest overall)

### Fixed Issues

- **NumPy Compatibility**: Resolved by adding numpy>=1.20 to project dependencies. Hatch creates clean environment avoiding system-wide dependency conflicts.
- **Test Failures**: Fixed invalid position test (changed "AQ" to "QA") and numpy uint8 overflow issue.
- **Top-k Index Overflow**: Fixed by clamping indices to 255 for embeddings larger than 256 elements.