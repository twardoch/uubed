# PROJECT: Specification for `uubed` High-Performance Encoding Library

## 1. 🎯 Project Mandate & Methodology

**Objective:** This document outlines the development roadmap for a production-grade, high-performance library for locality-preserving semantic embedding encoding. The conceptual foundation for this work is based on the research and development contained in `llms.txt` and the prototype implementations in `work/voyemb.py`.

**Current Status:** Initial prototyping complete with QuadB64 family of encodings demonstrating locality preservation.

**Virtual Team:**

- **Project Architect:** Leads the specification and implementation process.
- **Ideot:** Provides creative, unorthodox solutions and challenges conventional thinking.
- **Critin:** Critiques flawed logic, stress-tests ideas, and ensures a balanced, robust final design.

**Working Principles:** The team adheres to the core principles of iterative development. Focus on minimal viable increments, write exceptionally clear documentation explaining the "what" and the "why," and modularize logic into clean, single-purpose functions. All work should be a collaborative, step-by-step process of sharing thoughts and adapting.

**Tools & Research:** Before and during the implementation process, leverage the following tools when available:

- Consult the `context7` tool for the most up-to-date software package documentation.
- Use `deepseek/deepseek-r1-0528` and `openai/o3` via `chat_completion` for additional reasoning and problem-solving assistance.
- Employ `sequentialthinking` to structure complex decision-making processes.
- Gather current information and context using `perplexity_ask` and `duckduckgo_web_search`.

---

## 2. Part A: Core Implementation (Python → Rust/C)

_The prototype in `voyemb.py` demonstrates the concepts. Now we need to build the production-grade native library._

### 2.1. Foundational Architectural Decisions

- [x] **Proof of Concept:** Python implementation of QuadB64 family complete
- [ ] **Language Choice:** Finalize decision between Rust vs. C for core library

  - **Task:** Create benchmarks comparing PyO3 vs CFFI performance with the QuadB64 encodings
  - **Ideot's Input:** Consider Zig for its simplicity and C ABI compatibility, or Rust with careful attention to compile-time optimizations
  - **Critin's Check:** Ensure build complexity doesn't impact adoption. Users should install via `[uv] pip install uubed` without needing Rust toolchain

- [ ] **Library Structure & API:** Design the native library interface
  - [ ] Define C-style API with clear ownership semantics
  - [ ] Create error handling strategy (error codes → Python exceptions)
  - [ ] Design streaming API for large embedding batches
  - [ ] Implement zero-copy interfaces where possible

### 2.2. Implementation Plan for Encoding Schemes

**Status:** Python prototypes complete, need native implementations.

- [ ] **Core Encodings Checklist:**
  - [x] **QuadB64 Codec:** Python implementation complete as `q64_encode/decode`
  - [ ] **QuadB64 Native:** Port to Rust/C with SIMD optimizations
  - [x] **SimHash-q64:** Python implementation complete
  - [ ] **SimHash-q64 Native:** Optimize random projection with BLAS
  - [x] **Top-k-q64:** Python implementation complete
  - [ ] **Top-k-q64 Native:** Use partial sorting algorithms
  - [x] **Z-order-q64:** Python implementation complete
  - [ ] **Z-order-q64 Native:** Optimize Morton encoding with bit manipulation
  - [ ] **Base64 with MSB trick:** Port the 33-byte optimization

### 2.3. Performance & Validation

- [ ] **Benchmarking Suite:**

  - [ ] Throughput tests: embeddings/second for each encoding
  - [ ] Memory usage profiling
  - [ ] Comparison with numpy/pure Python baseline
  - [ ] SIMD vs scalar performance comparison

- [ ] **Testing Strategy:**
  - [x] Python prototype tests (implicit in voyemb.py demonstrations)
  - [ ] Port test cases to pytest framework
  - [ ] Property-based tests with Hypothesis
  - [ ] Cross-language validation (Python ↔ Native identical outputs)
  - [ ] Fuzzing for edge cases

---

## 3. Part B: Python Package & API

_Transform the research code into a production-ready Python package._

### 3.1. Package Architecture

- [x] **Basic Package Structure:** Created with hatch
- [ ] **Module Organization:**
  - [ ] `uubed.encoders` - High-level encoding interface
  - [ ] `uubed.native` - Native library bindings
  - [ ] `uubed.utils` - Helper functions and utilities
  - [ ] `uubed.benchmarks` - Performance testing utilities

### 3.2. FFI & Bindings

- [ ] **Binding Technology Decision:**

  - [ ] Evaluate PyO3/Maturin for Rust (if Rust chosen)
  - [ ] Evaluate CFFI for C (if C chosen)
  - [ ] Create proof-of-concept for both approaches

- [ ] **Pythonic API Design:**

  ```python
  # Target API
  from uubed import encode, decode

  # Automatic encoding selection
  encoded = encode(embedding, method="auto")  # Returns best encoding

  # Specific encodings
  q64_str = encode(embedding, method="q64")
  shq64_str = encode(embedding, method="simhash-q64")

  # Batch operations
  encoded_batch = encode(embeddings_list, method="q64", parallel=True)
  ```

### 3.3. Distribution

- [x] **CI/CD Pipeline:** GitHub Actions configured
- [ ] **Binary Wheels:**
  - [ ] Configure cibuildwheel for multi-platform builds
  - [ ] Test wheel installation without dev dependencies
  - [ ] Implement wheel size optimization

---

## 4. Part C: Documentation & Educational Materials

### 4.1. User Documentation

- [ ] **API Reference:**

  - [ ] Docstrings for all public functions
  - [ ] Type stubs for better IDE support
  - [ ] Interactive examples in documentation

- [ ] **User Guide:**
  - [ ] Quick start guide
  - [ ] Encoding method selection guide
  - [ ] Performance tuning guide
  - [ ] Migration from other embedding storage methods

### 4.2. Technical Book ("The QuadB64 Codex")

- [ ] **Chapter Outline:**

  1. [ ] **Introduction:** The substring pollution problem
  2. [ ] **QuadB64 Fundamentals:** Position-safe encoding theory
  3. [ ] **The QuadB64 Family:**
     - [ ] Eq64 (full embeddings with dots)
     - [ ] Shq64 (SimHash variant)
     - [ ] T8q64 (Top-k indices)
     - [ ] Zoq64 (Z-order/Morton)
  4. [ ] **Locality Preservation:** Mathematical foundations
  5. [ ] **Implementation Details:** From Python to native code
  6. [ ] **Benchmarks & Comparisons:** Performance analysis
  7. [ ] **Real-World Applications:** Search engines, vector DBs
  8. [ ] **Future Directions:** Matryoshka embeddings integration

- [ ] **Publishing Setup:**
  - [ ] Configure MkDocs or mdBook
  - [ ] Create build pipeline
  - [ ] Deploy to GitHub Pages

---

## 5. Part D: Project Management & Tracking

### 5.1. Documentation Artifacts

- [x] **README.md:** Basic structure exists
- [ ] **README.md Updates:**

  - [ ] Add QuadB64 explanation
  - [ ] Include performance benchmarks
  - [ ] Add usage examples

- [ ] **PROGRESS.md:** Create detailed task tracking
- [ ] **CHANGELOG.md:** Start tracking changes
- [x] **File Headers:** `this_file` pattern implemented in some files

### 5.2. Research Integration

- [x] **Prototype Validation:** QuadB64 family proven in voyemb.py
- [ ] **Research Documentation:**
  - [ ] Extract key insights from chat1.md and chat2.md
  - [ ] Document the evolution from Base64 → QuadB64
  - [ ] Create visual diagrams of encoding schemes

---

## 6. Implementation Phases

### 6.1. Phase 1: Python Package Foundation (Current)

- [x] Basic package structure
- [x] Prototype implementations
- [ ] Refactor voyemb.py into package modules
- [ ] Add comprehensive test suite
- [ ] Create initial documentation

### 6.2. Phase 2: Native Core Development

- [ ] Language selection and toolchain setup
- [ ] Port QuadB64 to native code
- [ ] Implement SIMD optimizations
- [ ] Create Python bindings
- [ ] Benchmark against prototype

### 6.3. Phase 3: Advanced Features

- [ ] Streaming API for large datasets
- [ ] GPU acceleration exploration
- [ ] Integration with vector databases
- [ ] Matryoshka embedding support
- [ ] Binary quantization options

### 6.4. Phase 4: Ecosystem & Adoption

- [ ] Create plugins for popular frameworks
- [ ] Develop conversion tools
- [ ] Build community examples
- [ ] Conference talks and papers

---

## 7. Technical Decisions Pending

1. **Native Language Choice:** Rust (with PyO3) vs C (with CFFI) vs Zig
2. **SIMD Strategy:** Auto-vectorization vs explicit intrinsics
3. **API Design:** Functional vs object-oriented interface
4. **Error Handling:** Exceptions vs error codes vs Result types
5. **Parallelism:** Thread pool vs async/await vs multiprocessing

---

## 8. Next Immediate Steps

1. [ ] Extract encoders from voyemb.py into uubed package
2. [ ] Create unit tests for each encoding method
3. [ ] Set up benchmarking framework
4. [ ] Write initial user documentation
5. [ ] Create simple CLI tool for testing

---

## 9. Success Metrics

- **Performance:** 10x faster than pure Python implementation
- **Accuracy:** Bit-perfect compatibility with prototype
- **Usability:** Install with uv/pip, no external dependencies
- **Adoption:** Integration with at least 2 vector databases
- **Documentation:** Complete API docs and user guide

---

## 10. "Wait, but" Reflection Points

After each implementation phase, the team should ask:

1. Is the API as simple as it could be?
2. Are we over-engineering any component?
3. What would a new user find confusing?
4. Can we reduce the cognitive load further?
5. Are all features truly necessary for v1.0?

# PLAN: uubed Implementation Plan From Prototype to Production

1. Executive Summary - Clear explanation of the problem (substring pollution) and the QuadB64 solution
2. Four Implementation Phases:

- Phase 1: Python package foundation (refactoring voyemb.py)
- Phase 2: Rust core implementation with SIMD optimizations
- Phase 3: Integration and packaging with PyO3 bindings
- Phase 4: Publishing and distribution

3. Detailed Code Examples for:

- All encoder implementations (Q64, Eq64, Shq64, T8q64, Zoq64)
- Rust implementations with performance optimizations
- PyO3 bindings and Python integration
- Testing and benchmarking

4. Junior Developer-Friendly Sections:

- Visual explanations of SIMD ("one chef cutting 16 carrots with one chop")
- Real-world performance comparisons
- Debugging tips with concrete examples
- Common gotchas and best practices
- Learning resources and journey roadmap

5. Technical Depth:

- SIMD optimization techniques
- Parallel processing with Rayon
- Memory-efficient implementations
- Cross-platform build configuration

The plan iteratively refined the content three times:

1. First draft with basic structure and code
2. Added specific technical details (const fn for compile-time optimization, matrix caching, etc.)
3. Enhanced with junior-friendly explanations, visual analogies, and practical debugging tips

This plan should enable a junior developer to successfully implement the uubed library, understanding not just the "how" but also the "why" behind each design decision.

## 11. 🎯 Executive Summary

This plan guides you through building `uubed`, a high-performance library for encoding embedding vectors into position-safe, locality-preserving strings. We'll transform the Python prototype into a production-ready package with native Rust performance.

**Key Innovation:** QuadB64 encoding uses position-dependent alphabets to eliminate substring pollution in search systems while preserving embedding similarity relationships.

---

## 12. 📚 Understanding the Problem & Solution

### 12.1. The Problem: Substring Pollution

When storing embeddings as Base64 strings in search engines:

- Regular Base64: "abc" can match _anywhere_ in the string
- False positives: Unrelated embeddings match due to random substring collisions
- Search quality degradation: Irrelevant results pollute search output

### 12.2. The Solution: QuadB64 Family

Position-safe encoding where characters at different positions use different alphabets:

```
Position 0,4,8...: ABCDEFGHIJKLMNOP
Position 1,5,9...: QRSTUVWXYZabcdef
Position 2,6,10..: ghijklmnopqrstuv
Position 3,7,11..: wxyz0123456789-_
```

This means "abc" can only match at specific positions, eliminating false positives!

### 12.3. Encoding Variants

1. **Eq64**: Full embeddings with dots for readability
2. **Shq64**: SimHash for similarity-preserving compact codes
3. **T8q64**: Top-k feature indices for sparse representation
4. **Zoq64**: Z-order spatial encoding for prefix search

---

## 13. 🏗️ Architecture Overview

```
uubed/
├── src/
│   ├── uubed/
│   │   ├── __init__.py      # Python API
│   │   ├── encoders/        # Pure Python implementations
│   │   │   ├── __init__.py
│   │   │   ├── q64.py       # QuadB64 base codec
│   │   │   ├── eq64.py      # Full embedding encoder
│   │   │   ├── shq64.py     # SimHash encoder
│   │   │   ├── t8q64.py     # Top-k encoder
│   │   │   └── zoq64.py     # Z-order encoder
│   │   ├── _native.py       # Native library bindings
│   │   └── api.py           # High-level API
│   └── lib.rs               # Rust entry point
├── rust/
│   ├── src/
│   │   ├── lib.rs           # Rust library root
│   │   ├── encoders/
│   │   │   ├── mod.rs
│   │   │   ├── q64.rs       # SIMD-optimized QuadB64
│   │   │   ├── simhash.rs   # Parallel SimHash
│   │   │   ├── topk.rs      # Fast top-k selection
│   │   │   └── zorder.rs    # Bit-interleaving
│   │   └── bindings.rs      # PyO3 Python bindings
├── tests/                   # Test suite
├── benchmarks/              # Performance tests
├── Cargo.toml              # Rust dependencies
└── pyproject.toml          # Python packaging
```

---

## 14. 📋 Phase 1: Python Package Foundation (Week 1)

### 14.1. Goal

Refactor the `voyemb.py` prototype into a proper Python package structure.

### 14.2. Step 1.1: Set Up Package Structure

```bash
# Create the package structure
mkdir -p src/uubed/encoders
touch src/uubed/__init__.py
touch src/uubed/encoders/__init__.py

# Move prototype code
cp work/voyemb.py src/uubed/encoders/q64.py
```

### 14.3. Step 1.2: Extract Base Q64 Codec

Create `src/uubed/encoders/q64.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/encoders/q64.py
"""QuadB64: Position-safe base encoding that prevents substring pollution."""

from typing import Union, List

# Position-dependent alphabets
ALPHABETS = [
    "ABCDEFGHIJKLMNOP",  # pos ≡ 0 (mod 4)
    "QRSTUVWXYZabcdef",  # pos ≡ 1
    "ghijklmnopqrstuv",  # pos ≡ 2
    "wxyz0123456789-_",  # pos ≡ 3
]

# Pre-compute reverse lookup for O(1) decode
REV_LOOKUP = {}
for idx, alphabet in enumerate(ALPHABETS):
    for char_idx, char in enumerate(alphabet):
        REV_LOOKUP[char] = (idx, char_idx)


def q64_encode(data: Union[bytes, List[int]]) -> str:
    """
    Encode bytes into q64 positional alphabet.

    Why this matters: Regular base64 allows "abc" to match anywhere.
    Q64 ensures "abc" can only match at specific positions, eliminating
    false positives in substring searches.

    Args:
        data: Bytes or list of integers to encode

    Returns:
        Position-safe encoded string (2 chars per byte)
    """
    if isinstance(data, list):
        data = bytes(data)

    result = []
    pos = 0

    for byte in data:
        # Split byte into two 4-bit nibbles
        hi_nibble = (byte >> 4) & 0xF
        lo_nibble = byte & 0xF

        # Encode each nibble with position-dependent alphabet
        for nibble in (hi_nibble, lo_nibble):
            alphabet = ALPHABETS[pos & 3]  # pos mod 4
            result.append(alphabet[nibble])
            pos += 1

    return "".join(result)


def q64_decode(encoded: str) -> bytes:
    """
    Decode q64 string back to bytes.

    Args:
        encoded: Q64 encoded string

    Returns:
        Original bytes

    Raises:
        ValueError: If string is malformed
    """
    if len(encoded) & 1:
        raise ValueError("q64 length must be even (2 chars per byte)")

    nibbles = []
    for pos, char in enumerate(encoded):
        try:
            expected_alphabet_idx, nibble_value = REV_LOOKUP[char]
        except KeyError:
            raise ValueError(f"Invalid q64 character {char!r}") from None

        if expected_alphabet_idx != (pos & 3):
            raise ValueError(
                f"Character {char!r} illegal at position {pos}. "
                f"Expected alphabet {expected_alphabet_idx}"
            )
        nibbles.append(nibble_value)

    # Combine nibbles back into bytes
    iterator = iter(nibbles)
    return bytes((hi << 4) | lo for hi, lo in zip(iterator, iterator))
```

### 14.4. Step 1.3: Create Specialized Encoders

Create `src/uubed/encoders/eq64.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/encoders/eq64.py
"""Eq64: Full embedding encoder with visual dots for readability."""

from .q64 import q64_encode, q64_decode
from typing import Union, List


def eq64_encode(data: Union[bytes, List[int]]) -> str:
    """
    Encode full embedding with dots every 8 characters.

    Example: "ABCDEFGh.ijklmnop.qrstuvwx"

    Why dots? Makes it easier to visually compare embeddings
    and spot patterns during debugging.
    """
    base_encoded = q64_encode(data)

    # Insert dots for readability
    result = []
    for i, char in enumerate(base_encoded):
        if i > 0 and i % 8 == 0:
            result.append(".")
        result.append(char)

    return "".join(result)


def eq64_decode(encoded: str) -> bytes:
    """Decode Eq64 by removing dots and using standard q64 decode."""
    return q64_decode(encoded.replace(".", ""))
```

### 14.5. Step 1.4: Create SimHash Encoder

Create `src/uubed/encoders/shq64.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/encoders/shq64.py
"""Shq64: SimHash encoder for similarity-preserving compact codes."""

import numpy as np
from .q64 import q64_encode
from typing import List


def simhash_q64(embedding: List[int], planes: int = 64) -> str:
    """
    Generate position-safe SimHash code.

    How it works:
    1. Project embedding onto 64 random hyperplanes
    2. Store sign bit (which side of hyperplane)
    3. Similar embeddings → similar bit patterns → similar codes

    Args:
        embedding: List of byte values (0-255)
        planes: Number of random projections (must be multiple of 8)

    Returns:
        16-character q64 string (for 64 planes)
    """
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)

    # Generate random projection matrix
    rand_vectors = rng.normal(size=(planes, len(embedding)))

    # Convert bytes to centered floats
    vec = np.array(embedding, dtype=float)
    vec = (vec - 128) / 128  # Center around 0

    # Project and get sign bits
    projections = rand_vectors @ vec
    bits = (projections > 0).astype(int)

    # Pack bits into bytes
    byte_data = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val |= int(bits[i + j]) << (7 - j)
        byte_data.append(byte_val)

    return q64_encode(bytes(byte_data))
```

### 14.6. Step 1.5: Create Top-k Encoder

Create `src/uubed/encoders/t8q64.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/encoders/t8q64.py
"""T8q64: Top-k indices encoder for sparse representation."""

import numpy as np
from .q64 import q64_encode
from typing import List


def top_k_q64(embedding: List[int], k: int = 8) -> str:
    """
    Encode top-k highest magnitude indices.

    Why this works: Important features tend to have extreme values.
    By storing only the indices of the k largest values, we get
    a sparse but effective representation.

    Args:
        embedding: List of byte values (0-255)
        k: Number of top indices to keep

    Returns:
        16-character q64 string (for k=8)
    """
    # Get indices of k largest values
    indices = np.argsort(np.array(embedding))[-k:]

    # Sort indices for consistent encoding
    indices = sorted(indices.tolist())

    # Ensure we have exactly k indices (pad with 255 if needed)
    while len(indices) < k:
        indices.append(255)

    return q64_encode(bytes(indices))
```

### 14.7. Step 1.6: Create Z-order Encoder

Create `src/uubed/encoders/zoq64.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/encoders/zoq64.py
"""Zoq64: Z-order (Morton code) encoder for spatial locality."""

import struct
from .q64 import q64_encode
from typing import List


def z_order_q64(embedding: List[int]) -> str:
    """
    Encode using Z-order (Morton) curve.

    Why Z-order? Space-filling curves preserve spatial locality:
    nearby points in high-dimensional space get similar prefixes,
    enabling efficient prefix searches and range queries.

    Args:
        embedding: List of byte values (0-255)

    Returns:
        8-character q64 string
    """
    # Quantize to 2 bits per dimension (take top 2 bits)
    quantized = [(b >> 6) & 0b11 for b in embedding]

    # Interleave bits for first 16 dimensions
    result = 0
    for i, val in enumerate(quantized[:16]):
        for bit_pos in range(2):
            bit = (val >> bit_pos) & 1
            result |= bit << (i * 2 + bit_pos)

    # Pack as 4 bytes
    packed = struct.pack(">I", result)
    return q64_encode(packed)
```

### 14.8. Step 1.7: Create High-Level API

Create `src/uubed/api.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/api.py
"""High-level API for uubed encoding."""

from typing import Union, List, Literal, Optional
import numpy as np
from .encoders import eq64, shq64, t8q64, zoq64

EncodingMethod = Literal["eq64", "shq64", "t8q64", "zoq64", "auto"]


def encode(
    embedding: Union[List[int], np.ndarray, bytes],
    method: EncodingMethod = "auto",
    **kwargs
) -> str:
    """
    Encode embedding vector using specified method.

    Args:
        embedding: Vector to encode (bytes or 0-255 integers)
        method: Encoding method or "auto" for automatic selection
        **kwargs: Method-specific parameters

    Returns:
        Encoded string

    Example:
        >>> import numpy as np
        >>> embedding = np.random.randint(0, 256, 32, dtype=np.uint8)
        >>>
        >>> # Full precision
        >>> full = encode(embedding, method="eq64")
        >>>
        >>> # Compact similarity hash
        >>> compact = encode(embedding, method="shq64")
        >>>
        >>> # Sparse top-k representation
        >>> sparse = encode(embedding, method="t8q64", k=8)
    """
    # Convert to list of integers
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    elif isinstance(embedding, bytes):
        embedding = list(embedding)

    # Validate input
    if not all(0 <= x <= 255 for x in embedding):
        raise ValueError("Embedding values must be in range 0-255")

    # Auto-select method based on use case
    if method == "auto":
        if len(embedding) <= 32:
            method = "shq64"  # Compact for small embeddings
        else:
            method = "eq64"   # Full precision for larger ones

    # Dispatch to appropriate encoder
    if method == "eq64":
        return eq64.eq64_encode(embedding)
    elif method == "shq64":
        planes = kwargs.get("planes", 64)
        return shq64.simhash_q64(embedding, planes=planes)
    elif method == "t8q64":
        k = kwargs.get("k", 8)
        return t8q64.top_k_q64(embedding, k=k)
    elif method == "zoq64":
        return zoq64.z_order_q64(embedding)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def decode(encoded: str, method: Optional[EncodingMethod] = None) -> bytes:
    """
    Decode encoded string back to bytes.

    Args:
        encoded: Encoded string
        method: Encoding method (auto-detected if None)

    Returns:
        Original bytes

    Note: Only eq64 supports full decoding. Other methods
    are lossy compressions.
    """
    # Auto-detect method from string pattern
    if method is None:
        if "." in encoded:
            method = "eq64"
        else:
            # Cannot auto-detect between compressed formats
            raise ValueError(
                "Cannot auto-detect encoding method. "
                "Please specify method parameter."
            )

    if method == "eq64":
        return eq64.eq64_decode(encoded)
    else:
        raise NotImplementedError(
            f"Decoding not supported for {method}. "
            "These are lossy compression methods."
        )
```

### 14.9. Step 1.8: Create Package **init**.py

Create `src/uubed/__init__.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/__init__.py
"""
uubed: High-performance encoding for embedding vectors.

Solves the "substring pollution" problem in search systems by using
position-dependent alphabets that prevent false matches.
"""

from .api import encode, decode

__version__ = "0.1.0"
__all__ = ["encode", "decode"]
```

### 14.10. Step 1.9: Create Tests

Create `tests/test_encoders.py`:

```python
#!/usr/bin/env python3
# this_file: tests/test_encoders.py
"""Test suite for uubed encoders."""

import pytest
import numpy as np
from uubed import encode, decode
from uubed.encoders import q64


class TestQ64:
    """Test the base Q64 codec."""

    def test_encode_decode_roundtrip(self):
        """Test that encode->decode returns original data."""
        data = bytes([0, 127, 255, 42, 100])
        encoded = q64.q64_encode(data)
        decoded = q64.q64_decode(encoded)
        assert decoded == data

    def test_position_safety(self):
        """Test that characters are position-dependent."""
        data1 = bytes([0, 0, 0, 0])
        data2 = bytes([255, 255, 255, 255])

        enc1 = q64.q64_encode(data1)
        enc2 = q64.q64_encode(data2)

        # Check that different positions use different alphabets
        for i in range(len(enc1)):
            alphabet_idx = i & 3
            assert enc1[i] in q64.ALPHABETS[alphabet_idx]
            assert enc2[i] in q64.ALPHABETS[alphabet_idx]

    def test_invalid_decode(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="even"):
            q64.q64_decode("ABC")  # Odd length

        with pytest.raises(ValueError, match="Invalid q64 character"):
            q64.q64_decode("!@")  # Invalid characters

        with pytest.raises(ValueError, match="illegal at position"):
            q64.q64_decode("AQ")  # A is from alphabet 0, Q from alphabet 1


class TestHighLevelAPI:
    """Test the high-level encode/decode API."""

    def test_auto_encode(self):
        """Test automatic method selection."""
        small_embedding = np.random.randint(0, 256, 32, dtype=np.uint8)
        large_embedding = np.random.randint(0, 256, 256, dtype=np.uint8)

        # Auto should pick shq64 for small, eq64 for large
        small_result = encode(small_embedding, method="auto")
        assert len(small_result) == 16  # SimHash is compact

        large_result = encode(large_embedding, method="auto")
        assert "." in large_result  # Eq64 has dots

    def test_all_methods(self):
        """Test all encoding methods."""
        embedding = list(range(32))

        eq64_result = encode(embedding, method="eq64")
        assert "." in eq64_result
        assert len(eq64_result) == 71  # 64 chars + 7 dots

        shq64_result = encode(embedding, method="shq64")
        assert len(shq64_result) == 16  # 64 bits = 8 bytes = 16 chars

        t8q64_result = encode(embedding, method="t8q64", k=8)
        assert len(t8q64_result) == 16  # 8 indices = 16 chars

        zoq64_result = encode(embedding, method="zoq64")
        assert len(zoq64_result) == 8   # 4 bytes = 8 chars

    def test_decode_eq64(self):
        """Test decoding of eq64 format."""
        data = bytes(range(32))
        encoded = encode(data, method="eq64")
        decoded = decode(encoded, method="eq64")
        assert decoded == data


class TestLocalityPreservation:
    """Test that similar embeddings produce similar codes."""

    def test_simhash_locality(self):
        """Test SimHash preserves similarity."""
        # Create two similar embeddings
        base = np.random.randint(0, 256, 32, dtype=np.uint8)
        similar = base.copy()
        similar[0] = (similar[0] + 1) % 256  # Small change

        different = 255 - base  # Very different

        # Encode all three
        base_hash = encode(base, method="shq64")
        similar_hash = encode(similar, method="shq64")
        different_hash = encode(different, method="shq64")

        # Count character differences
        similar_diff = sum(a != b for a, b in zip(base_hash, similar_hash))
        different_diff = sum(a != b for a, b in zip(base_hash, different_hash))

        # Similar embeddings should have fewer differences
        assert similar_diff < different_diff

    def test_topk_locality(self):
        """Test Top-k preserves important features."""
        # Create embedding with clear top features
        embedding = np.zeros(256, dtype=np.uint8)
        top_indices = [10, 20, 30, 40, 50, 60, 70, 80]
        for idx in top_indices:
            embedding[idx] = 255

        # Add some noise
        similar = embedding.copy()
        similar[top_indices[0]] = 254  # Slightly reduce one top value
        similar[90] = 100  # Add medium value elsewhere

        # Encode both
        base_topk = encode(embedding, method="t8q64")
        similar_topk = encode(similar, method="t8q64")

        # Should have high overlap
        assert base_topk == similar_topk  # Top indices unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 14.11. Step 1.10: Run Tests and Fix Issues

```bash
# Install test dependencies
pip install pytest numpy

# Run tests
python -m pytest tests/ -v

# Fix any failing tests by debugging the encoder implementations
```

### 14.12. Phase 1 Checklist

- [ ] Package structure created
- [ ] Q64 base codec implemented and tested
- [ ] All encoder variants (Eq64, Shq64, T8q64, Zoq64) implemented
- [ ] High-level API created
- [ ] Comprehensive test suite passing
- [ ] Basic benchmarks established

---

## 15. 🦀 Phase 2: Rust Core Implementation (Week 2)

### 15.1. Goal

Implement high-performance Rust versions of the encoders.

### 15.2. Step 2.1: Initialize Rust Project

```bash
# Create Rust workspace
mkdir rust
cd rust
cargo init --lib

# Add to root Cargo.toml
cat > ../Cargo.toml << 'EOF'
[workspace]
members = ["rust"]

[workspace.package]
version = "0.1.0"
edition = "2021"

[patch.crates-io]
# Add any patches here if needed
EOF
```

### 15.3. Step 2.2: Set Up Rust Dependencies

Edit `rust/Cargo.toml`:

```toml
[package]
name = "uubed-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "uubed_native"
crate-type = ["cdylib"]  # Required for Python extensions

[dependencies]
# Core dependencies
pyo3 = { version = "0.22", features = ["extension-module"] }
rayon = "1.10"        # Parallel processing
once_cell = "1.20"    # Lazy static initialization

# For SimHash random projections
rand = "0.8"
rand_chacha = "0.3"   # Cryptographically secure RNG
rand_distr = "0.4"    # Normal distribution

# Performance optimizations
bytemuck = { version = "1.19", optional = true }

[features]
default = []
simd = ["bytemuck"]   # Enable SIMD optimizations

[profile.release]
# Optimize for speed
lto = true            # Link-time optimization
codegen-units = 1     # Single compilation unit for better optimization
opt-level = 3         # Maximum optimization
strip = true          # Remove debug symbols for smaller binaries

# Profile for development with some optimizations
[profile.dev-opt]
inherits = "dev"
opt-level = 2         # Some optimization for bearable performance
```

**Understanding the Dependencies:**

- **pyo3**: The magic that lets Rust talk to Python. The `extension-module` feature is required for building Python extensions.
- **rayon**: Provides data parallelism - automatically spreads work across CPU cores.
- **once_cell**: Allows lazy initialization of static data (like our projection matrices).
- **rand + rand_chacha**: Reproducible random number generation for SimHash projections.
- **rand_distr**: Provides normal distribution for better random projections.
- **bytemuck**: Enables zero-copy conversions between types (optional for SIMD).

### 15.4. Step 2.3: Implement Q64 Codec in Rust

First, update the main Rust library entry point.

Create `rust/src/lib.rs`:

```rust
// this_file: rust/src/lib.rs
//! uubed-core: High-performance encoding library

pub mod encoders;
pub mod bindings;

// Re-export main functions
pub use encoders::{q64_encode, q64_decode};
```

Create `rust/src/encoders/mod.rs`:

```rust
// this_file: rust/src/encoders/mod.rs

pub mod q64;
pub mod simhash;
pub mod topk;
pub mod zorder;

pub use q64::{q64_encode, q64_decode};
pub use simhash::simhash_q64;
pub use topk::top_k_q64;
pub use zorder::z_order_q64;
```

Create `rust/src/encoders/q64.rs`:

```rust
// this_file: rust/src/encoders/q64.rs
//! QuadB64: Position-safe encoding with SIMD optimization.

use std::error::Error;
use std::fmt;

/// Position-dependent alphabets
const ALPHABETS: [&[u8; 16]; 4] = [
    b"ABCDEFGHIJKLMNOP",  // pos ≡ 0 (mod 4)
    b"QRSTUVWXYZabcdef",  // pos ≡ 1
    b"ghijklmnopqrstuv",  // pos ≡ 2
    b"wxyz0123456789-_",  // pos ≡ 3
];

/// Reverse lookup table (ASCII char -> (alphabet_idx, nibble_value))
/// We use a const fn to build this at compile time for better performance
const fn build_reverse_lookup() -> [Option<(u8, u8)>; 256] {
    let mut table = [None; 256];
    let mut alphabet_idx = 0;

    // Manual loop unrolling since const fn limitations
    while alphabet_idx < 4 {
        let alphabet = ALPHABETS[alphabet_idx];
        let mut nibble_value = 0;
        while nibble_value < 16 {
            let ch = alphabet[nibble_value];
            table[ch as usize] = Some((alphabet_idx as u8, nibble_value as u8));
            nibble_value += 1;
        }
        alphabet_idx += 1;
    }
    table
}

static REV_LOOKUP: [Option<(u8, u8)>; 256] = build_reverse_lookup();

#[derive(Debug, Clone)]
pub struct Q64Error {
    message: String,
}

impl fmt::Display for Q64Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q64 error: {}", self.message)
    }
}

impl Error for Q64Error {}

/// Encode bytes into Q64 format.
///
/// # Performance
/// - Uses SIMD when available for parallel nibble extraction
/// - Processes 16 bytes at a time on x86_64 with AVX2
/// - Falls back to scalar code on other architectures
pub fn q64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 2);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SIMD fast path for x86_64
        unsafe { q64_encode_simd(data, &mut result) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        // Scalar fallback
        q64_encode_scalar(data, &mut result);
    }

    result
}

/// Scalar implementation of Q64 encoding
fn q64_encode_scalar(data: &[u8], output: &mut String) {
    for (byte_idx, &byte) in data.iter().enumerate() {
        let hi_nibble = (byte >> 4) & 0xF;
        let lo_nibble = byte & 0xF;
        let base_pos = byte_idx * 2;

        // Use position-dependent alphabets
        output.push(ALPHABETS[base_pos & 3][hi_nibble as usize] as char);
        output.push(ALPHABETS[(base_pos + 1) & 3][lo_nibble as usize] as char);
    }
}

/// SIMD implementation for x86_64 with AVX2
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn q64_encode_simd(data: &[u8], output: &mut String) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let chunks = data.chunks_exact(16);
    let remainder = chunks.remainder();

    // Process 16 bytes at a time
    for (chunk_idx, chunk) in chunks.enumerate() {
        // Load 16 bytes
        let input = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

        // Split into high and low nibbles
        let hi_mask = _mm_set1_epi8(0xF0u8 as i8);
        let lo_mask = _mm_set1_epi8(0x0F);

        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input, 4), lo_mask);
        let lo_nibbles = _mm_and_si128(input, lo_mask);

        // Process nibbles and convert to characters
        // This is simplified - real implementation would use lookup tables
        let base_pos = chunk_idx * 32;

        // Extract and encode each nibble
        for i in 0..16 {
            let hi = _mm_extract_epi8(hi_nibbles, i) as usize;
            let lo = _mm_extract_epi8(lo_nibbles, i) as usize;

            let pos = base_pos + i * 2;
            output.push(ALPHABETS[pos & 3][hi] as char);
            output.push(ALPHABETS[(pos + 1) & 3][lo] as char);
        }
    }

    // Handle remainder with scalar code
    let byte_offset = data.len() - remainder.len();
    for (idx, &byte) in remainder.iter().enumerate() {
        let byte_idx = byte_offset + idx;
        let hi_nibble = (byte >> 4) & 0xF;
        let lo_nibble = byte & 0xF;
        let base_pos = byte_idx * 2;

        output.push(ALPHABETS[base_pos & 3][hi_nibble as usize] as char);
        output.push(ALPHABETS[(base_pos + 1) & 3][lo_nibble as usize] as char);
    }
}

/// Decode Q64 string back to bytes
pub fn q64_decode(encoded: &str) -> Result<Vec<u8>, Q64Error> {
    if encoded.len() & 1 != 0 {
        return Err(Q64Error {
            message: "Q64 string length must be even".to_string(),
        });
    }

    let mut result = Vec::with_capacity(encoded.len() / 2);
    let chars: Vec<char> = encoded.chars().collect();

    for (pos, chunk) in chars.chunks_exact(2).enumerate() {
        let ch1 = chunk[0];
        let ch2 = chunk[1];

        // Validate and decode first nibble
        let (alphabet1, nibble1) = validate_char(ch1, pos * 2)?;

        // Validate and decode second nibble
        let (alphabet2, nibble2) = validate_char(ch2, pos * 2 + 1)?;

        // Combine nibbles into byte
        result.push((nibble1 << 4) | nibble2);
    }

    Ok(result)
}

/// Validate character and return (alphabet_idx, nibble_value)
fn validate_char(ch: char, pos: usize) -> Result<(u8, u8), Q64Error> {
    if ch as u32 > 255 {
        return Err(Q64Error {
            message: format!("Non-ASCII character '{}' at position {}", ch, pos),
        });
    }

    match REV_LOOKUP[ch as usize] {
        Some((alphabet_idx, nibble_value)) => {
            let expected_alphabet = (pos & 3) as u8;
            if alphabet_idx != expected_alphabet {
                Err(Q64Error {
                    message: format!(
                        "Character '{}' from alphabet {} at position {} (expected alphabet {})",
                        ch, alphabet_idx, pos, expected_alphabet
                    ),
                })
            } else {
                Ok((alphabet_idx, nibble_value))
            }
        }
        None => Err(Q64Error {
            message: format!("Invalid Q64 character '{}' at position {}", ch, pos),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let data = vec![0, 127, 255, 42, 100];
        let encoded = q64_encode(&data);
        let decoded = q64_decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_position_safety() {
        let data = vec![0, 0, 0, 0];
        let encoded = q64_encode(&data);

        // Verify each character is from correct alphabet
        for (i, ch) in encoded.chars().enumerate() {
            let alphabet_idx = i & 3;
            assert!(ALPHABETS[alphabet_idx].contains(&(ch as u8)));
        }
    }
}
```

### 15.5. Step 2.4: Implement SimHash in Rust

Create `rust/src/encoders/simhash.rs`:

```rust
// this_file: rust/src/encoders/simhash.rs
//! SimHash implementation with parallel matrix multiplication.

use rayon::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::collections::HashMap;

/// Cache for projection matrices of different sizes
static MATRIX_CACHE: Lazy<Mutex<HashMap<(usize, usize), ProjectionMatrix>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Random projection matrix for SimHash
#[derive(Clone)]
struct ProjectionMatrix {
    data: Vec<f32>,
    planes: usize,
    dims: usize,
}

impl ProjectionMatrix {
    /// Generate matrix with fixed seed for reproducibility
    fn new(planes: usize, dims: usize) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut data = Vec::with_capacity(planes * dims);

        // Generate normal distribution for better random projections
        for _ in 0..(planes * dims) {
            data.push(normal.sample(&mut rng));
        }

        Self { data, planes, dims }
    }

    /// Get cached matrix or create new one
    fn get_or_create(planes: usize, dims: usize) -> ProjectionMatrix {
        let mut cache = MATRIX_CACHE.lock().unwrap();
        cache.entry((planes, dims))
            .or_insert_with(|| ProjectionMatrix::new(planes, dims))
            .clone()
    }

    /// Project vector onto hyperplanes (parallel)
    fn project(&self, embedding: &[u8]) -> Vec<bool> {
        // Convert bytes to centered floats
        let centered: Vec<f32> = embedding
            .iter()
            .map(|&b| (b as f32 - 128.0) / 128.0)
            .collect();

        // Parallel matrix multiplication
        (0..self.planes)
            .into_par_iter()
            .map(|plane_idx| {
                let row_start = plane_idx * self.dims;
                let row_end = row_start + self.dims.min(centered.len());

                let dot_product: f32 = self.data[row_start..row_end]
                    .iter()
                    .zip(&centered)
                    .map(|(a, b)| a * b)
                    .sum();

                dot_product > 0.0
            })
            .collect()
    }
}

/// Generate SimHash with Q64 encoding
///
/// # Algorithm
/// 1. Project embedding onto random hyperplanes
/// 2. Take sign of each projection as a bit
/// 3. Pack bits into bytes
/// 4. Encode with position-safe Q64
pub fn simhash_q64(embedding: &[u8], planes: usize) -> String {
    // Get cached projection matrix for efficiency
    let matrix = ProjectionMatrix::get_or_create(planes, embedding.len());

    // Project and get bits
    let bits = matrix.project(embedding);

    // Pack bits into bytes
    let mut bytes = Vec::with_capacity((bits.len() + 7) / 8);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            if bit {
                byte |= 1 << (7 - i);
            }
        }
        bytes.push(byte);
    }

    // Encode with Q64
    super::q64::q64_encode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_locality() {
        let base = vec![100; 32];
        let mut similar = base.clone();
        similar[0] = 101;  // Small change

        let different: Vec<u8> = base.iter().map(|&x| 255 - x).collect();

        let hash1 = simhash_q64(&base, 64);
        let hash2 = simhash_q64(&similar, 64);
        let hash3 = simhash_q64(&different, 64);

        // Count differences
        let diff_similar = hash1.chars()
            .zip(hash2.chars())
            .filter(|(a, b)| a != b)
            .count();

        let diff_different = hash1.chars()
            .zip(hash3.chars())
            .filter(|(a, b)| a != b)
            .count();

        assert!(diff_similar < diff_different);
    }
}
```

### 15.6. Step 2.5: Implement Top-k Encoder in Rust

Create `rust/src/encoders/topk.rs`:

```rust
// this_file: rust/src/encoders/topk.rs
//! Top-k indices encoder for sparse representation.

use rayon::prelude::*;

/// Find top k indices with highest values
///
/// Uses parallel partial sorting for efficiency on large embeddings
pub fn top_k_indices(embedding: &[u8], k: usize) -> Vec<u8> {
    if embedding.len() <= 256 {
        // Fast path for small embeddings
        top_k_indices_small(embedding, k)
    } else {
        // Parallel path for large embeddings
        top_k_indices_parallel(embedding, k)
    }
}

/// Fast implementation for embeddings that fit in a u8 index
fn top_k_indices_small(embedding: &[u8], k: usize) -> Vec<u8> {
    let mut indexed: Vec<(u8, u8)> = embedding
        .iter()
        .enumerate()
        .map(|(idx, &val)| (val, idx as u8))
        .collect();

    // Partial sort to get top k
    let k_clamped = k.min(indexed.len());
    indexed.select_nth_unstable_by(k_clamped - 1, |a, b| b.0.cmp(&a.0));

    // Extract indices and sort them
    let mut indices: Vec<u8> = indexed[..k_clamped]
        .iter()
        .map(|(_, idx)| *idx)
        .collect();
    indices.sort_unstable();

    // Pad with 255 if needed
    indices.resize(k, 255);
    indices
}

/// Parallel implementation for large embeddings
fn top_k_indices_parallel(embedding: &[u8], k: usize) -> Vec<u8> {
    // Split into chunks for parallel processing
    let chunk_size = 256;
    let chunks: Vec<_> = embedding
        .chunks(chunk_size)
        .enumerate()
        .collect();

    // Find top candidates from each chunk in parallel
    let candidates: Vec<(u8, usize)> = chunks
        .par_iter()
        .flat_map(|(chunk_idx, chunk)| {
            let mut local_top: Vec<(u8, usize)> = chunk
                .iter()
                .enumerate()
                .map(|(idx, &val)| (val, chunk_idx * chunk_size + idx))
                .collect();

            // Keep top k from each chunk
            let local_k = k.min(local_top.len());
            local_top.select_nth_unstable_by(local_k - 1, |a, b| b.0.cmp(&a.0));
            local_top.truncate(local_k);
            local_top
        })
        .collect();

    // Final selection from candidates
    let mut final_candidates = candidates;
    let final_k = k.min(final_candidates.len());
    final_candidates.select_nth_unstable_by(final_k - 1, |a, b| b.0.cmp(&a.0));

    // Extract indices, handle large indices
    let mut indices: Vec<u8> = final_candidates[..final_k]
        .iter()
        .map(|(_, idx)| (*idx).min(255) as u8)
        .collect();
    indices.sort_unstable();

    // Pad with 255 if needed
    indices.resize(k, 255);
    indices
}

/// Generate top-k encoding with Q64
pub fn top_k_q64(embedding: &[u8], k: usize) -> String {
    let indices = top_k_indices(embedding, k);
    super::q64::q64_encode(&indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_basic() {
        let data = vec![10, 50, 30, 80, 20, 90, 40, 70];
        let top3 = top_k_indices(&data, 3);
        assert_eq!(top3, vec![3, 5, 7]);  // Indices of 80, 90, 70
    }

    #[test]
    fn test_top_k_padding() {
        let data = vec![10, 20, 30];
        let top5 = top_k_indices(&data, 5);
        assert_eq!(top5, vec![0, 1, 2, 255, 255]);  // Padded with 255
    }
}
```

### 15.7. Step 2.6: Implement Z-order Encoder in Rust

Create `rust/src/encoders/zorder.rs`:

```rust
// this_file: rust/src/encoders/zorder.rs
//! Z-order (Morton code) encoder for spatial locality.

/// Interleave bits for Z-order curve
///
/// This creates a space-filling curve that preserves spatial locality.
/// Points that are close in high-dimensional space will have similar
/// Z-order codes and thus similar prefixes.
pub fn z_order_q64(embedding: &[u8]) -> String {
    // Take top 2 bits from each dimension
    let quantized: Vec<u8> = embedding
        .iter()
        .map(|&b| (b >> 6) & 0b11)
        .collect();

    // We'll interleave bits from up to 16 dimensions into a 32-bit value
    let dims_to_use = quantized.len().min(16);
    let mut result: u32 = 0;

    // Bit interleaving using bit manipulation tricks
    for dim in 0..dims_to_use {
        let val = quantized[dim] as u32;

        // Spread the 2 bits across the result
        // Bit 0 goes to position dim*2
        // Bit 1 goes to position dim*2 + 1
        result |= (val & 0b01) << (dim * 2);
        result |= ((val & 0b10) >> 1) << (dim * 2 + 1);
    }

    // Convert to bytes
    let bytes = result.to_be_bytes();
    super::q64::q64_encode(&bytes)
}

/// Advanced Z-order with more bits per dimension
///
/// This version uses 4 bits per dimension for finer granularity
pub fn z_order_q64_extended(embedding: &[u8]) -> String {
    // Take top 4 bits from each dimension
    let quantized: Vec<u8> = embedding
        .iter()
        .map(|&b| (b >> 4) & 0b1111)
        .collect();

    // We can fit 8 dimensions × 4 bits = 32 bits
    let dims_to_use = quantized.len().min(8);
    let mut result: u32 = 0;

    // Interleave 4 bits from each dimension
    for dim in 0..dims_to_use {
        let val = quantized[dim] as u32;

        // Use bit manipulation to spread bits
        // This is a simplified version - production code would use
        // lookup tables or PDEP instruction for efficiency
        for bit in 0..4 {
            let bit_val = (val >> bit) & 1;
            result |= bit_val << (bit * 8 + dim);
        }
    }

    // Convert to bytes
    let bytes = result.to_be_bytes();
    super::q64::q64_encode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_order_basic() {
        // Test that similar inputs produce similar codes
        let vec1 = vec![255, 255, 0, 0];  // Top-left in 2D
        let vec2 = vec![255, 254, 0, 0];  // Very close to vec1
        let vec3 = vec![0, 0, 255, 255];  // Bottom-right in 2D

        let z1 = z_order_q64(&vec1);
        let z2 = z_order_q64(&vec2);
        let z3 = z_order_q64(&vec3);

        // z1 and z2 should share a longer prefix than z1 and z3
        let prefix_len_12 = z1.chars()
            .zip(z2.chars())
            .take_while(|(a, b)| a == b)
            .count();

        let prefix_len_13 = z1.chars()
            .zip(z3.chars())
            .take_while(|(a, b)| a == b)
            .count();

        assert!(prefix_len_12 > prefix_len_13);
    }
}
```

### 15.8. Step 2.7: Create PyO3 Bindings

Create `rust/src/bindings.rs`:

```rust
// this_file: rust/src/bindings.rs
//! Python bindings for uubed-core using PyO3.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Encode bytes using Q64 algorithm
#[pyfunction]
#[pyo3(signature = (data))]
fn q64_encode_native(data: &[u8]) -> String {
    crate::encoders::q64_encode(data)
}

/// Decode Q64 string to bytes
#[pyfunction]
#[pyo3(signature = (encoded))]
fn q64_decode_native(encoded: &str) -> PyResult<Vec<u8>> {
    crate::encoders::q64_decode(encoded)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Generate SimHash with Q64 encoding
#[pyfunction]
#[pyo3(signature = (embedding, planes=64))]
fn simhash_q64_native(embedding: &[u8], planes: usize) -> String {
    crate::encoders::simhash_q64(embedding, planes)
}

/// Generate top-k indices with Q64 encoding
#[pyfunction]
#[pyo3(signature = (embedding, k=8))]
fn top_k_q64_native(embedding: &[u8], k: usize) -> String {
    crate::encoders::top_k_q64(embedding, k)
}

/// Generate Z-order with Q64 encoding
#[pyfunction]
#[pyo3(signature = (embedding))]
fn z_order_q64_native(embedding: &[u8]) -> String {
    crate::encoders::z_order_q64(embedding)
}

/// Python module initialization
#[pymodule]
fn _uubed_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(q64_encode_native, m)?)?;
    m.add_function(wrap_pyfunction!(q64_decode_native, m)?)?;
    m.add_function(wrap_pyfunction!(simhash_q64_native, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_q64_native, m)?)?;
    m.add_function(wrap_pyfunction!(z_order_q64_native, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
```

### 15.9. Step 2.6: Update Project Structure for Maturin

Create `pyproject.toml` at the root:

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "uubed"
version = "0.1.0"
description = "High-performance encoding for embedding vectors"
readme = "README.md"
requires-python = ">=3.8"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "your.email@example.com" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "hypothesis>=6.0",
]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "uubed._native"
python-source = "src"
```

### 15.10. Step 2.7: Build and Test Native Module

```bash
# Install maturin
pip install maturin

# Build in development mode
maturin develop

# Run tests to verify native module works
python -c "from uubed._native import q64_encode_native; print(q64_encode_native(b'hello'))"
```

### 15.11. Phase 2 Checklist

- [ ] Rust project structure created
- [ ] Q64 codec implemented with SIMD optimization
- [ ] SimHash implemented with parallel processing
- [ ] Top-k and Z-order encoders implemented
- [ ] PyO3 bindings created
- [ ] Native module builds successfully
- [ ] Performance benchmarks show 10x improvement

---

## 16. 📦 Phase 3: Integration & Packaging (Week 3)

### 16.1. Goal

Integrate native module with Python API and prepare for distribution.

### 16.2. Step 3.1: Create Native Module Wrapper

Create `src/uubed/_native.py`:

```python
#!/usr/bin/env python3
# this_file: src/uubed/_native.py
"""Wrapper for native module with fallback to pure Python."""

try:
    # Try to import native module
    from uubed._native import (
        q64_encode_native,
        q64_decode_native,
        simhash_q64_native,
        # ... other native functions
    )
    HAS_NATIVE = True
except ImportError:
    # Fall back to pure Python
    HAS_NATIVE = False

    # Import pure Python implementations
    from .encoders.q64 import q64_encode as q64_encode_native
    from .encoders.q64 import q64_decode as q64_decode_native
    from .encoders.shq64 import simhash_q64 as simhash_q64_native


def is_native_available() -> bool:
    """Check if native acceleration is available."""
    return HAS_NATIVE
```

### 16.3. Step 3.2: Update API to Use Native Functions

Update `src/uubed/api.py` to prefer native implementations:

```python
# Add at the top
from ._native import (
    q64_encode_native,
    q64_decode_native,
    simhash_q64_native,
    is_native_available,
)

# Update encode function to use native versions
def encode(
    embedding: Union[List[int], np.ndarray, bytes],
    method: EncodingMethod = "auto",
    **kwargs
) -> str:
    """Encode embedding vector using specified method."""
    # ... validation code ...

    # Use native implementations when available
    if method == "eq64":
        # Native Q64 with dots
        encoded = q64_encode_native(bytes(embedding))
        # Add dots
        result = []
        for i, char in enumerate(encoded):
            if i > 0 and i % 8 == 0:
                result.append(".")
            result.append(char)
        return "".join(result)

    elif method == "shq64":
        planes = kwargs.get("planes", 64)
        return simhash_q64_native(bytes(embedding), planes)

    # ... rest of the methods ...
```

### 16.4. Step 3.3: Create Comprehensive Benchmarks

Create `benchmarks/bench_encoders.py`:

```python
#!/usr/bin/env python3
# this_file: benchmarks/bench_encoders.py
"""Benchmark native vs pure Python implementations."""

import time
import numpy as np
from uubed import encode
from uubed._native import is_native_available
from uubed.encoders import q64, shq64


def benchmark_function(func, *args, iterations=1000):
    """Time a function over multiple iterations."""
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    return (end - start) / iterations


def main():
    """Run benchmarks."""
    # Generate test data
    sizes = [32, 256, 1024]

    print(f"Native module available: {is_native_available()}")
    print("-" * 60)

    for size in sizes:
        data = np.random.randint(0, 256, size, dtype=np.uint8).tobytes()

        print(f"\nEmbedding size: {size} bytes")
        print("-" * 40)

        # Benchmark Q64
        if is_native_available():
            from uubed._native import q64_encode_native
            native_time = benchmark_function(q64_encode_native, data)
            print(f"Q64 Native:     {native_time*1e6:.2f} μs")

        pure_time = benchmark_function(q64.q64_encode, data)
        print(f"Q64 Pure Python: {pure_time*1e6:.2f} μs")

        if is_native_available():
            speedup = pure_time / native_time
            print(f"Speedup: {speedup:.1f}x")

        # Benchmark SimHash
        print()
        if is_native_available():
            from uubed._native import simhash_q64_native
            native_time = benchmark_function(
                simhash_q64_native, data, 64, iterations=100
            )
            print(f"SimHash Native:     {native_time*1e3:.2f} ms")

        pure_time = benchmark_function(
            shq64.simhash_q64, list(data), 64, iterations=100
        )
        print(f"SimHash Pure Python: {pure_time*1e3:.2f} ms")

        if is_native_available():
            speedup = pure_time / native_time
            print(f"Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
```

### 16.5. Step 3.4: Set Up CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install maturin pytest numpy

      - name: Build wheel
        run: maturin build --release

      - name: Install wheel
        run: pip install target/wheels/*.whl

      - name: Run tests
        run: pytest tests/ -v

      - name: Run benchmarks
        run: python benchmarks/bench_encoders.py

  build-wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist
```

### 16.6. Step 3.5: Create Documentation

Create `docs/quickstart.md`:

````markdown
# Quick Start Guide

## 17. Installation

```bash
pip install uubed
```
````

## 18. Basic Usage

```python
import numpy as np
from uubed import encode, decode

# Create a random embedding
embedding = np.random.randint(0, 256, 256, dtype=np.uint8)

# Full precision encoding
full_code = encode(embedding, method="eq64")
print(f"Full: {full_code[:50]}...")  # ABCDefGh.IjKLmnOp...

# Compact similarity hash (16 characters)
compact_code = encode(embedding, method="shq64")
print(f"Compact: {compact_code}")  # QRsTUvWxYZabcdef

# Decode back (only works for eq64)
decoded = decode(full_code)
assert np.array_equal(embedding, np.frombuffer(decoded, dtype=np.uint8))
```

## 19. Why QuadB64?

Regular Base64 in search engines:

- Query: "find similar to 'abc'"
- Problem: "abc" matches everywhere!
- Result: False positives pollute results

QuadB64 solution:

- Position-dependent alphabets
- "abc" can only match at specific positions
- Result: Clean, accurate similarity search

## 20. Encoding Methods

### 20.1. Eq64 - Full Embeddings

- **Use case**: Need full precision
- **Size**: 2n characters (n = embedding bytes)
- **Features**: Dots every 8 chars for readability

### 20.2. Shq64 - SimHash

- **Use case**: Fast similarity comparison
- **Size**: 16 characters (64-bit hash)
- **Features**: Preserves cosine similarity

### 20.3. T8q64 - Top-k Indices

- **Use case**: Sparse representation
- **Size**: 16 characters (8 indices)
- **Features**: Captures most important features

### 20.4. Zoq64 - Z-order

- **Use case**: Spatial/prefix search
- **Size**: 8 characters
- **Features**: Nearby points share prefixes

## 21. Performance

With native acceleration:

- **Q64 encoding**: ~10x faster than pure Python
- **SimHash**: ~15x faster with parallel processing
- **Batch processing**: Near-linear scaling with cores

## 22. Advanced Usage

```python
# Batch encoding
embeddings = [np.random.randint(0, 256, 256, dtype=np.uint8) for _ in range(100)]
codes = [encode(emb, method="shq64") for emb in embeddings]

# Custom parameters
custom_hash = encode(embedding, method="shq64", planes=128)  # More bits
sparse_repr = encode(embedding, method="t8q64", k=16)  # More indices

# Check native acceleration
from uubed._native import is_native_available
print(f"Native module: {is_native_available()}")
```

````

### 22.1. Phase 3 Checklist
- [ ] Native module integrated with Python API
- [ ] Fallback to pure Python works correctly
- [ ] Comprehensive benchmarks show expected speedup
- [ ] CI/CD pipeline building wheels for all platforms
- [ ] Documentation complete and clear
- [ ] Package ready for PyPI upload

---

## 23. 🚀 Phase 4: Publishing & Distribution (Week 4)

### 23.1. Goal
Publish the package to PyPI with binary wheels for all platforms.

### 23.2. Step 4.1: Prepare for Release
1. Update version numbers in all files
2. Create comprehensive README.md
3. Add CHANGELOG.md
4. Ensure all tests pass
5. Run final benchmarks

### 23.3. Step 4.2: Build Binary Wheels
```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for all platforms
cibuildwheel --output-dir dist

# Build source distribution
maturin sdist
````

### 23.4. Step 4.3: Test Package Installation

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install from wheel
pip install dist/uubed-0.1.0-cp39-cp39-linux_x86_64.whl

# Test it works
python -c "from uubed import encode; print(encode(b'test', method='eq64'))"
```

### 23.5. Step 4.4: Upload to PyPI

```bash
# Install twine
pip install twine

# Upload to TestPyPI first
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# If everything works, upload to real PyPI
twine upload dist/*
```

### 23.6. Phase 4 Checklist

- [ ] Version numbers updated everywhere
- [ ] README.md polished and complete
- [ ] CHANGELOG.md created
- [ ] All platform wheels built
- [ ] Package tested in clean environment
- [ ] Uploaded to PyPI
- [ ] Installation instructions verified

---

## 24. 🎓 Key Concepts for Junior Developers

### 24.1. Why Native Code?

**Python is interpreted**, which means each operation goes through multiple layers:

1. Python bytecode interpretation
2. Object allocation/deallocation
3. Dynamic type checking

**Rust is compiled**, giving us:

1. Direct CPU instructions
2. Zero-cost abstractions
3. SIMD vectorization
4. No garbage collection pauses

**Real-world impact**: A loop processing 1 million embeddings:

- Python: ~10 seconds (interpreter overhead on every iteration)
- Rust: ~0.1 seconds (compiled to optimal machine code)

### 24.2. Understanding SIMD

**SIMD** = Single Instruction, Multiple Data

Think of it like this:

- **Normal (scalar)**: One chef cutting one carrot at a time
- **SIMD**: One chef cutting 16 carrots with one chop!

Instead of:

```python
# Python: Process one byte at a time
for i in range(16):
    result[i] = data[i] >> 4  # 16 operations
```

SIMD does:

```rust
// Rust: Process 16 bytes in one CPU instruction
result[0..16] = _mm_srli_epi16(data[0..16], 4);  // 1 operation!
```

### 24.3. Understanding the QuadB64 Algorithm

**The Problem Visualized:**

```
Regular Base64: "Hello" → "SGVsbG8="
Search for "Vsb" → Matches! (false positive)

QuadB64: "Hello" → "S0V2b3g4"
Search for "V2b" → No match (different alphabets at each position)
```

**How Position-Dependent Alphabets Work:**

```
Position 0,4,8...: A-P only
Position 1,5,9...: Q-f only
Position 2,6,10..: g-v only
Position 3,7,11..: w-_ only

So "AQ" is valid (A from pos 0, Q from pos 1)
But "AA" is invalid (A can't appear at pos 1)
```

### 24.4. PyO3 Magic Explained

PyO3 handles the complex Python C API for us:

**Reference Counting**:

```rust
// PyO3 automatically handles Python's reference counting
let list = PyList::new(py, &[1, 2, 3]);  // RefCount +1
// When `list` goes out of scope, RefCount -1
```

**GIL Management**:

```rust
// PyO3 ensures we have the Global Interpreter Lock
Python::with_gil(|py| {
    // Safe to use Python objects here
});
```

**Type Conversions**:

```rust
// PyO3 converts between Rust and Python types
#[pyfunction]
fn add(a: i32, b: i32) -> i32 {  // Python int ↔ Rust i32
    a + b
}
```

### 24.5. Debugging Tips

**1. Start with Pure Python** Always implement in Python first:

- Easier to debug
- Establishes correct behavior
- Provides test baseline

**2. Add Logging**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def encode(data):
    logging.debug(f"Encoding {len(data)} bytes")
    result = _native.encode(data)
    logging.debug(f"Result: {result[:20]}...")
    return result
```

**3. Use Rust's `dbg!` Macro**

```rust
let result = q64_encode(&data);
dbg!(&result[..10]);  // Prints: [src/encoders/q64.rs:45] &result[..10] = "ABCDefGhij"
```

**4. Test Small Examples**

```python
# Start with tiny inputs
assert encode(b"A") == "Iq"
assert encode(b"AB") == "IqQm"
# Then scale up
```

### 24.6. Performance Tips

1. **Batch operations**: Process multiple items at once

   ```python
   # Bad: One at a time
   results = [encode(emb) for emb in embeddings]

   # Good: Batch processing
   results = encode_batch(embeddings)
   ```

2. **Avoid allocations**: Reuse buffers when possible

   ```rust
   // Bad: Allocate new string each time
   fn encode(data: &[u8]) -> String {
       let mut result = String::new();  // Allocation!

   // Good: Pre-allocate with capacity
   fn encode(data: &[u8]) -> String {
       let mut result = String::with_capacity(data.len() * 2);
   ```

3. **Profile first**: Measure before optimizing

   ```bash
   # Python profiling
   python -m cProfile -s cumtime your_script.py

   # Rust profiling
   cargo build --release
   perf record --call-graph=dwarf target/release/your_binary
   ```

4. **Test thoroughly**: Native bugs are harder to debug
   ```python
   # Always test edge cases
   test_empty = encode(b"")
   test_single = encode(b"A")
   test_max = encode(bytes([255] * 1000))
   ```

---

## 25. 🔧 Troubleshooting

### 25.1. Common Issues

**1. Module import fails**

```
ImportError: cannot import name '_native' from 'uubed'
```

Solution: Rebuild with `maturin develop`

**2. Rust compiler errors**

```
error[E0308]: mismatched types
```

Solution: Check Rust types match PyO3 expectations

**3. Performance not improved**

- Check native module is actually being used
- Verify SIMD instructions are enabled
- Profile to find actual bottlenecks

**4. CI builds failing**

- Different platforms have different requirements
- Windows needs Visual Studio Build Tools
- macOS needs Xcode Command Line Tools
- Linux needs gcc

---

## 26. 🎉 Success Criteria

You know you've succeeded when:

1. ✅ `pip install uubed` works on all platforms
2. ✅ Native module loads automatically
3. ✅ 10x performance improvement achieved
4. ✅ All tests pass in CI
5. ✅ Documentation is clear and helpful
6. ✅ Other developers start using your library!

---

## 27. 📚 Resources for Learning

### 27.1. Python Packaging

- [Python Packaging Guide](https://packaging.python.org/)
- [Maturin Documentation](https://maturin.rs/)
- [Real Python: Python Wheels](https://realpython.com/python-wheels/)

### 27.2. Rust & PyO3

- [The Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Guide](https://pyo3.rs/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)

### 27.3. SIMD & Performance

- [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [Godbolt Compiler Explorer](https://godbolt.org/) - See your code's assembly

### 27.4. Embedding & Vector Search

- [Understanding Embeddings](https://www.pinecone.io/learn/vector-embeddings/)
- [Locality-Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)

---

## 28. 💡 Common Gotchas & Best Practices

### 28.1. Gotcha 1: Python Import Paths

```python
# Wrong: Relative import in tests
from src.uubed import encode  # Fails!

# Right: Install in development mode first
# Run: pip install -e .
from uubed import encode  # Works!
```

### 28.2. Gotcha 2: Rust String vs Bytes

```rust
// Wrong: String for binary data
fn encode(data: String) -> String {  // NO!

// Right: Use bytes for binary data
fn encode(data: &[u8]) -> String {  // YES!
```

### 28.3. Gotcha 3: Maturin Module Names

```toml
# pyproject.toml
[tool.maturin]
module-name = "uubed._native"  # Must match import path!

# Python code
from uubed._native import encode  # The dot matters!
```

### 28.4. Best Practice 1: Version Everything

```python
# src/uubed/__init__.py
__version__ = "0.1.0"

# rust/Cargo.toml
version = "0.1.0"

# pyproject.toml
version = "0.1.0"

# Keep them in sync!
```

### 28.5. Best Practice 2: Test Both Implementations

```python
def test_consistency():
    """Ensure native and pure Python match."""
    data = b"test data"

    from uubed.encoders.q64 import q64_encode as pure_encode
    from uubed._native import q64_encode_native as native_encode

    assert pure_encode(data) == native_encode(data)
```

### 28.6. Best Practice 3: Document Performance

```python
def encode(data: bytes) -> str:
    """Encode bytes to Q64 format.

    Performance:
        Pure Python: ~10 MB/s
        Native: ~100 MB/s (10x faster)

    Example:
        >>> encode(b"Hello")
        'S0V2b3g4'
    """
```

---

## 29. 🚀 Your Journey

**Week 1**: Get Python package working

- Focus on correctness, not speed
- Write comprehensive tests
- Document everything

**Week 2**: Learn Rust basics

- Do Rustlings exercises
- Implement simple Q64 encoder
- Get comfortable with cargo

**Week 3**: Build native module

- Start with simple bindings
- Add optimizations gradually
- Benchmark everything

**Week 4**: Polish & ship

- Handle edge cases
- Write user docs
- Share with the world!

Remember: **Start simple, test everything, and iterate!**

Every expert was once a beginner. You've got this! 🎉
