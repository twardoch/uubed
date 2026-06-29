---
title: Encoding Methods — Comparison
---

# Encoding Methods

All four methods encode `uint8` embedding vectors into plain ASCII strings using
the QuadB64 positional-alphabet scheme.  Choose based on whether you need
lossless round-trips, similarity preservation, sparsity, or spatial locality.

## At a glance

| Method | Output chars | Lossless? | Similarity | Use case |
|--------|-------------|-----------|------------|----------|
| **Eq64** | `2 × bytes` | Yes | Exact | Store and retrieve full embeddings |
| **Shq64** | 16 | No (projection) | Cosine ≈ | Fast approximate nearest-neighbour |
| **T8q64** | 16 | No (top-k) | Top dimensions | Sparse keyword-style retrieval |
| **Zoq64** | 8 | No (quantised) | Spatial prefix | Range / prefix search |

## Performance (Apple M1 Pro, 1 KB embedding)

| Method | Pure Python | Rust native | Speedup |
|--------|-------------|-------------|---------|
| Eq64 | 465 µs | 4.4 µs | ~105× |
| Shq64 | 1 431 µs | 140 µs | ~10× |
| T8q64 | 892 µs | 42 µs | ~21× |
| Zoq64 | 74 µs | 0.6 µs | ~117× |

## Eq64 — Full precision

Encodes every byte as two QuadB64 characters with a `.` separator every 8
characters for readability.  The only method that supports `decode()`.

```
input : [72, 101, 108, 108, 111]   # "Hello"
output: "HRmnQS.Xm..."
```

**When to use:** You need the exact original vector back, or you are indexing
by exact match only (not substring or similarity).

## Shq64 — SimHash (16 chars)

Projects the embedding through 64 random hyperplanes (fixed seed 42), takes the
sign of each dot product, and encodes the resulting 64-bit integer with QuadB64.

```
input : any float/uint8 vector
output: "AQghSTuvwx012345"   # always 16 chars
```

**When to use:** Cosine-similarity approximate search.  Two embeddings with high
cosine similarity will produce similar (not equal) Shq64 codes.

## T8q64 — Top-k indices (16 chars)

Finds the 8 dimensions with the largest values and encodes their indices as
8 bytes (padded with 255 if fewer than 8 non-zero dims).

```
input : sparse vector [0, 0, 255, 0, 128, ...]
output: "AgQhSTuv.wxyz012"   # 8 bytes × 2 chars
```

**When to use:** Sparse embeddings where only a handful of dimensions carry
signal.  Efficient for BM25-style retrieval in hybrid search pipelines.

## Zoq64 — Z-order / Morton code (8 chars)

Quantises each dimension to 2 bits, takes the first 16 dimensions, and
interleaves their bits into a 32-bit Morton code encoded as 4 QuadB64 bytes.

```
input : [128, 64, 192, 32, ...]
output: "AQghSTuv"   # always 8 chars
```

**When to use:** Spatial or range queries.  Nearby vectors in embedding space
share a common prefix of their Zoq64 codes, enabling efficient prefix scans.

## Choosing a method

```
Need exact reconstruction?          → eq64
Cosine-similarity ANN search?       → shq64
Sparse / keyword-style retrieval?   → t8q64
Spatial range or prefix queries?    → zoq64
Not sure?                           → eq64 (safest default)
```
