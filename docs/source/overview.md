# Overview

## What is uubed?

uubed (pronounced "you-you-bed") encodes embedding vectors into position-safe strings. It fixes substring pollution in search systems.

## The Problem: Substring Pollution

Base64 embedding strings can cause false matches in search engines:

```python
# Traditional encoding creates substring pollution
embedding1 = "dGhlIHF1aWNr..."  # "the quick brown fox"
embedding2 = "YnJvd24gZm94..."  # "brown fox jumps"

# Searching for embedding2 may incorrectly match embedding1
```

## The Solution: Position-Safe Encoding

uubed uses QuadB64 encoding with position-dependent alphabets:

- **Position 0**: A-Z
- **Position 1**: a-z  
- **Position 2**: Mixed case
- **Position 3**: Digits and symbols

This prevents one encoded string from being a substring of another.

## Encoding Methods

uubed offers four methods for different scenarios:

### 1. Eq64 (Full Precision)
- Preserves all embedding data
- Use for exact similarity search
- ~71 characters per 32 dimensions

### 2. Shq64 (SimHash)
- Locality-sensitive hashing
- Use for approximate nearest neighbor search
- 16 characters

### 3. T8q64 (Top-k)
- Encodes top-8 feature indices
- Use for sparse embeddings or feature analysis
- 16 characters

### 4. Zoq64 (Z-order)
- Spatial encoding via Morton codes
- Use for multi-dimensional range queries
- 8 characters

## Performance

Native Rust implementation delivers:

- 40-105x faster full precision encoding
- >230 MB/s throughput on modern hardware
- 60-1600x faster Z-order encoding
- Minimal memory overhead

## Use Cases

- **Vector Databases**: Store embeddings as searchable strings
- **Search Engines**: Index embeddings without pollution
- **Caching Systems**: Use encoded strings as cache keys
- **Data Pipelines**: Serialize embeddings efficiently