# uubed Glossary

A living glossary of terms, concepts, and technical definitions used throughout the uubed project.

## Core Concepts

### QuadB64
A family of position-safe Base64 encoding variants developed for uubed. Unlike standard Base64, QuadB64 encodings preserve locality and avoid substring pollution in search engines.

### Substring Pollution
When encoded data fragments match partial search queries, leading to false positives. Standard Base64 is especially prone to this.

### Position-Safe Encoding
An encoding scheme where character position affects value, preventing arbitrary substrings from being valid encodings.

## Encoding Variants

### Eq64 (Embeddings QuadB64)
Primary encoding for full embedding vectors. Uses dot separators every 4 characters to ensure position safety. Optimized for dense vector representations.

### Shq64 (SimHash QuadB64)
For SimHash fingerprints and binary hashes. Compact while keeping search-friendly properties.

### T8q64 (Top-k QuadB64)
Encodes only the indices and values of the k largest components in sparse vectors.

### Zoq64 (Z-order QuadB64)
Uses Z-order (Morton) encoding for spatial data and multi-dimensional indices. Preserves locality across dimensions.

## Technical Terms

### SIMD (Single Instruction, Multiple Data)
Parallel processing used in uubed to speed up encoding/decoding. Supports AVX2, AVX-512, and NEON.

### FFI (Foreign Function Interface)
Interface allowing the Rust core to be called from other languages—mainly Python via PyO3.

### Locality Preservation
Similar inputs produce similar outputs, maintaining neighborhood relationships.

### Matryoshka Embeddings
Nested embeddings where earlier dimensions carry more important information. Future uubed versions will support specialized encoding for these.

## Performance Terms

### Vectorization
Converting sequential operations into parallel SIMD operations for better performance.

### Zero-Copy Operations
Processing data without unnecessary memory allocations or copies—key for high-performance encoding.

### Batch Processing
Handling multiple vectors at once to reduce overhead and boost throughput.

## Implementation Details

### PyO3
Rust library used to create Python bindings for uubed-rs.

### Base Encoding Alphabet
Standard 64-character set: `A-Z`, `a-z`, `0-9`, `+`, `/`, with `=` for padding.

### Chunk Size
Number of bytes processed per unit during encoding. QuadB64 uses 4-character chunks with position markers.

## Search and Retrieval

### Vector Database
Databases like Pinecone, Weaviate, and Qdrant optimized for storing and searching high-dimensional vectors. Main use case for uubed encodings.

### Cosine Similarity
Standard metric for comparing embedding vectors, preserved by uubed's locality-preserving encodings.

### Approximate Nearest Neighbor (ANN)
Efficient search algorithms that find similar vectors, aided by uubed's locality preservation.

---

*Updated as the project evolves. Contributions welcome via pull requests.*