---
title: Shq64 Quickstart
---

# Shq64 — SimHash (16 characters)

Shq64 projects the embedding through 64 random hyperplanes (seeded with 42
for reproducibility), takes the sign of each projection, and packs the
resulting 64-bit hash into 16 QuadB64 characters.

**Always produces 16 characters regardless of embedding dimension.**

## Install

```bash
pip install uubed
```

## Encode

```python
import numpy as np
from uubed import encode

embedding = np.random.randint(0, 256, 256, dtype=np.uint8)
code = encode(embedding, method="shq64")

print(code)          # e.g. "AQghSTuvwxyz0123"
print(len(code))     # 16
```

## Similarity search (Pinecone example)

```python
import pinecone
from uubed import encode

index = pinecone.Index("my-index")
embedding = np.array([...], dtype=np.uint8)
code = encode(embedding, method="shq64")

# Store the compact hash as metadata
index.upsert([{
    "id": "doc-42",
    "values": embedding.tolist(),
    "metadata": {"shq64": code},
}])

# Pre-filter by approximate hash match before vector search
results = index.query(
    vector=query_embedding.tolist(),
    filter={"shq64": {"$eq": code}},
    top_k=10,
)
```

## Similarity property

Two embeddings with high cosine similarity produce Shq64 codes that differ
in fewer bit positions (Hamming distance).  You can use Hamming distance on
the codes as a fast pre-filter before computing exact cosine similarity.

```python
def hamming(a: str, b: str) -> int:
    """Approximate similarity: lower = more similar."""
    assert len(a) == len(b) == 16
    return sum(x != y for x, y in zip(a, b))
```

## Notes

- `decode()` is **not** supported for Shq64 (lossy compression).
- The random projection matrix is fixed (seed 42), so codes are
  deterministic and comparable across calls and processes.
- Dimensionality does not affect output size.
