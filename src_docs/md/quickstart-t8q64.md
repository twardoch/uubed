---
title: T8q64 Quickstart
---

# T8q64 — Top-k indices (16 characters)

T8q64 finds the 8 dimensions with the highest values and encodes their indices
as 8 bytes using QuadB64.  If there are fewer than 8 non-zero dimensions the
remainder is padded with byte value 255.

**Always produces 16 characters (8 bytes × 2 chars).**

## Install

```bash
pip install uubed
```

## Encode

```python
import numpy as np
from uubed import encode

# Sparse embedding: only a few dimensions are active
embedding = np.zeros(256, dtype=np.uint8)
embedding[[3, 17, 42, 99]] = [200, 180, 150, 210]

code = encode(embedding, method="t8q64")
print(code)       # e.g. "AgQhSTuv.wxyz012"
print(len(code))  # 16
```

## Hybrid search (BM25 + vector)

```python
from opensearchpy import OpenSearch
from uubed import encode

client = OpenSearch()
embedding = np.array([...], dtype=np.uint8)
code = encode(embedding, method="t8q64")

# Index
client.index(index="docs", body={
    "text": "document content",
    "t8q64": code,
})

# Keyword-style filter using top-k overlap
# (works because top dimensions rarely overlap between unrelated embeddings)
client.search(index="docs", body={
    "query": {
        "bool": {
            "should": [
                {"match": {"text": "query terms"}},
                {"term": {"t8q64.keyword": code}},
            ]
        }
    }
})
```

## Choosing k

The default is `k=8`.  Larger k improves recall at the cost of a longer code
(`k * 2` characters).  For embeddings with more than 256 dimensions, indices
are clamped to 255 to fit in a single byte.

## Notes

- `decode()` is **not** supported for T8q64 (lossy).
- Useful as a fast pre-filter in hybrid retrieval pipelines.
- Indices are sorted descending by value before encoding.
