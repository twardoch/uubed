---
title: Zoq64 Quickstart
---

# Zoq64 — Z-order / Morton code (8 characters)

Zoq64 quantises each dimension to 2 bits, takes the first 16 dimensions, and
interleaves their bits into a 32-bit Morton code.  The result is encoded as
4 QuadB64 bytes (8 characters).

**Always produces 8 characters regardless of embedding dimension.**

The key property: vectors that are spatially close in the original embedding
space share a common *prefix* of their Zoq64 code.  This enables efficient
prefix-range scans in databases that support ordered string indices.

## Install

```bash
pip install uubed
```

## Encode

```python
import numpy as np
from uubed import encode

embedding = np.array([128, 64, 192, 32, 255, 0, 100, 200,
                      50, 150, 75, 225, 10, 180, 90, 140],
                     dtype=np.uint8)

code = encode(embedding, method="zoq64")
print(code)       # e.g. "AQghSTuv"
print(len(code))  # 8
```

## Prefix range query (Redis sorted-set example)

```python
import redis
from uubed import encode

r = redis.Redis()
embedding = np.array([...], dtype=np.uint8)
code = encode(embedding, method="zoq64")

# Store with the code as sort key
r.zadd("embeddings", {f"{code}:{doc_id}": 0})

# Retrieve all embeddings whose Zoq64 prefix matches the first 4 chars
prefix = code[:4]
members = r.zrangebylex("embeddings", f"[{prefix}", f"[{prefix}\xff")
```

## Prefix range query (SQLite example)

```python
import sqlite3
from uubed import encode

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE docs (id TEXT, zoq64 TEXT, content TEXT)")
conn.execute("CREATE INDEX idx_zoq64 ON docs (zoq64)")

embedding = np.array([...], dtype=np.uint8)
code = encode(embedding, method="zoq64")
conn.execute("INSERT INTO docs VALUES (?, ?, ?)", ("doc-1", code, "text"))

# Prefix scan — very fast on the sorted index
prefix = code[:4]
rows = conn.execute(
    "SELECT id FROM docs WHERE zoq64 BETWEEN ? AND ?",
    (prefix, prefix + "\xff"),
).fetchall()
```

## Notes

- Only the first 16 dimensions affect the code; later dimensions are ignored.
- 2-bit quantisation means each dimension is bucketed into 0–3.
- `decode()` is **not** supported (lossy).
- Zoq64 is the fastest encoder (~117× speedup over pure Python).
