---
title: Eq64 Quickstart
---

# Eq64 — Full-precision lossless encoding

Eq64 is the default method.  Every input byte becomes two QuadB64 characters;
a `.` is inserted every 8 characters for readability.  It is the **only**
method that supports `decode()`.

## Install

```bash
pip install uubed
```

## Encode and decode

```python
import numpy as np
from uubed import encode, decode

embedding = np.array([72, 101, 108, 108, 111, 32, 87, 111], dtype=np.uint8)

code = encode(embedding, method="eq64")
print(code)
# e.g. "HRmnQSXm"  (8 bytes × 2 chars = 16 chars, no dot needed yet)

recovered_bytes = decode(code)
recovered = np.frombuffer(recovered_bytes, dtype=np.uint8)
assert np.array_equal(embedding, recovered)
print("Round-trip OK")
```

## Output size formula

```
output_chars = input_bytes * 2 + floor(input_bytes * 2 / 8)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  one dot per 8 chars
```

For a 256-dimension `float32` embedding stored as 1 024 bytes:
`1024 * 2 + 128 = 2176 characters`.

## Store in Elasticsearch

```python
from elasticsearch import Elasticsearch
from uubed import encode

es = Elasticsearch()
embedding = np.random.randint(0, 256, 256, dtype=np.uint8)
code = encode(embedding, method="eq64")

es.index(index="embeddings", id="doc-1", body={
    "embedding_code": code,
    "text": "some document",
})

# Exact-match retrieval — no substring pollution
results = es.search(index="embeddings", body={
    "query": {"term": {"embedding_code.keyword": code}}
})
```

## Notes

- Values outside `[0, 255]` raise `ValueError`.
- The `.` separators are stripped automatically during `decode()`.
- Output is pure ASCII; safe to store in any varchar / text field.
