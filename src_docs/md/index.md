---
title: uubed — Position-Safe Embedding Encoding
---

# uubed

**uubed** encodes embedding vectors into strings that are safe to store in
full-text search engines without causing false-positive substring matches.

!!! warning "Two packages, one name prefix"
    `pip install uubed` installs the **Python library** (what you want).
    `pip install uubed-project` installs this **coordination hub** only
    (no encoding API). See the [README](https://github.com/twardoch/uubed).

## The problem in one line

Standard Base64 stored in Elasticsearch matches substrings of other embeddings,
producing ghost results. QuadB64 uses four position-dependent alphabets so no
two positions share a character; substring matches become structurally impossible.

## Encoding methods at a glance

| Method | Output size | Use case |
|--------|-------------|----------|
| [Eq64](quickstart-eq64.md) | 2 chars/byte | Lossless round-trip |
| [Shq64](quickstart-shq64.md) | 16 chars | Fast cosine-similarity search |
| [T8q64](quickstart-t8q64.md) | 16 chars | Sparse top-k retrieval |
| [Zoq64](quickstart-zoq64.md) | 8 chars | Spatial/prefix search |

## Installation

```bash
pip install uubed        # Python bindings + Rust core
```

## Quick example

```python
import numpy as np
from uubed import encode, decode

embedding = np.random.randint(0, 256, 256, dtype=np.uint8)

code = encode(embedding, method="eq64")
recovered = decode(code)
assert np.array_equal(embedding, np.frombuffer(recovered, dtype=np.uint8))
```

## Repository map

| Repo | Purpose |
|------|---------|
| [uubed](https://github.com/twardoch/uubed) | This hub — coordination & docs |
| [uubed-rs](https://github.com/twardoch/uubed-rs) | Rust core (crates.io: `uubed`) |
| [uubed-py](https://github.com/twardoch/uubed-py) | Python bindings (PyPI: `uubed`) |
| [uubed-docs](https://github.com/twardoch/uubed-docs) | Extended documentation book |
