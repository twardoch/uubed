---
title: Architecture
---

# Architecture

## Layer diagram

```
┌─────────────────────────────────────────────────────────┐
│  Your application                                        │
│  numpy array / list[int] / bytes                         │
└────────────────────────┬────────────────────────────────┘
                         │ encode(embedding, method=...)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  uubed-py  (PyPI: uubed)                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  api.py — encode() / decode()                    │   │
│  │  Validates input, selects encoder, normalises IO │   │
│  └──────┬──────────────────────────────────────────┘   │
│         │ tries native first, falls back to pure Python  │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │  _native  (optional, loaded at import time)      │   │
│  │  PyO3 bindings → uubed-rs                        │   │
│  └──────┬──────────────────────────────────────────┘   │
│         │ if _native unavailable                         │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │  Pure-Python fallback encoders                   │   │
│  │  encoders/q64.py  eq64  shq64  t8q64  zoq64      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  uubed-rs  (crates.io: uubed)                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  uubed-core  — const lookup tables, SIMD paths   │   │
│  │  Encoders: q64, eq64, shq64, t8q64, zoq64        │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  uubed-pyo3  — maturin wheel, #[pyfunction]      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Data flow

1. **Input normalisation** — `encode()` accepts `np.ndarray`, `list[int]`,
   or `bytes`.  Values must be in `[0, 255]`.
2. **Method dispatch** — selects `eq64`, `shq64`, `t8q64`, or `zoq64`.
3. **Native path** — PyO3 extension calls into Rust; single memcopy, no GIL
   contention for the encoding loop.
4. **Python fallback** — pure-Python encoders activate if the wheel is absent
   (e.g. source installs on unsupported platforms).
5. **Output** — a plain ASCII `str`, safe to store in any database field.

## Position-safe alphabet design

QuadB64 splits the Base64 character space into four disjoint alphabets,
one per modular position:

```
Position mod 4 = 0  →  ABCDEFGHIJKLMNOP   (16 chars)
Position mod 4 = 1  →  QRSTUVWXYZabcdef   (16 chars)
Position mod 4 = 2  →  ghijklmnopqrstuv   (16 chars)
Position mod 4 = 3  →  wxyz0123456789-_   (16 chars)
```

A character valid at position 0 is invalid at position 1, 2, or 3.
Therefore any substring of a QuadB64 string that is shorter than 4 characters
is either all from one alphabet slot or crosses a boundary — neither can appear
at the same position in a *different* embedding.  Elasticsearch full-text
tokenisation (which splits on non-word characters, spaces, etc.) cannot produce
false matches.

## Repository relationships

```
uubed  (this hub)
├── orchestrates CI across sub-repos
├── aggregates changelogs
└── hosts community templates

uubed-rs ──builds──► uubed-py ──publishes──► PyPI: uubed
                                └──docs──► uubed-docs
```
