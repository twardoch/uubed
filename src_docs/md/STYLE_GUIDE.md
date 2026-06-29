---
title: Style Guide
---

# Style Guide

Conventions for code, documentation, and commit messages in the uubed project.

## Code style

### Python (uubed-py, uubed hub)

- **Formatter:** `ruff format` (line length 88, Black-compatible).
- **Linter:** `ruff check` with the `E`, `F`, `I`, `UP`, `B` rule sets.
- **Type hints:** required on all public functions and methods.
  Use the built-in generic syntax (`list[str]`, `dict[str, int]`, `X | None`)
  — no `Optional`, `List`, `Dict` from `typing` unless targeting Python 3.9.
- **Docstrings:** one-line summary, blank line, extended explanation if needed.
  Use plain English; avoid restating the function name.
- **`this_file` comment:** every source file must contain a comment near the
  top (after shebangs) with the path relative to the project root:
  ```python
  # this_file: path/to/file.py
  ```
- **No magic numbers:** extract constants with descriptive names.
- **Function length:** aim for under 20 lines; split if longer.
- **File length:** aim for under 200 lines; split into modules if longer.

### Rust (uubed-rs)

- **Formatter:** `rustfmt` (default settings).
- **Linter:** `clippy --all-targets -- -D warnings`.
- **`this_file` comment:** `//! this_file: path/to/file.rs` in the module doc.
- Prefer `const` lookup tables over runtime computation.
- All public items need `///` doc comments.
- Use `thiserror` for library errors; `anyhow` is fine in binaries/tests.

## Documentation style

- Write for a developer who has never heard of QuadB64.
- Lead with the problem, not the solution.
- Concrete examples before abstract explanations.
- Tables before prose when comparing options.
- Keep pages under 200 lines of Markdown.
- Admonitions (`!!! note`, `!!! warning`) for things that trip people up.

## Commit message style

```
<type>(<scope>): <short summary in imperative mood>

[optional body — why, not what]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

Scopes: `hub`, `py`, `rs`, `docs`, `ci`.

Examples:
```
feat(hub): switch version source to hatch-vcs
fix(py): clamp T8q64 indices to 255 for embeddings >256 dims
docs(hub): add package-name disambiguation to README
test(hub): add smoke import tests for research module
```

## Versioning

- Semantic versioning (`MAJOR.MINOR.PATCH`).
- Tags are the single source of truth — `hatch-vcs` reads them at build time.
- Tagging order: `uubed-rs` → `uubed-py` → `uubed` (hub) → `uubed-docs`.
- The hub version tracks `uubed-py` (the user-facing package).

## File structure conventions

```
project-root/
├── research/          # Research prototypes (part of uubed-project wheel)
├── scripts/           # Developer utilities (not shipped)
├── tests/             # pytest test suite
├── src_docs/          # MkDocs source (md/ + mkdocs.yaml)
│   └── md/            # Markdown pages
├── docs/              # Built MkDocs output (git-ignored or gh-pages)
├── examples/          # Runnable usage examples (also serve as smoke tests)
├── pyproject.toml     # Single build config; version from hatch-vcs
└── CHANGELOG.md       # Keep a Changelog format
```

## Testing

- `uvx hatch test` — canonical test command; runs pytest in an isolated env.
- Every public function needs at least one test.
- Test naming: `test_<function>_when_<condition>_then_<result>`.
- Include helpful assertion messages: `assert x == y, f"expected {y}, got {x}"`.
- Tests that require external services (APIs, network) must be skipped
  gracefully when the service is unavailable.
