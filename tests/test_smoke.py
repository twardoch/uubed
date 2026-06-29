#!/usr/bin/env python3
"""
Smoke tests: verify the hub package can be imported and exposes a version.

These tests are intentionally minimal — they confirm that the build pipeline
produced a working installation, not that the encoding logic is correct
(that is covered by test_core_functions.py and test_edge_cases.py).
"""
# this_file: tests/test_smoke.py


def test_import_research() -> None:
    """research module imports without raising."""
    import research  # noqa: F401


def test_research_has_version() -> None:
    """research.__version__ is present."""
    import research
    assert hasattr(research, "__version__"), "research must expose __version__"


def test_research_version_is_nonempty_string() -> None:
    """research.__version__ is a non-empty string."""
    import research
    version = research.__version__
    assert isinstance(version, str), f"__version__ must be str, got {type(version)}"
    assert len(version) > 0, "__version__ must not be empty"


def test_research_has_docstring() -> None:
    """research module has a module-level docstring."""
    import research
    assert research.__doc__ is not None, "research module must have a docstring"
    assert research.__doc__.strip(), "research docstring must not be blank"
