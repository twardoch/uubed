"""
uubed research module - contains research code and prototypes for position-safe embedding encoding.
"""
# this_file: research/__init__.py

try:
    from ._version import __version__
except ImportError:
    # Fallback when package is not built via hatch-vcs (e.g. running from source)
    __version__ = "1.0.5"