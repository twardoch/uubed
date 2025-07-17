#!/usr/bin/env python3
"""
Test for research module initialization and version info.
"""

import os
import sys

# Add the research directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research'))

def test_research_module_import():
    """Test that the research module can be imported."""
    try:
        import research
        assert hasattr(research, '__version__')
        assert isinstance(research.__version__, str)
        print(f"✅ PASS: research module version: {research.__version__}")
    except ImportError as e:
        assert False, f"Failed to import research module: {e}"

def test_research_module_version():
    """Test that the research module has a valid version."""
    try:
        import research
        version = research.__version__
        # Check that version follows semantic versioning pattern
        parts = version.split('.')
        assert len(parts) >= 2, f"Version should have at least 2 parts, got: {version}"
        assert all(part.isdigit() or part.replace('-', '').replace('+', '').isalnum() 
                  for part in parts), f"Invalid version format: {version}"
        print(f"✅ PASS: Valid version format: {version}")
    except ImportError as e:
        assert False, f"Failed to import research module: {e}"

def test_research_module_docstring():
    """Test that the research module has a docstring."""
    try:
        import research
        assert research.__doc__ is not None
        assert len(research.__doc__.strip()) > 0
        print(f"✅ PASS: research module has docstring: {research.__doc__.strip()}")
    except ImportError as e:
        assert False, f"Failed to import research module: {e}"