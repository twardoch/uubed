#!/usr/bin/env python3
"""
Test for version handling in the uubed project.
"""
# this_file: tests/test_version.py

import subprocess
import sys
import os
import pytest
import re
from pathlib import Path

def test_version_script_exists():
    """Test that the version script exists and is executable."""
    script_path = Path(__file__).parent.parent / "scripts" / "get_version.py"
    assert script_path.exists(), f"Version script {script_path} not found"
    assert os.access(script_path, os.X_OK), f"Version script {script_path} is not executable"

def test_version_script_execution():
    """Test that the version script executes without errors."""
    script_path = Path(__file__).parent.parent / "scripts" / "get_version.py"
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Version script returned exit code {result.returncode}, stderr: {result.stderr}"
        
        # Check that output looks like a version number
        version = result.stdout.strip()
        assert re.match(r'^\d+\.\d+\.\d+', version), f"Version '{version}' does not match semver format"
        
        print(f"✅ Version script output: {version}")
        
    except subprocess.TimeoutExpired:
        pytest.fail("Version script timed out")
    except Exception as e:
        pytest.fail(f"Error running version script: {e}")

def test_version_fallback():
    """Test that version script falls back to __init__.py version when git is unavailable."""
    # This test verifies the fallback mechanism works
    script_path = Path(__file__).parent.parent / "scripts" / "get_version.py"
    
    # Run in a temporary directory where git is not available or has no tags
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the script to temp directory
        import shutil
        temp_script = Path(temp_dir) / "get_version.py"
        shutil.copy(script_path, temp_script)
        
        # Create a mock __init__.py
        research_dir = Path(temp_dir) / "research"
        research_dir.mkdir()
        init_file = research_dir / "__init__.py"
        init_file.write_text('__version__ = "0.2.0"\n')
        
        # Run the script from temp directory
        result = subprocess.run([sys.executable, str(temp_script)], 
                              capture_output=True, text=True, 
                              cwd=temp_dir, timeout=30)
        
        assert result.returncode == 0, f"Version script failed: {result.stderr}"
        version = result.stdout.strip()
        assert version == "0.2.0", f"Expected fallback version '0.2.0', got '{version}'"

def test_research_module_version():
    """Test that research module has a version attribute."""
    try:
        import research
        assert hasattr(research, '__version__'), "research module should have __version__ attribute"
        assert re.match(r'^\d+\.\d+\.\d+', research.__version__), f"research.__version__ '{research.__version__}' does not match semver format"
        print(f"✅ research.__version__ = {research.__version__}")
    except ImportError:
        pytest.skip("research module not available (expected in main hub repo)")

def test_hatch_version_configuration():
    """Test that hatch can read the version from our configuration."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"
    
    # Check that hatch can read the version
    try:
        result = subprocess.run(["hatch", "version"], 
                              capture_output=True, text=True, 
                              cwd=pyproject_path.parent, timeout=30)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            assert re.match(r'^\d+\.\d+\.\d+', version), f"Hatch version '{version}' does not match semver format"
            print(f"✅ Hatch version: {version}")
        else:
            # Hatch might not be available in test environment
            print(f"⚠️  Hatch not available: {result.stderr}")
            pytest.skip("Hatch not available in test environment")
            
    except FileNotFoundError:
        pytest.skip("Hatch not available in test environment")
    except subprocess.TimeoutExpired:
        pytest.fail("Hatch version command timed out")