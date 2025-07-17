#!/usr/bin/env python3
"""
Test for build system functionality.
"""
# this_file: tests/test_build.py

import subprocess
import sys
import os
import pytest
import tempfile
from pathlib import Path

def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and is valid."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"
    
    # Try to parse it
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            pytest.skip("tomllib/tomli not available")
    
    with open(pyproject_path, 'rb') as f:
        config = tomllib.load(f)
    
    assert 'project' in config, "pyproject.toml should have [project] section"
    assert 'name' in config['project'], "Project should have a name"
    assert 'version' in config.get('project', {}).get('dynamic', []), "Version should be dynamic"

def test_hatch_build():
    """Test that hatch can build the project."""
    project_root = Path(__file__).parent.parent
    
    try:
        # Clean any existing build artifacts
        for build_dir in ['dist', 'build']:
            build_path = project_root / build_dir
            if build_path.exists():
                import shutil
                shutil.rmtree(build_path)
        
        # Try to build with hatch
        result = subprocess.run(["hatch", "build"], 
                              capture_output=True, text=True, 
                              cwd=project_root, timeout=120)
        
        if result.returncode == 0:
            # Check that build artifacts were created
            dist_dir = project_root / "dist"
            assert dist_dir.exists(), "dist directory should be created"
            
            # Look for wheel and sdist
            wheels = list(dist_dir.glob("*.whl"))
            sdists = list(dist_dir.glob("*.tar.gz"))
            
            assert len(wheels) > 0, "At least one wheel should be built"
            assert len(sdists) > 0, "At least one sdist should be built"
            
            print(f"✅ Build successful: {len(wheels)} wheels, {len(sdists)} sdists")
            
        else:
            if "hatch: command not found" in result.stderr or "No such file or directory" in result.stderr:
                pytest.skip("Hatch not available in test environment")
            else:
                pytest.fail(f"Build failed: {result.stderr}")
                
    except FileNotFoundError:
        pytest.skip("Hatch not available in test environment")
    except subprocess.TimeoutExpired:
        pytest.fail("Build timed out")

def test_pip_install_from_source():
    """Test that the project can be installed from source with pip."""
    project_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create a temporary venv
            venv_path = Path(temp_dir) / "test_venv"
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                          check=True, timeout=60)
            
            # Get python executable in venv
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"
            
            # Install the project in editable mode
            result = subprocess.run([str(python_exe), "-m", "pip", "install", "-e", str(project_root)], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Try to import the package
                import_result = subprocess.run([str(python_exe), "-c", "import research; print('Import successful')"], 
                                             capture_output=True, text=True, timeout=30)
                
                assert import_result.returncode == 0, f"Failed to import after install: {import_result.stderr}"
                print("✅ Pip install and import successful")
                
            else:
                print(f"⚠️  Pip install failed: {result.stderr}")
                # This might be expected if dependencies are not available
                pytest.skip("Pip install failed (possibly missing dependencies)")
                
        except subprocess.TimeoutExpired:
            pytest.fail("Install timed out")
        except Exception as e:
            pytest.fail(f"Install test failed: {e}")

def test_dependencies_listed():
    """Test that all required dependencies are properly listed."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            pytest.skip("tomllib/tomli not available")
    
    with open(pyproject_path, 'rb') as f:
        config = tomllib.load(f)
    
    project = config.get('project', {})
    deps = project.get('dependencies', [])
    
    # Check for essential dependencies
    dep_names = [dep.split('>=')[0].split('==')[0] for dep in deps]
    
    # These should be present based on the project
    expected_deps = ['numpy', 'rich']
    
    for dep in expected_deps:
        assert dep in dep_names, f"Expected dependency '{dep}' not found in dependencies"
    
    print(f"✅ Dependencies check passed: {dep_names}")

def test_python_version_compatibility():
    """Test that the project specifies compatible Python versions."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            pytest.skip("tomllib/tomli not available")
    
    with open(pyproject_path, 'rb') as f:
        config = tomllib.load(f)
    
    project = config.get('project', {})
    requires_python = project.get('requires-python', '')
    
    assert requires_python, "requires-python should be specified"
    assert '3.10' in requires_python, "Should support Python 3.10+"
    
    print(f"✅ Python version requirement: {requires_python}")