#!/usr/bin/env python3
"""
Test script for the uubed project.
This script runs the test suite with various options.
"""
# this_file: scripts/test.py

import subprocess
import sys
import os
import argparse
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors."""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return False

def check_test_dependencies():
    """Check that required test dependencies are available."""
    print("🔍 Checking test dependencies...")
    
    # Check for pytest
    try:
        import pytest
        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not found")
        print("   Install with: pip install pytest")
        return False
    
    # Check for hatch (optional, for running tests in environments)
    if shutil.which('hatch'):
        print("✅ hatch available for environment management")
    else:
        print("⚠️  hatch not found (optional for advanced testing)")
    
    return True

def run_tests(coverage=False, verbose=False, pattern=None):
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=research", "--cov-report=term-missing"])
    
    if pattern:
        cmd.extend(["-k", pattern])
    
    # Add test directory
    cmd.append("tests")
    
    return run_command(cmd)

def run_tests_with_hatch(coverage=False, verbose=False, pattern=None):
    """Run tests using hatch environments."""
    print("🧪 Running tests with hatch...")
    
    # Build hatch command
    cmd = ["hatch", "run"]
    
    if coverage:
        cmd.append("test-cov")
    else:
        cmd.append("test")
    
    # Add pytest args
    if verbose:
        cmd.append("-v")
    
    if pattern:
        cmd.extend(["-k", pattern])
    
    return run_command(cmd)

def lint_code():
    """Run code linting."""
    print("🔍 Running code linting...")
    
    linters = [
        # Try different linters in order of preference
        ["python", "-m", "ruff", "check", "."],
        ["python", "-m", "flake8", "."],
        ["python", "-m", "pylint", "research"],
    ]
    
    for linter in linters:
        try:
            # Check if linter is available
            result = subprocess.run([linter[0], "-m", linter[1].split()[1], "--help"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Using {linter[1]} for linting")
                return run_command(linter, check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("⚠️  No linter found - install ruff, flake8, or pylint")
    return True  # Don't fail if no linter is available

def format_code():
    """Run code formatting."""
    print("🎨 Running code formatting...")
    
    formatters = [
        ["python", "-m", "ruff", "format", "."],
        ["python", "-m", "black", "."],
    ]
    
    for formatter in formatters:
        try:
            # Check if formatter is available
            result = subprocess.run([formatter[0], "-m", formatter[1].split()[1], "--help"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Using {formatter[1]} for formatting")
                return run_command(formatter, check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("⚠️  No formatter found - install ruff or black")
    return True  # Don't fail if no formatter is available

def type_check():
    """Run type checking."""
    print("🔍 Running type checking...")
    
    type_checkers = [
        ["python", "-m", "mypy", "research"],
        ["python", "-m", "pyright", "research"],
    ]
    
    for checker in type_checkers:
        try:
            # Check if type checker is available
            result = subprocess.run([checker[0], "-m", checker[1].split()[1], "--help"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Using {checker[1]} for type checking")
                return run_command(checker, check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("⚠️  No type checker found - install mypy or pyright")
    return True  # Don't fail if no type checker is available

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests for the uubed project")
    parser.add_argument("--coverage", action="store_true", 
                       help="Run tests with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Run tests in verbose mode")
    parser.add_argument("--pattern", "-k", 
                       help="Run only tests matching this pattern")
    parser.add_argument("--use-hatch", action="store_true", 
                       help="Use hatch to run tests in managed environment")
    parser.add_argument("--lint", action="store_true", 
                       help="Run linting")
    parser.add_argument("--format", action="store_true", 
                       help="Run code formatting")
    parser.add_argument("--type-check", action="store_true", 
                       help="Run type checking")
    parser.add_argument("--all", action="store_true", 
                       help="Run all checks (tests, lint, format, type-check)")
    
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"🧪 Testing uubed project from {project_root}")
    
    # Check dependencies first
    if not check_test_dependencies():
        sys.exit(1)
    
    success = True
    
    # Run tests
    if not any([args.lint, args.format, args.type_check]) or args.all:
        if args.use_hatch and shutil.which('hatch'):
            success &= run_tests_with_hatch(coverage=args.coverage, 
                                           verbose=args.verbose, 
                                           pattern=args.pattern)
        else:
            success &= run_tests(coverage=args.coverage, 
                               verbose=args.verbose, 
                               pattern=args.pattern)
    
    # Run linting
    if args.lint or args.all:
        success &= lint_code()
    
    # Run formatting
    if args.format or args.all:
        success &= format_code()
    
    # Run type checking
    if args.type_check or args.all:
        success &= type_check()
    
    if success:
        print("🎉 All tests and checks passed!")
    else:
        print("❌ Some tests or checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()