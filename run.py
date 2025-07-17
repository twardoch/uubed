#!/usr/bin/env python3
"""
Convenience runner script for uubed project.
This provides a simple interface to common development tasks.
"""
# this_file: run.py

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convenience runner for uubed project development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  test        - Run the test suite
  test-all    - Run tests with linting and formatting
  build       - Build the project
  clean       - Clean build artifacts
  install     - Install the project in development mode
  dev         - Set up development environment
  lint        - Run code linting
  format      - Format code
  type-check  - Run type checking
  release     - Create a release (dry run)
  version     - Show current version
  ci          - Run CI-like checks (test-all + build)
        """
    )
    
    parser.add_argument("command", help="Command to run")
    parser.add_argument("args", nargs="*", help="Additional arguments")
    
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent
    
    # Map commands to their implementations
    commands = {
        "test": ["python", "scripts/test.py"],
        "test-all": ["python", "scripts/test.py", "--all"],
        "test-coverage": ["python", "scripts/test.py", "--coverage"],
        "build": ["python", "scripts/build.py"],
        "build-clean": ["python", "scripts/build.py", "--clean"],
        "build-release": ["python", "scripts/build.py", "--release", "--clean"],
        "clean": ["python", "-c", """
import shutil
import os
from pathlib import Path

# Remove build artifacts
for pattern in ['dist/', 'build/', '*.egg-info/']:
    for path in Path('.').glob(pattern):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

# Remove Python cache
for root, dirs, files in os.walk('.'):
    for d in dirs[:]:
        if d == '__pycache__':
            shutil.rmtree(os.path.join(root, d))
            dirs.remove(d)
    for f in files:
        if f.endswith('.pyc'):
            os.remove(os.path.join(root, f))

print('🧹 Cleaned build artifacts')
        """],
        "install": ["pip", "install", "-e", "."],
        "dev": ["python", "-c", """
import subprocess
import sys

commands = [
    [sys.executable, "-m", "pip", "install", "-e", "."],
    [sys.executable, "-m", "pip", "install", "pytest", "pytest-cov", "ruff", "black", "mypy"]
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print('✅ Development environment set up')
        """],
        "lint": ["python", "scripts/test.py", "--lint"],
        "format": ["python", "scripts/test.py", "--format"],
        "type-check": ["python", "scripts/test.py", "--type-check"],
        "release": ["python", "scripts/release.py", "--dry-run"],
        "release-real": ["python", "scripts/release.py"],
        "version": ["python", "scripts/get_version.py"],
        "check-deps": ["python", "scripts/build.py", "--check-deps"],
        "ci": ["python", "-c", """
import subprocess
import sys

commands = [
    [sys.executable, "scripts/test.py", "--all"],
    [sys.executable, "scripts/build.py", "--clean"]
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
        sys.exit(1)

print('✅ CI checks passed')
        """]
    }
    
    if args.command not in commands:
        print(f"❌ Unknown command: {args.command}")
        print("Available commands:")
        for cmd in commands:
            print(f"  {cmd}")
        sys.exit(1)
    
    # Build command with additional args
    cmd = commands[args.command] + args.args
    
    # Run the command
    success = run_command(cmd, cwd=script_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()