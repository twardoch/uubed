#!/usr/bin/env python3
"""
Get version from git tags or fallback to default version.
This script is used to dynamically determine the version for builds.
"""
# this_file: scripts/get_version.py

import subprocess
import sys
import re
from pathlib import Path

def get_git_version():
    """Get version from git tags."""
    try:
        # Get the latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True
        )
        tag = result.stdout.strip()
        
        # Remove 'v' prefix if present
        if tag.startswith('v'):
            tag = tag[1:]
        
        # Validate semver format
        if re.match(r'^\d+\.\d+\.\d+$', tag):
            return tag
        else:
            print(f"Warning: Tag '{tag}' is not valid semver format", file=sys.stderr)
            return None
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_version_from_init():
    """Get version from research/__init__.py as fallback."""
    try:
        init_file = Path(__file__).parent.parent / "research" / "__init__.py"
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"Error reading __init__.py: {e}", file=sys.stderr)
    
    return "0.1.0"

def main():
    """Main entry point."""
    # Try to get version from git tags first
    version = get_git_version()
    
    if version is None:
        # Fallback to version in __init__.py
        version = get_version_from_init()
        print(f"Using fallback version: {version}", file=sys.stderr)
    else:
        print(f"Using git tag version: {version}", file=sys.stderr)
    
    # Output the version (this will be captured by the build system)
    print(version)

if __name__ == "__main__":
    main()