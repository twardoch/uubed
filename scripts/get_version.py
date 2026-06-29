#!/usr/bin/env python3
"""
Get version from git tags or fallback to research/__init__.py.

Used as a standalone utility; NOT the hatchling build-time version source
(that role belongs to hatch-vcs / [tool.hatch.version] source = "vcs").
"""
# this_file: scripts/get_version.py

import os
import re
import subprocess
import sys
from pathlib import Path


def get_git_version() -> str | None:
    """Return the most recent git tag as a semver string, or None."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        tag = result.stdout.strip()
        if tag.startswith("v"):
            tag = tag[1:]
        if re.match(r"^\d+\.\d+\.\d+", tag):
            return tag
        print(f"Warning: tag '{tag}' is not semver", file=sys.stderr)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_version_from_init() -> str | None:
    """
    Read __version__ from research/__init__.py.

    Searches in two locations so the script works both from its installed
    position (scripts/get_version.py, i.e. one directory below project root)
    and when copied directly into a directory alongside research/ (e.g. in
    test_version_fallback).
    """
    script_dir = Path(__file__).parent
    candidates: list[Path] = [
        # Script sitting next to research/ (test / copied scenario)
        script_dir / "research" / "__init__.py",
        # Script in scripts/ subdirectory (normal installed layout)
        script_dir.parent / "research" / "__init__.py",
        # cwd-relative fallback
        Path(os.getcwd()) / "research" / "__init__.py",
    ]
    for init_file in candidates:
        if init_file.exists():
            try:
                content = init_file.read_text(encoding="utf-8")
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
            except OSError as exc:
                print(f"Error reading {init_file}: {exc}", file=sys.stderr)
    return None


def main() -> None:
    """Print the project version to stdout."""
    version = get_git_version()
    if version is None:
        version = get_version_from_init()
        if version is None:
            version = "0.1.0"
        print(f"Using fallback version: {version}", file=sys.stderr)
    else:
        print(f"Using git tag version: {version}", file=sys.stderr)
    print(version)


if __name__ == "__main__":
    main()
