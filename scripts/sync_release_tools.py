#!/usr/bin/env python3
# this_file: scripts/sync_release_tools.py
"""Copy the canonical release helpers into sibling repositories."""

import argparse
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="uubed-project checkout directory")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    for repo in (
        "uubed-all/uubed-docs",
        "uubed-all/uubed-py",
        "uubed-all/uubed-rs",
        "vexy-paraltext",
        "abersetz",
    ):
        for name in ("release.py", "release_artifacts.py"):
            target = args.workspace / repo / "scripts" / name
            if args.check:
                if target.read_bytes() != (source / name).read_bytes():
                    raise SystemExit(f"Release helper differs: {target}")
            else:
                shutil.copy2(source / name, target)
