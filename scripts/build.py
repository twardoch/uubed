#!/usr/bin/env python3
"""
Build script for the uubed project.
This script handles building the project for different platforms and configurations.
"""
# this_file: scripts/build.py

import glob
import subprocess
import sys
import os
import argparse
import shutil
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> bool:
    """Run a command and handle errors; return True on success."""
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


def clean_build_artifacts() -> None:
    """Clean existing build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    artifacts = ['dist', 'build', '.eggs', '*.egg-info']

    for artifact in artifacts:
        if artifact.startswith('*'):
            # Handle glob patterns
            for path in glob.glob(artifact):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            path = Path(artifact)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

def check_dependencies() -> bool:
    """Check that required build dependencies are available."""
    print("🔍 Checking build dependencies...")

    required_tools = ['hatch']

    for tool in required_tools:
        if not shutil.which(tool):
            print(f"❌ Required tool '{tool}' not found in PATH")
            print(f"   Install with: pip install {tool}")
            return False

    print("✅ All required tools are available")
    return True


def build_project(release: bool = False) -> bool:
    """Build the project using hatch."""
    print("🏗️  Building project...")
    
    # Build with hatch
    cmd = ["hatch", "build"]
    
    if not run_command(cmd):
        return False
    
    # Check that artifacts were created
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory created")
        return False
    
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    
    if not wheels:
        print("❌ No wheel files created")
        return False
    
    if not sdists:
        print("❌ No source distribution created")
        return False
    
    print(f"✅ Build successful: {len(wheels)} wheels, {len(sdists)} source distributions")
    
    # List created artifacts
    print("\n📦 Created artifacts:")
    for wheel in wheels:
        print(f"   🎡 {wheel.name}")
    for sdist in sdists:
        print(f"   📄 {sdist.name}")
    
    return True

def check_version() -> str | None:
    """Check and display current version; return it or None on failure."""
    print("📏 Checking version...")

    try:
        result = subprocess.run(
            ["hatch", "version"], capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        print(f"✅ Current version: {version}")
        return version
    except subprocess.CalledProcessError:
        print("❌ Failed to get version")
        return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build the uubed project")
    parser.add_argument("--release", action="store_true", 
                       help="Build for release (optimized)")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean build artifacts before building")
    parser.add_argument("--check-deps", action="store_true", 
                       help="Only check dependencies")
    
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"🚀 Building uubed project from {project_root}")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ Dependency check completed successfully")
        return
    
    # Display version
    version = check_version()
    if not version:
        sys.exit(1)
    
    # Clean if requested
    if args.clean:
        clean_build_artifacts()
    
    # Build the project
    if not build_project(release=args.release):
        print("❌ Build failed")
        sys.exit(1)
    
    print("🎉 Build completed successfully!")

if __name__ == "__main__":
    main()