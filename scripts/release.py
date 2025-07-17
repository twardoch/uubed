#!/usr/bin/env python3
"""
Release script for the uubed project.
This script handles the complete release process including building, testing, and publishing.
"""
# this_file: scripts/release.py

import subprocess
import sys
import os
import argparse
import shutil
import re
from pathlib import Path

def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True)
        if capture_output:
            return result.stdout.strip() if result.returncode == 0 else None
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False if not capture_output else None
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return False if not capture_output else None

def check_git_status():
    """Check that git working directory is clean."""
    print("🔍 Checking git status...")
    
    # Check if we're in a git repository
    if not run_command(["git", "rev-parse", "--git-dir"], capture_output=True):
        print("❌ Not in a git repository")
        return False
    
    # Check for uncommitted changes
    status = run_command(["git", "status", "--porcelain"], capture_output=True)
    if status:
        print("❌ Working directory has uncommitted changes:")
        print(status)
        return False
    
    print("✅ Git working directory is clean")
    return True

def get_current_version():
    """Get the current version from hatch."""
    print("📏 Getting current version...")
    
    version = run_command(["hatch", "version"], capture_output=True)
    if version:
        print(f"✅ Current version: {version}")
        return version
    else:
        print("❌ Failed to get current version")
        return None

def validate_version(version):
    """Validate that version follows semantic versioning."""
    if not re.match(r'^\d+\.\d+\.\d+$', version):
        print(f"❌ Version '{version}' does not follow semantic versioning (X.Y.Z)")
        return False
    
    print(f"✅ Version '{version}' is valid")
    return True

def check_tag_exists(tag):
    """Check if a git tag already exists."""
    result = run_command(["git", "tag", "-l", tag], capture_output=True)
    if result:
        print(f"❌ Tag '{tag}' already exists")
        return True
    return False

def create_git_tag(version, dry_run=False):
    """Create a git tag for the version."""
    tag = f"v{version}"
    
    if check_tag_exists(tag):
        return False
    
    print(f"🏷️  Creating git tag: {tag}")
    
    if dry_run:
        print("   (dry run - not actually creating tag)")
        return True
    
    # Create annotated tag
    if not run_command(["git", "tag", "-a", tag, "-m", f"Release {version}"]):
        return False
    
    print(f"✅ Created tag: {tag}")
    return True

def push_tag(version, dry_run=False):
    """Push the git tag to remote."""
    tag = f"v{version}"
    
    print(f"📤 Pushing tag to remote: {tag}")
    
    if dry_run:
        print("   (dry run - not actually pushing)")
        return True
    
    return run_command(["git", "push", "origin", tag])

def run_tests():
    """Run the test suite before release."""
    print("🧪 Running tests before release...")
    
    # Use our test script
    test_script = Path(__file__).parent / "test.py"
    
    return run_command([sys.executable, str(test_script), "--all"])

def build_release():
    """Build the release artifacts."""
    print("🏗️  Building release artifacts...")
    
    # Use our build script
    build_script = Path(__file__).parent / "build.py"
    
    return run_command([sys.executable, str(build_script), "--release", "--clean"])

def check_release_artifacts():
    """Check that release artifacts were created properly."""
    print("📦 Checking release artifacts...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory found")
        return False
    
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    
    if not wheels:
        print("❌ No wheel files found")
        return False
    
    if not sdists:
        print("❌ No source distribution found")
        return False
    
    print(f"✅ Found {len(wheels)} wheels and {len(sdists)} source distributions")
    return True

def publish_to_pypi(dry_run=False):
    """Publish to PyPI using twine."""
    print("📤 Publishing to PyPI...")
    
    if dry_run:
        print("   (dry run - not actually publishing)")
        return True
    
    # Check if twine is available
    if not shutil.which('twine'):
        print("❌ twine not found. Install with: pip install twine")
        return False
    
    # Upload to PyPI
    cmd = ["twine", "upload", "dist/*"]
    
    print("⚠️  This will publish to PyPI. Make sure you have:")
    print("   - Set up your PyPI credentials (~/.pypirc or environment variables)")
    print("   - Verified the package contents")
    
    confirm = input("Continue with PyPI upload? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Upload cancelled by user")
        return False
    
    return run_command(cmd)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Release the uubed project")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Perform a dry run without making changes")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests")
    parser.add_argument("--skip-build", action="store_true", 
                       help="Skip building (use existing artifacts)")
    parser.add_argument("--skip-publish", action="store_true", 
                       help="Skip publishing to PyPI")
    parser.add_argument("--version", 
                       help="Override version (otherwise use current version)")
    
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"🚀 Starting release process for uubed project")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    # Check git status
    if not check_git_status():
        sys.exit(1)
    
    # Get version
    if args.version:
        version = args.version
        if not validate_version(version):
            sys.exit(1)
    else:
        version = get_current_version()
        if not version:
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("❌ Tests failed - aborting release")
            sys.exit(1)
    
    # Build release
    if not args.skip_build:
        if not build_release():
            print("❌ Build failed - aborting release")
            sys.exit(1)
    
    # Check artifacts
    if not check_release_artifacts():
        print("❌ Release artifacts check failed - aborting release")
        sys.exit(1)
    
    # Create git tag
    if not create_git_tag(version, dry_run=args.dry_run):
        print("❌ Failed to create git tag - aborting release")
        sys.exit(1)
    
    # Push tag
    if not push_tag(version, dry_run=args.dry_run):
        print("❌ Failed to push git tag - aborting release")
        sys.exit(1)
    
    # Publish to PyPI
    if not args.skip_publish:
        if not publish_to_pypi(dry_run=args.dry_run):
            print("❌ Failed to publish to PyPI")
            sys.exit(1)
    
    print("🎉 Release process completed successfully!")
    print(f"   Version: {version}")
    print(f"   Tag: v{version}")
    
    if not args.dry_run:
        print("   🚀 Package is now available on PyPI")
        print("   🏷️  Git tag has been pushed to remote")

if __name__ == "__main__":
    main()