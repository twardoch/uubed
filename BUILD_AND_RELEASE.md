# Build and Release Guide

This document describes the build and release process for the uubed project, including git-tag-based semversioning, testing, and multiplatform releases.

## Overview

The project uses a git-tag-based semversioning system where:
- Version numbers are determined by git tags (e.g., `v1.2.3`)
- The build system automatically detects the version from the latest git tag
- Releases are triggered by pushing a git tag to the repository
- All builds are tested on multiple platforms (Linux, Windows, macOS) and Python versions (3.10, 3.11, 3.12)

## Local Development

### Prerequisites

- Python 3.10 or higher
- Git
- [Hatch](https://hatch.pypa.io/) for build management

Install Hatch:
```bash
pip install hatch
```

### Local Scripts

The project includes several convenience scripts in the `scripts/` directory:

#### `scripts/get_version.py`
Determines the current version from git tags or falls back to the version in `research/__init__.py`.

```bash
python scripts/get_version.py
```

#### `scripts/test.py`
Runs the comprehensive test suite with various options.

```bash
# Run all tests
python scripts/test.py

# Run tests with coverage
python scripts/test.py --coverage

# Run tests with linting and formatting
python scripts/test.py --all

# Run specific test pattern
python scripts/test.py --pattern test_version

# Use hatch environment
python scripts/test.py --use-hatch
```

#### `scripts/build.py`
Builds the project with different configurations.

```bash
# Standard build
python scripts/build.py

# Release build with cleanup
python scripts/build.py --release --clean

# Check dependencies only
python scripts/build.py --check-deps
```

#### `scripts/release.py`
Manages the complete release process.

```bash
# Dry run (recommended first)
python scripts/release.py --dry-run

# Full release
python scripts/release.py

# Release with custom version
python scripts/release.py --version 1.2.3

# Skip certain steps
python scripts/release.py --skip-tests --skip-publish
```

### Development Workflow

1. **Make changes** to the codebase
2. **Run tests** to ensure everything works:
   ```bash
   python scripts/test.py --all
   ```
3. **Build locally** to verify:
   ```bash
   python scripts/build.py --clean
   ```
4. **Commit and push** changes
5. **Create a release** (see release process below)

## Release Process

### 1. Prepare for Release

1. Ensure all changes are committed and pushed
2. Verify tests pass locally:
   ```bash
   python scripts/test.py --all
   ```
3. Update version in `research/__init__.py` if needed
4. Run a dry-run release to check everything:
   ```bash
   python scripts/release.py --dry-run
   ```

### 2. Create Release

1. **Tag the release**:
   ```bash
   git tag -a v1.2.3 -m "Release 1.2.3"
   git push origin v1.2.3
   ```

2. **Or use the release script**:
   ```bash
   python scripts/release.py
   ```

### 3. Automated Release Process

When a tag is pushed, GitHub Actions will automatically:

1. **Build on multiple platforms** (Linux, Windows, macOS)
2. **Test on multiple Python versions** (3.10, 3.11, 3.12)
3. **Verify version consistency** between tag and package
4. **Create wheel and source distributions**
5. **Run the full test suite**
6. **Create a GitHub release** with artifacts
7. **Publish to PyPI** (if configured)

## GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)
- Runs on every push to `main` and PRs
- Tests on multiple OS and Python versions
- Runs comprehensive test suite
- Builds project to verify it works

### Release Workflow (`.github/workflows/release.yml`)
- Triggered by git tags (`v*`)
- Builds on multiple platforms
- Verifies version consistency
- Creates GitHub releases
- Publishes to PyPI

### Build Wheels Workflow (`.github/workflows/build-wheels.yml`)
- Builds wheels for all platforms
- Can be triggered manually
- Tests wheel installation
- Collects all artifacts

## Version Management

### Git Tag-Based Versioning

The project uses git tags to determine versions:

1. **Latest tag**: The version is extracted from the latest git tag
2. **Tag format**: Tags should follow `vX.Y.Z` format (e.g., `v1.2.3`)
3. **Fallback**: If no tags exist, falls back to version in `research/__init__.py`

### Version Script Behavior

The `scripts/get_version.py` script:
- Looks for the latest git tag
- Strips the `v` prefix if present
- Validates semver format (X.Y.Z)
- Falls back to `research/__init__.py` if no valid tags

### Hatch Integration

The `pyproject.toml` is configured to use the version script:
```toml
[tool.hatch.version]
source = "code"
path = "scripts/get_version.py"
```

## Testing

### Test Structure

- `tests/test_research.py` - Tests for research code
- `tests/test_version.py` - Tests for version handling
- `tests/test_build.py` - Tests for build system

### Test Categories

1. **Unit tests** - Test individual components
2. **Integration tests** - Test build and version systems
3. **Installation tests** - Test pip installation
4. **Wheel tests** - Test wheel creation and installation

### Coverage

Tests include coverage reporting. Coverage reports are uploaded to Codecov in CI.

## Build Artifacts

### Local Build Artifacts

Running `python scripts/build.py` creates:
- `dist/` directory with wheels and source distributions
- Wheel files (`.whl`) for easy installation
- Source distribution (`.tar.gz`) for pip install from source

### GitHub Release Artifacts

Each release includes:
- Source distribution (`.tar.gz`)
- Wheel files for different platforms
- Release notes generated from commits

## PyPI Publishing

### Prerequisites

To publish to PyPI, you need:
1. A PyPI account
2. API token configured as `PYPI_TOKEN` secret in GitHub
3. The package name available on PyPI

### Manual Publishing

You can also publish manually:
```bash
# Build first
python scripts/build.py --clean

# Upload to PyPI
twine upload dist/*
```

## Troubleshooting

### Common Issues

1. **Version mismatch**: Ensure git tags match package version
2. **Missing dependencies**: Install hatch and other build tools
3. **Test failures**: Run tests locally first
4. **Permission errors**: Check PyPI credentials

### Debug Commands

```bash
# Check current version
python scripts/get_version.py
hatch version

# Check git tags
git tag -l
git describe --tags --abbrev=0

# Check build artifacts
ls -la dist/

# Test wheel installation
pip install dist/*.whl
```

## Best Practices

1. **Always test** before releasing
2. **Use semantic versioning** (major.minor.patch)
3. **Create meaningful tag messages**
4. **Check CI status** before releasing
5. **Use dry-run** for release script first
6. **Keep changelog updated**

## Security

- Never commit secrets or API keys
- Use GitHub secrets for sensitive data
- Verify package contents before publishing
- Use trusted publishing when possible

## Support

For issues with the build and release process:
1. Check the GitHub Actions logs
2. Run scripts locally to debug
3. Verify all prerequisites are installed
4. Check the troubleshooting section above