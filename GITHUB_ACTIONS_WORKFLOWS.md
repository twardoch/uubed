# GitHub Actions Workflows

Due to permission restrictions, the GitHub Actions workflow files need to be added manually to the repository. Here are the workflow files that should be created:

## Required Workflow Files

### 1. Update `.github/workflows/ci.yml`

Replace the existing CI workflow with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for version detection

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install hatch

      - name: Check version
        run: |
          python scripts/get_version.py
          hatch version

      - name: Run tests
        run: |
          python scripts/test.py --all

      - name: Build project
        run: |
          python scripts/build.py --clean

      - name: Upload coverage to Codecov
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

### 2. Update `.github/workflows/release.yml`

Replace the existing release workflow with:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for version detection

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install hatch

      - name: Check version matches tag
        run: |
          VERSION=$(python scripts/get_version.py)
          TAG_VERSION=${GITHUB_REF#refs/tags/v}
          echo "Package version: $VERSION"
          echo "Tag version: $TAG_VERSION"
          if [ "$VERSION" != "$TAG_VERSION" ]; then
            echo "Error: Version mismatch between tag and package"
            exit 1
          fi

      - name: Run tests
        run: |
          python scripts/test.py --all

      - name: Build project
        run: |
          python scripts/build.py --release --clean

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist-${{ matrix.os }}-${{ matrix.python-version }}
          path: dist/

  release:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install hatch twine

      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: dist-*
          merge-multiple: true
          path: dist/

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/*
          generate_release_notes: true
          draft: false
          prerelease: false

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: |
          python -m twine upload dist/*.tar.gz dist/*.whl --skip-existing
```

### 3. Create `.github/workflows/build-wheels.yml`

Create a new workflow file for building wheels:

```yaml
name: Build Wheels

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      skip_tests:
        description: 'Skip tests'
        required: false
        default: 'false'

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for version detection

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install hatch build wheel

      - name: Check version
        run: |
          VERSION=$(python scripts/get_version.py)
          echo "Building version: $VERSION"
          echo "PACKAGE_VERSION=$VERSION" >> $GITHUB_ENV

      - name: Run tests
        if: github.event.inputs.skip_tests != 'true'
        run: |
          python scripts/test.py --all

      - name: Build wheel and sdist
        run: |
          python scripts/build.py --release --clean

      - name: Verify wheel
        run: |
          pip install dist/*.whl
          python -c "import research; print(f'Successfully imported research v{research.__version__}')"

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-py${{ matrix.python-version }}
          path: dist/

  build_sdist:
    name: Build source distribution
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install hatch build

      - name: Build sdist
        run: |
          python -m build --sdist

      - name: Upload sdist
        uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz

  # Job to collect all artifacts and prepare for release
  collect_artifacts:
    name: Collect all build artifacts
    runs-on: ubuntu-latest
    needs: [build_wheels, build_sdist]
    
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          merge-multiple: true
          path: dist/

      - name: Download sdist
        uses: actions/download-artifact@v4
        with:
          name: sdist
          path: dist/

      - name: List artifacts
        run: |
          echo "Build artifacts:"
          ls -la dist/

      - name: Upload combined artifacts
        uses: actions/upload-artifact@v4
        with:
          name: all-dist-files
          path: dist/

  # Test installation from wheels on different platforms
  test_wheels:
    name: Test wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    needs: build_wheels
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Download wheels
        uses: actions/download-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-py${{ matrix.python-version }}
          path: dist/

      - name: Install from wheel
        run: |
          pip install dist/*.whl

      - name: Test installation
        run: |
          python -c "import research; print(f'Successfully imported research v{research.__version__}')"
          python -m pytest tests/ -v

      - name: Test wheel metadata
        run: |
          pip show research
```

## Setup Instructions

1. **Create/Update workflows**: Add or update the above workflow files in your repository
2. **Configure secrets**: Add `PYPI_TOKEN` to your GitHub repository secrets for PyPI publishing
3. **Test the setup**: Push a tag to trigger the release workflow

## Notes

- The workflows will be triggered automatically when you push a git tag (e.g., `v1.0.6`)
- The `build-wheels.yml` workflow can also be triggered manually from the GitHub Actions tab
- Make sure to configure the `PYPI_TOKEN` secret in your repository settings before releasing