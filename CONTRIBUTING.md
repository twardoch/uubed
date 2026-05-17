# Contributing to uubed

Thanks for considering a contribution to uubed. This guide explains how to contribute effectively across our multi-repository setup.

## Project Structure

The uubed project spans four repositories:

- **[uubed](https://github.com/twardoch/uubed)** – Project home, coordination, and high-level docs
- **[uubed-rs](https://github.com/twardoch/uubed-rs)** – Rust implementation (performance-critical code)
- **[uubed-py](https://github.com/twardoch/uubed-py)** – Python bindings and high-level API
- **[uubed-docs](https://github.com/twardoch/uubed-docs)** – Documentation and technical book

## Where to Contribute

### Bug Reports and Feature Requests

Report issues in the relevant repository:

- **Rust implementation** → [uubed-rs/issues](https://github.com/twardoch/uubed-rs/issues)
- **Python API** → [uubed-py/issues](https://github.com/twardoch/uubed-py/issues)
- **Documentation** → [uubed-docs/issues](https://github.com/twardoch/uubed-docs/issues)
- **General or cross-component concerns** → [uubed/issues](https://github.com/twardoch/uubed/issues)

### Code Contributions

1. **Rust Core Development**
   - Performance improvements
   - New encoding schemes
   - SIMD implementations
   - Repository: [uubed-rs](https://github.com/twardoch/uubed-rs)

2. **Python Development**
   - API enhancements
   - ML framework integrations
   - CLI improvements
   - Repository: [uubed-py](https://github.com/twardoch/uubed-py)

3. **Documentation**
   - API docs
   - Tutorials and examples
   - Technical explanations
   - Repository: [uubed-docs](https://github.com/twardoch/uubed-docs)

## Getting Started

### Prerequisites

- **Rust development**: Rust 1.70+ with cargo
- **Python development**: Python 3.8+ with pip
- **Documentation**: Node.js for MkDocs

### Development Setup

1. Fork the relevant repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/uubed-COMPONENT.git
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Making Changes

- Follow the existing code style
- Add tests for new features
- Make sure all tests pass
- Update documentation if needed
- Keep commits focused and write clear messages

### Testing

- **Rust**: Run `cargo test` in uubed-rs
- **Python**: Run `pytest` in uubed-py
- **Docs**: Build locally with `mkdocs serve`

### Submitting Pull Requests

1. Push changes to your fork
2. Open a PR against the `main` branch
3. Fill out the PR template
4. Wait for CI checks to complete
5. Respond to feedback quickly

## Code Style

### Rust
- Format with `cargo fmt`
- Lint with `cargo clippy`
- Write idiomatic Rust

### Python
- Follow PEP 8
- Use type hints
- Format with `black`, lint with `ruff`

### Documentation
- Be clear and concise
- Include working code examples
- Match the existing structure

## Performance Considerations

uubed is built for speed. When changing performance-sensitive parts:

- Run benchmarks using the existing suite
- Watch memory usage and allocation
- Profile for bottlenecks
- Note any performance trade-offs

## Community

- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Engage helpfully in discussions
- Share real-world use cases

## Questions?

1. Check existing issues and discussions
2. Start a discussion in [uubed](https://github.com/twardoch/uubed/discussions)
3. Contact maintainers directly on GitHub

Every contribution counts. Thank you.