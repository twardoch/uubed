# TODO

## Completed Tasks ✅
- [x] Implement `get_available_encoders` function in `encoders/__init__.py`
- [x] Move `struct` import to module level in `gpu.py`
- [x] Replace `type(None)` with `Any` from typing in `gpu.py`
- [x] Verify Git remotes for all child repositories
- [x] Ensure all sub-repos have updated `AGENTS.md`, `PLAN.md`, `TODO.md`

## Outstanding Tasks

### Cross-Repository Coordination (Medium Priority)
- [ ] Provide GitHub Actions that trigger downstream builds
- [ ] Establish release tagging policy
- [ ] Create comprehensive API reference using Sphinx
- [ ] Add more code examples in docstrings
- [ ] Create troubleshooting guide for common errors

### Performance and Testing (High Priority)
- [ ] Profile the code to identify bottlenecks
- [ ] Consider implementing parallel processing for batch operations
- [ ] Add caching for frequently encoded embeddings
- [ ] Add unit tests for all identified edge cases
- [ ] Create integration tests for streaming operations
- [ ] Add performance benchmarks for different encoding methods