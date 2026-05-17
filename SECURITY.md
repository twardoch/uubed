# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅                 |
| 0.x.x   | ❌                 |

## Reporting a Vulnerability

If you've found a security issue, don't post it publicly. Email us at **security@uubed.dev** or use GitHub's [private vulnerability reporting](https://github.com/twardoch/uubed/security/advisories/new).

Include:
- A clear description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Suggested fixes (if any)

### Response Timeline

- Initial response: within 48 hours
- Status update: within 7 days
- Fix delivery: 30–90 days, depending on severity

## Security Considerations

### Input Validation

uubed handles binary data and embedding vectors. It's built to be safe, but validate your inputs:

- Check that embedding dimensions match expected sizes
- Sanitize encoded strings before using them in queries
- Apply rate limits to encoding operations in production

### Memory Safety

- Core logic is written in Rust, which prevents memory errors by design
- Python bindings via PyO3 include bounds checking and safe memory handling
- Large inputs may still cause high memory usage — monitor accordingly

### Cryptographic Considerations

uubed is **not** a cryptography tool. Its encoding methods are for performance and avoiding substring pollution, not for securing data.

Do **not**:
- Encode passwords, keys, or other secrets
- Expect tamper resistance — always use proper authentication for integrity
- Treat output as encrypted — it's reversible and offers zero confidentiality

### Supply Chain Security

Our practices include:
- Regular dependency audits
- Reproducible builds
- Signed releases
- Minimal external dependencies

## Security Best Practices for Users

### Deployment

1. Use the latest stable version
2. Scan dependencies regularly
3. Validate all inputs before processing
4. Handle errors properly to prevent information leaks

### Integration Security

#### Search Queries

Use parameterized queries:
```python
# Good
cursor.execute("SELECT * FROM docs WHERE embedding_code = %s", (encoded,))

# Bad — opens door to SQL injection
cursor.execute(f"SELECT * FROM docs WHERE embedding_code = '{encoded}'")
```

#### API Exposure

When wrapping uubed in an API:
- Enforce authentication and authorization
- Rate-limit requests
- Sanitize inputs
- Use HTTPS everywhere

#### Data Handling

- Don’t log encoded embeddings
- Restrict access to encoded data
- Encrypt at rest if needed

## Known Issues

### Resource Exhaustion

Large batch operations can eat up memory. Limit batch sizes and concurrent jobs if resources are tight.

### Timing Attacks

Encoding time may leak information based on input. For critical systems, consider constant-time execution requirements.

### Dependencies

We audit dependencies regularly:
- Rust: `cargo audit`
- Python: automated scanners
- Build tools: kept up to date

## Acknowledgments

Thanks for reporting issues responsibly. Valid findings will earn you:
- Credit in security advisories (with permission)
- A mention in the CHANGELOG
- An invite to our advisory team (optional)

## Questions?

For general security questions, use [GitHub Discussions](https://github.com/twardoch/uubed/discussions). For sensitive topics, email directly.