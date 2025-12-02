# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for
receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private security advisory
on GitHub. You can do this by navigating to the "Security" tab of this repository
and clicking "Report a vulnerability".

When reporting a vulnerability, please include:

1. A description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact
4. Any suggested fixes (if applicable)

We will respond to security reports within 48 hours and will keep you informed
of our progress toward a fix.

## Security Best Practices

### Model Loading

When loading pre-trained models, be cautious about loading pickle files from
untrusted sources. Pickle files can contain arbitrary code that will be executed
during loading. Only load models from trusted sources.

### Dependencies

This project uses automated dependency updates via Dependabot to help ensure
dependencies are kept up to date with security patches.

### CI/CD Security

- GitHub Actions are pinned to specific commit SHAs to prevent supply chain attacks
- Minimal permissions are used in workflows
- Third-party actions are reviewed before use

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1, 0.1.2) and
will be documented in the release notes.
