# Security Policy

Thank you for helping keep MultiMind SDK and its users safe!

## Supported Versions

| Version | Supported           |
| ------- | ------------------- |
| 0.3.x   | Yes                 |
| 0.2.x   | Security fixes only |
| < 0.2   | No                  |

## Reporting a Vulnerability
**Do NOT open a public GitHub issue for security vulnerabilities.**
If you discover a security vulnerability, please report it privately and responsibly:

- **Email:** [dev@multimind.dev](mailto:dev@multimind.dev)
- **Do not** open a public issue for security vulnerabilities.
- Provide as much detail as possible, including:
  - A description of the vulnerability
  - Steps to reproduce
  - Affected versions
  - Any relevant logs or screenshots

We will acknowledge your report within 3 business days and work with you to resolve the issue promptly. Once the vulnerability is resolved, we will coordinate a public disclosure with credit to the reporter (unless you request otherwise).


## Security Practices

- All dependencies are monitored with [Dependabot](.github/dependabot.yml)
- CI runs, on every PR:
  - **bandit** static security analysis — high-severity findings fail the
    build (medium/low are reported but don't block; see the note below)
  - **pip-audit** dependency vulnerability scanning (advisory — surfaced in
    logs so we can prioritize upgrades)
  - linting and the automated test suite
- API keys are read from environment variables and are never intentionally
  logged or persisted in plain text
- The compliance module is designed with OWASP guidance in mind

You can run the same scans locally with `make security`.

> **Known security debt:** bandit currently reports ~190 medium-severity
> findings, dominated by HuggingFace model-download revision pinning (B615)
> and SQL-string construction in vector-store backends (B608). These are
> tracked for a dedicated hardening pass and do not block CI today.

> **Note on cryptographic features:** when the optional `cryptography.zkp`
> dependency is not installed, zero-knowledge-proof functionality falls back
> to a clearly-named dummy implementation that emits a `UserWarning`. This
> fallback is **not** production-grade — install the real dependency for any
> guarantee that must survive an audit.

## Responsible Disclosure

We ask that you:

- Do not publicly disclose the vulnerability before it has been resolved
- Do not exploit the vulnerability beyond what is necessary to demonstrate it
- Act in good faith to avoid privacy violations, data destruction, or
  service disruption

## Scope

This policy applies to the MultiMind SDK codebase and all official
repositories under the [multimindlab](https://github.com/multimindlab)
organization.


We appreciate the efforts of security researchers and users who responsibly disclose 
vulnerabilities.


Thank you for helping make MultiMind SDK more secure!
