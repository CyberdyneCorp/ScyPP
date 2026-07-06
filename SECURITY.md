# Security Policy

SciPP is a numerical computing library (a C++20 port of SciPy). The most relevant risks are
memory-safety issues (out-of-bounds access, use-after-free) reachable through the numerical
routines, and denial-of-service from untrusted input (e.g. a pathological matrix size, a
malformed sparse structure, or extreme solver parameters).

## Supported versions

The project is pre-1.0-stability: it follows semantic versioning but the API is still settling
and minor releases may carry breaking changes. Security fixes are applied to `main` and
released in the next tag. There is no long-term-support branch yet.

| Version | Supported |
|---------|-----------|
| `main`  | ✅ |
| latest tag | ✅ |
| older tags | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately via GitHub's [**Report a vulnerability**](https://github.com/CyberdyneCorp/SciPP/security/advisories/new)
(Security → Advisories), or email the maintainer at **leonardoaraujo.santos@gmail.com** with:

- a description of the issue and its impact,
- steps or a minimal input that reproduces it (a code snippet driving the routine, plus the
  matrix/array or parameters involved),
- affected version/commit.

We aim to acknowledge a report within a few days and to agree on a disclosure timeline with
you. Please give us a reasonable window to ship a fix before any public disclosure.

## Scope

In scope: the core library and its public C++ API. Out of scope: vulnerabilities in
third-party dependencies (NumPP and any BLAS/LAPACK/GPU runtime — report those upstream) and
issues that require running deliberately hostile build tooling.
