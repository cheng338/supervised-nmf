# supervised-nmf 0.3.2

supervised-nmf 0.3.2 simplifies source packaging and updates research
documentation. Numerical algorithms, classifier options, defaults and runtime
dependencies are unchanged.

- Use directory patterns for source packaging instead of maintaining a list of
  individual files.
- Remove the additional archive-rebuild and unlisted-file checks introduced in
  0.3.1. Retain import, classifier, metadata, license and required-source checks.
- Clarify research documentation and use generic interpreter commands in
  examples. Study decisions, reported results and original-source provenance
  are preserved.

Python 3.11 or later is required. Chang Cheng's original work remains All Rights
Reserved, with separate third-party notices preserved. See the
[0.3.0 release notes](release-0.3.0.md) for numerical behavior and the SPA
training-coordinate migration.

```shell
python -m pip install --upgrade supervised_nmf-0.3.2-py3-none-any.whl
```

[GitHub CI](https://github.com/cheng338/supervised-nmf/actions/runs/39535516660)
passed on Windows and Ubuntu with Python 3.11 and 3.12, with 353 tests passing
in each environment. Both the wheel and source archive were built and checked.
Checksums and build evidence accompany the release. No new dataset study was
run for this patch.
