# supervised-nmf 0.3.2

October 20, 2026

This patch simplifies source packaging and edits the research documentation.
The source manifest uses directory patterns instead of an explicit list of
individual files. Distribution verification retains the existing import,
classifier, metadata, license and source-content checks; the additional
archive-rebuild and unlisted-file checks introduced in 0.3.1 are removed.
The separate PC throughput benchmark remains excluded from current source.

Research documentation now describes study decisions directly and uses generic
interpreter commands. The methodological choices, their
timing, numerical results and original-source provenance are preserved.

The numerical algorithms, classifier options, defaults and runtime dependencies
are unchanged. Python 3.11 or later is required. Chang Cheng's original work
remains All Rights Reserved, with separate third-party notices preserved.
See the [0.3.0 release notes](release-0.3.0.md) for current numerical behavior
and the SPA training-coordinate migration.

```shell
python -m pip install --upgrade supervised_nmf-0.3.2-py3-none-any.whl
```

Build-verification results accompany the GitHub release. No new dataset study
was run for this patch.
