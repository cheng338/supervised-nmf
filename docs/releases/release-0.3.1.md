# supervised-nmf 0.3.1

October 20, 2026 (America/Indianapolis). Author: Chang Cheng.

This patch removes the separate PC throughput benchmark from the public source
tree and source distribution. The NMF algorithms, screening, classifiers,
tuning APIs, numerical safeguards and runtime dependencies are unchanged from
0.3.0. The installed wheel never contained the portable speedtest.

## Removed development tooling

The removed `portable_speedtest/` directory contained a machine-comparison
benchmark, launchers, environment snapshots, PC profiles, archived benchmark
manifests and instructions for running it on another PC. Its three dedicated
test modules and the release check that required the benchmark are also removed.
Distribution checks now reject that directory and the removed test modules in
both the wheel and source archive.

The source manifest now names each approved file explicitly. This prevents
untracked research or machine-management scripts from being swept into a local
source build. Wheel and source-archive checks enforce their separate file sets;
the manifest alone does not control Python modules in a directly built wheel.

The benchmark source and tests were preserved before removal. Existing result
files, Git history and earlier release assets remain unchanged. The lower test
count in this release reflects removal of benchmark-tool tests; the classifier,
numerical and scientific-study regression tests remain.

## Package-content review

The review covered the installed package, runtime and optional dependencies,
examples, scientific scripts, tests, documentation, requirements and original
source provenance. The following material remains because it supports the
classification method or its reproducibility:

- Classification, screening, factorization, grouped calibration, evaluation and
  numerical safeguards, including their regression tests.
- Simulation and CelebA examples, explicit parameter grids, solver-comparison
  scripts and their recorded protocols. ANLS-BPP remains an experimental study
  comparator, outside the public classifier's solver choices.
- Scientific reports, original research scripts and their provenance, historical
  environment records, authorship and third-party notices.

Public documentation now distinguishes maintainer machine-selection rules from
package use. Personal interpreter commands for unpublished historical workflows
were replaced with labeled workflow records. An obsolete CI-account note and
an unrelated DWD-package implementation-reference paragraph were removed;
the full dissertation citation and scientific results remain.

The internal recovery-report helper is retained because separately maintained
research scripts use its approved numerical fix. It is outside classifier
execution. Removing it would require a coordinated change to those callers.

No DWD implementation, dataset images, dissertation PDFs, committee documents,
credentials or SSH launchers were found in the reviewed distributions. Local
research outputs and supplementary development files remain outside releases.

Python 3.11 or later is required. Chang Cheng's original work remains All Rights
Reserved; the separate third-party notices retain their own terms. The
[0.3.0 release notes](release-0.3.0.md) describe the current numerical behavior,
SPA training-coordinate migration and research-validation limitations.

Download the wheel from [GitHub releases](https://github.com/cheng338/supervised-nmf/releases):

```shell
python -m pip install --upgrade supervised_nmf-0.3.1-py3-none-any.whl
```

Build-verification results are recorded with the GitHub release and its attached
validation receipt. The content review does not constitute a new dataset study.
