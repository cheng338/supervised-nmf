# Research records

The [validation history](validation.md) describes the datasets, evaluation rules and scope of the reported results. For current package usage, see the [documentation index](../docs/README.md).

| Subject | Records |
| --- | --- |
| CelebA | [Default selection protocol](default-selection-protocol.md), [results](celeba-default-results.md), [identity-disjoint validation](celeba-identity-validation.md) |
| Solver and dataset comparisons | [Four-solver study](nmf-solver-comparison.md), [Fashion-MNIST protocol](fashion-mnist-protocol.md), [multi-dataset results](multi-dataset-results.md) |
| Wang validation studies | [Study report](wang-validation-report.md), [simulations](wang-simulation-validation.md), [real-data validation](wang-real-data-validation.md) |
| Numerical implementation | [BLAS optimization](engineering/blas-optimization.md), [AOCL workaround](engineering/aocl-nnls-workaround.md), [projection changes](engineering/implementation-improvements.md) |
| Development decisions | [Development protocol](engineering/development-protocol.md), [tuning and reporting](engineering/tuning-improvements.md) |

## Finding a moved document

Study protocols and results previously under `docs/` are here under the same filenames. The five engineering documents linked in the last two rows are in `research/engineering/`. Release notes moved from `docs/release-<version>.md` to `docs/releases/release-<version>.md`; the [changelog](../CHANGELOG.md) lists them by version. Links to files in earlier release tags still use their original paths. Bookmarks and external links to the old paths on the default branch may need updating.
