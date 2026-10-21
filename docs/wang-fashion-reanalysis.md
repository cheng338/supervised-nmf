# Reused Fashion-MNIST evidence: paired uncertainty

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng.

All 21 historical evaluation receipts passed checksum, qualification, paired-ID and recomputed-metric checks. No models were refitted and no official-test images were accessed.

The comparison uses 8,000 training rows, seed 43, and 2,000 development-validation images for each of three binary tasks. Every method uses KNN with its own historical training-CV-selected parameters.

| Task | SPA minus comparator | Difference (BA percentage points) | Pointwise 95% bootstrap interval |
| --- | --- | ---: | ---: |
| tshirt-shirt | SPA minus Screened PCA | -1.250 | [-2.650, 0.200] |
| tshirt-shirt | SPA minus Screening only | -0.850 | [-2.000, 0.350] |
| tshirt-shirt | SPA minus Raw KNN | -1.300 | [-2.600, 0.050] |
| pullover-coat | SPA minus Screened PCA | -1.750 | [-3.350, -0.250] |
| pullover-coat | SPA minus Screening only | -1.750 | [-3.251, -0.300] |
| pullover-coat | SPA minus Raw KNN | -1.750 | [-3.251, -0.300] |
| sneaker-boot | SPA minus Screened PCA | -1.650 | [-2.500, -0.850] |
| sneaker-boot | SPA minus Screening only | -1.550 | [-2.450, -0.700] |
| sneaker-boot | SPA minus Raw KNN | -0.950 | [-1.850, -0.150] |
| equal mean of these three fixed tasks | SPA minus Screened PCA | -1.550 | [-2.300, -0.817] |
| equal mean of these three fixed tasks | SPA minus Screening only | -1.383 | [-2.083, -0.717] |
| equal mean of these three fixed tasks | SPA minus Raw KNN | -1.333 | [-2.067, -0.633] |

Pointwise percentile intervals for paired class-stratified resampling of validation images; conditional on the fixed fitted models and tasks. No training, model-selection or task-population uncertainty, no multiplicity adjustment, and no confirmatory interpretation of reused development labels.

Negative differences favor the comparator. Across the matched three-task average, the historical SPA sNMF procedure trails screened PCA, screening alone and raw KNN. This evidence should be retained rather than selecting a new dataset to conceal it.

Reproduce: `python scripts/reanalyze_wang_fashion.py`. Machine-readable results and individual source hashes are in `runs/wang-fashion-reanalysis-20261006/report.json`.
