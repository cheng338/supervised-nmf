# Original research source

Chang Cheng is the author of the original research scripts and this package.
`original/sim.py` and `original/snmf_am.py` are unmodified snapshots supplied by
the author. `sources.json` records their SHA-256 hashes and modification times.
No dataset, dissertation PDF, or committee review is bundled with the package.

Do not run these snapshots as package examples: they preserve the historical
behavior, including warning suppression, infeasible grid combinations, and
fixed output filenames. New implementation and audit tools live under `src/`
and `examples/`. Their existence does not establish reproduction of any published
numerical result. The historical software environment and complete run records
were not available at initial development.

The source methodology is Chapter 7 of Chang Cheng's dissertation, now titled
*Ensemble Kernel Distance-Weighted Discrimination and Supervised Nonnegative
Matrix Factorization for Classification*. The original methodological reference
was the October 1, 2026 dissertation version. The study draft updated October 2
and Wang's earlier committee review are supporting references.

Current studies use explicit parameter grids for the package's NMF/PCA,
screening and KNN methods. Supplementary planning archives are excluded from
distributions and do not define the package's methods or dependencies.

