# AOCL-specific NNLS multiplication workaround

Author: Chang Cheng. October 29, 2026.

This is a temporary compatibility workaround for a native-library defect.
It is intended to be retired once AMD's correction is available and verified
in the AOCL builds used by the package; padding is not part of the NMF method.

The package detects AOCL in the loaded BLAS inventory and applies real
zero-padding to the eight explicit matrix products in its SPA NNLS helper.
This follows the environment policy that AOCL, when present, supplies NumPy's
BLAS. Other loaded libraries do not override that policy. MKL-only, OpenBLAS,
and generic BLIS environments retain direct NumPy multiplication. Detection
uses the AOCL version or library prefix reported by `threadpoolctl`, not the
environment directory name or CPU vendor. Configure native libraries at
process startup; discovery is cached per process by the existing controller.

For an M-by-K times K-by-N product, both outer dimensions are rounded up to
multiples of eight. New operand entries are zero and K is unchanged. The
returned M-by-N crop is an owned C-contiguous array. Already aligned products
use direct multiplication. Padding changes the executed matrix dimensions,
rather than allocating spare bytes around an invalid output access. In exact
arithmetic the original product and NNLS objective are unchanged. Floating
point rounding can differ, so original-design KKT checks and public SciPy
fallback remain in place.

The workaround addresses an observed AOCL double-GEMM 5-by-3 fringe overread.
The inspected single-thread Zen3 blocking path cannot produce that odd fringe
from these padded outer dimensions. This argument does not establish safety
for every internal kernel, architecture, build, or thread partition. Detection
is deliberately AOCL-wide; empirical qualification remains bounded by the
recorded experiments. The workaround covers explicit products in the package
NNLS helper, including QR and its fallbacks. It does not replace internal
NumPy/SciPy products, other NMF solvers, or report-only residual products.

The prior local comparison covered AOCL 5.2 and 5.3 on Windows with Zen3
dispatch and one native thread. Padding passed the recorded guarded cases and
numerical checks. Multiples-of-eight padding added about 4% to the sum of
seven helper-case medians, and about 11–12% on the two captured workloads.
Those timings support a correctness tradeoff, not a universal performance
claim. No memory optimization motivated adoption. No AOCL source or installed
binary is changed, and no new dependency or estimator option is introduced.

The preserved alternative evaluation is in
`output/aocl-alternatives-20261029/REPORT.txt`. Integration receipts are stored
separately in `output/aocl-only-padding-20261029`; they must be interpreted
with their recorded source hashes, runtime, and validation status. Earlier
study checkpoints and results retain their original implementation identity.

The integrated implementation passed 155 focused tests plus 113 subtests in
each of the three local environments: AOCL 5.2, AOCL 5.3, and MKL 2026.2.
Each environment also passed seven NNLS cases and two complete SPA fit/query
comparisons. MKL results were bitwise identical to the preserved implementation
in all nine comparisons, with zero calls to the padding helper. Its ratio of
summed case medians was 0.992, consistent with no material change in this
bounded timing experiment. Corresponding AOCL ratios were 1.057 and 1.049.
These are measured implementation costs, not classifier-accuracy evidence.

All 30 guarded product checks passed: five shapes in each environment at
native thread budgets of 1 and 20. They checked the actual helper's enlarged
dimensions, zero padding, scalar reference values, preserved inputs, and
allocation canaries. The MKL controls retained their original dimensions.
Passing these cases at a 20-thread budget does not extend the single-thread
tile-parity argument into a guarantee for arbitrary native scheduling.

## Retirement policy

The current implementation enables padding for every positively identified
AOCL runtime. No fixed AOCL build has been qualified in this package's recorded
workaround evidence, so there is no automatic version exemption yet.
Qualification is a development and release activity, not a self-test performed
at import, estimator construction, or each fit.

When AMD supplies a correction:

1. Identify the upstream correction and the distributed AOCL version/build
   containing it. Confirm the actual loaded library and applicable dispatch
   paths; a larger version number or an unrelated release note is insufficient.
   Preserve the loaded binary's identity and hash with the qualification record.
2. Test direct, unpadded multiplication against the preserved crash reproductions
   and guarded fringe cases, then check NNLS numerical qualification, complete
   SPA fitting/prediction, and runtime on the affected workloads. Record the
   tested builds, platforms, dispatch paths and thread settings.
   Run native crash and guarded-memory checks in isolated processes. Require
   both memory checks and numerical checks to pass; a process that merely avoids
   crashing has not established that an out-of-bounds access is gone.
3. Bypass padding for verified fixed runtimes. Keep the compatibility path for
   supported affected or unqualified runtimes, including ambiguous version/build
   metadata. AOCL remains the selected library; retiring padding does not switch
   the environment to MKL or another backend.
   Add an exemption in a subsequent package release only when runtime metadata
   can reliably identify the qualified coverage. Do not infer that all newer
   versions, other binary distributions, or other dispatch paths are fixed.
4. Remove the padding helper and its dispatch integration entirely when the
   package's supported AOCL policy no longer requires affected or unqualified
   runtimes. Retain numerical and native-crash regression coverage and preserve
   historical study sources, checkpoints and results.

Verification follows the project's machine-selection rule before any new run.
Retirement must not require a new estimator parameter or changes to fitted-model
semantics. Announce the verified fixed-build coverage and any changed support
requirements in the package release notes.
