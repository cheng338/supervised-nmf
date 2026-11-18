# AOCL-specific NNLS multiplication workaround

This is a temporary compatibility workaround for a native-library defect.
It is intended to be retired after a correction is verified in the AOCL builds
used by the package; padding is not part of the NMF method.

## Current implementation

The package detects AOCL in the loaded BLAS inventory and selects protected
multiplication for covered products in its SPA NNLS helper, experimental
ANLS-BPP solver and experimental projected-gradient diagnostics.
This follows the environment policy that AOCL, when present, supplies NumPy's
BLAS. Other loaded libraries do not override that policy. MKL-only, OpenBLAS,
and generic BLIS environments retain direct NumPy multiplication. Detection
uses the AOCL version or library prefix reported by `threadpoolctl`, not the
environment directory name or CPU vendor. Configure native libraries at
process startup; discovery is cached per process by the existing controller.

For an M-by-K times K-by-N product, protected native products retain the full
reduction dimension K and have outer dimensions divisible by eight. The
full-padding route rounds M and N upward, fills added entries with zeros and
returns an owned C-contiguous M-by-N crop. Already aligned products use direct
multiplication. In a one-sided padding case, the operand whose outer dimension
already fits can be borrowed when it is a plain ndarray with native float64
dtype, dtype-aligned storage and canonical C-contiguous strides. Read-only
inputs and nonowning C-contiguous views can qualify.

When both outer dimensions need padding and M and N are at least eight, an
additional route is available for plain aligned float64 ndarrays whose right
operand has canonical C-contiguous strides. It pads the full left operand,
borrows the right operand's aligned column core, and copies only the remaining
columns into a zero-padded eight-column fringe. Two products use the same padded
left operand and full K; their crops fill disjoint columns of a fresh owned
C-contiguous result. The borrowed core retains its parent's row stride. Other
operands retain the full-padding allocation, float64 conversion and zero-fill
path. These routes do not change backend selection or padding widths.

The generic multiplication helper does not modify inputs or retain them between
calls. Experimental ANLS fitting additionally uses owned prepared copies of
fixed training right-hand operands and scoped native-output workspaces. The
prepared-operand route retains one full padded product; the output workspaces
follow the generic helper's geometry and reuse native buffers while returning
fresh independent logical results. These roles are local to a fit and do not
become fitted-model state or a global cache for arbitrary inputs.

Padding changes the executed matrix dimensions, rather than allocating spare
bytes around an invalid output access. The column split preserves full-K
reductions and disjoint output coverage. In exact arithmetic the original
product and NNLS objective are unchanged. Floating-point rounding can differ,
so SPA's original-design KKT checks and public SciPy fallback remain in place.

Covered SPA products include the package's batched NNLS and QR-reduction paths.
Batched qualification reuses the existing cross product for the exact admitted
design/target block; QR qualification still recomputes that scale when it is not
supplied. Conditional branches, repeated iterations and split products mean
there is no fixed count of executed native products per solve. The policy also
covers the experimental solver's selected products and its projected-gradient
residual/gradient calculations. It does not intercept NumPy, SciPy or
scikit-learn internals, the CD/MU solvers' internal products, or the separate
package reconstruction-error reporter. Reference SciPy solves retain their own
internal arithmetic.

The workaround addresses an observed AOCL double-GEMM 5-by-3 fringe overread.
The inspected single-thread Zen3 blocking path cannot produce that odd fringe
from these padded outer dimensions. This argument does not establish safety
for every internal kernel, architecture, build, or thread partition. Detection
is deliberately AOCL-wide; empirical qualification remains bounded by the
recorded experiments.

## Original evaluation

Author: Chang Cheng. October 29, 2026.

The original workaround covered eight explicit source sites in SPA's NNLS
helper. The following counts and measurements describe that original
integration and its recorded sources. They do not measure the later
[0.4.2 implementation changes](release-0.4.2.md#numerical-work-and-experimental-stopping)
or the current routes described above.

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

The original integrated implementation passed 155 focused tests plus 113 subtests in
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

Before retiring the compatibility path for a corrected build:

1. Identify the upstream correction and the distributed AOCL version/build
   containing it. Confirm the actual loaded library and applicable dispatch
   paths; a larger version number or an unrelated release note is insufficient.
2. Test direct, unpadded multiplication against the preserved crash reproductions
   and guarded fringe cases, then check NNLS numerical qualification, complete
   SPA fitting/prediction, and runtime on the affected workloads. Record the
   tested builds, platforms, dispatch paths and thread settings.
3. Bypass padding for verified fixed runtimes. Keep the compatibility path for
   supported affected or unqualified runtimes, including ambiguous version/build
   metadata. AOCL remains the selected library; retiring padding does not switch
   the environment to MKL or another backend.
4. Remove the padding helper and its dispatch integration entirely when the
   package's supported AOCL policy no longer requires affected or unqualified
   runtimes. Retain numerical and native-crash regression coverage and preserve
   historical study sources, checkpoints and results.

Verification follows the project's machine-selection rule before any new run.
Retirement must not require a new estimator parameter or changes to fitted-model
semantics. Announce the verified fixed-build coverage and any changed support
requirements in the package release notes.
