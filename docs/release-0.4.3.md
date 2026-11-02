# supervised-nmf 0.4.3

Version 0.4.3 corrects sample and subject identifier handling, preserves NumPy
extended-precision scalars in reports, and stabilizes calibration checkpoint
identities when native libraries load lazily. SPA, absolute screening and
unscaled KNN remain the defaults; factorization and prediction algorithms are
unchanged.

## Identifiers and grouped evaluation

Identifiers must be Python/NumPy integers, finite floating-point scalars or
nonblank strings. Boolean, complex, temporal and unsupported object identifiers
raise `ValueError` before fitting or prediction. Large Python integers retain
their values without a NumPy finiteness check.

Uniqueness, overlap, subject counts and grouped CV use exact represented numeric
values. This prevents mixed NumPy/Python comparisons from merging distinct groups
or missing equal IDs because their hashes differ. Original identifiers remain
available in metadata and reports. Integer and floating-point IDs with equal
values still identify the same sample or subject. These checks cannot recover
type information lost through earlier caller coercion. See the
[identifier policy](api.md#calibration-and-reporting).

## Reports and checkpoint reuse

NumPy `longdouble` values that cannot unbox to Python scalars now serialize as
`longdouble-v1` JSON objects, using an exact binary significand and exponent.
Adjacent values, extreme finite magnitudes and signed zero remain distinguishable.
Extended complex report values use `clongdouble-v1`; complex identifiers remain
invalid. Ordinary scalar encodings are unchanged. Consumers of extended-scalar
reports should follow the [documented representation](api.md#calibration-and-reporting).

General calibration initializes the numerical-library inventory before recording
checkpoint identity, matching shared-grid calibration. Lazy BLAS/OpenMP loading
therefore does not invalidate an otherwise identical resume. Source and runtime
changes still prevent reuse; this is not a migration of earlier checkpoints.

Padding-workspace tests cover ownership, reuse, failure cleanup and recovery.
Maintained AOCL operand tests no longer pin a historical implementation hash.
The [benchmark documentation](nmf-solver-comparison.md) clarifies historical
environment restrictions and the limits of the separate study runners.

## Performance compared with 0.4.2

The comparison covered ordinary fit/predict and evaluation workflows, SPA/CD/MU
controls, rank-one and extreme-range projection cases, grouped and explicit-fold
validation, identifier precision and overlap failures, extended-scalar reports,
and fresh, resumed and changed-input checkpoints through all three calibration
APIs. Serial workers on an AMD Ryzen 9 6900HS Windows PC used one logical CPU
and one native thread in each unchanged local MKL and AOCL environment.
The main comparison used six order-balanced
version pairs with baseline-versus-baseline controls; calibration received eight
additional pairs per environment. Workflow timings cover complete public calls;
metadata and serialization microbenchmarks time those operations separately.
Percentages below are changes in median paired runtime ratios, not ratios of
the separately reported runtime medians.

The original full-feature Fashion inputs used 8,000 training and 2,000 evaluation
rows with 784 features; CelebA used 8,000 and 4,000 rows with 4,096 features.
Both used their actual historical split identifiers, with integer IDs for
Fashion and string image IDs for CelebA. Default SPA fit-and-evaluate workflows
showed no material slowdown in these measurements. Selected fitted arrays,
predictions and non-timing diagnostics matched exactly within each backend.

| Original real-data workflow | MKL paired runtime change | AOCL paired runtime change |
| --- | ---: | ---: |
| Fashion fit and evaluate | -0.6% | -2.0% |
| Fashion evaluation only | -7.3% | -9.3% |
| CelebA fit and evaluate | -0.6% | -1.7% |
| CelebA evaluation only | +0.9% | -0.8% |

Negative changes indicate shorter runtime. These results do not establish
performance equivalence for every workload. Identifier-validation
microbenchmarks retained measurable costs:

- At 10,000 IDs, ordinary floating-point validation took about 1.1 ms longer
  (+9-10%); object arrays of NumPy numeric scalars took 2.0-2.2 ms longer
  (+16-17%).
- At 1,000, 10,000 and 100,000 `longdouble` IDs, paired runtime ratios were
  2.38-2.42×, 2.44-2.47× and 3.35-3.40×, respectively, with median paired
  additions of about 1.8, 18 and 300 ms. The 100,000-ID case was stress coverage.
- Integer validation improved: the 10,000-ID case fell from separate runtime
  medians of about 12 ms to 4 ms.

The eight-pair calibration follow-up confirmed costs on small synthetic
workloads. Warm-process general calibration added 3.3-5.5 ms per fresh call or
resume (+6-14% in paired medians). Shared-grid and rank calibration were more variable:
their main-run median paired additions ranged from 0.9 to 11.5 ms; follow-up
differences ranged from -1.4 to +10.5 ms. First-call AOCL timing remained variable
across 14 paired runs, so no precise cold-start speed change is claimed;
the targeted follow-up measured about 2.9 ms (+7.8%) additional resume time.

Correct identifier and checkpoint behavior motivates this release despite those
measured costs. Cases where 0.4.2 failed or gave incorrect results were checked
as corrected behavior and were not reported as
speedups. Factorization and prediction algorithms remain unchanged.

## Validation and remaining limits

Pre-release validation of the production changes used the existing local Windows
Python 3.12 MKL and AOCL environments. MKL passed 1,909 tests and 724 subtests;
AOCL passed 1,890 tests and 712 subtests. Each environment skipped 25 cases that
require wider native `longdouble` precision or range. AOCL also retained 19
historical exclusions: 17 legacy experimental tests with unprotected products
and two MKL-emulation cases. Those 19 tests passed on MKL. Isolated installation
checks of the validation build passed 152 metadata tests with the same 25
capability skips in each environment. Installed runtimes were unchanged.

Native `longdouble` precision remains platform-dependent.
The [0.4.2 compatibility and native-library limits](release-0.4.2.md#compatibility-and-remaining-limits)
continue to apply, including uncovered AOCL paths inside other libraries.
Python 3.11 or later is required.
