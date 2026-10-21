# Project rules

These rules were set by Chang Cheng on October 20, 2026. They apply to future
package development and validation, and supersede conflicting earlier project
guidance. Preserve historical study configurations and results as records of
what was actually run.

1. **Minor performance differences do not decide adoption by themselves.**
   A small gain or loss in predictive performance or execution speed does not
   automatically justify accepting, rejecting, or making a proposed fix or
   improvement optional. Consider correctness, fidelity to the methodology,
   theoretical support, numerical reliability, practical benefit, maintenance
   cost, and the strength and uncertainty of the evidence. Explain the decision.
   Use mathematical reasoning, relevant theory, numerical analysis, and coding
   knowledge alongside accuracy, runtime, and memory measurements. Examine the
   assumptions, mathematical equivalence or changes to the objective, numerical
   stability, algorithmic complexity, implementation correctness, and software
   design. Distinguish analytical guarantees from empirical observations;
   performance measurements alone do not establish correctness or theoretical
   validity.

2. **Do not sacrifice speed for memory savings.**
   Memory reduction is not an acceptable reason to make execution slower.
   Preserve or improve speed when proposing a memory optimization. If lower
   memory use requires a slowdown, report the tradeoff instead of adopting it,
   including as an optional implementation. Rule 1 does not waive this rule.

3. **Confirm the validation machine before execution.**
   Before starting each new validation run or agreed batch, ask Chang Cheng
   whether to use the local PC or Robust and wait for an explicit choice.
   This includes unit tests, smoke checks, numerical checks, benchmarks, and
   dataset validation studies. One confirmation covers the specified batch;
   previous unrelated runs, an available environment, or a request to implement
   a change do not authorize selecting the machine for a new validation run.
   Reading files, reviewing existing results, and preparing code or plans may
   proceed without launching validation.

4. **Default Robust to 20 total compute threads.**
   Robust has 20 physical cores and 40 logical threads. Use a default budget of
   20 compute threads for new runs, accounting for concurrent workers and their
   native numerical threads together. For example, use 20 workers with one
   numerical thread each, not 20 workers with 20 numerical threads each.
   Do not infer a 40-thread default from the logical CPU count. A different
   budget requires an explicit user instruction. Older launchers and frozen
   protocols may still record 40; override the runtime setting or prepare an
   updated launch configuration before a new run, without rewriting historical
   evidence. Confirm the machine under rule 3 regardless of this default.

5. **Python 3.10 support has ended.**
   Current development requires Python 3.11 or later. Keep Python 3.12 in
   `py12_df_aocl` as the primary development environment. Do not add Python 3.10
   compatibility workarounds or include it in new validation plans. Preserve
   previous Python 3.10 requirements, results, and released artifacts as
   historical records; they do not establish support for current source.
   This policy does not authorize validation without the rule 3 confirmation.

See [the development protocol](docs/development-protocol.md) for methodology,
authorship, study history, and evidence requirements.
