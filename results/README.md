# Aggregate experimental results

These CSV files are derived, non-checkpoint artifacts from the frozen
`PR-DCKD-EXP-V1` protocol. Formal comparisons use 50,000 generated samples,
generation seed 271828, final EMA checkpoints at step 30,000, and PRDC
`nearest_k=5`.

Key files:

- `formal_results.csv`: all unique formal 50k evaluations.
- `primary_comparison_summary.csv`: released fixed, DCKD-Global-MS, and tuned
  fixed C1 summaries over seeds 123, 2026, and 3407.
- `factorial_results.csv` and `factorial_summary.csv`: complete local/global
  by single/multi-scale comparison.
- `resource_results.csv` and `resource_summary.csv`: constrained-bank study.
- `inference_efficiency_summary.csv`: parameter, NFE, FLOP, memory, latency,
  and throughput comparison.
- `mechanism_stages.csv`: radius and clamp diagnostics.

The decisive reporting boundary is unchanged: DCKD-Global-MS improves the
released fixed radii for all three observed seeds, while tuned fixed C1 has
lower FID for all three matched seeds. DCKD lowers Recall in the main and
constrained comparisons. With only two or three training seeds, these are
descriptive results rather than statistical-significance claims.
