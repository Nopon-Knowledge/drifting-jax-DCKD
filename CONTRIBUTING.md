# Contributing

Contributions should preserve the fixed-kernel path and the documented
evaluation semantics.

1. Create a focused branch and keep generated data, checkpoints, credentials,
   cluster paths, and logs out of commits.
2. Add or update tests for changes to the loss, memory bank, inference
   manifests, FID, or PRDC.
3. Run the CPU checks:

   ```bash
   JAX_PLATFORMS=cpu python -m pytest -q \
     tests/test_drift_loss.py tests/test_prdc.py
   ```

4. For scientific changes, record the configuration, training seed,
   generation seed, checkpoint rule, and evaluator version. Do not discard
   finite unfavorable results.

By contributing, you confirm that you have the right to submit the change.
Repository-wide licensing remains subject to `NOTICE.md`.
