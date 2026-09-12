# Execution record

Executed in the preparation environment:

- Syntax compilation of all Python source files.
- Full default synthetic data generation: 2,000 participants, 20,000 sessions.
- Random-forest benchmark on 1,400 training, 300 validation and 300 test participants.
- Four retrained RF modality-removal ablations.
- Five-fold RF cross-validation limited to the training partition with preprocessing fitted inside each fold.
- 200 participant-bootstrap draws for the packaged demonstration (CLI default is 1,000 for subsequent runs).
- Class-specific metrics, calibration, decision curves and subgroup outputs.
- Independent checks of label rules, counts, partition identities, probability sums, confusion-matrix/accuracy consistency and byte-identical seeded CSV regeneration.

Command:

```bash
python run_study.py --data data --out results_verified --mode rf --cv --bootstrap 200
python validate_outputs.py
```

Observed RF test accuracy: 0.5966666667; macro F1: approximately 0.595063. These are NEW simulation results and are not comparable evidence for the manuscript's proposed-framework figures. They are included to make execution reviewable, not to match a target outcome.

Not executed: optional PyTorch neural models, full probability fusion, and SHAP/LIME routes because torch, shap and lime were unavailable in the preparation environment. Their Python syntax was checked, but runtime behavior and results remain unverified. No claim of a completed four-model reproduction is made. Install requirements-optional.txt and execute these routes before relying on them for submission.

The included dependency ranges are not an exact lock for optional routes. See VERIFIED_ENVIRONMENT.txt for actual core versions used.
