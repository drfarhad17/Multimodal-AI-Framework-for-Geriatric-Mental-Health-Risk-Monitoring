# Mapping to the supplied article and unresolved issues

| Article component | This package | Status |
|---|---|---|
| 2,000 participants and 20,000 sessions | Same default counts and balanced labels | Implemented, newly generated |
| Age mean 74.6 ± 8.1 | Truncated random draws; actual moments logged | Does not force reported moments |
| GDS/GAD/MMSE/MoCA label thresholds | Same eligibility cutoffs | Severity normalization newly specified |
| Conditional simulation | Transparent equations and noise distributions | Newly chosen; original distributions unavailable |
| Four missing-data rates | Target rates with maximum 2 missing sessions per modality | Realized rates logged, not exact paper rates |
| 70/15/15 participant split | Stratified 1,400/300/300 split | Implemented |
| 5-fold cross-validation | Training-only RF with fold-specific preprocessing | Full neural/fusion CV not implemented |
| Random forest | 200 trees, leaf size 2, session-feature flattening | Unspecified hyperparameters newly chosen |
| LSTM 128 units | PyTorch 128-unit LSTM | Article states TensorFlow; disclosed substitution |
| CNN 3x3 | Session-feature matrix CNN | Not raw-signal/spectrogram feature extraction |
| BERT conversational semantics | Small numerical-feature Transformer | NOT BERT; transcripts unavailable |
| Learned representation fusion | Validation-fitted convex probability fusion | Approximation, not exact paper architecture |
| SHAP and LIME | Kernel SHAP and tabular LIME on final predictions | Optional dependency route; approximate perturbations |
| Removing SHAP/LIME reduces accuracy | Identical predictions when post-hoc XAI removed | Article needs training-feedback mechanism to support a decrease |
| AES-256 and role-based controls | Not included in offline code | Do not claim implemented encryption/access control |
| Real-time alerts/interventions | Not included | No clinical deployment or prospective evaluation |
| Statistical values and plots | Recomputed outputs, never hard-coded target values | Existing manuscript numbers not validated |

## Issues to reconcile before submission

The paper describes confidence intervals and other results as illustrative in some passages while other passages claim experimentally confirmed clinical applicability. Use consistent simulation-only wording and actual prediction-derived values.

The stated held-out participant count is 300, but the confusion-matrix narrative lists diagonal counts 184, 176, 171, 178 and 183, totaling 892 correct cases. Those cannot be counts from one prediction per 300 test participants. If the matrix is session-level or another split, identify that explicitly and compute uncertainty at participant level.

Balanced class test supports imply macro recall equals accuracy for that evaluation. The reported 96.2% accuracy and 94.8% recall require an explanation of different averaging, supports or evaluation units. Check against original predictions.

The appendix repeats entries and mixes AUC rows with ablation results. Rebuild it from verified output tables.

The manuscript does not define an unambiguous binary risk endpoint for Brier/calibration/decision curves. This code defines any non-normal label; choose and justify the actual scientific endpoint before comparing results.

No source data, saved weights, original code, transcript corpus, audio files or full hyperparameter specification were attached. Exact reproduction is therefore not possible from this document alone. This is a usable new experimental foundation with explicit scope limitations, not recovered evidence for prior results.
