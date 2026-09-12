# SMGMH reference implementation

Code accompanying development of “Explainable Multimodal AI Framework for Geriatric Mental Health Risk Monitoring: A Simulation Study”.

## Status and intended use

This package is NEW code reconstructed from the supplied Article_R(3).docx narrative. It is not the original experimental code or original raw dataset. It does not reproduce or substantiate the article's reported 96.2% accuracy, AUC 0.972, confidence intervals, SHAP values, or timings. Every result produced by this package is calculated from a new simulation. Read IMPLEMENTATION_NOTES.md before journal submission. Use revised, actually computed results if this implementation becomes the study's experimental basis.

No clinical use. All individuals, features, screening scores and labels are synthetic. No recordings, transcripts, patient information, or clinical diagnoses are included.

## Quick start

Python 3.11 recommended. Unzip, open a terminal in SMGMH_Code, then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_synthetic_data.py --out data --participants 2000 --seed 42
python run_study.py --data data --out results --mode rf --cv
```

This runs the random forest, RF modality ablations, five-fold RF training-only cross-validation, and test evaluation. It does NOT run or label an RF as the complete proposed fusion model.

For the four-model reference implementation:

```bash
python -m pip install -r requirements-optional.txt
python run_study.py --data data --out results_full --mode full --epochs 30 --cv
python explain.py --results results_full --samples 5
```

Optional PyTorch, SHAP and LIME dependencies need installation access and are not required by the RF route. CPU execution is the default. SHAP can be slow. Increase explained participants for a more representative global summary; five cases only demonstrate the workflow. Use separate output directories for different configurations to avoid mixing saved files.

## Dataset information

Default: 2,000 participants, five balanced classes (400 each), 10 observations each, 20,000 rows, 52% female. Age draws are truncated normal with underlying mean 74.6 and SD 8.1; truncation means the realized mean and SD DIFFER from the article. Actual demographics and missingness are recorded in data/generation_metadata.json.

participants.csv: participant_id (synthetic key), age (years), sex (synthetic female/male), gds15 (0–15), gad7 (0–21), mmse and moca (simulated here 10–30), label (0–4), class_name. Class order: normal, depression, anxiety, cognitive_decline, dementia. These mutually exclusive simulation labels are not clinical diagnostic criteria.

synthetic_raw_data.csv: participant_id, session (0–9), timestamp (synthetic five-minute spacing), and 13 numerical features described in data_dictionary.csv. Empty CSV fields are missing values. Screening scores, demographics, identifiers and labels are excluded from model inputs. Each feature modality has a capped missingness count of at most two sessions per person. The script records actual missingness rates.

The generator uses screening eligibility thresholds from the article, with an explicitly chosen normalized-severity rule to resolve overlaps. Rejection sampling enforces class balance. Conditional features depend on screening-derived severities, participant random effects and correlated session noise. This designed relationship can make synthetic classification easier than real clinical prediction. Synthetic features are proxies; no audio waveforms or text are generated.

## Code information and methodology

- generate_synthetic_data.py: deterministic data generator, clinical-screen eligibility rule, feature schema.
- run_study.py: participant-stratified 70/15/15 splits; offline short-gap interpolation inside participants; train-only median imputation and min-max scaling; random forest; optional neural baselines; fusion; metrics and ablations.
- models.py: optional PyTorch LSTM (128 units), 3x3 CNN over session-feature matrices, and numerical-feature Transformer. Adam learning rate 0.001, batch size 32. Fixed 30 epochs by default; no claim of optimized hyperparameters.
- explain.py: model-agnostic SHAP and LIME for the saved RF or complete probability fusion, with real computed attribution outputs.
- requirements.txt: core dependency ranges. VERIFIED_ENVIRONMENT.txt records actual locally tested versions.

Full-mode fusion is a convex weighted average of class probabilities. Weights minimize validation log loss. This is a disclosed approximation, not the unspecified learned representation fusion in the article. Component-removal ablations re-optimize remaining weights on validation predictions; base learners are held fixed. RF modality ablations retrain separate RF models. Five-fold CV covers RF only and re-fits preprocessing in each training fold; it is not full-architecture cross-validation.

Each test prediction uses all ten observed sessions of one held-out participant. This is retrospective sequence classification, not forecasting, continuous online validation, or external clinical validation. Offline interpolation may use later observations within that sequence.

## Outputs

results/ contains splits.csv, test_predictions.csv, metrics.csv, class_metrics.csv, confusion_matrix.csv, bootstrap_ci.csv, subgroups.csv, calibration_any_condition.csv, decision_curve_any_condition.csv, timings.csv, preprocessing.joblib, saved models, explanation_inputs.npz and run_metadata.json. CV adds rf_training_cv.csv. Full mode adds fusion_weights.json and neural weights. explain.py adds SHAP arrays and summary, LIME HTML and explanation metadata.

Metrics use macro precision/recall/F1; multiclass AUC is macro one-vs-rest. Multiclass Brier is mean SUM of squared class probability errors (range 0–2), not a binary Brier. MAE/RMSE are on probabilities versus one-hot labels, not arbitrary numeric class codes. Binary calibration and decision curves define the event as ANY non-normal reference label (80% prevalence by construction); they cannot reproduce an article curve assuming another endpoint/prevalence. Sensitivity and specificity are one-vs-rest. Confidence intervals use resampling of held-out participants with fixed fitted models; they do not include retraining uncertainty. Subgroup metrics are descriptive; no fairness or significance claim follows from them.

## Reproducibility and verification

Data generation is seeded. Exact neural outputs may differ by dependency versions/platform. Timings describe the current run and hardware, not the article's Colab T4 environment. This package's RF workflow was run on the complete default dataset. See VALIDATION.md for which routes were actually executed. To verify generation repeatability, run the generator into a second directory with the same seed and compare CSV hashes.

## Before PeerJ resubmission

Resolve the differences in IMPLEMENTATION_NOTES.md, run the final agreed configuration, update the manuscript from actual output predictions, and provide the real files supporting the revised results. Do not describe this newly generated data as original raw data supporting the existing numerical claims. The original audio/BERT pipeline needs original assets/code or corresponding manuscript revisions. No automatic journal upload or public publication is performed.

## Citation, license and contributions

Cite the manuscript by its actual authors and title; add its DOI only if assigned. This package has no assigned DOI. License is pending author choice; no third-party license rights are asserted. Before public repository release, select a code license and a synthetic-data license and record them. Contributions should state assumptions, include reproducible commands and update the validation record. Direct project questions to the manuscript's corresponding author.
