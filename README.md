# Multiphase MRI-based prediction of extrahepatic recurrence after HCC resection

Minimal research code for MRI preprocessing, single-modality survival models,
imaging fusion (IFM), clinical-radiologic modeling (CRM), and statistical validation.
The endpoint is **extrahepatic recurrence (EHR)**, regardless of preceding or
concurrent intrahepatic recurrence. Legacy `Distant_Metastasis` and
`Time_to_DM_Months` column names are retained for compatibility, not as a different endpoint.

**No patient-level data, medical images, trained weights, pretrained weights,
manuscript files, or patient predictions are included.** This is research code,
not a validated clinical deployment or a CT surveillance recommendation.

## Installation and checks

Run from the repository root. The tested release environment uses Python 3.11.
It is not a byte-identical record of the original Python 3.10 training environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_competing_risk_metrics.py tests/test_clinical_preprocessing.py -q
```

For the image branches and full test suite, install the CPU build below, or choose
an appropriate GPU build from the official PyTorch installation guide:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ -q
python tools/smoke_test.py
python tools/audit_release.py
```

For radiomics extraction, install `requirements-radiomics.txt`. PyRadiomics 3.0.1
may require a C compiler when a wheel is unavailable. The tested installation
instructions and any platform limits are recorded in `VERIFICATION.md`.
R with `cmprsk` is needed only for the Gray-test script.

## Inputs and paths

Use deidentified local inputs, following the files under `schemas/`.
`configs/example_paths.env` lists environment variables; placeholder paths must
be replaced with your own. No code uploads clinical data to a service.

1. `clinical.xlsx`: one row per patient, seven clinical variables and endpoint records.
2. `splits.csv`: patient identifier, `Set`, and fixed development `Fold` (0-4).
3. `processing_list.csv`: patient identifier and local MRI directory.
4. AP/PVP/DP images with corresponding reviewed tumor masks.
5. `events.csv`: frozen competing-event time and status, as described in
   `schemas/competing_events_schema.md`.

An external patient must not appear in a development fold. Do not rebuild study
partitions from dates or infer institutional membership from filenames.

## Workflow

```bash
python 01_data_preparation/preprocess_multiphase_mri.py --roi-mode tumor_plus_5mm --list-file data/processing_list.csv --output-dl-dir data/dl_tumor_plus5mm --output-rad-dir data/radiomics_nifti
python 01_data_preparation/extract_multiphase_radiomics.py
python 02_single_modality_models/screen_image_models.py --save-final-checkpoints
python 02_single_modality_models/screen_radiomics_models.py
python 02_single_modality_models/select_single_modality_models.py
python 03_imaging_fusion/train_late_score_fusion.py
python 04_clinical_radiologic_model/train_clinical_radiologic_model.py --input-dir outputs/imaging_fusion/late/<run_tag>
```

After setting `MR_PREDICTIONS_CSV` to the selected CRM `final_predictions.csv`:

```bash
python 05_statistical_validation/evaluate_model_performance.py
python 05_statistical_validation/evaluate_competing_risk_metrics.py --input outputs/statistical_validation/analysis_dataset_with_clinical_baselines.csv --events-csv data/events.csv --output outputs/statistical_validation/primary
Rscript 05_statistical_validation/evaluate_competing_risks.R outputs/statistical_validation/primary/competing_risk_input.csv outputs/statistical_validation/primary
```

For early-feature fusion, set `MR_LATE_PREDICTIONS_CSV` to the selected IFM output
and run `03_imaging_fusion/compare_early_fusion.py --save-embeddings`. It requires
the locally generated fold checkpoints, branch-selection table and radiomics
features; it cannot run against the publication weights because those are not provided.

## Method boundaries

- Phase-specific crops have a nominal 5-mm bounding-box margin after 1-mm
  isotropic resampling. No interphase registration is performed.
- Radiomics uses 107 fixed Original features per phase (321 total), without
  outcome-based feature selection.
- Image selection uses final-epoch development OOF C-index. External branch
  scores average the five fold models. Model selection does not use external outcomes.
- Fusion and CRM reuse upstream OOF scores; these stages are not fully nested
  end-to-end cross-validation. Preprocessing statistics must be fitted on training data only.
- The primary probability layer combines cause-specific Cox models for EHR
  and competing death. Monthly IPCW Brier scores are integrated over months 1-36.
  A three-horizon arithmetic mean is not an integrated Brier score.
- Development probability mappings are fitted and evaluated in the same
  development cohort and are not fully cross-fitted. Bootstrap intervals are
  conditional on the fitted models and probability mappings.
- Calibration against Kaplan-Meier and decision curves generated by
  `evaluate_model_performance.py` are **net-risk sensitivity analyses**, treating
  competing death as censoring. They are not competing-risk CIF calibration or
  a validation of a chest CT schedule.
- Exploratory groups use model-specific development OOF 86th-percentile
  cutoffs, applied unchanged externally. This post hoc choice was informed by
  previously examined external subgroup results; it is not a prespecified or
  independently validated clinical decision threshold.

## Reproducibility and exclusions

Tests use synthetic inputs and verify execution, transformations, joins and
metric calculations. Passing them does **not** demonstrate independent replication
of manuscript performance. Exact numerical reproduction requires the frozen
cohort, reviewed masks, preprocessing cache, fold assignments and model weights.
Changes in hardware or software can also affect retraining results.

Pretrained initialization weights are not redistributed. Obtain ImageNet/Med3D
weights from their licensed sources, then set `MR_PRETRAIN_DIR`. Do not interpret
a random-initialization test as reproduction of pretrained study models.

Exploratory RNA-seq scripts, manuscript plotting/assembly, private cohort audits
and clinical applications are outside this minimal release. Data and weight
requests remain subject to institutional review; see `MODEL_WEIGHTS_ACCESS.md`.
Third-party notices are listed in `THIRD_PARTY_NOTICES.md`.
