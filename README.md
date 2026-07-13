# Multiphase MRI Model for Distant Metastasis after HCC Resection

This repository contains the minimal analysis code supporting the manuscript workflow. The code is ordered by the Methods narrative.

## Study workflow

1. Prepare AP, PVP, and DP MRI inputs using a tumor-plus-5-mm region of interest.
2. Screen 2D, 2.5D, 3D, and fixed-feature radiomics models in five-fold cross-validation.
3. Compare input-matched deep-feature early fusion with four-score late fusion.
4. Combine the selected imaging-fusion score with seven prespecified clinical variables using Coxnet.
5. Evaluate discrimination, calibration, decision-curve utility, absolute risk, and competing risks.

Model and preprocessing selection use development-cohort out-of-fold results. External outcomes are not used to select architectures, fusion learners, clinical preprocessing, or risk thresholds.

## Repository layout

```text
01_data_preparation/             Multiphase MRI and radiomics inputs
02_single_modality_models/       2D, 2.5D, 3D, and radiomics screening
03_imaging_fusion/               Early-feature and late-score fusion
04_clinical_radiologic_model/    Imaging plus clinical Coxnet modeling
05_statistical_validation/       Performance and competing-risk analyses
configs/                          Path and radiomics examples
schemas/                          Required input columns and formats
tests/                            Leakage and fold-provenance tests
tools/                            Public-release audit
```

Manuscript plotting and figure-assembly scripts are intentionally excluded.

## Environment

```bash
conda env create -f environment.yml
conda activate hcc-dm-mri
```

The manuscript statistical snapshot used Python 3.11.5. The public environment uses Python 3.10. Competing-risk summaries require R and the `cmprsk` package.

Copy `configs/example_paths.env` to `.env.local`, replace the example paths, and load it before running the workflow:

```bash
set -a
source .env.local
set +a
```

## Restricted inputs

The following inputs are not distributed because they contain clinical or imaging data:

```text
data/clinical.xlsx
data/splits.csv
data/processing_list.csv
data/raw_mri/<case_id>/{AP,PVP,DP}.nii.gz
data/raw_mri/<case_id>/{AP,PVP,DP}_mask.nii.gz
data/radiomics_features_raw.csv
```

Column requirements are documented in `schemas/`. Configure local paths with environment variables shown in `configs/example_paths.env`.
ImageNet weights are downloaded by the training code. Med3D initialization weights are not redistributed; place the appropriately licensed files in `MR_PRETRAIN_DIR` using the names documented in `02_single_modality_models/model_components.py`, or the 3D candidates will use random initialization.

## 1. Data preparation

```bash
python 01_data_preparation/preprocess_multiphase_mri.py \
  --roi-mode tumor_plus_5mm \
  --list-file data/processing_list.csv \
  --output-dl-dir data/dl_tumor_plus5mm \
  --output-rad-dir data/radiomics_nifti

python 01_data_preparation/extract_multiphase_radiomics.py
```

The deep-learning array contains AP/PVP/DP inputs. Radiomics extraction uses the tumor segmentation within the same preprocessed crop and retains 107 `Original` PyRadiomics features per phase, giving 321 features without outcome-based feature selection.

## 2. Single-modality models

```bash
python 02_single_modality_models/screen_image_models.py \
  --save-final-checkpoints
python 02_single_modality_models/screen_radiomics_models.py
python 02_single_modality_models/select_single_modality_models.py
```

Candidate image models are trained for 50 epochs in fixed five-fold cross-validation. Selection uses final-epoch pooled out-of-fold C-index. The canonical image selection table is written to `outputs/single_modality_models/selected/selected_models.csv`. The retained branches are represented as `Risk_2D`, `Risk_2p5D`, `Risk_3D`, and `Risk_Rad`.

## 3. Imaging fusion

```bash
python 03_imaging_fusion/train_late_score_fusion.py
export MR_LATE_PREDICTIONS_CSV=outputs/imaging_fusion/late/<run_tag>/all_selected_predictions.csv
python 03_imaging_fusion/compare_early_fusion.py --save-embeddings
```

Early fusion concatenates fold-specific deep embeddings with 321 radiomics features. Late fusion combines the four branch risk scores. CoxPH, random survival forest, survival support vector machine, gradient boosting, and ExtraST are compared using development-cohort out-of-fold C-index.

## 4. Clinical-radiologic model

```bash
python 04_clinical_radiologic_model/train_clinical_radiologic_model.py \
  --input-dir outputs/imaging_fusion/late/<run_tag>
```

The clinical-radiologic model combines the imaging-fusion score with age, sex, the Child-Pugh variable, alpha-fetoprotein, BCLC stage, tumor size, and tumor number. Preprocessing and the Coxnet learner are selected using development-cohort pooled out-of-fold C-index; external metrics are diagnostic only. The retained code-facing field name is documented in `schemas/clinical_data_dictionary.csv`. The selected pipeline writes `final_predictions.csv` in its run directory.

## 5. Statistical validation

```bash
export MR_PREDICTIONS_CSV=outputs/clinical_radiologic_model/<run_tag>/final_predictions.csv
python 05_statistical_validation/evaluate_model_performance.py
python 05_statistical_validation/calibrate_absolute_risk.py
python 05_statistical_validation/evaluate_ipcw_brier.py
Rscript 05_statistical_validation/evaluate_competing_risks.R \
  outputs/statistical_validation/competing_risk_input.csv \
  outputs/statistical_validation/competing_risk
```

The competing event is death before distant metastasis. Cumulative incidence is compared with Gray's test. The fixed high-risk operating point is a Cox-calibrated 3-year absolute distant-metastasis risk of at least 20%.

## Verification

Run the package checks before release:

```bash
python tools/audit_release.py
pytest tests/ -q
```

## Data and model availability

De-identified clinical and imaging data are not publicly distributed. Model weights may be requested from the corresponding author and are released only after institutional review and execution of an appropriate data-use agreement. See `MODEL_WEIGHTS_ACCESS.md`.

The code is provided under the license in `LICENSE`.
