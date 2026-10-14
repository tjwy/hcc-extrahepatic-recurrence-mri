# MRI-based prediction of extrahepatic recurrence in HCC

Code for predicting extrahepatic recurrence after hepatocellular carcinoma
resection using multiphase MRI. The workflow includes 2D, 2.5D, and 3D deep-learning
models, radiomics, imaging fusion, clinical integration, and survival analysis.

Patient data, medical images, and model weights are not included.

## Installation

Python 3.11 is used for this code package.

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

The command above installs CPU builds. Use a compatible GPU build for training.
Radiomics extraction additionally requires `requirements-radiomics.txt`;
the Gray-test script requires R and `cmprsk`.

## Usage

Configure the environment variables listed in `configs/example_paths.env` for
your local data. Input formats are described in `schemas/`.
Run the scripts in the following order:

| Directory | Contents |
| --- | --- |
| `01_data_preparation` | MRI preprocessing and radiomics extraction |
| `02_single_modality_models` | Image and radiomics model development |
| `03_imaging_fusion` | Early-feature and late-score fusion |
| `04_clinical_radiologic_model` | Integration of imaging and clinical variables |
| `05_statistical_validation` | Discrimination, competing risks, and prediction error |

For example, after generating the branch scores:

```bash
python 03_imaging_fusion/train_late_score_fusion.py
python 04_clinical_radiologic_model/train_clinical_radiologic_model.py --input-dir outputs/imaging_fusion/late/<run_tag>
```

The legacy columns `Distant_Metastasis` and `Time_to_DM_Months` refer to the EHR
event indicator and observation time. They are retained for input compatibility.

## Tests

```bash
python -m pytest tests/ -q
python tools/smoke_test.py
python tools/audit_release.py
```

Tests use synthetic data. See [VERIFICATION.md](VERIFICATION.md) for coverage,
implementation notes, and optional dependencies.

## Data and weights

This repository contains code only. Transcriptomic analyses and manuscript
figure-generation scripts are not included. See
[MODEL_WEIGHTS_ACCESS.md](MODEL_WEIGHTS_ACCESS.md) for access conditions and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party software notices.
