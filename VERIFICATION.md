# Tests and implementation notes

## Test coverage

Tests cover MRI preprocessing, model input construction, survival-model
interfaces, clinical-variable preprocessing, competing-risk metrics, and a
synthetic branch-score to IFM/CRM workflow. They do not reproduce the study's
numerical results without the study data and model checkpoints.

The Windows check on 2026-10-14 completed with 42 tests passed and 2 skipped.
The skipped checks required Rscript and PyRadiomics, which were unavailable in
that process. Linux checks are reported under the repository's Actions tab.

## Optional dependencies

PyRadiomics 3.0.1 may require a C compiler for installation. After installing
the base requirements, use:

```bash
python -m pip install wheel==0.45.1
python -m pip install --no-build-isolation -r requirements-radiomics.txt
```

The Gray-test script requires R with `cmprsk`. Tests that require unavailable
optional dependencies are reported as skipped.

ImageNet and Med3D initialization weights are obtained separately from their
original sources. Set `MR_PRETRAIN_DIR` to the local weights directory.

## Implementation notes

- MRI crops use a nominal 5-mm margin after 1-mm isotropic resampling, without
  interphase registration. Radiomics uses 107 Original features per phase.
- Image-model selection uses final-epoch development OOF C-index. External
  branch scores average predictions from the five fold-specific models.
- Fusion and CRM reuse upstream OOF scores without refitting the upstream
  models within each fusion-validation split. This is not fully nested
  end-to-end cross-validation.
- Early-feature fusion follows the original checkpoint-specific folds. Late-score
  learner selection uses repeated five-fold validation on fixed OOF scores;
  the two approaches use different patient partitions.
- The primary probability layer combines cause-specific Cox models for EHR and
  competing death. IPCW Brier scores are integrated over monthly horizons from
  1 to 36 months. Development probability mappings are not fully cross-fitted.
- Calibration and decision curves from `evaluate_model_performance.py` are
  net-risk sensitivity analyses that treat competing death as censoring.
- Exploratory risk groups use model-specific development OOF 86th-percentile
  cutoffs, applied unchanged externally. The cutoff choice was informed by
  previously examined external subgroup results and is not an independently
  validated clinical decision threshold.

Patient data, medical images, model weights, transcriptomic analyses, and
manuscript figure-generation scripts are not distributed in this package.
