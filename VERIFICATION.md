# Verification scope

## Local checks

The release candidate was checked in an isolated Windows Python 3.11 environment
using the pinned base dependencies and PyTorch 2.6.0 / torchvision 0.21.0 CPU builds.
The local suite completed with **43 tests passed and 1 skipped** on 2026-10-08;
the skipped test is PyRadiomics extraction. The synthetic workflow and `pip check`
also passed. Linux CI results should be read from the repository Actions page.

On 2026-10-14, the local suite was rerun after recurrence-terminology updates:
42 tests passed and 2 were skipped. The skipped checks were the optional
Rscript/Gray-test interface and PyRadiomics extraction, which were unavailable
in that local process. This rerun did not use private patient data or retrain
the study models.

Checks cover:

- Synthetic MRI preprocessing and explicit rejection of image/mask geometry mismatch.
- 3-channel 2D, 9-channel 2.5D, and 3-channel 3D input construction.
- Forward passes of the selected VGG19-BN, MobileNetV2, and 3D ResNet10 architectures.
- Finite Cox-loss gradients; interface tests use explicit random initialization only.
- Development-only imputation, fixed categorical encoding, and preservation of missingness.
- OOF feature provenance and internal-only learner selection.
- Cause-specific Cox CIF mapping, Aalen-Johansen state probabilities, competing-death controls,
  IPCW AUC/Brier calculations, and normalized monthly trapezoidal IBS integration.
- Gray-test execution using R 4.4.1 and cmprsk 2.2.12, including a one-group edge case.
- A 120-subject synthetic branch-score -> IFM -> CRM -> statistical validation workflow.
- Python syntax, dependency checks, source-file hashes, and sensitive-content scanning of Git history.

## Platform limitations

Installing PyRadiomics 3.0.1 from source in the isolated Windows environment
requires Microsoft C++ Build Tools, which were not installed as part of this review.
Its extraction test is skipped locally rather than reported as passed.
The optional Linux CI job installs a C-compiler-capable runtime and tests extraction
of all 107 Original features per phase, as well as rejecting incomplete phases.

For a source install, first install the base requirements and wheel, then:

```bash
python -m pip install wheel==0.45.1
python -m pip install --no-build-isolation -r requirements-radiomics.txt
```

A failed installation is an environment/dependency failure, not evidence that
feature extraction executed successfully. No system-wide compiler or packages
were installed or modified by these checks.

## What passing tests does not establish

No private patient data, original images, manuscript weights, or full 50-epoch
training runs were used. These checks are not strict numerical replication of
the reported C-indexes, AUCs, confidence intervals, or RNA-seq results.
Release software differs from historical training software. The frozen study
inputs and models remain necessary for exact study-output reproduction.

This candidate removes obsolete three-year 20% risk grouping and sparse-horizon
Brier scripts. The current exploratory cutoff policy and competing-risk IBS
implementation are documented separately from net-risk sensitivity analyses.
Interface corrections do not update the frozen manuscript results or clinical records.
