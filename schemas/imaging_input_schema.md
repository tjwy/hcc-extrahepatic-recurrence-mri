# Imaging input schema

Each case must have a unique `MRI_id` and three contrast-enhanced MRI phases:

```text
raw_mri/<MRI_id>/AP.nii.gz
raw_mri/<MRI_id>/PVP.nii.gz
raw_mri/<MRI_id>/DP.nii.gz
raw_mri/<MRI_id>/AP_mask.nii.gz
raw_mri/<MRI_id>/PVP_mask.nii.gz
raw_mri/<MRI_id>/DP_mask.nii.gz
```

The locked workflow uses the PVP tumor segmentation to define the maximal tumor slice and tumor-centered views. Each phase is cropped separately with a nominal five-voxel margin after 1-mm isotropic resampling. No interphase registration is performed. The resized AP, PVP, and DP arrays are used jointly by the image branches; equal tensor dimensions do not imply anatomical registration.

`DP` is a legacy input filename. Record the actual contrast agent and acquisition time in your study metadata; a gadoxetate transitional phase is not a conventional extracellular-agent delayed phase.

Radiomics uses the tumor segmentation within the same resampled tumor-plus-5-mm crop and produces columns matching:

```text
^(AP|PVP|DP)_original_
```

Diagnostic PyRadiomics columns are excluded. The expected feature count is 107 per phase and 321 in total.
