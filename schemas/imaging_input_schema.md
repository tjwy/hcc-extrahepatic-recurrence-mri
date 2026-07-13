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

The locked workflow uses the PVP tumor segmentation to define the maximal tumor slice and tumor-centered views. The region of interest is expanded by 5 mm in physical space. AP, PVP, and DP images are spatially aligned and used jointly by the 2D, 2.5D, and 3D branches.

Radiomics uses the tumor segmentation within the same resampled tumor-plus-5-mm crop and produces columns matching:

```text
^(AP|PVP|DP)_original_
```

Diagnostic PyRadiomics columns are excluded. The expected feature count is 107 per phase and 321 in total.
