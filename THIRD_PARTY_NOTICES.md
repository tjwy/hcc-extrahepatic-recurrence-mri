# Third-party components

This release depends on PyTorch/torchvision, scikit-survival, scikit-learn,
lifelines, NumPy, SciPy, pandas, nibabel, SimpleITK and PyRadiomics. Their
original licenses and citation requirements apply independently of this repository.

The 3D residual-network implementation follows the MedicalNet/Med3D architecture
and common ResNet building blocks. See https://github.com/Tencent/MedicalNet
for the original project, license, and pretrained-weight distribution conditions.
No third-party pretrained weights are included or relicensed here.
The MedicalNet and 3D-ResNets-PyTorch notices are retained in
`third_party/MedicalNet_LICENSE.txt`.

ImageNet initialization uses torchvision's official model implementations and
weight distribution. The original datasets and software are not owned by this repository.

The cmprsk package is distributed separately through CRAN under its original license.
