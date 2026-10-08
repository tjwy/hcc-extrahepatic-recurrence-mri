import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def synthetic_images(folder):
    folder.mkdir()
    image = np.random.default_rng(13).normal(size=(24, 24, 12)).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[6:12, 6:12, 3:7] = 1
    affine = np.diag([2., 2., 2., 1.])
    for index, phase in enumerate(("AP", "PVP", "DP")):
        nib.save(nib.Nifti1Image(image + index, affine), folder / f"{phase}.nii.gz")
        nib.save(nib.Nifti1Image(mask, affine), folder / f"{phase}_mask.nii.gz")


def test_preprocessing_generates_finite_three_phase_tensor(tmp_path):
    prep = load("01_data_preparation/preprocess_multiphase_mri.py", "prep_synthetic")
    source, dl, rad = tmp_path / "source", tmp_path / "dl", tmp_path / "rad"
    dl.mkdir()
    synthetic_images(source)
    assert prep.process_case("SYN-001", str(source), str(dl), str(rad), roi_margin=5) == "Success"
    values = np.load(dl / "SYN-001.npy")
    assert values.shape == (3, 32, 128, 128)
    assert np.isfinite(values).all()
    assert np.allclose(values.mean(axis=(1, 2, 3)), 0, atol=1e-4)
    assert np.allclose(np.sqrt(np.sum(nib.load(rad / "SYN-001/AP.nii.gz").affine[:3, :3] ** 2, axis=0)), 1)


def test_preprocessing_rejects_mask_geometry_mismatch(tmp_path):
    prep = load("01_data_preparation/preprocess_multiphase_mri.py", "prep_bad_geometry")
    source, dl, rad = tmp_path / "source", tmp_path / "dl", tmp_path / "rad"
    dl.mkdir()
    synthetic_images(source)
    mask = nib.load(source / "AP_mask.nii.gz")
    affine = mask.affine.copy()
    affine[0, 3] = 20.
    nib.save(nib.Nifti1Image(np.asarray(mask.dataobj), affine), source / "AP_mask.nii.gz")
    assert "geometry mismatch" in prep.process_case("SYN-001", str(source), str(dl), str(rad), roi_margin=5)
    assert not (dl / "SYN-001.npy").exists()


def test_radiomics_produces_exactly_107_features_per_phase(tmp_path, monkeypatch):
    pytest.importorskip("radiomics")
    source = tmp_path / "images" / "SYN-001"
    source.parent.mkdir()
    synthetic_images(source)
    monkeypatch.setenv("MR_RADIOMICS_DATA_DIR", str(source.parent))
    monkeypatch.setenv("MR_RADIOMICS_OUTPUT_FILE", str(tmp_path / "synthetic_features.csv"))
    monkeypatch.setenv("MR_RADIOMICS_LOG_FILE", str(tmp_path / "synthetic.log"))
    rad = load("01_data_preparation/extract_multiphase_radiomics.py", "radiomics_synthetic")
    result = rad.extract_features("SYN-001")
    assert result is not None
    assert len(result) == 322
    for phase in ("AP", "PVP", "DP"):
        assert len([key for key in result if key.startswith(phase + "_original_")]) == 107
    (source / "PVP_mask.nii.gz").unlink()
    assert rad.extract_features("SYN-001") is None
