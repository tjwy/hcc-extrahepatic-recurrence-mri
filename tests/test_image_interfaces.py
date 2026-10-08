import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def components(tmp_path, monkeypatch):
    monkeypatch.setenv("MR_TORCH_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("MR_PRETRAIN_DIR", str(tmp_path / "weights"))
    torch.set_num_threads(1)
    result = load("02_single_modality_models/model_components.py", "image_components_test")
    # Explicit random initialization for interface tests only; no weight downloads.
    monkeypatch.setattr(result, "load_pretrained_state_once", lambda name: None)
    monkeypatch.setattr(result, "load_med3d_weights", lambda *args: None)
    trainer = load("02_single_modality_models/screen_image_models.py", "image_trainer_test")
    trainer.install_multichannel_risk2d(result)
    trainer.install_multichannel_risk3d(result)
    return result, trainer


@pytest.mark.parametrize("backbone,channels", [("vgg19_bn", 3), ("mobilenet_v2", 9)])
def test_selected_2d_and_multiplanar_forward(components, backbone, channels):
    models, _ = components
    model = models.Risk2DNet(backbone=backbone, dropout=0., input_channels=channels).eval()
    with torch.no_grad():
        prediction = model(torch.randn(1, channels, 128, 128))
    assert prediction.shape == (1, 1)
    assert torch.isfinite(prediction).all()


def test_selected_3d_forward(components):
    models, _ = components
    model = models.Risk3DNet(backbone="resnet10_3d", dropout=0., input_channels=3).eval()
    with torch.no_grad():
        prediction = model(torch.randn(1, 3, 32, 32, 32))
    assert prediction.shape == (1, 1)
    assert torch.isfinite(prediction).all()


def test_maskaware_input_channels(components, tmp_path, monkeypatch):
    models, trainer = components
    monkeypatch.setattr(models, "IMG_DIR", str(tmp_path))
    np.save(tmp_path / "SYN-001.npy", np.random.default_rng(1).normal(size=(3, 32, 128, 128)).astype(np.float32))
    frame = pd.DataFrame({"MRI_id": ["SYN-001"], "Time_to_DM_Months": [12.], "Distant_Metastasis": [1]})
    trainer.install_maskaware_dataset(models, {"SYN-001": {"MaskAware_Axial_Z": 16,
                                                            "MaskAware_Coronal_Y": 64,
                                                            "MaskAware_Sagittal_X": 64}})
    for family, expected in [("2d", (3, 256, 256)), ("2p5d", (9, 256, 256)), ("3d", (3, 64, 64, 64))]:
        item = models.HCCImageDataset(frame, mode=family)[0]
        assert tuple(item["x"].shape) == expected
        assert torch.isfinite(item["x"]).all()


def test_cox_loss_backward_is_finite(components):
    models, _ = components
    risks = torch.tensor([.1, .3, -.1], requires_grad=True)
    loss = models.cox_ph_loss(risks, torch.tensor([1., 0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(risks.grad).all()


def test_selected_batch_sizes_match_checkpoint_metadata(components):
    _, trainer = components
    assert trainer.BATCH_SIZE_OVERRIDES[("2d", "vgg19_bn")] == 8
    assert trainer.BATCH_SIZE_OVERRIDES[("2p5d", "mobilenet_v2")] == 4
    assert trainer.BATCH_SIZE_OVERRIDES[("3d", "resnet10_3d")] == 2
