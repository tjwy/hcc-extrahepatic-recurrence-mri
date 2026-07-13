import os
import argparse
import datetime
import random
import warnings
import shutil
import urllib.request
import re
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM

import nibabel as nib
from torchvision import models as tv_models

warnings.filterwarnings("ignore")

CLINICAL_FIXED_7 = ['Age', 'Sex', 'Child_Pugh_Score', 'AFP', 'BCLC', 'Tumor_Size', 'Tumor_Number']

ROOT = os.environ.get("MR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
SINGLE_MODALITY_ROOT = os.path.join(ROOT, "outputs", "single_modality_models")


def _resolve_output_dir(default_name):
    run_output_root = str(os.environ.get("MR_RUN_OUTPUT_ROOT", "")).strip()
    if run_output_root == "":
        return os.path.join(SINGLE_MODALITY_ROOT, default_name)
    out = os.path.join(run_output_root, "stage_outputs", default_name)
    os.makedirs(out, exist_ok=True)
    return out

EXCEL_PATH = os.environ.get("MR_CLINICAL_EXCEL", os.path.join(ROOT, "data", "clinical.xlsx"))
SPLIT_PATH = os.environ.get("MR_SPLIT_CSV", os.path.join(ROOT, "data", "splits.csv"))
IMG_DIR = os.environ.get("MR_IMG_DIR", os.path.join(ROOT, "data", "dl_tumor_plus5mm"))
RAD_PRED_DIR = _resolve_output_dir("results_radiomics")
RAD_VAL_PRED = os.path.join(RAD_PRED_DIR, "radiomics_val_predictions.csv")
RAD_TEST_PRED = os.path.join(RAD_PRED_DIR, "radiomics_test_predictions.csv")
RAD_SUMMARY_DIR = _resolve_output_dir("results_radiomics")

OUT_DIR = _resolve_output_dir("results_fusion")
LOG_DIR = _resolve_output_dir("logs")
TORCH_HOME_DIR = os.environ.get("MR_TORCH_HOME", os.path.join(ROOT, "cache_torch"))
PRETRAIN_DIR = os.environ.get("MR_PRETRAIN_DIR", os.path.join(TORCH_HOME_DIR, "hub", "checkpoints"))
USER_PRETRAIN_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints")
EXTERNAL_SET_VALUES = ["External_Zhongshan", "External_Test"]
os.environ.setdefault("TORCH_HOME", TORCH_HOME_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PRETRAIN_DIR, exist_ok=True)

TV_PRETRAIN_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}
TV_STATE_CACHE = {}

MED3D_WEIGHT_CANDIDATES = {
    "resnet10_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_10_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_10.pth"),
    ],
    "resnet18_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_18_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_18.pth"),
    ],
    "resnet34_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_34_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_34.pth"),
    ],
    "resnet50_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_50_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_50.pth"),
    ],
    "resnet101_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_101_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_101.pth"),
    ],
    "resnet152_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_152_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_152.pth"),
    ],
    "resnet200_3d": [
        os.path.join(PRETRAIN_DIR, "resnet_200_23dataset.pth"),
        os.path.join(PRETRAIN_DIR, "resnet_200.pth"),
    ],
}


def sanitize_tag(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unspecified"


def infer_roi_mode():
    explicit = str(os.environ.get("MR_ROI_MODE", "")).strip()
    if explicit:
        return explicit

    base = os.path.basename(os.path.normpath(IMG_DIR))
    mapping = {
        "Processed_Data_DL": "tumor_plus_10mm",
        "Processed_Data_DL_tumor_only": "tumor_only",
        "Processed_Data_DL_tumor_plus5mm": "tumor_plus_5mm",
    }
    return mapping.get(base, base or "unspecified_roi")


def get_run_context():
    roi_mode = infer_roi_mode()
    return {
        "roi_mode": roi_mode,
        "roi_tag": sanitize_tag(roi_mode),
        "img_dir": IMG_DIR,
        "split_path": SPLIT_PATH,
    }


def build_tagged_output_path(out_dir, stem, now_tag, ext):
    ctx = get_run_context()
    return os.path.join(out_dir, f"{stem}_{ctx['roi_tag']}_{now_tag}.{ext}")


def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size, shuffle, num_workers, seed, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def cox_ph_loss(risk_pred, event, eps=1e-7):
    risk_pred = risk_pred.view(-1)
    event = event.view(-1)
    if event.sum() < 1:
        return torch.tensor(0.0, requires_grad=True, device=risk_pred.device)
    log_h = risk_pred
    log_cumsum_h = torch.logcumsumexp(log_h, dim=0)
    loss = -torch.sum(event * (log_h - log_cumsum_h)) / (event.sum() + eps)
    return loss


def ensure_local_pretrain_file(fname):
    """Keep torchvision checkpoints inside workspace cache and reuse user cache if present."""
    local_path = os.path.join(PRETRAIN_DIR, fname)
    if os.path.exists(local_path):
        return local_path

    user_cache_path = os.path.join(USER_PRETRAIN_CACHE, fname)
    if os.path.exists(user_cache_path):
        try:
            shutil.copy2(user_cache_path, local_path)
            print(f"[{ts()}] Migrated user cache to workspace: {user_cache_path} -> {local_path}")
            return local_path
        except Exception as e:
            print(f"[{ts()}] Warning: failed to migrate user cache {user_cache_path}: {e}")

    return local_path


def _load_torch_checkpoint_safely(path):
    try:
        state = torch.load(path, map_location="cpu")
        return state, None
    except Exception as e:
        return None, e


def load_pretrained_state_once(backbone):
    if backbone in TV_STATE_CACHE:
        return TV_STATE_CACHE[backbone]

    url = TV_PRETRAIN_URLS.get(backbone)
    if url is None:
        TV_STATE_CACHE[backbone] = None
        return None

    fname = os.path.basename(url)
    cached = ensure_local_pretrain_file(fname)
    if not os.path.exists(cached):
        print(f"[{ts()}] No local pretrained cache for {backbone} at {cached}; using random init.")
        TV_STATE_CACHE[backbone] = None
        return None

    state, err = _load_torch_checkpoint_safely(cached)
    if err is None:
        TV_STATE_CACHE[backbone] = state
        print(f"[{ts()}] Loaded local pretrained {backbone} from {cached}")
        return state

    print(f"[{ts()}] Warning: failed to read local pretrained {backbone} ({cached}): {err}. The file might be corrupted.")
    try:
        os.remove(cached)
        print(f"[{ts()}] Removed corrupted cache for {backbone}: {cached}")
    except Exception as remove_err:
        print(f"[{ts()}] Warning: failed to remove corrupted cache for {backbone}: {remove_err}")
        TV_STATE_CACHE[backbone] = None
        return None

    try:
        torch.hub.load_state_dict_from_url(url, model_dir=PRETRAIN_DIR, map_location="cpu", progress=True)
    except Exception as e:
        try:
            urllib.request.urlretrieve(url, cached)
            print(f"[{ts()}] Re-downloaded pretrained cache for {backbone} via urllib fallback")
        except Exception as e2:
            print(f"[{ts()}] Warning: failed to refresh {backbone} cache after corruption: {e}; fallback failed: {e2}")
            TV_STATE_CACHE[backbone] = None
            return None

    state, err = _load_torch_checkpoint_safely(cached)
    if err is None:
        TV_STATE_CACHE[backbone] = state
        print(f"[{ts()}] Refreshed and loaded local pretrained {backbone} from {cached}")
        return state

    print(f"[{ts()}] Warning: refreshed cache for {backbone} is still unreadable: {err}")
    TV_STATE_CACHE[backbone] = None
    return None


def ensure_imagenet_cache(backbones):
    """Download missing torchvision weights into local cache before training."""
    uniq = sorted(set(backbones))
    if len(uniq) == 0:
        return
    print(f"[{ts()}] Precaching ImageNet weights for {len(uniq)} backbones at {PRETRAIN_DIR}...")
    for b in uniq:
        url = TV_PRETRAIN_URLS.get(b)
        if url is None:
            continue
        fname = os.path.basename(url)
        cached = ensure_local_pretrain_file(fname)
        if os.path.exists(cached):
            _, err = _load_torch_checkpoint_safely(cached)
            if err is None:
                print(f"[{ts()}] Cache exists for {b}: {cached}")
                continue
            print(f"[{ts()}] Detected corrupted cache for {b}: {cached} | {err}")
            try:
                os.remove(cached)
            except Exception as remove_err:
                print(f"[{ts()}] Warning: failed to remove corrupted cache for {b}: {remove_err}")
                continue
        try:
            torch.hub.load_state_dict_from_url(url, model_dir=PRETRAIN_DIR, map_location="cpu", progress=True)
            print(f"[{ts()}] Cached ImageNet weights for {b}")
        except Exception as e:
            # Fallback downloader avoids torch.hub-specific write issues on some filesystems.
            try:
                urllib.request.urlretrieve(url, cached)
                print(f"[{ts()}] Cached ImageNet weights for {b} via urllib fallback")
            except Exception as e2:
                print(f"[{ts()}] Warning: failed to cache {b} from {url}: {e}; fallback failed: {e2}")


def load_med3d_weights(backbone_module, weight_path, backbone_name):
    if not os.path.exists(weight_path):
        print(f"[{ts()}] MED3D weights not found for {backbone_name}: {weight_path}; using random init.")
        return

    try:
        state = torch.load(weight_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        filtered = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            if nk.startswith("fc") or nk.startswith("classifier"):
                continue
            filtered[nk] = v

        missing, unexpected = backbone_module.load_state_dict(filtered, strict=False)
        print(f"[{ts()}] Loaded MED3D for {backbone_name}: {weight_path} | missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[{ts()}] Warning: failed loading MED3D for {backbone_name} from {weight_path}: {e}")


def resolve_med3d_weight_path(backbone_name):
    cands = MED3D_WEIGHT_CANDIDATES.get(backbone_name, [])
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0] if len(cands) > 0 else None


def ensure_med3d_weights(backbones):
    uniq = sorted(set(backbones))
    if len(uniq) == 0:
        return
    print(f"[{ts()}] Checking local MED3D weights for {len(uniq)} backbones...")
    for b in uniq:
        p = resolve_med3d_weight_path(b)
        if p is None:
            print(f"[{ts()}] Warning: no MED3D weight mapping for {b}; random init will be used.")
            continue
        if os.path.exists(p):
            print(f"[{ts()}] MED3D cache ready for {b}: {p}")
        else:
            print(f"[{ts()}] Warning: MED3D weights missing for {b}: {p}; random init will be used.")


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample_basic_block(x, out_channels, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    ch = out_channels - out.size(1)
    if ch <= 0:
        return out
    pad = torch.zeros(
        out.size(0),
        ch,
        out.size(2),
        out.size(3),
        out.size(4),
        device=out.device,
        dtype=out.dtype,
    )
    return torch.cat([out, pad], dim=1)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3DAny(nn.Module):
    def __init__(self, block, layers, input_channels=1, shortcut_type="B"):
        super().__init__()
        self.inplanes = 64
        self.shortcut_type = shortcut_type
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            out_channels = planes * block.expansion
            if self.shortcut_type == "A":
                downsample = lambda x: downsample_basic_block(x, out_channels, stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.inplanes, out_channels, stride),
                    nn.BatchNorm3d(out_channels),
                )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class HCCImageDataset(Dataset):
    def __init__(self, df, mode="2d", image_size=256):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_phase_volume(self, pid):
        # Primary format in this project: .npy with shape (3, D, H, W)
        npy_path = os.path.join(IMG_DIR, f"{pid}.npy")
        if os.path.exists(npy_path):
            try:
                arr = np.load(npy_path).astype(np.float32)
                if arr.ndim == 4 and arr.shape[0] >= 1:
                    # Use AP phase (index 0), aligned with arterial-phase modeling.
                    ap = arr[0]
                    if ap.std() < 1e-6:
                        return np.zeros((32, 128, 128), dtype=np.float32)
                    return ap
            except Exception:
                pass

        # Fallback format: .nii.gz
        nii_path = os.path.join(IMG_DIR, f"{pid}.nii.gz")
        if os.path.exists(nii_path):
            try:
                vol = nib.load(nii_path).get_fdata().astype(np.float32)
                if vol.ndim == 4:
                    if vol.shape[0] <= 4:
                        vol = vol[0]
                    else:
                        vol = vol[..., 0]
                if vol.ndim == 3:
                    m, s = vol.mean(), vol.std()
                    if s < 1e-6:
                        return np.zeros_like(vol)
                    return (vol - m) / (s + 1e-6)
            except Exception:
                pass

        return np.zeros((32, 128, 128), dtype=np.float32)

    def _resize_2d(self, arr2d):
        x = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["MRI_id"]).strip()
        t = np.float32(row["Time_to_DM_Months"])
        e = np.float32(row["Distant_Metastasis"])

        vol = self._load_phase_volume(pid)
        d, h, w = vol.shape

        # Pick the most informative axial slice by maximal |intensity| energy.
        energy = np.abs(vol).sum(axis=(1, 2))
        axial_idx = int(np.argmax(energy)) if np.max(energy) > 0 else d // 2

        if self.mode == "2d":
            ax = vol[axial_idx, :, :]
            x = self._resize_2d(ax)
            x = x.repeat(3, 1, 1)
        elif self.mode == "2p5d":
            ax = vol[axial_idx, :, :]
            cor = vol[:, h // 2, :]
            sag = vol[:, :, w // 2]
            ax = self._resize_2d(ax)
            cor = self._resize_2d(cor)
            sag = self._resize_2d(sag)
            x = torch.cat([ax, cor, sag], dim=0)
        else:
            x3 = torch.from_numpy(vol).float().unsqueeze(0)
            x3 = F.interpolate(x3.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False).squeeze(0)
            x = x3

        return {
            "x": x,
            "time": torch.tensor(t, dtype=torch.float32),
            "event": torch.tensor(e, dtype=torch.float32),
            "pid": pid,
        }


class Risk2DNet(nn.Module):
    def __init__(self, backbone="resnet18", dropout=0.3):
        super().__init__()
        # Build bare architecture first, then load cached pretrained weights once.
        if backbone == "resnet18":
            net = tv_models.resnet18(weights=None)
        elif backbone == "resnet34":
            net = tv_models.resnet34(weights=None)
        elif backbone == "resnet50":
            net = tv_models.resnet50(weights=None)
        elif backbone == "vgg16_bn":
            net = tv_models.vgg16_bn(weights=None)
        elif backbone == "vgg19_bn":
            net = tv_models.vgg19_bn(weights=None)
        elif backbone == "densenet121":
            net = tv_models.densenet121(weights=None)
        elif backbone == "densenet169":
            net = tv_models.densenet169(weights=None)
        elif backbone == "efficientnet_b0":
            net = tv_models.efficientnet_b0(weights=None)
        elif backbone == "efficientnet_b1":
            net = tv_models.efficientnet_b1(weights=None)
        elif backbone == "mobilenet_v2":
            net = tv_models.mobilenet_v2(weights=None)
        else:
            raise ValueError(f"Unsupported 2D backbone: {backbone}")

        state = load_pretrained_state_once(backbone)
        if state is not None:
            net.load_state_dict(state, strict=False)

        if backbone in ["resnet18", "resnet34", "resnet50"]:
            net.fc = nn.Identity()
        else:
            net.classifier = nn.Identity()

        self.backbone = net

        # Robustly infer feature dimension after replacing classifier/fc with Identity.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = int(self.backbone(dummy).shape[1])

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.backbone(x)
        r = self.head(z)
        return r


class Risk3DNet(nn.Module):
    def __init__(self, backbone="resnet18_3d", dropout=0.3):
        super().__init__()
        self.backbone_name = backbone
        feat_dim = 512

        if backbone == "resnet10_3d":
            self.backbone = ResNet3DAny(BasicBlock3D, [1, 1, 1, 1], input_channels=1, shortcut_type="B")
            load_med3d_weights(self.backbone, resolve_med3d_weight_path("resnet10_3d"), backbone)
        elif backbone == "resnet18_3d":
            self.backbone = ResNet3DAny(BasicBlock3D, [2, 2, 2, 2], input_channels=1, shortcut_type="A")
            load_med3d_weights(self.backbone, resolve_med3d_weight_path("resnet18_3d"), backbone)
        elif backbone == "resnet34_3d":
            self.backbone = ResNet3DAny(BasicBlock3D, [3, 4, 6, 3], input_channels=1, shortcut_type="A")
            load_med3d_weights(self.backbone, resolve_med3d_weight_path("resnet34_3d"), backbone)
        elif backbone in ["resnet50_3d", "resnet101_3d", "resnet152_3d", "resnet200_3d"]:
            layer_cfg = {
                "resnet50_3d": [3, 4, 6, 3],
                "resnet101_3d": [3, 4, 23, 3],
                "resnet152_3d": [3, 8, 36, 3],
                "resnet200_3d": [3, 24, 36, 3],
            }
            self.backbone = ResNet3DAny(Bottleneck3D, layer_cfg[backbone], input_channels=1, shortcut_type="B")
            load_med3d_weights(self.backbone, resolve_med3d_weight_path(backbone), backbone)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported 3D backbone: {backbone}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.backbone(x)
        r = self.head(z)
        return r


def evaluate_c_index(model, loader, device):
    model.eval()
    preds, times, events = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            r = model(x).view(-1)
            preds.extend(r.cpu().numpy().tolist())
            times.extend(batch["time"].numpy().tolist())
            events.extend(batch["event"].numpy().tolist())
    try:
        return concordance_index(times, -np.array(preds), events), np.array(preds)
    except Exception:
        return 0.5, np.array(preds)


def fit_single_cv_model(df_traincv, df_external, family, backbone, cfg, device, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_rows = []
    ext_fold_preds = []
    fold_best_val = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(df_traincv), 1):
        fold_seed = int(seed) + fold * 1000
        set_global_seed(fold_seed)

        tr = df_traincv.iloc[tr_idx].copy().reset_index(drop=True)
        va = df_traincv.iloc[va_idx].copy().reset_index(drop=True)

        train_ds = HCCImageDataset(tr, mode=family)
        val_ds = HCCImageDataset(va, mode=family)
        ext_ds = HCCImageDataset(df_external, mode=family)

        train_loader = build_loader(train_ds, cfg["batch_size"], True, 2, fold_seed, drop_last=False)
        val_loader = build_loader(val_ds, cfg["batch_size"], False, 2, fold_seed + 1)
        ext_loader = build_loader(ext_ds, cfg["batch_size"], False, 2, fold_seed + 2)

        if family in ["2d", "2p5d"]:
            model = Risk2DNet(backbone=backbone, dropout=cfg["dropout"]).to(device)
        else:
            model = Risk3DNet(backbone=backbone, dropout=cfg["dropout"]).to(device)

        opt_name = cfg.get("optim", "adamw").lower()
        if family in ["2d", "2p5d"]:
            param_groups = [
                {"params": model.backbone.parameters(), "lr": cfg.get("bb_lr", cfg["lr"] * 0.1)},
                {"params": model.head.parameters(), "lr": cfg.get("head_lr", cfg["lr"])}
            ]
        else:
            param_groups = [{"params": model.parameters(), "lr": cfg["lr"]}]

        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=cfg.get("momentum", 0.9),
                nesterov=cfg.get("nesterov", True),
                weight_decay=cfg["wd"],
            )
        else:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["wd"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

        best_val = -1.0
        best_state = None
        bad_epochs = 0
        use_early_stopping = cfg.get("use_early_stopping", False)

        for epoch in range(cfg["epochs"]):
            model.train()
            total_loss = 0.0
            used_batches = 0

            for batch in train_loader:
                x = batch["x"].to(device)
                t = batch["time"].to(device)
                e = batch["event"].to(device)

                order = torch.argsort(t, descending=True)
                x = x[order]
                e = e[order]

                optimizer.zero_grad()
                risk = model(x)
                loss = cox_ph_loss(risk, e)
                if loss.item() == 0.0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                used_batches += 1

            scheduler.step()

            c_val, val_pred = evaluate_c_index(model, val_loader, device)
            now = datetime.datetime.now().strftime("%H:%M:%S")
            avg_loss = total_loss / max(1, used_batches)
            pred_std = float(np.std(val_pred)) if len(val_pred) > 0 else 0.0
            print(f"[{now}] Seed:{fold_seed} | {family}/{backbone} Fold{fold} Ep{epoch+1:02d} | Optim:{opt_name} | Loss:{avg_loss:.4f} | ValC:{c_val:.4f} | ValPredStd:{pred_std:.6f}")

            if c_val > best_val:
                best_val = c_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if use_early_stopping and bad_epochs >= cfg["patience"]:
                break

        fold_best_val.append(best_val)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device)
                r = model(x).view(-1).cpu().numpy()
                for pid, rp in zip(b["pid"], r):
                    oof_rows.append({
                        "MRI_id": str(pid).strip(),
                        "Fold": fold,
                        "Risk_Score": float(rp),
                    })

        ext_preds = []
        with torch.no_grad():
            for b in ext_loader:
                x = b["x"].to(device)
                r = model(x).view(-1).cpu().numpy()
                ext_preds.extend(r.tolist())
        ext_fold_preds.append(np.array(ext_preds))

    oof_df = pd.DataFrame(oof_rows)
    ext_mean_pred = np.mean(np.stack(ext_fold_preds, axis=0), axis=0)
    return oof_df, ext_mean_pred, float(np.mean(fold_best_val))


def load_base_dataframe():
    df_clin = pd.read_excel(EXCEL_PATH)
    df_clin["MRI_id"] = df_clin["MRI_id"].astype(str).str.strip()
    df_clin = df_clin.dropna(subset=["Time_to_DM_Months", "Distant_Metastasis"])

    df_split = pd.read_csv(SPLIT_PATH)
    df_split["MRI_id"] = df_split["MRI_id"].astype(str).str.strip()

    return pd.merge(df_clin, df_split[["MRI_id", "Set"]], on="MRI_id", how="inner")
