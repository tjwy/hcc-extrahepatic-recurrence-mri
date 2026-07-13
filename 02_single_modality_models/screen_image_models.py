from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from lifelines.utils import concordance_index
from scipy.ndimage import zoom


ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SCRIPT31 = ROOT / "02_single_modality_models" / "model_components.py"
CANDIDATE_REGISTRY = ROOT / "02_single_modality_models" / "candidate_architectures.py"
SPLIT_CSV = Path(os.environ.get("MR_SPLIT_CSV", ROOT / "data" / "splits.csv"))
CLINICAL_EXCEL = Path(os.environ.get("MR_CLINICAL_EXCEL", ROOT / "data" / "clinical.xlsx"))
IMG_DIR = Path(os.environ.get("MR_IMG_DIR", ROOT / "data" / "dl_tumor_plus5mm"))
MASK_RAD_DIR = Path(os.environ.get("MR_MASK_RAD_DIR", ROOT / "data" / "radiomics_nifti"))
OUT_ROOT = Path(os.environ.get("MR_IMAGE_MODEL_OUT", ROOT / "outputs" / "single_modality_models" / "image"))

TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
ID_COL = "MRI_id"
BATCH_SIZE_OVERRIDES = {
    ("2d", "resnet50"): 24,
    ("2d", "vgg16_bn"): 24,
    ("2d", "vgg19_bn"): 24,
    ("2d", "densenet121"): 24,
    ("2d", "densenet169"): 24,
    ("2d", "efficientnet_b0"): 24,
    ("2d", "efficientnet_b1"): 24,
    ("2d", "mobilenet_v2"): 24,
    ("2p5d", "resnet18"): 24,
    ("2p5d", "resnet34"): 24,
    ("2p5d", "resnet50"): 24,
    ("2p5d", "densenet121"): 24,
    ("2p5d", "densenet169"): 24,
    ("2p5d", "efficientnet_b0"): 24,
    ("2p5d", "efficientnet_b1"): 24,
    ("2p5d", "mobilenet_v2"): 24,
    ("3d", "resnet10_3d"): 2,
    ("3d", "resnet18_3d"): 2,
    ("3d", "resnet34_3d"): 2,
    ("3d", "resnet50_3d"): 2,
    ("3d", "resnet101_3d"): 2,
    ("3d", "resnet152_3d"): 2,
    ("3d", "resnet200_3d"): 2,
}


def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def output_prefix(roi_mode: str) -> str:
    return f"{roi_mode}_image_finalepoch_fixedfold"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tumor-plus-5mm image single-modality screening using fixed split folds and final epoch checkpoints."
    )
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--mask-rad-dir", type=Path, default=MASK_RAD_DIR)
    parser.add_argument("--mask-phase", default="PVP", choices=["AP", "PVP", "DP"])
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    parser.add_argument("--clinical-excel", type=Path, default=CLINICAL_EXCEL)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--roi-mode", default="tumor_plus_5mm")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--families", default="2d,2p5d,3d")
    parser.add_argument("--backbones", default="", help="Optional comma-separated backbone filter.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional cap for test runs; 0 means no cap.")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Optional upper bound for candidate batch sizes; useful when sharing GPU memory.",
    )
    parser.add_argument(
        "--disable-batch-size-overrides",
        action="store_true",
        help="Use candidate-table batch sizes exactly as defined by the prespecified candidate registry.",
    )
    parser.add_argument("--resume-existing", action="store_true", help="Skip candidates with complete existing output CSVs.")
    parser.add_argument(
        "--save-final-checkpoints",
        action="store_true",
        help="Save one final-epoch model checkpoint per fold. Required before embedding-level early fusion.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write manifest and candidate table without training.")
    return parser.parse_args()


def resize_mask_to_dl(mask_xyz: np.ndarray, target_shape: tuple[int, int, int] = (32, 128, 128)) -> np.ndarray:
    """Match Phase2 DL layout: NIfTI (X,Y,Z) -> model volume (Z,Y,X)."""
    mask_zyx = (mask_xyz > 0).astype(np.float32).transpose(2, 1, 0)
    factors = np.array(target_shape, dtype=np.float32) / np.array(mask_zyx.shape, dtype=np.float32)
    return zoom(mask_zyx, factors, order=0) > 0.5


def compute_mask_view_indices(pid: str, mask_rad_dir: Path, mask_phase: str) -> dict:
    mask_path = mask_rad_dir / pid / f"{mask_phase}_mask.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing {mask_phase} mask for {pid}: {mask_path}")
    mask = resize_mask_to_dl(nib.load(str(mask_path)).get_fdata())
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError(f"Empty resized {mask_phase} mask for {pid}: {mask_path}")
    axial_area = mask.sum(axis=(1, 2))
    axial_idx = int(np.argmax(axial_area))
    centroid = coords.mean(axis=0)
    coronal_idx = int(np.clip(round(float(centroid[1])), 0, mask.shape[1] - 1))
    sagittal_idx = int(np.clip(round(float(centroid[2])), 0, mask.shape[2] - 1))
    return {
        ID_COL: pid,
        "Mask_Phase": mask_phase,
        "Mask_Path": str(mask_path),
        "MaskAware_Axial_Z": axial_idx,
        "MaskAware_Coronal_Y": coronal_idx,
        "MaskAware_Sagittal_X": sagittal_idx,
        "Mask_Max_Axial_Area": int(axial_area[axial_idx]),
        "Mask_Voxel_Count_Resized": int(mask.sum()),
    }


def build_mask_index(df: pd.DataFrame, mask_rad_dir: Path, mask_phase: str, out_root: Path, run_id: str) -> tuple[dict[str, dict], Path]:
    rows = []
    for pid in sorted(df[ID_COL].astype(str).str.strip().unique()):
        rows.append(compute_mask_view_indices(pid, mask_rad_dir, mask_phase))
    index_df = pd.DataFrame(rows)
    index_path = out_root / f"maskaware_{mask_phase.lower()}_view_index_{run_id}.csv"
    index_df.to_csv(index_path, index=False)
    index_df.to_csv(out_root / "maskaware_view_index.csv", index=False)
    return {str(row[ID_COL]).strip(): row for row in rows}, index_path


def install_maskaware_dataset(components, mask_index: dict[str, dict]) -> None:
    base_cls = components.HCCImageDataset

    class MaskAwareHCCImageDataset(base_cls):
        def _load_phase_stack(self, pid: str) -> np.ndarray:
            npy_path = Path(components.IMG_DIR) / f"{pid}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing DL npy for MRI_id={pid}: {npy_path}")
            arr = np.load(npy_path).astype(np.float32)
            if arr.ndim != 4 or arr.shape[0] < 3:
                raise ValueError(f"Expected {npy_path} shape (3,D,H,W); got {arr.shape}")
            return arr[:3]

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            pid = str(row[ID_COL]).strip()
            t = np.float32(row[TIME_COL])
            e = np.float32(row[EVENT_COL])

            stack = self._load_phase_stack(pid)
            _, d, h, w = stack.shape
            if pid not in mask_index:
                raise KeyError(f"Missing mask-aware view indices for MRI_id={pid}")
            meta = mask_index[pid]
            axial_idx = int(np.clip(meta["MaskAware_Axial_Z"], 0, d - 1))
            coronal_idx = int(np.clip(meta["MaskAware_Coronal_Y"], 0, h - 1))
            sagittal_idx = int(np.clip(meta["MaskAware_Sagittal_X"], 0, w - 1))

            if self.mode == "2d":
                # 2D = AP/PVP/DP channels at the same maximal-tumor axial slice.
                x = torch.cat([self._resize_2d(stack[p, axial_idx, :, :]) for p in range(3)], dim=0)
            elif self.mode == "2p5d":
                # 2.5D = three phases x three orthogonal tumor-centered views, 9 channels.
                views = []
                for phase_idx in range(3):
                    vol = stack[phase_idx]
                    views.extend(
                        [
                            self._resize_2d(vol[axial_idx, :, :]),
                            self._resize_2d(vol[:, coronal_idx, :]),
                            self._resize_2d(vol[:, :, sagittal_idx]),
                        ]
                    )
                x = torch.cat(views, dim=0)
            else:
                # 3D = AP/PVP/DP full tumor-plus-5mm volumes, 3 channels.
                x3 = torch.from_numpy(stack).float()
                x3 = components.F.interpolate(
                    x3.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False
                ).squeeze(0)
                x = x3

            return {
                "x": x,
                "time": torch.tensor(t, dtype=torch.float32),
                "event": torch.tensor(e, dtype=torch.float32),
                "pid": pid,
            }

    components.HCCImageDataset = MaskAwareHCCImageDataset


def _expand_conv2d(components, old_conv, input_channels: int):
    if old_conv.in_channels == input_channels:
        return old_conv
    new_conv = components.nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        old_w = old_conv.weight.data
        if input_channels % old_w.shape[1] == 0:
            repeats = input_channels // old_w.shape[1]
            new_conv.weight.copy_(old_w.repeat(1, repeats, 1, 1) / float(repeats))
        else:
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_w.repeat(1, input_channels, 1, 1))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.data)
    return new_conv


def _set_first_conv2d(components, net, backbone: str, input_channels: int) -> None:
    if input_channels == 3:
        return
    if backbone in {"resnet18", "resnet34", "resnet50"}:
        net.conv1 = _expand_conv2d(components, net.conv1, input_channels)
    elif backbone in {"vgg16_bn", "vgg19_bn"}:
        net.features[0] = _expand_conv2d(components, net.features[0], input_channels)
    elif backbone in {"densenet121", "densenet169"}:
        net.features.conv0 = _expand_conv2d(components, net.features.conv0, input_channels)
    elif backbone in {"efficientnet_b0", "efficientnet_b1", "mobilenet_v2"}:
        net.features[0][0] = _expand_conv2d(components, net.features[0][0], input_channels)
    else:
        raise ValueError(f"Unsupported 2D backbone for input-channel adaptation: {backbone}")


def install_multichannel_risk2d(components) -> None:
    class Risk2DNetMultiChannel(components.nn.Module):
        def __init__(self, backbone="resnet18", dropout=0.3, input_channels=3):
            super().__init__()
            if backbone == "resnet18":
                net = components.tv_models.resnet18(weights=None)
            elif backbone == "resnet34":
                net = components.tv_models.resnet34(weights=None)
            elif backbone == "resnet50":
                net = components.tv_models.resnet50(weights=None)
            elif backbone == "vgg16_bn":
                net = components.tv_models.vgg16_bn(weights=None)
            elif backbone == "vgg19_bn":
                net = components.tv_models.vgg19_bn(weights=None)
            elif backbone == "densenet121":
                net = components.tv_models.densenet121(weights=None)
            elif backbone == "densenet169":
                net = components.tv_models.densenet169(weights=None)
            elif backbone == "efficientnet_b0":
                net = components.tv_models.efficientnet_b0(weights=None)
            elif backbone == "efficientnet_b1":
                net = components.tv_models.efficientnet_b1(weights=None)
            elif backbone == "mobilenet_v2":
                net = components.tv_models.mobilenet_v2(weights=None)
            else:
                raise ValueError(f"Unsupported 2D backbone: {backbone}")

            state = components.load_pretrained_state_once(backbone)
            if state is not None:
                net.load_state_dict(state, strict=False)
            _set_first_conv2d(components, net, backbone, int(input_channels))

            if backbone in {"resnet18", "resnet34", "resnet50"}:
                net.fc = components.nn.Identity()
            else:
                net.classifier = components.nn.Identity()

            self.backbone = net
            with torch.no_grad():
                dummy = torch.zeros(1, int(input_channels), 224, 224)
                feat_dim = int(self.backbone(dummy).shape[1])

            self.head = components.nn.Sequential(
                components.nn.Dropout(dropout),
                components.nn.Linear(feat_dim, 64),
                components.nn.ReLU(inplace=True),
                components.nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    components.Risk2DNet = Risk2DNetMultiChannel


def _expand_conv3d(components, old_conv, input_channels: int):
    if old_conv.in_channels == input_channels:
        return old_conv
    new_conv = components.nn.Conv3d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        old_w = old_conv.weight.data
        if input_channels % old_w.shape[1] == 0:
            repeats = input_channels // old_w.shape[1]
            new_conv.weight.copy_(old_w.repeat(1, repeats, 1, 1, 1) / float(repeats))
        else:
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_w.repeat(1, input_channels, 1, 1, 1))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.data)
    return new_conv


def install_multichannel_risk3d(components) -> None:
    class Risk3DNetMultiChannel(components.nn.Module):
        def __init__(self, backbone="resnet18_3d", dropout=0.3, input_channels=3):
            super().__init__()
            self.backbone_name = backbone
            feat_dim = 512

            if backbone == "resnet10_3d":
                self.backbone = components.ResNet3DAny(
                    components.BasicBlock3D, [1, 1, 1, 1], input_channels=1, shortcut_type="B"
                )
                components.load_med3d_weights(self.backbone, components.resolve_med3d_weight_path("resnet10_3d"), backbone)
            elif backbone == "resnet18_3d":
                self.backbone = components.ResNet3DAny(
                    components.BasicBlock3D, [2, 2, 2, 2], input_channels=1, shortcut_type="A"
                )
                components.load_med3d_weights(self.backbone, components.resolve_med3d_weight_path("resnet18_3d"), backbone)
            elif backbone == "resnet34_3d":
                self.backbone = components.ResNet3DAny(
                    components.BasicBlock3D, [3, 4, 6, 3], input_channels=1, shortcut_type="A"
                )
                components.load_med3d_weights(self.backbone, components.resolve_med3d_weight_path("resnet34_3d"), backbone)
            elif backbone in ["resnet50_3d", "resnet101_3d", "resnet152_3d", "resnet200_3d"]:
                layer_cfg = {
                    "resnet50_3d": [3, 4, 6, 3],
                    "resnet101_3d": [3, 4, 23, 3],
                    "resnet152_3d": [3, 8, 36, 3],
                    "resnet200_3d": [3, 24, 36, 3],
                }
                self.backbone = components.ResNet3DAny(
                    components.Bottleneck3D, layer_cfg[backbone], input_channels=1, shortcut_type="B"
                )
                components.load_med3d_weights(self.backbone, components.resolve_med3d_weight_path(backbone), backbone)
                feat_dim = 2048
            else:
                raise ValueError(f"Unsupported 3D backbone: {backbone}")

            self.backbone.conv1 = _expand_conv3d(components, self.backbone.conv1, int(input_channels))
            self.head = components.nn.Sequential(
                components.nn.Dropout(dropout),
                components.nn.Linear(feat_dim, 64),
                components.nn.ReLU(inplace=True),
                components.nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    components.Risk3DNet = Risk3DNetMultiChannel


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_components(img_dir: Path, split_csv: Path, clinical_excel: Path, out_root: Path, roi_mode: str):
    os.environ["MR_SPLIT_CSV"] = str(split_csv)
    os.environ["MR_IMG_DIR"] = str(img_dir)
    os.environ["MR_ROI_MODE"] = roi_mode
    os.environ["MR_RUN_OUTPUT_ROOT"] = str(out_root)
    components = load_module(SCRIPT31, "components_final_epoch")
    components.EXCEL_PATH = str(clinical_excel)
    return components


def load_candidates(
    families: set[str],
    backbones: set[str],
    max_candidates: int,
    apply_batch_size_overrides: bool,
) -> list[tuple[str, str, dict]]:
    registry = load_module(CANDIDATE_REGISTRY, "candidate_registry")
    candidates = [
        (fam, bb, cfg)
        for fam, bb, cfg in registry.build_candidates()
        if fam in families and (not backbones or bb in backbones)
    ]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError("No candidates selected.")
    adjusted = []
    for fam, bb, cfg in candidates:
        cfg = cfg.copy()
        override = BATCH_SIZE_OVERRIDES.get((fam, bb)) if apply_batch_size_overrides else None
        if override is not None:
            cfg["batch_size"] = override
        if fam == "2d":
            cfg["input_channels"] = 3
            cfg["input_definition"] = "AP/PVP/DP at PVP-mask maximal axial tumor slice"
        elif fam == "2p5d":
            cfg["input_channels"] = 9
            cfg["input_definition"] = "AP/PVP/DP x axial/coronal/sagittal tumor-centered views"
        elif fam == "3d":
            cfg["input_channels"] = 3
            cfg["input_definition"] = "AP/PVP/DP full tumor-plus-5mm volumes"
        adjusted.append((fam, bb, cfg))
    candidates = adjusted
    return candidates


def load_base_dataframe(components, split_csv: Path) -> pd.DataFrame:
    base = components.load_base_dataframe().copy()
    base[ID_COL] = base[ID_COL].astype(str).str.strip()
    split = pd.read_csv(split_csv)
    split[ID_COL] = split[ID_COL].astype(str).str.strip()
    base = base.drop(columns=["Fold"], errors="ignore")
    base = base.merge(split[[ID_COL, "Fold"]], on=ID_COL, how="left")
    return base


def c_index(df: pd.DataFrame, risk_col: str = "Risk_Score") -> float:
    return float(
        concordance_index(
            df[TIME_COL].values,
            -df[risk_col].values,
            df[EVENT_COL].astype(bool).values,
        )
    )


def pooled_oof_c(df_train: pd.DataFrame, oof_df: pd.DataFrame) -> float:
    oof = oof_df.copy()
    oof[ID_COL] = oof[ID_COL].astype(str).str.strip()
    agg = oof.groupby(ID_COL, as_index=False)["Risk_Score"].mean()
    labels = df_train[[ID_COL, TIME_COL, EVENT_COL]].copy()
    labels[ID_COL] = labels[ID_COL].astype(str).str.strip()
    merged = labels.merge(agg, on=ID_COL, how="inner")
    if len(merged) != len(labels):
        raise RuntimeError(f"OOF merge mismatch: merged={len(merged)} expected={len(labels)}")
    return c_index(merged)


def predict_rows(components, model, loader, device, include_labels: bool) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            risks = model(x).view(-1).detach().cpu().numpy()
            for i, pid in enumerate(batch["pid"]):
                row = {ID_COL: str(pid).strip(), "Risk_Score": float(risks[i])}
                if include_labels:
                    row[TIME_COL] = float(batch["time"][i].item())
                    row[EVENT_COL] = int(batch["event"][i].item())
                rows.append(row)
    return rows


def fit_final_epoch_fixed_folds(
    components,
    df_train: pd.DataFrame,
    df_ext: pd.DataFrame,
    family: str,
    backbone: str,
    cfg: dict,
    device,
    seed: int,
    checkpoint_dir: Path | None = None,
):
    folds = sorted(int(x) for x in df_train["Fold"].dropna().unique())
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Unexpected Train_CV folds: {folds}")

    oof_rows: list[dict] = []
    ext_fold_preds = []
    fold_rows: list[dict] = []

    for fold in folds:
        fold_seed = int(seed) + (fold + 1) * 1000
        components.set_global_seed(fold_seed)
        tr = df_train[df_train["Fold"].astype(int) != fold].copy().reset_index(drop=True)
        va = df_train[df_train["Fold"].astype(int) == fold].copy().reset_index(drop=True)

        train_ds = components.HCCImageDataset(tr, mode=family)
        val_ds = components.HCCImageDataset(va, mode=family)
        ext_ds = components.HCCImageDataset(df_ext, mode=family)
        train_loader = components.build_loader(train_ds, cfg["batch_size"], True, 2, fold_seed, drop_last=False)
        val_loader = components.build_loader(val_ds, cfg["batch_size"], False, 2, fold_seed + 1)
        ext_loader = components.build_loader(ext_ds, cfg["batch_size"], False, 2, fold_seed + 2)

        if family in {"2d", "2p5d"}:
            model = components.Risk2DNet(
                backbone=backbone,
                dropout=cfg["dropout"],
                input_channels=int(cfg.get("input_channels", 3)),
            ).to(device)
            param_groups = [
                {"params": model.backbone.parameters(), "lr": cfg.get("bb_lr", cfg["lr"] * 0.1)},
                {"params": model.head.parameters(), "lr": cfg.get("head_lr", cfg["lr"])},
            ]
        else:
            model = components.Risk3DNet(
                backbone=backbone,
                dropout=cfg["dropout"],
                input_channels=int(cfg.get("input_channels", 3)),
            ).to(device)
            param_groups = [{"params": model.parameters(), "lr": cfg["lr"]}]

        if cfg.get("optim", "sgd").lower() != "sgd":
            raise ValueError("This strict rerun expects SGD configs only.")
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=cfg.get("momentum", 0.9),
            nesterov=cfg.get("nesterov", True),
            weight_decay=cfg["wd"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))

        best_val = -1.0
        best_epoch = -1
        final_val = float("nan")
        for epoch in range(int(cfg["epochs"])):
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
                loss = components.cox_ph_loss(risk, e)
                if float(loss.item()) == 0.0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                used_batches += 1

            scheduler.step()
            val_c, _ = components.evaluate_c_index(model, val_loader, device)
            final_val = float(val_c)
            if final_val > best_val:
                best_val = final_val
                best_epoch = epoch + 1
            print(
                f"[{ts()}] {family}/{backbone} fold={fold} epoch={epoch + 1:02d}/{cfg['epochs']} "
                f"loss={total_loss / max(1, used_batches):.4f} val_c={final_val:.4f}",
                flush=True,
            )

        for row in predict_rows(components, model, val_loader, device, include_labels=False):
            row["Fold"] = int(fold)
            oof_rows.append(row)

        ext_rows = predict_rows(components, model, ext_loader, device, include_labels=False)
        ext_fold_preds.append(pd.DataFrame(ext_rows).rename(columns={"Risk_Score": f"Fold{fold}_Risk"}))
        checkpoint_path = ""
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(checkpoint_dir / f"{family}_{backbone}_fold{fold}_final_epoch.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "family": family,
                    "backbone": backbone,
                    "fold": int(fold),
                    "seed": int(fold_seed),
                    "config": cfg,
                    "final_epoch": int(cfg["epochs"]),
                    "final_epoch_val_c": float(final_val),
                    "best_epoch_audit_not_used": int(best_epoch),
                    "best_val_c_audit_not_used": float(best_val),
                    "checkpoint_policy": "final_epoch_only",
                },
                checkpoint_path,
            )

        fold_rows.append(
            {
                "Family": family,
                "Backbone": backbone,
                "Fold": int(fold),
                "Train_N": int(len(tr)),
                "Val_N": int(len(va)),
                "Final_Epoch": int(cfg["epochs"]),
                "Final_Epoch_Val_C": final_val,
                "Best_Epoch_Audit_Not_Used": int(best_epoch),
                "Best_Val_C_Audit_Not_Used": float(best_val),
                "Final_Checkpoint_Path": checkpoint_path,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ext_merged = ext_fold_preds[0]
    for frame in ext_fold_preds[1:]:
        ext_merged = ext_merged.merge(frame, on=ID_COL, how="inner")
    risk_cols = [c for c in ext_merged.columns if c.endswith("_Risk")]
    ext_merged["Risk_Score"] = ext_merged[risk_cols].mean(axis=1)
    ext_merged = df_ext[[ID_COL, TIME_COL, EVENT_COL]].merge(ext_merged[[ID_COL, "Risk_Score"]], on=ID_COL, how="inner")

    return pd.DataFrame(oof_rows), ext_merged, pd.DataFrame(fold_rows)


def find_existing_outputs(out_dir: Path, roi_mode: str, family: str, backbone: str) -> tuple[Path, Path, Path] | None:
    pattern = f"{roi_mode}_{family}_{backbone}_*_finalepoch_fixedfold_oof.csv"
    for oof_path in sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
        stem = oof_path.name.removesuffix("_oof.csv")
        ext_path = out_dir / f"{stem}_external.csv"
        fold_path = out_dir / f"{stem}_fold_history.csv"
        if ext_path.exists() and fold_path.exists():
            return oof_path, ext_path, fold_path
    return None


def checkpoints_complete(out_dir: Path, roi_mode: str, family: str, backbone: str, run_id_from_stem: str) -> bool:
    checkpoint_dir = out_dir / "checkpoints" / f"{roi_mode}_{family}_{backbone}_{run_id_from_stem}"
    return all((checkpoint_dir / f"{family}_{backbone}_fold{fold}_final_epoch.pth").exists() for fold in range(5))


def run_id_from_oof_path(roi_mode: str, family: str, backbone: str, oof_path: Path) -> str:
    prefix = f"{roi_mode}_{family}_{backbone}_"
    suffix = "_finalepoch_fixedfold_oof.csv"
    name = oof_path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return ""
    return name[len(prefix) : -len(suffix)]


def summarize_existing_outputs(
    df_train: pd.DataFrame,
    roi_mode: str,
    family: str,
    backbone: str,
    cfg: dict,
    paths: tuple[Path, Path, Path],
) -> dict:
    oof_path, ext_path, fold_path = paths
    oof_df = pd.read_csv(oof_path)
    ext_df = pd.read_csv(ext_path)
    fold_df = pd.read_csv(fold_path)
    checkpoint_dir = ""
    if "Final_Checkpoint_Path" in fold_df.columns:
        nonempty = fold_df["Final_Checkpoint_Path"].dropna().astype(str)
        if len(nonempty) > 0:
            checkpoint_dir = str(Path(nonempty.iloc[0]).parent)
    return {
        "ROI_Mode": roi_mode,
        "Family": family,
        "Backbone": backbone,
        "Checkpoint_Policy": "final_epoch_only",
        "Fold_Policy": "fixed_split_fold_column",
        "Mean_FinalEpoch_Val_C": float(fold_df["Final_Epoch_Val_C"].mean()),
        "Mean_BestEpoch_Val_C_Audit_Not_Used": float(fold_df["Best_Val_C_Audit_Not_Used"].mean()),
        "Pooled_OOF_C": float(pooled_oof_c(df_train, oof_df)),
        "External_C_Diagnostic_Not_For_Selection": float(c_index(ext_df)),
        "Config": json.dumps(cfg, sort_keys=True),
        "OOF_CSV": str(oof_path),
        "External_CSV": str(ext_path),
        "Fold_History_CSV": str(fold_path),
        "Checkpoint_Dir": checkpoint_dir,
    }


def write_manifest(
    args: argparse.Namespace,
    candidates: list[tuple[str, str, dict]],
    df: pd.DataFrame,
    run_id: str,
    out_root: Path,
    mask_index_path: Path,
) -> None:
    train = df[df["Set"].eq("Train_CV")]
    manifest = {
        "run_id": run_id,
        "created_at": ts(),
        "roi_mode": args.roi_mode,
        "checkpoint_policy": "final_epoch_only",
        "fold_policy": "use_existing_Fold_column_from_split_csv",
        "selection_metric_after_run": "Pooled_OOF_C",
        "input_selection_policy": "mask-aware for 2d/2p5d; PVP tumor mask selects max axial tumor slice and tumor-centroid coronal/sagittal views",
        "image_volume_policy": "2d uses AP/PVP/DP axial channels; 2p5d uses AP/PVP/DP x axial/coronal/sagittal channels; 3d uses AP/PVP/DP full volumes",
        "two_d_policy": "2d tensor shape is (3,H,W): AP/PVP/DP at the same PVP-mask maximal axial tumor slice",
        "two_p5d_policy": "2p5d tensor shape is (9,H,W): for each AP/PVP/DP phase, axial maximal tumor slice plus coronal/sagittal tumor-centroid views",
        "three_d_policy": "3d tensor shape is (3,D,H,W): AP/PVP/DP full tumor-plus-5mm volumes resized to (64,64,64); no slice selection is applied",
        "mask_phase_for_view_selection": args.mask_phase,
        "mask_rad_dir": str(args.mask_rad_dir),
        "mask_index_csv": str(mask_index_path),
        "split_csv": str(args.split_csv),
        "split_md5": md5(args.split_csv),
        "clinical_excel": str(args.clinical_excel),
        "clinical_excel_md5": md5(args.clinical_excel),
        "img_dir": str(args.img_dir),
        "img_dir_exists": args.img_dir.exists(),
        "train_n": int(len(train)),
        "external_n": int(df["Set"].eq("External_Test").sum()),
        "train_fold_counts": train["Fold"].value_counts().sort_index().astype(int).to_dict(),
        "candidate_count": len(candidates),
        "candidates": [{"family": fam, "backbone": bb, "config": cfg} for fam, bb, cfg in candidates],
        "dry_run": bool(args.dry_run),
        "save_final_checkpoints": bool(args.save_final_checkpoints),
        "batch_size_overrides_applied": not bool(args.disable_batch_size_overrides),
    }
    (out_root / f"final_epoch_screening_manifest_{run_id}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_root / "singlemod_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    families = {x.strip() for x in args.families.split(",") if x.strip()}
    backbones = {x.strip() for x in args.backbones.split(",") if x.strip()}
    candidates = load_candidates(
        families,
        backbones,
        args.max_candidates,
        apply_batch_size_overrides=not args.disable_batch_size_overrides,
    )
    if args.max_batch_size > 0:
        capped = []
        for fam, bb, cfg in candidates:
            cfg = cfg.copy()
            cfg["batch_size"] = min(int(cfg["batch_size"]), int(args.max_batch_size))
            capped.append((fam, bb, cfg))
        candidates = capped

    components = load_model_components(args.img_dir, args.split_csv, args.clinical_excel, args.out_root, args.roi_mode)
    df = load_base_dataframe(components, args.split_csv)
    df_train = df[df["Set"].eq("Train_CV")].copy().reset_index(drop=True)
    df_ext = df[df["Set"].eq("External_Test")].copy().reset_index(drop=True)

    run_id = run_tag()
    mask_index, mask_index_path = build_mask_index(df, args.mask_rad_dir, args.mask_phase, args.out_root, run_id)
    install_maskaware_dataset(components, mask_index)
    install_multichannel_risk2d(components)
    install_multichannel_risk3d(components)
    write_manifest(args, candidates, df, run_id, args.out_root, mask_index_path)
    pd.DataFrame(
        [{"Family": fam, "Backbone": bb, "Config": json.dumps(cfg, sort_keys=True)} for fam, bb, cfg in candidates]
    ).to_csv(args.out_root / f"candidate_table_{run_id}.csv", index=False)

    print(
        f"[{ts()}] final-epoch fixed-fold screening | roi={args.roi_mode} | "
        f"Train_CV={len(df_train)} | External_Test={len(df_ext)} | candidates={len(candidates)}",
        flush=True,
    )
    if args.dry_run:
        print(f"[{ts()}] Dry run complete. No training started.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components.ensure_imagenet_cache([bb for fam, bb, _ in candidates if fam in {"2d", "2p5d"}])
    components.ensure_med3d_weights([bb for fam, bb, _ in candidates if fam == "3d"])

    rows: list[dict] = []
    for idx, (family, backbone, cfg) in enumerate(candidates, 1):
        if args.resume_existing:
            existing = find_existing_outputs(out_dir, args.roi_mode, family, backbone)
            if existing is not None:
                existing_run_id = run_id_from_oof_path(args.roi_mode, family, backbone, existing[0])
                has_checkpoints = checkpoints_complete(out_dir, args.roi_mode, family, backbone, existing_run_id)
                if (not args.save_final_checkpoints) or has_checkpoints:
                    print(f"[{ts()}] Candidate {idx}/{len(candidates)}: {family}/{backbone} | resume skip existing", flush=True)
                    rows.append(summarize_existing_outputs(df_train, args.roi_mode, family, backbone, cfg, existing))
                    pd.DataFrame(rows).to_csv(args.out_root / f"{output_prefix(args.roi_mode)}_summary_{run_id}.csv", index=False)
                    continue
                print(
                    f"[{ts()}] Candidate {idx}/{len(candidates)}: {family}/{backbone} | existing CSVs lack checkpoints; rerun",
                    flush=True,
                )
        print(f"[{ts()}] Candidate {idx}/{len(candidates)}: {family}/{backbone}", flush=True)
        candidate_run_id = run_id
        checkpoint_dir = None
        if args.save_final_checkpoints:
            checkpoint_dir = out_dir / "checkpoints" / f"{args.roi_mode}_{family}_{backbone}_{candidate_run_id}"
        oof_df, ext_df, fold_df = fit_final_epoch_fixed_folds(
            components,
            df_train,
            df_ext,
            family,
            backbone,
            cfg.copy(),
            device,
            args.seed,
            checkpoint_dir=checkpoint_dir,
        )
        pooled = pooled_oof_c(df_train, oof_df)
        ext_c = c_index(ext_df)
        mean_final = float(fold_df["Final_Epoch_Val_C"].mean())
        mean_best_audit = float(fold_df["Best_Val_C_Audit_Not_Used"].mean())

        stem = f"{args.roi_mode}_{family}_{backbone}_{run_id}_finalepoch_fixedfold"
        oof_path = out_dir / f"{stem}_oof.csv"
        ext_path = out_dir / f"{stem}_external.csv"
        fold_path = out_dir / f"{stem}_fold_history.csv"
        oof_df.to_csv(oof_path, index=False)
        ext_df.to_csv(ext_path, index=False)
        fold_df.to_csv(fold_path, index=False)

        rows.append(
            {
                "ROI_Mode": args.roi_mode,
                "Family": family,
                "Backbone": backbone,
                "Checkpoint_Policy": "final_epoch_only",
                "Fold_Policy": "fixed_split_fold_column",
                "Mean_FinalEpoch_Val_C": mean_final,
                "Mean_BestEpoch_Val_C_Audit_Not_Used": mean_best_audit,
                "Pooled_OOF_C": float(pooled),
                "External_C_Diagnostic_Not_For_Selection": float(ext_c),
                "Config": json.dumps(cfg, sort_keys=True),
                "OOF_CSV": str(oof_path),
                "External_CSV": str(ext_path),
                "Fold_History_CSV": str(fold_path),
                "Checkpoint_Dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
            }
        )
        pd.DataFrame(rows).to_csv(args.out_root / f"{output_prefix(args.roi_mode)}_summary_{run_id}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary_path = args.out_root / f"{output_prefix(args.roi_mode)}_summary_{run_id}.csv"
    summary.to_csv(summary_path, index=False)
    selected = summary.sort_values(["Family", "Pooled_OOF_C"], ascending=[True, False]).groupby("Family").head(1)
    selected.to_csv(args.out_root / f"{output_prefix(args.roi_mode)}_selected_by_pooled_oof_{run_id}.csv", index=False)
    print(f"[{ts()}] Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
