#!/usr/bin/env python3
"""Input-matched deep-feature early-fusion comparator."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv


ID_COL = "MRI_id"
TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
PROJECT_ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
TRAINER_PATH = PROJECT_ROOT / "02_single_modality_models" / "screen_image_models.py"
SELECTED_CSV = Path(os.environ.get("MR_IMAGE_SELECTED_CSV", PROJECT_ROOT / "outputs" / "single_modality_models" / "selected" / "selected_models.csv"))
SPLIT_CSV = Path(os.environ.get("MR_SPLIT_CSV", PROJECT_ROOT / "data" / "splits.csv"))
CLINICAL_EXCEL = Path(os.environ.get("MR_CLINICAL_EXCEL", PROJECT_ROOT / "data" / "clinical.xlsx"))
IMG_DIR = Path(os.environ.get("MR_IMG_DIR", PROJECT_ROOT / "data" / "dl_tumor_plus5mm"))
MASK_INDEX_CSV = Path(os.environ.get("MR_VIEW_INDEX_CSV", PROJECT_ROOT / "outputs" / "single_modality_models" / "image" / "maskaware_view_index.csv"))
RADIOMICS_RAW_CSV = Path(os.environ.get("MR_RADIOMICS_CSV", PROJECT_ROOT / "data" / "radiomics_features_raw.csv"))
LATE_PREDICTIONS_CSV = Path(os.environ.get("MR_LATE_PREDICTIONS_CSV", PROJECT_ROOT / "derived_inputs" / "late_predictions.csv"))
OUTPUT_ROOT = Path(os.environ.get("MR_EARLY_FUSION_OUT", PROJECT_ROOT / "outputs" / "imaging_fusion" / "early"))
FEATURE_RE = re.compile(r"^(AP|PVP|DP)_original_")
LOCKED_MODELS = {
    "2d": "vgg19_bn",
    "2p5d": "mobilenet_v2",
    "3d": "resnet10_3d",
}
MODEL_ORDER = ["CoxPH", "RSF", "SVM", "GB", "ExtraST"]
PCA_COMPONENTS = 50
EXPECTED_TRAIN_N = 732
EXPECTED_EXTERNAL_N = 227


@dataclass(frozen=True)
class LockedBranch:
    family: str
    backbone: str
    checkpoint_dir: Path
    checkpoint_paths: tuple[Path, ...]
    config: dict[str, object]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_locked_branches(selected_csv: Path) -> list[LockedBranch]:
    selected = pd.read_csv(selected_csv)
    branches: list[LockedBranch] = []
    for family, backbone in LOCKED_MODELS.items():
        row = selected[
            selected["Family"].astype(str).eq(family)
            & selected["Backbone"].astype(str).eq(backbone)
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one locked row for {family}/{backbone}; observed={len(row)}")
        record = row.iloc[0]
        checkpoint_dir = Path(str(record["Checkpoint_Dir"]))
        checkpoint_paths = tuple(
            checkpoint_dir / f"{family}_{backbone}_fold{fold}_final_epoch.pth"
            for fold in range(5)
        )
        missing = [str(path) for path in checkpoint_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing locked checkpoints for {family}/{backbone}: {missing}")
        config = json.loads(str(record["Config"]))
        branches.append(
            LockedBranch(
                family=family,
                backbone=backbone,
                checkpoint_dir=checkpoint_dir,
                checkpoint_paths=checkpoint_paths,
                config=config,
            )
        )
    return branches


def validate_oof_coverage(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    expected_ids = expected[ID_COL].astype(str).str.strip()
    observed_ids = observed[ID_COL].astype(str).str.strip()
    if expected_ids.nunique() != len(expected) or observed_ids.nunique() != len(observed):
        raise ValueError("OOF coverage mismatch: duplicate MRI_id")
    if set(expected_ids) != set(observed_ids):
        raise ValueError("OOF coverage mismatch: missing or unexpected MRI_id")
    expected_fold = dict(zip(expected_ids, expected["Fold"].astype(int)))
    observed_fold = dict(zip(observed_ids, observed["Source_Fold"].astype(int)))
    wrong = [pid for pid in expected_fold if expected_fold[pid] != observed_fold[pid]]
    if wrong:
        raise ValueError(f"OOF checkpoint fold mismatch for {len(wrong)} cases")


def select_early_candidate(grid: pd.DataFrame) -> pd.Series:
    ranked = grid.copy()
    ranked["Model_Order"] = ranked["Model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    if ranked["Model_Order"].isna().any():
        raise ValueError("Candidate grid contains unsupported model")
    ranked = ranked.sort_values(
        ["Pooled_OOF_C", "Model_Order"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return ranked.iloc[0]


def fit_fold_preprocessor(
    x_train_raw: np.ndarray,
    x_pred_raw: np.ndarray,
    requested_components: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
    x_pred = scaler.transform(imputer.transform(x_pred_raw))
    n_components = min(int(requested_components), x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(x_train), pca.transform(x_pred), n_components


def validate_fold_feature_provenance(
    features: pd.DataFrame,
    expected_ids: list[str],
    outer_fold: int,
) -> None:
    observed_ids = features[ID_COL].astype(str).str.strip()
    if observed_ids.nunique() != len(features) or set(observed_ids) != set(map(str, expected_ids)):
        raise ValueError("fold feature MRI_id mismatch")
    source_folds = set(features["Source_Fold"].astype(int).tolist())
    if source_folds != {int(outer_fold)}:
        raise ValueError(
            f"source checkpoint mismatch: expected fold {outer_fold}, observed={sorted(source_folds)}"
        )


def merge_fold_features(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise ValueError("No branch feature tables supplied")
    expected_ids = set(tables[0][ID_COL].astype(str).str.strip())
    for table in tables[1:]:
        if set(table[ID_COL].astype(str).str.strip()) != expected_ids:
            raise ValueError("branch MRI_id mismatch")
    merged = tables[0].copy()
    for table in tables[1:]:
        merged = merged.merge(table, on=ID_COL, how="inner", validate="one_to_one")
    return merged


def c_index_safe(times, risk, events) -> float:
    return float(
        concordance_index(
            np.asarray(times, dtype=float),
            -np.asarray(risk, dtype=float),
            np.asarray(events, dtype=int),
        )
    )


def make_survival_model(name: str):
    if name == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=0.01)
    if name == "RSF":
        return RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "SVM":
        return FastSurvivalSVM(max_iter=4000, tol=1e-5, random_state=42)
    if name == "GB":
        return GradientBoostingSurvivalAnalysis(
            random_state=42,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
        )
    if name == "ExtraST":
        return ExtraSurvivalTrees(
            n_estimators=300,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def paired_bootstrap_delta(
    times,
    events,
    early_risk,
    late_risk,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> dict[str, float | int]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    early_risk = np.asarray(early_risk, dtype=float)
    late_risk = np.asarray(late_risk, dtype=float)
    observed = c_index_safe(times, early_risk, events) - c_index_safe(times, late_risk, events)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(times), size=len(times))
        try:
            delta = c_index_safe(times[idx], early_risk[idx], events[idx]) - c_index_safe(
                times[idx], late_risk[idx], events[idx]
            )
        except ZeroDivisionError:
            continue
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        raise RuntimeError("No valid paired bootstrap replicates")
    values = np.asarray(deltas, dtype=float)
    lower, upper = np.percentile(values, [2.5, 97.5])
    p_two_sided = min(
        1.0,
        2.0
        * min(
            (float(np.sum(values <= 0)) + 1.0) / (len(values) + 1.0),
            (float(np.sum(values >= 0)) + 1.0) / (len(values) + 1.0),
        ),
    )
    return {
        "Delta_C_Early_Minus_Late": float(observed),
        "CI95_Lower": float(lower),
        "CI95_Upper": float(upper),
        "P_Two_Sided": float(p_two_sided),
        "N_Bootstrap_Valid": int(len(values)),
        "Seed": int(seed),
    }


def load_base_and_runtime(output_dir: Path):
    trainer = load_module(TRAINER_PATH, "corrected_early_trainer")
    components = trainer.load_model_components(IMG_DIR, SPLIT_CSV, CLINICAL_EXCEL, output_dir, "tumor_plus_5mm")
    base = trainer.load_base_dataframe(components, SPLIT_CSV)
    base[ID_COL] = base[ID_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    target = base[base["Set"].isin(["Train_CV", "External_Test"])].copy().reset_index(drop=True)

    mask_table = pd.read_csv(MASK_INDEX_CSV)
    mask_table[ID_COL] = mask_table[ID_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    missing_masks = sorted(set(target[ID_COL]) - set(mask_table[ID_COL]))
    if missing_masks:
        raise ValueError(f"Mask-index coverage mismatch: missing={missing_masks[:10]}")
    mask_index = {
        str(row[ID_COL]).strip(): row.to_dict()
        for _, row in mask_table[mask_table[ID_COL].isin(target[ID_COL])].iterrows()
    }
    trainer.install_maskaware_dataset(components, mask_index)
    trainer.install_multichannel_risk2d(components)
    trainer.install_multichannel_risk3d(components)
    return trainer, components, base


def build_branch_model(components, branch: LockedBranch, fold: int, device):
    checkpoint_path = branch.checkpoint_paths[fold]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(branch.config)
    config.update(checkpoint.get("config", {}))
    input_channels = int(config.get("input_channels", 9 if branch.family == "2p5d" else 3))
    dropout = float(config.get("dropout", 0.0))
    if branch.family in {"2d", "2p5d"}:
        model = components.Risk2DNet(
            backbone=branch.backbone,
            dropout=dropout,
            input_channels=input_channels,
        )
    else:
        model = components.Risk3DNet(
            backbone=branch.backbone,
            dropout=dropout,
            input_channels=input_channels,
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def extract_embeddings(model, loader, device) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    arrays: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            z = model.backbone(x).detach().cpu().numpy().astype(np.float32)
            ids.extend(str(pid).strip() for pid in batch["pid"])
            arrays.append(z)
    if not arrays:
        raise RuntimeError("No embeddings extracted")
    return ids, np.concatenate(arrays, axis=0)


def save_embedding_cache(
    path: Path,
    development_ids: list[str],
    development_features: np.ndarray,
    external_ids: list[str],
    external_features: np.ndarray,
    source_fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    np.save(temp / "development_ids.npy", np.asarray(development_ids, dtype=str), allow_pickle=False)
    np.save(
        temp / "development_features.npy",
        np.asarray(development_features, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(temp / "external_ids.npy", np.asarray(external_ids, dtype=str), allow_pickle=False)
    np.save(
        temp / "external_features.npy",
        np.asarray(external_features, dtype=np.float32),
        allow_pickle=False,
    )
    (temp / "metadata.json").write_text(
        json.dumps({"source_fold": int(source_fold)}) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def load_embedding_cache(path: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray, int]:
    if path.is_dir():
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return (
            np.load(path / "development_ids.npy", allow_pickle=False).astype(str).tolist(),
            np.load(path / "development_features.npy", allow_pickle=False).astype(np.float32),
            np.load(path / "external_ids.npy", allow_pickle=False).astype(str).tolist(),
            np.load(path / "external_features.npy", allow_pickle=False).astype(np.float32),
            int(metadata["source_fold"]),
        )
    cached = np.load(path, allow_pickle=False)
    return (
        cached["development_ids"].astype(str).tolist(),
        cached["development_features"].astype(np.float32),
        cached["external_ids"].astype(str).tolist(),
        cached["external_features"].astype(np.float32),
        int(cached["source_fold"][0]),
    )


def branch_fold_embeddings(
    components,
    branch: LockedBranch,
    fold: int,
    development: pd.DataFrame,
    external: pd.DataFrame,
    output_dir: Path,
    device,
    reuse: bool,
    save_cache: bool,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict[str, object]]:
    cache_path = output_dir / "embeddings" / f"fold{fold}_{branch.family}_{branch.backbone}"
    fallback_cache = cache_path.with_suffix(".npz")
    reusable_cache = cache_path if cache_path.exists() else fallback_cache
    if reuse and reusable_cache.exists():
        dev_ids, dev_x, ext_ids, ext_x, source_fold = load_embedding_cache(reusable_cache)
        if source_fold != fold:
            raise ValueError(f"Embedding cache source fold mismatch: {reusable_cache}")
        config = dict(branch.config)
    else:
        model, config = build_branch_model(components, branch, fold, device)
        batch_size = int(config.get("batch_size", 2))
        dev_loader = components.build_loader(
            components.HCCImageDataset(development, mode=branch.family),
            batch_size,
            False,
            2,
            9100 + fold,
        )
        ext_loader = components.build_loader(
            components.HCCImageDataset(external, mode=branch.family),
            batch_size,
            False,
            2,
            9200 + fold,
        )
        dev_ids, dev_x = extract_embeddings(model, dev_loader, device)
        ext_ids, ext_x = extract_embeddings(model, ext_loader, device)
        if save_cache:
            save_embedding_cache(cache_path, dev_ids, dev_x, ext_ids, ext_x, fold)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validate_fold_feature_provenance(
        pd.DataFrame({ID_COL: dev_ids, "Source_Fold": fold}),
        development[ID_COL].astype(str).tolist(),
        fold,
    )
    if ext_ids != external[ID_COL].astype(str).tolist():
        raise ValueError(f"External embedding order mismatch for {branch.family}/{branch.backbone} fold={fold}")
    manifest = {
        "family": branch.family,
        "backbone": branch.backbone,
        "source_fold": fold,
        "checkpoint": str(branch.checkpoint_paths[fold]),
        "checkpoint_md5": file_md5(branch.checkpoint_paths[fold]),
        "cache": (
            str(reusable_cache)
            if reuse and reusable_cache.exists()
            else (str(cache_path) if save_cache else None)
        ),
        "development_n": len(dev_ids),
        "external_n": len(ext_ids),
        "feature_dim": int(dev_x.shape[1]),
        "config": config,
    }
    return dev_ids, dev_x, ext_ids, ext_x, manifest


def load_radiomics() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(RADIOMICS_RAW_CSV)
    raw[ID_COL] = raw[ID_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    feature_cols = [column for column in raw.columns if FEATURE_RE.match(str(column))]
    phase_counts = {
        phase: sum(column.startswith(f"{phase}_original_") for column in feature_cols)
        for phase in ("AP", "PVP", "DP")
    }
    if len(feature_cols) != 321 or set(phase_counts.values()) != {107}:
        raise ValueError(f"Expected fixed radiomics 107 x 3; observed={phase_counts}")
    return raw[[ID_COL] + feature_cols].copy(), feature_cols


def align_matrix(ids: list[str], features: np.ndarray, target_ids: list[str]) -> np.ndarray:
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate MRI_id in embedding matrix")
    positions = {pid: idx for idx, pid in enumerate(ids)}
    missing = [pid for pid in target_ids if pid not in positions]
    if missing:
        raise ValueError(f"Embedding MRI_id mismatch: missing={missing[:10]}")
    return features[[positions[pid] for pid in target_ids]]


def build_survival(frame: pd.DataFrame):
    return Surv.from_arrays(
        event=frame[EVENT_COL].astype(bool).to_numpy(),
        time=frame[TIME_COL].astype(float).to_numpy(),
    )


def fit_predict_model(name: str, x_train: np.ndarray, train_labels: pd.DataFrame, x_pred: np.ndarray) -> np.ndarray:
    model = make_survival_model(name)
    model.fit(x_train, build_survival(train_labels))
    prediction = model.predict(x_pred)
    if np.ndim(prediction) == 2:
        prediction = prediction[:, -1]
    return np.asarray(prediction, dtype=float)


def evaluate_candidates(
    base: pd.DataFrame,
    fold_matrices: dict[int, tuple[np.ndarray, np.ndarray]],
    requested_components: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    development = base[base["Set"].eq("Train_CV")].copy().reset_index(drop=True)
    external = base[base["Set"].eq("External_Test")].copy().reset_index(drop=True)
    candidate_oof = {name: np.full(len(development), np.nan, dtype=float) for name in MODEL_ORDER}
    candidate_external = {
        name: np.full((5, len(external)), np.nan, dtype=float) for name in MODEL_ORDER
    }
    fold_rows: list[dict[str, object]] = []

    for fold in range(5):
        x_dev_raw, x_ext_raw = fold_matrices[fold]
        train_idx = np.flatnonzero(development["Fold"].astype(int).to_numpy() != fold)
        val_idx = np.flatnonzero(development["Fold"].astype(int).to_numpy() == fold)
        combined_pred = np.concatenate([x_dev_raw[val_idx], x_ext_raw], axis=0)
        x_train, x_pred, actual_components = fit_fold_preprocessor(
            x_dev_raw[train_idx], combined_pred, requested_components
        )
        x_val = x_pred[: len(val_idx)]
        x_ext = x_pred[len(val_idx) :]
        train_labels = development.iloc[train_idx]
        val_labels = development.iloc[val_idx]
        for name in MODEL_ORDER:
            log(f"outer fold={fold} learner={name} PCA={actual_components}")
            val_pred = fit_predict_model(name, x_train, train_labels, x_val)
            ext_pred = fit_predict_model(name, x_train, train_labels, x_ext)
            candidate_oof[name][val_idx] = val_pred
            candidate_external[name][fold] = ext_pred
            fold_rows.append(
                {
                    "Model": name,
                    "Outer_Fold": fold,
                    "Val_N": len(val_idx),
                    "Val_C": c_index_safe(val_labels[TIME_COL], val_pred, val_labels[EVENT_COL]),
                    "PCA_Components": actual_components,
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    grid_rows = []
    for name in MODEL_ORDER:
        if np.isnan(candidate_oof[name]).any() or np.isnan(candidate_external[name]).any():
            raise RuntimeError(f"Incomplete predictions for {name}")
        model_folds = fold_table[fold_table["Model"].eq(name)]
        grid_rows.append(
            {
                "Model": name,
                "Pooled_OOF_C": c_index_safe(
                    development[TIME_COL], candidate_oof[name], development[EVENT_COL]
                ),
                "Mean_Fold_C": float(model_folds["Val_C"].mean()),
                "SD_Fold_C": float(model_folds["Val_C"].std(ddof=1)),
                "Min_Fold_C": float(model_folds["Val_C"].min()),
                "Max_Fold_C": float(model_folds["Val_C"].max()),
            }
        )
    grid = pd.DataFrame(grid_rows)
    return grid, candidate_oof, candidate_external, fold_table


def load_late_ifm_predictions(base: pd.DataFrame, set_name: str) -> pd.DataFrame:
    predictions = pd.read_csv(LATE_PREDICTIONS_CSV)
    predictions[ID_COL] = predictions[ID_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    risk_col = next(
        (column for column in ["IFM_Risk", "Risk_IFM", "IFM_Score", "IFM"] if column in predictions.columns),
        None,
    )
    if risk_col is None:
        raise ValueError(f"Cannot identify IFM risk column in {LATE_PREDICTIONS_CSV}")
    labels = base[base["Set"].eq(set_name)][[ID_COL, TIME_COL, EVENT_COL]].copy()
    late = predictions[predictions["Set"].eq(set_name)][[ID_COL, risk_col]].copy()
    return labels.merge(late, on=ID_COL, how="inner", validate="one_to_one").rename(
        columns={risk_col: "Late_IFM_Risk"}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    parser.add_argument("--smoke-fold", type=int, choices=range(5), default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    locked = load_locked_branches(SELECTED_CSV)
    trainer, components, base = load_base_and_runtime(args.output_dir)
    development = base[base["Set"].eq("Train_CV")].copy().reset_index(drop=True)
    external = base[base["Set"].eq("External_Test")].copy().reset_index(drop=True)
    if len(development) != EXPECTED_TRAIN_N or len(external) != EXPECTED_EXTERNAL_N:
        raise ValueError(
            f"Cohort size mismatch: Train_CV={len(development)}, External_Test={len(external)}"
        )
    folds = sorted(development["Fold"].astype(int).unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Unexpected fixed folds: {folds}")

    radiomics, radiomics_cols = load_radiomics()
    rad_lookup = radiomics.set_index(ID_COL)
    dev_ids = development[ID_COL].astype(str).tolist()
    ext_ids = external[ID_COL].astype(str).tolist()
    x_rad_dev = rad_lookup.loc[dev_ids, radiomics_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x_rad_ext = rad_lookup.loc[ext_ids, radiomics_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components.ensure_imagenet_cache([branch.backbone for branch in locked if branch.family in {"2d", "2p5d"}])
    components.ensure_med3d_weights([branch.backbone for branch in locked if branch.family == "3d"])
    target_folds = [args.smoke_fold] if args.smoke_fold is not None else list(range(5))
    fold_matrices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    branch_manifest: list[dict[str, object]] = []

    for fold in target_folds:
        log(f"extracting corrected same-checkpoint embeddings for outer fold={fold} on {device}")
        dev_parts: list[np.ndarray] = []
        ext_parts: list[np.ndarray] = []
        for branch in locked:
            source_dev_ids, x_dev, source_ext_ids, x_ext, record = branch_fold_embeddings(
                components,
                branch,
                fold,
                development,
                external,
                args.output_dir,
                device,
                args.reuse_embeddings,
                args.save_embeddings,
            )
            dev_parts.append(align_matrix(source_dev_ids, x_dev, dev_ids))
            ext_parts.append(align_matrix(source_ext_ids, x_ext, ext_ids))
            branch_manifest.append(record)
        fold_matrices[fold] = (
            np.concatenate(dev_parts + [x_rad_dev], axis=1),
            np.concatenate(ext_parts + [x_rad_ext], axis=1),
        )

    if args.smoke_fold is not None:
        fold = args.smoke_fold
        x_dev, x_ext = fold_matrices[fold]
        smoke = {
            "outer_fold": fold,
            "development_shape": list(x_dev.shape),
            "external_shape": list(x_ext.shape),
            "radiomics_feature_count": len(radiomics_cols),
            "branches": branch_manifest,
        }
        path = args.output_dir / f"smoke_fold{fold}_manifest.json"
        path.write_text(json.dumps(smoke, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"smoke test saved: {path}")
        return args.output_dir

    grid, candidate_oof, candidate_external, fold_table = evaluate_candidates(
        base, fold_matrices, args.pca_components
    )
    selected = select_early_candidate(grid)
    selected_model = str(selected["Model"])
    external_risk = candidate_external[selected_model].mean(axis=0)
    external_c = c_index_safe(external[TIME_COL], external_risk, external[EVENT_COL])
    grid["Selected"] = grid["Model"].eq(selected_model)
    grid["External_C_Selected_Only"] = np.where(grid["Selected"], external_c, np.nan)

    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    ID_COL: dev_ids,
                    "Set": "Train_CV",
                    TIME_COL: development[TIME_COL].to_numpy(),
                    EVENT_COL: development[EVENT_COL].to_numpy(),
                    "Model": selected_model,
                    "Risk": candidate_oof[selected_model],
                    "Prediction_Policy": "outer-fold held-out",
                }
            ),
            pd.DataFrame(
                {
                    ID_COL: ext_ids,
                    "Set": "External_Test",
                    TIME_COL: external[TIME_COL].to_numpy(),
                    EVENT_COL: external[EVENT_COL].to_numpy(),
                    "Model": selected_model,
                    "Risk": external_risk,
                    "Prediction_Policy": "mean of five outer-fold models",
                }
            ),
        ],
        ignore_index=True,
    )
    late_train = load_late_ifm_predictions(base, "Train_CV")
    late_external = load_late_ifm_predictions(base, "External_Test")
    late_c_latest_endpoint = c_index_safe(
        late_train[TIME_COL], late_train["Late_IFM_Risk"], late_train[EVENT_COL]
    )
    late_external_c_latest_endpoint = c_index_safe(
        late_external[TIME_COL], late_external["Late_IFM_Risk"], late_external[EVENT_COL]
    )
    paired = pd.DataFrame(
        {ID_COL: dev_ids, "Early_IFM_Risk": candidate_oof[selected_model]}
    ).merge(late_train, on=ID_COL, how="inner", validate="one_to_one")
    delta_result = paired_bootstrap_delta(
        paired[TIME_COL],
        paired[EVENT_COL],
        paired["Early_IFM_Risk"],
        paired["Late_IFM_Risk"],
    )
    delta_table = pd.DataFrame([{"Comparison": "Early minus late IFM", **delta_result}])
    comparison = pd.DataFrame(
        [
            {
                "Fusion_Level": "Early feature fusion",
                "Model": selected_model,
                "Internal_Pooled_OOF_C": float(selected["Pooled_OOF_C"]),
                "External_C": external_c,
                "Selection_Status": "input-matched comparator; does not automatically replace mainline",
            },
            {
                "Fusion_Level": "Late score fusion",
                "Model": "ExtraST",
                "Internal_Pooled_OOF_C": late_c_latest_endpoint,
                "External_C": late_external_c_latest_endpoint,
                "Selection_Status": "locked mainline",
            },
        ]
    )

    fold_table.to_csv(args.output_dir / "early_fusion_fold_metrics.csv", index=False)
    grid.to_csv(args.output_dir / "early_fusion_candidate_grid.csv", index=False)
    predictions.to_csv(args.output_dir / "early_fusion_selected_predictions.csv", index=False)
    comparison.to_csv(args.output_dir / "early_vs_locked_late_internal_comparison.csv", index=False)
    delta_table.to_csv(args.output_dir / "early_vs_late_paired_bootstrap_delta.csv", index=False)
    manifest = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_role": "input-matched comparator; no automatic mainline replacement",
        "selection_metric": "development fixed-fold pooled OOF C-index",
        "external_policy": "evaluated only after internal learner selection",
        "outer_fold_policy": "checkpoint k supplies outer-train, held-out fold k, and external embeddings",
        "preprocessing_policy": "median imputation, standardization, and PCA fitted inside outer training fold",
        "pca_components_requested": args.pca_components,
        "candidate_models": MODEL_ORDER,
        "selected_model": selected_model,
        "train_cv_n": len(development),
        "external_n": len(external),
        "raw_feature_count": int(fold_matrices[0][0].shape[1]),
        "radiomics_feature_count": len(radiomics_cols),
        "source_files": {
            "selected_branches": {"path": str(SELECTED_CSV), "md5": file_md5(SELECTED_CSV)},
            "split": {"path": str(SPLIT_CSV), "md5": file_md5(SPLIT_CSV)},
            "clinical_endpoint": {"path": str(CLINICAL_EXCEL), "md5": file_md5(CLINICAL_EXCEL)},
            "mask_index": {"path": str(MASK_INDEX_CSV), "md5": file_md5(MASK_INDEX_CSV)},
            "radiomics_raw": {"path": str(RADIOMICS_RAW_CSV), "md5": file_md5(RADIOMICS_RAW_CSV)},
            "locked_late_predictions": {"path": str(LATE_PREDICTIONS_CSV), "md5": file_md5(LATE_PREDICTIONS_CSV)},
        },
        "endpoint_note": "Downstream early-fusion fitting and all reported C-index calculations use the frozen corrected endpoint table.",
        "branches": branch_manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = [
        "# Corrected Multiphase Early-Fusion Comparator",
        "",
        "This is an input-matched comparator and does not automatically replace the locked late-score IFM/CRM.",
        "",
        comparison.to_markdown(index=False),
        "",
        delta_table.to_markdown(index=False),
        "",
        "Selection used fixed development folds and pooled OOF C-index. External outcomes were not used for selection.",
    ]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    log(f"completed comparator: {args.output_dir}")
    return args.output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
