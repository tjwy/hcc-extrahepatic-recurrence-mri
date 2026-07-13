#!/usr/bin/env python3
"""Shared survival learners and cross-validation utilities for imaging fusion.

This script does not retrain image or radiomics models. It recombines existing
OOF/external risk scores, locks one candidate per image branch by Train_CV-only
OOF performance, then fits a forced 4-input LFM and downstream CRM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv


PROJECT_ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

SPLIT_CSV = Path(os.environ.get("MR_SPLIT_CSV", PROJECT_ROOT / "data" / "splits.csv"))
EXPECTED_SPLIT_MD5 = os.environ.get("MR_EXPECTED_SPLIT_MD5", "")
EXCEL_PATH = Path(os.environ.get("MR_CLINICAL_EXCEL", PROJECT_ROOT / "data" / "clinical.xlsx"))

RAD_DIR = Path(os.environ.get("MR_RADIOMICS_OUT_ROOT", PROJECT_ROOT / "outputs" / "single_modality_models" / "radiomics"))
RAD_SUMMARY = Path(os.environ.get("MR_RADIOMICS_SUMMARY_CSV", RAD_DIR / "radiomics_model_summary.csv"))
RAD_VAL = Path(os.environ.get("MR_RADIOMICS_OOF_CSV", RAD_DIR / "radiomics_val_predictions.csv"))
RAD_EXT = Path(os.environ.get("MR_RADIOMICS_EXTERNAL_CSV", RAD_DIR / "radiomics_test_predictions.csv"))

ID_COL = "MRI_id"
TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
EXTERNAL_SET = "External_Test"
CLINICAL_FIXED_7 = [
    "Age",
    "Sex",
    "Child_Pugh_Score",
    "AFP",
    "BCLC",
    "Tumor_Size",
    "Tumor_Number",
]
FOUR_INPUT_FEATURES = ["Risk_2D", "Risk_2p5D", "Risk_3D", "Risk_Rad"]
CRM_FEATURES = ["Final_Imaging_Score"] + CLINICAL_FIXED_7
LFM_MODEL_NAMES = [
    "CoxPH",
    "RSF",
    "SVM",
    "GB",
    "ExtraST",
]
CRM_MODEL_NAMES = list(LFM_MODEL_NAMES)

def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def c_index_safe(times: Iterable[float], risk: Iterable[float], events: Iterable[int]) -> float:
    try:
        return float(concordance_index(np.asarray(times, dtype=float), -np.asarray(risk, dtype=float), np.asarray(events, dtype=int)))
    except Exception:
        return 0.5


def normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ID_COL] = out[ID_COL].astype(str).str.strip()
    return out


def load_base_dataframe() -> pd.DataFrame:
    clinical = normalize_id(pd.read_excel(EXCEL_PATH))
    clinical = clinical.dropna(subset=[TIME_COL, EVENT_COL])
    split = normalize_id(pd.read_csv(SPLIT_CSV))
    base = clinical.merge(split[[ID_COL, "Set"]], on=ID_COL, how="inner")
    for col in CLINICAL_FIXED_7:
        base[col] = pd.to_numeric(base[col], errors="coerce")
        base[col] = base[col].fillna(base[col].median())
    return base


def select_branch_candidate(
    summary_df: pd.DataFrame,
    family: str,
    tie_margin: float = 0.002,
) -> pd.Series:
    """Select one image candidate by Train_CV-only OOF, breaking near ties by fold-best Val.

    External_C is intentionally ignored here; tests enforce this.
    """
    sub = summary_df[summary_df["Family"].astype(str).str.lower() == family.lower()].copy()
    if sub.empty:
        raise ValueError(f"No candidates found for family={family}")
    sub["Pooled_OOF_C"] = pd.to_numeric(sub["Pooled_OOF_C"], errors="coerce")
    sub["Mean_FoldBest_Val_C"] = pd.to_numeric(sub["Mean_FoldBest_Val_C"], errors="coerce")
    best_oof = float(sub["Pooled_OOF_C"].max())
    tied = sub[sub["Pooled_OOF_C"] >= best_oof - tie_margin].copy()
    return tied.sort_values(["Mean_FoldBest_Val_C", "Pooled_OOF_C"], ascending=[False, False]).iloc[0]


def make_model(model_name: str):
    if model_name == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=0.001)
    if model_name == "RSF":
        return RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "SVM":
        return FastSurvivalSVM(max_iter=2000, tol=1e-5, random_state=42)
    if model_name == "GB":
        return GradientBoostingSurvivalAnalysis(
            random_state=42,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
        )
    if model_name == "ExtraST":
        return ExtraSurvivalTrees(
            n_estimators=300,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def predict_fold_or_full(
    X_train: np.ndarray,
    times_train: np.ndarray,
    events_train: np.ndarray,
    X_pred: np.ndarray,
    model_name: str,
) -> np.ndarray:
    y_train = Surv.from_arrays(event=events_train.astype(bool), time=times_train.astype(float))
    model = make_model(model_name)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_pred), dtype=float)


def oof_predictions(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, float]:
    X_raw = df_train[feature_cols].astype(float).to_numpy()
    times = df_train[TIME_COL].astype(float).to_numpy()
    events = df_train[EVENT_COL].astype(int).to_numpy()
    oof = np.zeros(len(df_train), dtype=float)
    seen = np.zeros(len(df_train), dtype=int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(X_raw):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[train_idx])
        X_val = scaler.transform(X_raw[val_idx])
        oof[val_idx] = predict_fold_or_full(
            X_tr,
            times[train_idx],
            events[train_idx],
            X_val,
            model_name,
        )
        seen[val_idx] += 1
    valid = seen > 0
    return oof, c_index_safe(times[valid], oof[valid], events[valid])


def fit_full_predict(
    df_train: pd.DataFrame,
    df_external: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_raw = df_train[feature_cols].astype(float).to_numpy()
    X_ext_raw = df_external[feature_cols].astype(float).to_numpy()
    times_train = df_train[TIME_COL].astype(float).to_numpy()
    events_train = df_train[EVENT_COL].astype(int).to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_ext = scaler.transform(X_ext_raw)
    train_pred = predict_fold_or_full(X_train, times_train, events_train, X_train, model_name)
    ext_pred = predict_fold_or_full(X_train, times_train, events_train, X_ext, model_name)
    return np.asarray(train_pred, dtype=float), np.asarray(ext_pred, dtype=float)


def summarize_repeated_cv(
    df_train: pd.DataFrame,
    df_external: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
    base_seed: int,
    n_repeats: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = []
    repeat_oof: dict[tuple[int, str], np.ndarray] = {}
    for repeat_idx in range(n_repeats):
        seed = base_seed + repeat_idx
        repeat_rows = []
        for model_name in model_names:
            oof, val_c = oof_predictions(df_train, feature_cols, model_name, seed, n_splits)
            repeat_oof[(repeat_idx, model_name)] = oof
            repeat_rows.append(
                {
                    "Repeat": repeat_idx,
                    "Seed": seed,
                    "Model": model_name,
                    "Feature_Set": "+".join(feature_cols),
                    "N_Features": len(feature_cols),
                    "Val_C": val_c,
                }
            )
        repeat_df = pd.DataFrame(repeat_rows).sort_values(["Val_C", "Model"], ascending=[False, True]).reset_index(drop=True)
        repeat_df["Rank"] = np.arange(1, len(repeat_df) + 1)
        rows.extend(repeat_df.to_dict("records"))

    repeat_table = pd.DataFrame(rows)
    stability = (
        repeat_table.groupby(["Model", "Feature_Set", "N_Features"], as_index=False)
        .agg(
            Mean_Val_C=("Val_C", "mean"),
            Std_Val_C=("Val_C", "std"),
            Min_Val_C=("Val_C", "min"),
            Max_Val_C=("Val_C", "max"),
            Top1_Freq=("Rank", lambda s: float((s == 1).mean())),
            Top3_Freq=("Rank", lambda s: float((s <= 3).mean())),
            N_Repeats=("Repeat", "nunique"),
        )
        .fillna({"Std_Val_C": 0.0})
    )
    stability["Model_Order"] = stability["Model"].map({name: idx for idx, name in enumerate(model_names)})
    stability = stability.sort_values(
        ["Mean_Val_C", "Std_Val_C", "Model_Order"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    locked_model = str(stability.iloc[0]["Model"])
    locked_oof, locked_oof_c = oof_predictions(df_train, feature_cols, locked_model, base_seed, n_splits)
    train_full, external_pred = fit_full_predict(df_train, df_external, feature_cols, locked_model)
    train_c = c_index_safe(df_train[TIME_COL], train_full, df_train[EVENT_COL])
    external_c = c_index_safe(df_external[TIME_COL], external_pred, df_external[EVENT_COL])
    locked = {
        "Model": locked_model,
        "Feature_Set": "+".join(feature_cols),
        "OOF_Pred": locked_oof,
        "OOF_C_base_seed": locked_oof_c,
        "Train_Full_Pred": train_full,
        "External_Pred": external_pred,
        "Train_C_full": train_c,
        "External_C": external_c,
        "Mean_Val_C": float(stability.iloc[0]["Mean_Val_C"]),
        "Std_Val_C": float(stability.iloc[0]["Std_Val_C"]),
        "Top1_Freq": float(stability.iloc[0]["Top1_Freq"]),
        "Top3_Freq": float(stability.iloc[0]["Top3_Freq"]),
    }
    return repeat_table, stability, locked


def load_image_candidate(row: pd.Series, base: pd.DataFrame, risk_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = normalize_id(pd.read_csv(row["OOF_CSV"]))
    ext = normalize_id(pd.read_csv(row["External_CSV"]))
    train_base = base[base["Set"] == "Train_CV"][[ID_COL, TIME_COL, EVENT_COL]].copy()
    ext_base = base[base["Set"] == EXTERNAL_SET][[ID_COL, TIME_COL, EVENT_COL]].copy()
    train = train_base.merge(oof[[ID_COL, "Risk_Score"]], on=ID_COL, how="inner").rename(columns={"Risk_Score": risk_name})
    external = ext_base.merge(ext[[ID_COL, "Risk_Score"]], on=ID_COL, how="inner").rename(columns={"Risk_Score": risk_name})
    return train, external


def load_rad_candidate(model_name: str, base: pd.DataFrame, risk_name: str = "Risk_Rad") -> tuple[pd.DataFrame, pd.DataFrame]:
    val = normalize_id(pd.read_csv(RAD_VAL))
    ext = normalize_id(pd.read_csv(RAD_EXT))
    score_col = f"{model_name}_Score"
    if score_col not in val.columns or score_col not in ext.columns:
        raise ValueError(f"Missing radiomics score column for {model_name}")
    train_base = base[base["Set"] == "Train_CV"][[ID_COL, TIME_COL, EVENT_COL]].copy()
    ext_base = base[base["Set"] == EXTERNAL_SET][[ID_COL, TIME_COL, EVENT_COL]].copy()
    train = train_base.merge(val[[ID_COL, score_col]], on=ID_COL, how="inner").rename(columns={score_col: risk_name})
    external = ext_base.merge(ext[[ID_COL, score_col]], on=ID_COL, how="inner").rename(columns={score_col: risk_name})
    return train, external


def merge_feature_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = tables[0].copy()
    for table in tables[1:]:
        keep_cols = [ID_COL] + [col for col in table.columns if col not in {ID_COL, TIME_COL, EVENT_COL}]
        merged = merged.merge(table[keep_cols], on=ID_COL, how="inner")
    return merged


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    try:
        value = spearmanr(a.astype(float), b.astype(float), nan_policy="omit").correlation
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


