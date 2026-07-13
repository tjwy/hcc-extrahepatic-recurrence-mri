#!/usr/bin/env python3
"""Select preprocessing and survival learner for the clinical-radiologic model.

This keeps the locked full4 IFM predictions fixed and only varies the CRM
clinical preprocessing plus survival learner. External v20/227 metrics are
reported as diagnostics and are not used for model fitting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv


ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
INPUT_DIR = Path(os.environ.get("MR_CRM_INPUT_DIR", ROOT / "outputs" / "imaging_fusion" / "late"))
OUT_ROOT = Path(os.environ.get("MR_CRM_OUT", ROOT / "outputs" / "clinical_radiologic_model"))
SPLIT_CSV = Path(os.environ.get("MR_SPLIT_CSV", ROOT / "data" / "splits.csv"))
CLINICAL_EXCEL = Path(os.environ.get("MR_CLINICAL_EXCEL", ROOT / "data" / "clinical.xlsx"))

ID_COL = "MRI_id"
TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
CLINICAL_FIXED_7 = [
    "Age",
    "Sex",
    "Child_Pugh_Score",
    "AFP",
    "BCLC",
    "Tumor_Size",
    "Tumor_Number",
]
CATEGORICAL_COLS = ["Sex", "Child_Pugh_Score", "BCLC", "Tumor_Number"]


@dataclass(frozen=True)
class PreprocessSpec:
    name: str
    categorical: bool
    log_afp: bool
    log_tumor_size: bool
    afp_high400: bool
    winsorize_afp: bool


PREPROCESS_SPECS = [
    PreprocessSpec("numeric_raw", False, False, False, False, False),
    PreprocessSpec("numeric_log_afp", False, True, False, False, False),
    PreprocessSpec("numeric_log_afp_size", False, True, True, False, False),
    PreprocessSpec("cat_log_afp", True, True, False, False, False),
    PreprocessSpec("cat_log_afp_size", True, True, True, False, False),
    PreprocessSpec("cat_log_afp_high400", True, True, False, True, False),
    PreprocessSpec("cat_log_afp_winsor", True, True, False, False, True),
    PreprocessSpec("cat_log_afp_size_high400", True, True, True, True, False),
]

PREPROCESS_PRESETS = {
    "numeric": ["numeric_raw", "numeric_log_afp", "numeric_log_afp_size"],
    "all": [p.name for p in PREPROCESS_SPECS],
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: dict


MODEL_SPECS = [
    ModelSpec("CoxPH_a0.001", "CoxPH", {"alpha": 0.001}),
    ModelSpec("CoxPH_a0.01", "CoxPH", {"alpha": 0.01}),
    ModelSpec("CoxPH_a0.1", "CoxPH", {"alpha": 0.1}),
    ModelSpec("Coxnet_l10.1_a0.01", "Coxnet", {"l1_ratio": 0.1, "alphas": [0.01]}),
    ModelSpec("Coxnet_l10.5_a0.01", "Coxnet", {"l1_ratio": 0.5, "alphas": [0.01]}),
    ModelSpec("Coxnet_l10.9_a0.01", "Coxnet", {"l1_ratio": 0.9, "alphas": [0.01]}),
    ModelSpec("Coxnet_l10.5_a0.1", "Coxnet", {"l1_ratio": 0.5, "alphas": [0.1]}),
    ModelSpec("SVM", "SVM", {}),
    ModelSpec("RSF_leaf8", "RSF", {"min_samples_leaf": 8, "max_features": "sqrt"}),
    ModelSpec("RSF_leaf15", "RSF", {"min_samples_leaf": 15, "max_features": "sqrt"}),
    ModelSpec("RSF_leaf25", "RSF", {"min_samples_leaf": 25, "max_features": "sqrt"}),
    ModelSpec("RSF_leaf15_allfeat", "RSF", {"min_samples_leaf": 15, "max_features": None}),
    ModelSpec("ExtraST_leaf8", "ExtraST", {"min_samples_leaf": 8, "max_features": "sqrt"}),
    ModelSpec("ExtraST_leaf15", "ExtraST", {"min_samples_leaf": 15, "max_features": "sqrt"}),
    ModelSpec("GB_d1_lr0.03", "GB", {"max_depth": 1, "learning_rate": 0.03, "n_estimators": 300}),
    ModelSpec("GB_d2_lr0.03", "GB", {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 300}),
    ModelSpec("GB_d1_lr0.01", "GB", {"max_depth": 1, "learning_rate": 0.01, "n_estimators": 500}),
    ModelSpec("GB_d2_lr0.01", "GB", {"max_depth": 2, "learning_rate": 0.01, "n_estimators": 500}),
]

MODEL_PRESETS = {
    "linear": [
        "CoxPH_a0.001",
        "CoxPH_a0.1",
        "Coxnet_l10.1_a0.01",
        "Coxnet_l10.5_a0.01",
        "Coxnet_l10.9_a0.01",
        "Coxnet_l10.5_a0.1",
        "SVM",
    ],
    "focused": [
        "CoxPH_a0.1",
        "Coxnet_l10.5_a0.1",
        "SVM",
        "RSF_leaf25",
        "ExtraST_leaf8",
        "ExtraST_leaf15",
        "GB_d1_lr0.03",
        "GB_d2_lr0.03",
    ],
    "all": [m.name for m in MODEL_SPECS],
}


def c_index_safe(times: np.ndarray, risk: np.ndarray, events: np.ndarray) -> float:
    try:
        return float(concordance_index(times.astype(float), -risk.astype(float), events.astype(int)))
    except Exception:
        return float("nan")


def select_candidate(summary: pd.DataFrame) -> pd.Series:
    """Select the CRM pipeline using development-cohort OOF C-index only."""
    eligible = summary.dropna(subset=["OOF_C_base_seed"]).copy()
    if eligible.empty:
        raise RuntimeError("No valid CRM candidates were fitted")
    return eligible.sort_values(
        ["OOF_C_base_seed", "Mean_Val_C", "Preprocess", "Model"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]


def make_model(spec: ModelSpec):
    if spec.family == "CoxPH":
        return CoxPHSurvivalAnalysis(**spec.params)
    if spec.family == "Coxnet":
        return CoxnetSurvivalAnalysis(**spec.params)
    if spec.family == "SVM":
        return FastSurvivalSVM(max_iter=2000, tol=1e-5, random_state=42)
    if spec.family == "RSF":
        return RandomSurvivalForest(n_estimators=300, random_state=42, n_jobs=-1, **spec.params)
    if spec.family == "ExtraST":
        return ExtraSurvivalTrees(n_estimators=300, random_state=42, n_jobs=-1, **spec.params)
    if spec.family == "GB":
        return GradientBoostingSurvivalAnalysis(random_state=42, **spec.params)
    raise ValueError(spec)


class CRMPreprocessor:
    def __init__(self, spec: PreprocessSpec):
        self.spec = spec

    def fit(self, df: pd.DataFrame) -> "CRMPreprocessor":
        self.medians_ = {c: pd.to_numeric(df[c], errors="coerce").median() for c in CLINICAL_FIXED_7}
        self.modes_ = {}
        self.categories_ = {}
        for c in CATEGORICAL_COLS:
            s = pd.to_numeric(df[c], errors="coerce")
            mode = s.mode(dropna=True)
            self.modes_[c] = float(mode.iloc[0]) if len(mode) else self.medians_[c]
            self.categories_[c] = sorted(s.dropna().unique().tolist())
        afp = pd.to_numeric(df["AFP"], errors="coerce").fillna(self.medians_["AFP"])
        self.afp_p99_ = float(afp.quantile(0.99))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["Final_Imaging_Score"] = pd.to_numeric(df["Final_Imaging_Score"], errors="coerce").astype(float)

        age = pd.to_numeric(df["Age"], errors="coerce").fillna(self.medians_["Age"])
        afp = pd.to_numeric(df["AFP"], errors="coerce").fillna(self.medians_["AFP"]).clip(lower=0)
        if self.spec.winsorize_afp:
            afp = afp.clip(upper=self.afp_p99_)
        tumor_size = pd.to_numeric(df["Tumor_Size"], errors="coerce").fillna(self.medians_["Tumor_Size"]).clip(lower=0)

        out["Age"] = age
        out["AFP_log1p" if self.spec.log_afp else "AFP"] = np.log1p(afp) if self.spec.log_afp else afp
        out["Tumor_Size_log1p" if self.spec.log_tumor_size else "Tumor_Size"] = (
            np.log1p(tumor_size) if self.spec.log_tumor_size else tumor_size
        )
        if self.spec.afp_high400:
            out["AFP_high400"] = (afp > 400).astype(float)

        if self.spec.categorical:
            for col in CATEGORICAL_COLS:
                s = pd.to_numeric(df[col], errors="coerce").fillna(self.modes_[col])
                for cat in self.categories_[col]:
                    out[f"{col}_{cat:g}"] = (s == cat).astype(float)
        else:
            for col in CATEGORICAL_COLS:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(self.medians_[col]).astype(float)

        return out


def prepare_crm_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(input_dir / "all_selected_predictions.csv")
    pred[ID_COL] = pred[ID_COL].astype(str).str.strip()
    pred = pred[pred["Feature_Set_Label"] == "all4"].copy()
    pred = pred.rename(columns={"IFM_Risk": "Final_Imaging_Score"})

    split = pd.read_csv(SPLIT_CSV)
    split[ID_COL] = split[ID_COL].astype(str).str.strip()
    clinical = pd.read_excel(CLINICAL_EXCEL)
    clinical[ID_COL] = clinical[ID_COL].astype(str).str.strip()
    base = clinical.merge(split[[ID_COL, "Set"]], on=ID_COL, how="inner")
    keep = [ID_COL, "Set", TIME_COL, EVENT_COL, "Cohort"] + CLINICAL_FIXED_7
    base = base[keep].copy()

    merged = pred[[ID_COL, "Set", TIME_COL, EVENT_COL, "Final_Imaging_Score"]].merge(
        base[[ID_COL, "Cohort"] + CLINICAL_FIXED_7],
        on=ID_COL,
        how="inner",
    )
    train = merged[merged["Set"] == "Train_CV"].reset_index(drop=True)
    external = merged[merged["Set"] == "External_Test"].reset_index(drop=True)
    if len(train) != 732 or len(external) != 227:
        raise RuntimeError(f"Unexpected Train_CV/External sizes: {len(train)}/{len(external)}")
    return train, external


def predict_with_pipeline(train: pd.DataFrame, pred: pd.DataFrame, prep_spec: PreprocessSpec, model_spec: ModelSpec) -> np.ndarray:
    prep = CRMPreprocessor(prep_spec).fit(train)
    x_train_raw = prep.transform(train)
    x_pred_raw = prep.transform(pred)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_pred = scaler.transform(x_pred_raw)
    y = Surv.from_arrays(
        event=train[EVENT_COL].astype(int).to_numpy().astype(bool),
        time=train[TIME_COL].astype(float).to_numpy(),
    )
    model = make_model(model_spec)
    model.fit(x_train, y)
    return np.asarray(model.predict(x_pred), dtype=float)


def oof_predictions(
    train: pd.DataFrame,
    prep_spec: PreprocessSpec,
    model_spec: ModelSpec,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, float]:
    out = np.zeros(len(train), dtype=float)
    seen = np.zeros(len(train), dtype=int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(train):
        tr = train.iloc[tr_idx].reset_index(drop=True)
        va = train.iloc[va_idx].reset_index(drop=True)
        try:
            out[va_idx] = predict_with_pipeline(tr, va, prep_spec, model_spec)
        except Exception as exc:
            raise RuntimeError(
                f"Failed fold prediction for preprocess={prep_spec.name}, model={model_spec.name}"
            ) from exc
        seen[va_idx] += 1
    valid = seen > 0
    return out, c_index_safe(
        train.loc[valid, TIME_COL].to_numpy(),
        out[valid],
        train.loc[valid, EVENT_COL].to_numpy(),
    )


def run(args: argparse.Namespace) -> Path:
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train, external = prepare_crm_tables(args.input_dir)
    selected_preprocess_names = set(PREPROCESS_PRESETS[args.preprocess_preset])
    preprocess_specs = [p for p in PREPROCESS_SPECS if p.name in selected_preprocess_names]
    selected_model_names = set(MODEL_PRESETS[args.model_preset])
    model_specs = [m for m in MODEL_SPECS if m.name in selected_model_names]
    summary_path = out_dir / "full4_crm_preprocess_sensitivity_summary.csv"

    rows = []
    full_rows = []
    for prep_spec in preprocess_specs:
        print(f"[{dt.datetime.now():%F %T}] preprocess={prep_spec.name}", flush=True)
        for model_spec in model_specs:
            val_cs = []
            failed = False
            for repeat_idx in range(args.repeats):
                try:
                    _, val_c = oof_predictions(train, prep_spec, model_spec, args.seed + repeat_idx, args.splits)
                except Exception as exc:
                    print(f"  {model_spec.name}: failed repeat {repeat_idx}: {exc}", flush=True)
                    failed = True
                    break
                val_cs.append(val_c)

            if failed:
                rows.append(
                    {
                        "Preprocess": prep_spec.name,
                        "Model": model_spec.name,
                        "Model_Family": model_spec.family,
                        "Mean_Val_C": np.nan,
                        "Std_Val_C": np.nan,
                        "Min_Val_C": np.nan,
                        "Max_Val_C": np.nan,
                        "OOF_C_base_seed": np.nan,
                        "Train_C_full": np.nan,
                        "Overfit_Gap": np.nan,
                        "External_C_Diagnostic": np.nan,
                        "External_TCGA_C_Diagnostic": np.nan,
                        "External_NonTCGA_C_Diagnostic": np.nan,
                        "N_Repeats": args.repeats,
                        "N_Splits": args.splits,
                    }
                )
                pd.DataFrame(rows).sort_values(
                    ["OOF_C_base_seed", "Mean_Val_C", "Preprocess", "Model"],
                    ascending=[False, False, True, True],
                ).to_csv(summary_path, index=False)
                continue

            base_oof, oof_c = oof_predictions(train, prep_spec, model_spec, args.seed, args.splits)
            train_pred = predict_with_pipeline(train, train, prep_spec, model_spec)
            ext_pred = predict_with_pipeline(train, external, prep_spec, model_spec)
            train_c = c_index_safe(train[TIME_COL].to_numpy(), train_pred, train[EVENT_COL].to_numpy())
            ext_c = c_index_safe(external[TIME_COL].to_numpy(), ext_pred, external[EVENT_COL].to_numpy())

            tcga = external["Cohort"].astype(str).str.contains("TCGA", case=False, na=False)
            ext_tcga_c = c_index_safe(
                external.loc[tcga, TIME_COL].to_numpy(),
                ext_pred[tcga.to_numpy()],
                external.loc[tcga, EVENT_COL].to_numpy(),
            )
            ext_nontcga_c = c_index_safe(
                external.loc[~tcga, TIME_COL].to_numpy(),
                ext_pred[(~tcga).to_numpy()],
                external.loc[~tcga, EVENT_COL].to_numpy(),
            )

            row = {
                "Preprocess": prep_spec.name,
                "Model": model_spec.name,
                "Model_Family": model_spec.family,
                "Mean_Val_C": float(np.nanmean(val_cs)),
                "Std_Val_C": float(np.nanstd(val_cs, ddof=1)),
                "Min_Val_C": float(np.nanmin(val_cs)),
                "Max_Val_C": float(np.nanmax(val_cs)),
                "OOF_C_base_seed": oof_c,
                "Train_C_full": train_c,
                "Overfit_Gap": train_c - oof_c,
                "External_C_Diagnostic": ext_c,
                "External_TCGA_C_Diagnostic": ext_tcga_c,
                "External_NonTCGA_C_Diagnostic": ext_nontcga_c,
                "N_Repeats": args.repeats,
                "N_Splits": args.splits,
            }
            rows.append(row)
            pd.DataFrame(rows).sort_values(
                ["OOF_C_base_seed", "Mean_Val_C", "Preprocess", "Model"],
                ascending=[False, False, True, True],
            ).to_csv(summary_path, index=False)
            full_rows.append(
                pd.DataFrame(
                    {
                        ID_COL: train[ID_COL],
                        "Set": "Train_CV",
                        "Preprocess": prep_spec.name,
                        "Model": model_spec.name,
                        "OOF_Pred_base_seed": base_oof,
                        "Train_Full_Pred": train_pred,
                    }
                )
            )
            full_rows.append(
                pd.DataFrame(
                    {
                        ID_COL: external[ID_COL],
                        "Set": "External_Test",
                        "Preprocess": prep_spec.name,
                        "Model": model_spec.name,
                        "OOF_Pred_base_seed": np.nan,
                        "Train_Full_Pred": ext_pred,
                    }
                )
            )
            print(
                f"  {model_spec.name}: mean={row['Mean_Val_C']:.4f} "
                f"oof={oof_c:.4f} train={train_c:.4f} ext={ext_c:.4f}",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values(
        ["OOF_C_base_seed", "Mean_Val_C", "Preprocess", "Model"],
        ascending=[False, False, True, True],
    )
    summary.to_csv(summary_path, index=False)
    all_predictions = pd.concat(full_rows, ignore_index=True)
    all_predictions.to_csv(out_dir / "full4_crm_preprocess_sensitivity_predictions.csv", index=False)

    selected = select_candidate(summary)
    selected_predictions = all_predictions[
        all_predictions["Preprocess"].eq(selected["Preprocess"])
        & all_predictions["Model"].eq(selected["Model"])
    ].copy()
    selected_predictions["CRM_Risk_For_Analysis"] = selected_predictions["OOF_Pred_base_seed"].where(
        selected_predictions["Set"].eq("Train_CV"),
        selected_predictions["Train_Full_Pred"],
    )
    selected_predictions["CRM_Risk_FullFit"] = selected_predictions["Train_Full_Pred"]
    ifm = pd.read_csv(args.input_dir / "all_selected_predictions.csv")
    ifm[ID_COL] = ifm[ID_COL].astype(str).str.strip()
    labels = pd.concat(
        [
            train[[ID_COL, "Set", TIME_COL, EVENT_COL]],
            external[[ID_COL, "Set", TIME_COL, EVENT_COL]],
        ],
        ignore_index=True,
    )
    final_predictions = labels.merge(
        ifm[[ID_COL, "IFM_Risk"]], on=ID_COL, how="inner", validate="one_to_one"
    ).merge(
        selected_predictions[[ID_COL, "CRM_Risk_For_Analysis", "CRM_Risk_FullFit"]],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
    final_predictions.to_csv(out_dir / "final_predictions.csv", index=False)
    manifest = {
        "timestamp": dt.datetime.now().strftime("%F %T"),
        "input_dir": str(args.input_dir),
        "selection_scope": "Train_CV repeated CV; external diagnostic only",
        "external_set": "v20/227",
        "ifm_route": "fixed full4 5mm all4 IFM predictions",
        "crm_features": ["Final_Imaging_Score"] + CLINICAL_FIXED_7,
        "preprocess_preset": args.preprocess_preset,
        "preprocess_specs": [spec.__dict__ for spec in preprocess_specs],
        "model_preset": args.model_preset,
        "model_specs": [{"name": m.name, "family": m.family, "params": m.params} for m in model_specs],
        "seed": args.seed,
        "repeats": args.repeats,
        "splits": args.splits,
        "train_cv_n": int(len(train)),
        "external_n": int(len(external)),
        "train_cv_events": int(train[EVENT_COL].sum()),
        "external_events": int(external[EVENT_COL].sum()),
        "selection_metric": "development pooled OOF C-index at the prespecified base seed",
        "selected_preprocess": str(selected["Preprocess"]),
        "selected_model": str(selected["Model"]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full4 CRM preprocessing sensitivity.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="focused")
    parser.add_argument("--preprocess-preset", choices=sorted(PREPROCESS_PRESETS), default="all")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
