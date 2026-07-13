import datetime
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv


warnings.filterwarnings("ignore")

ROOT = os.environ.get(
    "MR_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[1]),
)
OUT_ROOT = os.path.join(ROOT, "outputs", "single_modality_models", "radiomics")
OUT_ROOT = os.environ.get("MR_RADIOMICS_OUT_ROOT", OUT_ROOT)
RADIOMICS_CSV = os.path.join(
    ROOT,
    "data",
    "radiomics_features_raw.csv",
)
RADIOMICS_CSV = os.environ.get("MR_RADIOMICS_CSV", RADIOMICS_CSV)
SPLIT_CSV = os.environ.get("MR_SPLIT_CSV", os.path.join(ROOT, "data", "splits.csv"))
CLINICAL_EXCEL = os.environ.get("MR_CLINICAL_EXCEL", os.path.join(ROOT, "data", "clinical.xlsx"))

ID_COL = "MRI_id"
TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
EXTERNAL_SET_VALUES = ["External_Test", "External_Zhongshan"]
ADJUVANT_SET_VALUES = ["Adjuvant_Test"]
FEATURE_RE = re.compile(r"^(AP|PVP|DP)_original_")


def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_surv(df):
    return Surv.from_arrays(
        event=df[EVENT_COL].astype(bool).values,
        time=df[TIME_COL].astype(float).values,
    )


def c_index_lifelines(df, score_col):
    try:
        return float(concordance_index(df[TIME_COL].values, -df[score_col].values, df[EVENT_COL].values))
    except Exception:
        return 0.5


def c_index_sksurv_safe(y, preds):
    try:
        return float(concordance_index_censored(y["event"], y["time"], preds)[0])
    except Exception:
        return 0.5


def build_model(name):
    if name == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=0.001)
    if name == "RSF":
        return RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "GB":
        return GradientBoostingSurvivalAnalysis(
            random_state=42,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
        )
    if name == "SVM":
        return FastSurvivalSVM(max_iter=2000, tol=1e-5, random_state=42)
    raise ValueError(name)


def load_data():
    for path in [RADIOMICS_CSV, SPLIT_CSV, CLINICAL_EXCEL]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    rad = pd.read_csv(RADIOMICS_CSV)
    split = pd.read_csv(SPLIT_CSV)
    clin = pd.read_excel(CLINICAL_EXCEL)

    for df in [rad, split, clin]:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()

    feature_cols = [c for c in rad.columns if FEATURE_RE.match(str(c))]
    phase_counts = {
        phase: int(sum(c.startswith(f"{phase}_original_") for c in feature_cols))
        for phase in ["AP", "PVP", "DP"]
    }
    if len(feature_cols) != 321 or sorted(phase_counts.values()) != [107, 107, 107]:
        raise RuntimeError(f"Expected 107 original radiomics features per phase, got {phase_counts}")

    df = rad[[ID_COL] + feature_cols].merge(clin[[ID_COL, TIME_COL, EVENT_COL]], on=ID_COL, how="inner")
    df = df.merge(split[[ID_COL, "Set", "Fold"]], on=ID_COL, how="inner")
    df = df.dropna(subset=[TIME_COL, EVENT_COL]).copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log(f"Loaded data shape={df.shape}")
    log(f"Set counts={df['Set'].value_counts().to_dict()}")
    log(f"Fixed original radiomics features={len(feature_cols)} phase_counts={phase_counts}")
    return df, feature_cols, phase_counts


def fit_predict(model_name, X_tr, y_tr, X_pred):
    model = build_model(model_name)
    model.fit(X_tr, y_tr)
    return model, np.asarray(model.predict(X_pred), dtype=float)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    run_tag = tag()
    df, feature_cols, phase_counts = load_data()
    traincv = df[df["Set"] == "Train_CV"].copy().reset_index(drop=True)
    external = df[df["Set"].isin(EXTERNAL_SET_VALUES)].copy().reset_index(drop=True)
    adjuvant = df[df["Set"].isin(ADJUVANT_SET_VALUES)].copy().reset_index(drop=True)
    folds = sorted(traincv["Fold"].dropna().astype(int).unique().tolist())
    model_names = ["CoxPH", "RSF", "GB", "SVM"]

    val_records = {m: [] for m in model_names}
    ext_records = {m: [] for m in model_names}
    fold_rows = []

    for fold in folds:
        tr = traincv[traincv["Fold"].astype(int) != fold].copy().reset_index(drop=True)
        va = traincv[traincv["Fold"].astype(int) == fold].copy().reset_index(drop=True)

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr = pd.DataFrame(imp.fit_transform(tr[feature_cols]), columns=feature_cols)
        X_va = pd.DataFrame(imp.transform(va[feature_cols]), columns=feature_cols)
        X_ext = pd.DataFrame(imp.transform(external[feature_cols]), columns=feature_cols)
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols)
        X_va = pd.DataFrame(scaler.transform(X_va), columns=feature_cols)
        X_ext = pd.DataFrame(scaler.transform(X_ext), columns=feature_cols)

        y_tr = build_surv(tr)
        y_va = build_surv(va)
        y_ext = build_surv(external)
        log(f"Fold {fold}: train={len(tr)} val={len(va)} features={len(feature_cols)}")

        for model_name in model_names:
            try:
                model, va_pred = fit_predict(model_name, X_tr, y_tr, X_va)
                ext_pred = np.asarray(model.predict(X_ext), dtype=float)
            except Exception as exc:
                log(f"Fold {fold} | {model_name} failed: {exc}")
                va_pred = np.zeros(len(va), dtype=float)
                ext_pred = np.zeros(len(external), dtype=float)

            fold_val_c = c_index_sksurv_safe(y_va, va_pred)
            fold_ext_c = c_index_sksurv_safe(y_ext, ext_pred)
            fold_rows.append(
                {
                    "Fold": int(fold),
                    "Model": model_name,
                    "Fold_Val_C": fold_val_c,
                    "Fold_External_C": fold_ext_c,
                    "N_Features": len(feature_cols),
                }
            )
            log(f"Fold {fold} | {model_name}: ValC={fold_val_c:.4f} ExtC={fold_ext_c:.4f}")

            val_records[model_name].append(
                pd.DataFrame({ID_COL: va[ID_COL].values, "Fold": int(fold), "Score": va_pred})
            )
            ext_records[model_name].append(
                pd.DataFrame({ID_COL: external[ID_COL].values, "Fold": int(fold), "Score": ext_pred})
            )

    summary_rows = []
    val_wide = traincv[[ID_COL, TIME_COL, EVENT_COL]].copy()
    ext_wide = external[[ID_COL, TIME_COL, EVENT_COL]].copy()
    best_model = None
    best_val = -1.0
    best_oof = None
    best_ext = None

    for model_name in model_names:
        val_df = pd.concat(val_records[model_name], ignore_index=True)
        ext_df = pd.concat(ext_records[model_name], ignore_index=True)
        oof = val_df.groupby(ID_COL, as_index=False)["Score"].mean().rename(columns={"Score": "Rad_Score"})
        ext = ext_df.groupby(ID_COL, as_index=False)["Score"].mean().rename(columns={"Score": "Rad_Score"})
        oof_eval = traincv[[ID_COL, TIME_COL, EVENT_COL]].merge(oof, on=ID_COL, how="inner")
        ext_eval = external[[ID_COL, TIME_COL, EVENT_COL]].merge(ext, on=ID_COL, how="inner")
        val_c = c_index_lifelines(oof_eval, "Rad_Score")
        ext_c = c_index_lifelines(ext_eval, "Rad_Score")
        summary_rows.append({"Model": model_name, "Val_C": val_c, "External_C": ext_c, "N_Features": len(feature_cols)})
        val_wide = val_wide.merge(oof.rename(columns={"Rad_Score": f"{model_name}_Score"}), on=ID_COL, how="left")
        ext_wide = ext_wide.merge(ext.rename(columns={"Rad_Score": f"{model_name}_Score"}), on=ID_COL, how="left")
        if val_c > best_val:
            best_model = model_name
            best_val = val_c
            best_oof = oof_eval
            best_ext = ext_eval

    summary = pd.DataFrame(summary_rows).sort_values("Val_C", ascending=False)
    summary_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_model_summary_{run_tag}.csv")
    fold_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_fold_metrics_{run_tag}.csv")
    val_wide_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_val_predictions_{run_tag}.csv")
    ext_wide_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_test_predictions_{run_tag}.csv")
    compat_val_path = os.path.join(OUT_ROOT, "radiomics_val_predictions.csv")
    compat_test_path = os.path.join(OUT_ROOT, "radiomics_test_predictions.csv")
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    val_wide.to_csv(val_wide_path, index=False)
    ext_wide.to_csv(ext_wide_path, index=False)
    best_oof[[ID_COL, "Rad_Score"]].to_csv(compat_val_path, index=False)
    best_ext[[ID_COL, "Rad_Score"]].to_csv(compat_test_path, index=False)

    full_sets = [traincv, external]
    if len(adjuvant) > 0:
        full_sets.append(adjuvant)
    full_pred_base = pd.concat(full_sets, ignore_index=True)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_all = pd.DataFrame(imp.fit_transform(traincv[feature_cols]), columns=feature_cols)
    X_all = pd.DataFrame(scaler.fit_transform(X_all), columns=feature_cols)
    y_all = build_surv(traincv)
    final_model = build_model(best_model)
    final_model.fit(X_all, y_all)
    final_pred_rows = []
    for set_name, subset in full_pred_base.groupby("Set", sort=False):
        X_sub = pd.DataFrame(imp.transform(subset[feature_cols]), columns=feature_cols)
        X_sub = pd.DataFrame(scaler.transform(X_sub), columns=feature_cols)
        pred = np.asarray(final_model.predict(X_sub), dtype=float)
        out = subset[[ID_COL, "Set", TIME_COL, EVENT_COL]].copy()
        out["Rad_Score"] = pred
        out["Best_Model"] = best_model
        final_pred_rows.append(out)
    fullfit_pred = pd.concat(final_pred_rows, ignore_index=True)
    fullfit_pred_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_{best_model}_fullfit_predictions_{run_tag}.csv")
    fullfit_pred.to_csv(fullfit_pred_path, index=False)

    fullfit_metrics = []
    for set_name, subset in fullfit_pred.groupby("Set", sort=False):
        fullfit_metrics.append(
            {
                "Set": set_name,
                "N": int(len(subset)),
                "C_Index": c_index_lifelines(subset, "Rad_Score"),
                "Model": best_model,
            }
        )
    fullfit_metrics_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_{best_model}_fullfit_metrics_{run_tag}.csv")
    pd.DataFrame(fullfit_metrics).to_csv(fullfit_metrics_path, index=False)

    feature_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_fixed_features_{run_tag}.json")
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_policy": "Fixed AP/PVP/DP original PyRadiomics features only; no univariate or LASSO outcome-based feature screening.",
                "n_features": len(feature_cols),
                "phase_counts": phase_counts,
                "features": feature_cols,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    meta_path = os.path.join(OUT_ROOT, f"radiomics_original107x3_run_metadata_{run_tag}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_tag": run_tag,
                "timestamp": ts(),
                "command": [sys.executable] + sys.argv,
                "radiomics_csv": RADIOMICS_CSV,
                "radiomics_csv_md5": file_md5(RADIOMICS_CSV),
                "split_csv": SPLIT_CSV,
                "split_csv_md5": file_md5(SPLIT_CSV),
                "clinical_excel": CLINICAL_EXCEL,
                "clinical_excel_md5": file_md5(CLINICAL_EXCEL),
                "feature_policy": "AP/PVP/DP original PyRadiomics 107 features per phase, fixed before outcome modeling.",
                "model_selection_metric": "Pooled OOF Val_C on Train_CV only.",
                "best_model": best_model,
                "best_val_c": float(best_val),
                "output_summary": summary_path,
                "output_fullfit_predictions": fullfit_pred_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log(f"Saved summary: {summary_path}")
    log(f"Saved fixed feature manifest: {feature_path}")
    log(f"Saved fullfit predictions: {fullfit_pred_path}")
    log(f"Best radiomics model by pooled OOF Val_C: {best_model} ({best_val:.4f})")


if __name__ == "__main__":
    main()
