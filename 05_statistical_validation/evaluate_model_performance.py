#!/usr/bin/env python3
"""Statistical validation of locked distant-metastasis predictions.

The script evaluates the locked distant-metastasis endpoint predictions:
IFM, CRM, clinical-only, and BCLC-only.
"""

from __future__ import annotations

import json
import math
import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("MR_VALIDATION_OUT", ROOT / "outputs" / "statistical_validation"))
PRED_PATH = Path(os.environ.get("MR_PREDICTIONS_CSV", ROOT / "derived_inputs" / "predictions.csv"))
CLIN_PATH = Path(os.environ.get("MR_CLINICAL_EXCEL", ROOT / "data" / "clinical.xlsx"))
SPLIT_PATH = Path(os.environ.get("MR_SPLIT_CSV", ROOT / "data" / "splits.csv"))

TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
HORIZONS = [12, 24, 36]
BOOTSTRAPS = 1000
SEED = 20260623

PALETTE = {
    "CRM": "#2A6FBB",
    "IFM": "#379A8B",
    "Clinical-only": "#8E6BBE",
    "BCLC": "#C5793D",
    "Treat all": "#777777",
    "Treat none": "#111111",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


def norm_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def risk_c_index(df: pd.DataFrame, risk_col: str) -> float:
    sub = df[[TIME_COL, EVENT_COL, risk_col]].dropna()
    if sub[EVENT_COL].sum() == 0 or len(sub) < 3:
        return np.nan
    try:
        return float(concordance_index(sub[TIME_COL], -sub[risk_col], sub[EVENT_COL]))
    except ZeroDivisionError:
        return np.nan


@dataclass
class ClinicalPreprocessor:
    continuous: tuple[str, ...] = ("Age", "AFP_log1p", "Tumor_Size")
    categorical: tuple[str, ...] = ("Sex", "Child_Pugh_Score", "BCLC", "Tumor_Number")
    cont_median: dict[str, float] | None = None
    cont_mean: dict[str, float] | None = None
    cont_std: dict[str, float] | None = None
    cat_mode: dict[str, object] | None = None
    columns: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        work = df.copy()
        work["AFP_log1p"] = np.log1p(pd.to_numeric(work["AFP"], errors="coerce").clip(lower=0))
        self.cont_median, self.cont_mean, self.cont_std = {}, {}, {}
        self.cat_mode = {}
        for col in self.continuous:
            x = pd.to_numeric(work[col], errors="coerce")
            med = float(x.median())
            filled = x.fillna(med)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.cont_median[col] = med
            self.cont_mean[col] = mean
            self.cont_std[col] = std if std > 1e-8 else 1.0
        for col in self.categorical:
            mode = work[col].dropna().mode()
            self.cat_mode[col] = mode.iloc[0] if len(mode) else "Missing"
        X = self.transform(df, fitting=True)
        self.columns = X.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        work = df.copy()
        work["AFP_log1p"] = np.log1p(pd.to_numeric(work["AFP"], errors="coerce").clip(lower=0))
        parts = []
        for col in self.continuous:
            x = pd.to_numeric(work[col], errors="coerce").fillna(self.cont_median[col])
            z = (x - self.cont_mean[col]) / self.cont_std[col]
            parts.append(z.rename(col))
        cat_frame = pd.DataFrame(index=work.index)
        for col in self.categorical:
            cat_frame[col] = work[col].fillna(self.cat_mode[col]).astype(str)
        dummies = pd.get_dummies(cat_frame, columns=list(self.categorical), drop_first=True, dtype=float)
        X = pd.concat(parts + [dummies], axis=1)
        if not fitting:
            X = X.reindex(columns=self.columns, fill_value=0.0)
        return X.astype(float)


def prepare_bclc(df: pd.DataFrame, train_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    x = pd.to_numeric(df["BCLC"], errors="coerce")
    fill = float(pd.to_numeric(train_ref["BCLC"], errors="coerce").median()) if train_ref is not None else float(x.median())
    out["BCLC"] = x.fillna(fill)
    return out


def fit_cox_risk(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_builder: str,
    penalizer: float = 0.1,
) -> np.ndarray:
    if feature_builder == "clinical":
        prep = ClinicalPreprocessor().fit(train_df)
        X_train = prep.transform(train_df)
        X_test = prep.transform(test_df)
    elif feature_builder == "bclc":
        X_train = prepare_bclc(train_df, train_df)
        X_test = prepare_bclc(test_df, train_df)
    else:
        raise ValueError(feature_builder)

    train_fit = pd.concat(
        [X_train.reset_index(drop=True), train_df[[TIME_COL, EVENT_COL]].reset_index(drop=True)], axis=1
    )
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_fit, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
    return cph.predict_log_partial_hazard(X_test).to_numpy(dtype=float)


def add_clinical_and_bclc_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Clinical_only_Risk"] = np.nan
    out["BCLC_Risk"] = np.nan
    train_mask = out["Set"].eq("Train_CV")
    ext_mask = out["Set"].eq("External_Test")
    folds = sorted(out.loc[train_mask, "Fold"].dropna().unique())

    for fold in folds:
        fold_mask = train_mask & out["Fold"].eq(fold)
        fit_mask = train_mask & ~out["Fold"].eq(fold)
        out.loc[fold_mask, "Clinical_only_Risk"] = fit_cox_risk(out.loc[fit_mask], out.loc[fold_mask], "clinical")
        out.loc[fold_mask, "BCLC_Risk"] = fit_cox_risk(out.loc[fit_mask], out.loc[fold_mask], "bclc")

    out.loc[ext_mask, "Clinical_only_Risk"] = fit_cox_risk(out.loc[train_mask], out.loc[ext_mask], "clinical")
    out.loc[ext_mask, "BCLC_Risk"] = fit_cox_risk(out.loc[train_mask], out.loc[ext_mask], "bclc")

    out.loc[train_mask, "Clinical_only_Fullfit_Risk"] = fit_cox_risk(out.loc[train_mask], out.loc[train_mask], "clinical")
    out.loc[train_mask, "BCLC_Fullfit_Risk"] = fit_cox_risk(out.loc[train_mask], out.loc[train_mask], "bclc")
    out.loc[ext_mask, "Clinical_only_Fullfit_Risk"] = out.loc[ext_mask, "Clinical_only_Risk"]
    out.loc[ext_mask, "BCLC_Fullfit_Risk"] = out.loc[ext_mask, "BCLC_Risk"]
    return out


def load_analysis_table() -> pd.DataFrame:
    pred = pd.read_csv(PRED_PATH)
    clin = pd.read_excel(CLIN_PATH)
    split = pd.read_csv(SPLIT_PATH)
    for frame in (pred, clin, split):
        frame["MRI_id_norm"] = frame["MRI_id"].map(norm_id)
    clin_cols = [
        "MRI_id_norm",
        "Age",
        "Sex",
        "AFP",
        "Child_Pugh_Score",
        "BCLC",
        "Tumor_Size",
        "Tumor_Number",
        "RFS_Months",
        "Recurrence_status",
        "OS_Months",
        "OS_Status",
    ]
    merged = pred.merge(clin[clin_cols], on="MRI_id_norm", how="left", validate="one_to_one")
    merged = merged.merge(split[["MRI_id_norm", "Fold"]], on="MRI_id_norm", how="left")
    missing = merged[["Age", "AFP", "BCLC", "Fold"]].isna().sum().to_dict()
    if missing["Age"] > 0:
        raise RuntimeError(f"Clinical merge has unexpected missing Age values: {missing}")
    (OUTDIR / "clinical_missingness_audit.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    merged["CRM_Risk"] = merged["CRM_Risk_For_Analysis"]
    merged.loc[merged["Set"].eq("External_Test"), "CRM_Risk"] = merged.loc[
        merged["Set"].eq("External_Test"), "CRM_Risk_FullFit"
    ]
    merged["Model_IFM"] = merged["IFM_Risk"]
    merged["Model_CRM"] = merged["CRM_Risk"]
    return add_clinical_and_bclc_predictions(merged)


def bootstrap_cindex(
    df: pd.DataFrame,
    risk_col: str,
    n_boot: int = BOOTSTRAPS,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sub = df[[TIME_COL, EVENT_COL, risk_col]].dropna().reset_index(drop=True)
    point = risk_c_index(sub, risk_col)
    vals = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = sub.iloc[idx]
        if b[EVENT_COL].sum() == 0:
            continue
        val = risk_c_index(b, risk_col)
        if not math.isnan(val):
            vals.append(val)
    return {
        "C_index": point,
        "CI_low": float(np.percentile(vals, 2.5)),
        "CI_high": float(np.percentile(vals, 97.5)),
        "N_boot_valid": len(vals),
    }


def bootstrap_cindex_diff(
    df: pd.DataFrame,
    risk_col_a: str,
    risk_col_b: str,
    n_boot: int = BOOTSTRAPS,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 17)
    sub = df[[TIME_COL, EVENT_COL, risk_col_a, risk_col_b]].dropna().reset_index(drop=True)
    point = risk_c_index(sub, risk_col_a) - risk_c_index(sub, risk_col_b)
    vals = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = sub.iloc[idx]
        if b[EVENT_COL].sum() == 0:
            continue
        da = risk_c_index(b, risk_col_a)
        db = risk_c_index(b, risk_col_b)
        if not math.isnan(da) and not math.isnan(db):
            vals.append(da - db)
    arr = np.asarray(vals)
    p_two = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0)) if len(arr) else np.nan
    return {
        "Delta_C": float(point),
        "CI_low": float(np.percentile(arr, 2.5)),
        "CI_high": float(np.percentile(arr, 97.5)),
        "Bootstrap_p_two_sided": float(min(p_two, 1.0)),
        "N_boot_valid": int(len(arr)),
    }


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()
    return SEED + int(digest[:6], 16) % 100000


def fit_risk_calibrator(train_df: pd.DataFrame, risk_col: str) -> CoxPHFitter:
    fit_df = train_df[[TIME_COL, EVENT_COL, risk_col]].dropna().rename(columns={risk_col: "risk"})
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
    return cph


def predict_absolute_risk(cph: CoxPHFitter, df: pd.DataFrame, risk_col: str, horizon: int) -> np.ndarray:
    x = pd.DataFrame({"risk": df[risk_col].to_numpy(dtype=float)})
    surv = cph.predict_survival_function(x, times=[horizon]).T.iloc[:, 0].to_numpy(dtype=float)
    return np.clip(1 - surv, 0, 1)


def km_risk_at_horizon(times: np.ndarray, events: np.ndarray, horizon: int) -> float:
    if len(times) == 0 or np.sum(events) == 0:
        return np.nan
    km = KaplanMeierFitter()
    km.fit(times, event_observed=events)
    return float(1 - km.predict(horizon))


def calibration_table(df: pd.DataFrame, pred_col: str, set_name: str, model_name: str, horizon: int) -> pd.DataFrame:
    sub = df[[TIME_COL, EVENT_COL, pred_col]].dropna().copy()
    n_bins = 5 if len(sub) >= 120 else 4
    try:
        sub["bin"] = pd.qcut(sub[pred_col], q=n_bins, duplicates="drop")
    except ValueError:
        sub["bin"] = pd.cut(sub[pred_col], bins=n_bins)
    rows = []
    for idx, (_, g) in enumerate(sub.groupby("bin", observed=False), 1):
        rows.append(
            {
                "Set": set_name,
                "Model": model_name,
                "Horizon_months": horizon,
                "Bin": idx,
                "N": len(g),
                "Predicted_risk_mean": float(g[pred_col].mean()),
                "Observed_KM_risk": km_risk_at_horizon(g[TIME_COL].to_numpy(), g[EVENT_COL].to_numpy(), horizon),
            }
        )
    return pd.DataFrame(rows)


def censor_survival_function(times: np.ndarray, events: np.ndarray):
    censor_event = 1 - events.astype(int)
    km = KaplanMeierFitter()
    km.fit(times, event_observed=censor_event)

    def g(t: float) -> float:
        val = float(km.predict(max(t, 1e-6)))
        return max(val, 1e-6)

    return g


def dca_table(df: pd.DataFrame, pred_col: str, set_name: str, model_name: str, horizon: int) -> pd.DataFrame:
    sub = df[[TIME_COL, EVENT_COL, pred_col]].dropna().copy()
    times = sub[TIME_COL].to_numpy(dtype=float)
    events = sub[EVENT_COL].to_numpy(dtype=int)
    pred = sub[pred_col].to_numpy(dtype=float)
    n = len(sub)
    g = censor_survival_function(times, events)
    case_w = np.zeros(n)
    control_w = np.zeros(n)
    for i, (time, event) in enumerate(zip(times, events)):
        if event == 1 and time <= horizon:
            case_w[i] = 1.0 / g(time)
        elif time > horizon:
            control_w[i] = 1.0 / g(horizon)
    rows = []
    thresholds = np.round(np.arange(0.05, 0.501, 0.025), 3)
    all_tp = case_w.sum() / n
    all_fp = control_w.sum() / n
    for pt in thresholds:
        odds = pt / (1 - pt)
        high = pred >= pt
        tp = case_w[high].sum() / n
        fp = control_w[high].sum() / n
        rows.append(
            {
                "Set": set_name,
                "Model": model_name,
                "Horizon_months": horizon,
                "Threshold": pt,
                "Net_Benefit": float(tp - fp * odds),
                "Treat_All_Net_Benefit": float(all_tp - all_fp * odds),
                "Treat_None_Net_Benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_abs_risk_tables(df: pd.DataFrame, model_cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    train = work[work["Set"].eq("Train_CV")].copy()
    calibration_rows, dca_rows = [], []
    for model, risk_col in model_cols.items():
        cph = fit_risk_calibrator(train, risk_col)
        for horizon in HORIZONS:
            pred_col = f"{model.replace('-', '_')}_Risk_{horizon}m"
            work[pred_col] = predict_absolute_risk(cph, work, risk_col, horizon)
            for set_name in ["Train_CV", "External_Test"]:
                sub = work[work["Set"].eq(set_name)]
                calibration_rows.append(calibration_table(sub, pred_col, set_name, model, horizon))
                dca_rows.append(dca_table(sub, pred_col, set_name, model, horizon))
    abs_cols = ["MRI_id", "Set", "Cohort", TIME_COL, EVENT_COL] + [
        c for c in work.columns if c.endswith("m") and "_Risk_" in c
    ]
    work[abs_cols].to_csv(OUTDIR / "absolute_risk_predictions_1y2y3y.csv", index=False)
    return pd.concat(calibration_rows, ignore_index=True), pd.concat(dca_rows, ignore_index=True)


def plot_calibration(calib: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for r, set_name in enumerate(["Train_CV", "External_Test"]):
        for c, horizon in enumerate(HORIZONS):
            ax = axes[r, c]
            sub = calib[(calib["Set"].eq(set_name)) & (calib["Horizon_months"].eq(horizon))]
            ax.plot([0, 0.55], [0, 0.55], color="#777777", lw=0.8, ls="--")
            for model, g in sub.groupby("Model"):
                g = g.sort_values("Predicted_risk_mean")
                ax.plot(
                    g["Predicted_risk_mean"],
                    g["Observed_KM_risk"],
                    marker="o",
                    ms=3,
                    lw=1.2,
                    color=PALETTE[model],
                    label=model,
                )
            ax.set_title(f"{set_name}, {horizon // 12} year")
            ax.set_xlim(0, 0.55)
            ax.set_ylim(0, 0.55)
            if c == 0:
                ax.set_ylabel("Observed DM risk")
            if r == 1:
                ax.set_xlabel("Predicted DM risk")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Calibration of absolute distant-metastasis risk", y=1.0, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(OUTDIR / "calibration_curves_train_external_1y2y3y.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTDIR / "calibration_curves_train_external_1y2y3y.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dca(dca: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    set_name = "External_Test"
    for ax, horizon in zip(axes, HORIZONS):
        sub = dca[(dca["Set"].eq(set_name)) & (dca["Horizon_months"].eq(horizon))]
        for model, g in sub.groupby("Model"):
            g = g.sort_values("Threshold")
            ax.plot(g["Threshold"], g["Net_Benefit"], color=PALETTE[model], lw=1.4, label=model)
        base = sub[sub["Model"].eq("CRM")].sort_values("Threshold")
        ax.plot(base["Threshold"], base["Treat_All_Net_Benefit"], color=PALETTE["Treat all"], lw=1, ls="--", label="Treat all")
        ax.axhline(0, color=PALETTE["Treat none"], lw=1, ls=":", label="Treat none")
        ax.set_title(f"External, {horizon // 12} year")
        ax.set_xlabel("Threshold probability")
        ax.set_xlim(0.05, 0.50)
        ax.set_ylim(-0.08, 0.22)
    axes[0].set_ylabel("Net benefit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Decision curve analysis for distant-metastasis risk", y=1.02, fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    fig.savefig(OUTDIR / "decision_curve_external_1y2y3y.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTDIR / "decision_curve_external_1y2y3y.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(ci: pd.DataFrame) -> None:
    order = ["BCLC", "Clinical-only", "IFM", "CRM"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharex=True)
    for ax, set_name in zip(axes, ["Train_CV", "External_Test"]):
        sub = ci[ci["Set"].eq(set_name)].set_index("Model").loc[order].reset_index()
        y = np.arange(len(sub))
        xerr = np.vstack([sub["C_index"] - sub["CI_low"], sub["CI_high"] - sub["C_index"]])
        ax.errorbar(sub["C_index"], y, xerr=xerr, fmt="o", ms=4, lw=1.1, capsize=2, color="#222222")
        for i, row in sub.iterrows():
            ax.text(row["CI_high"] + 0.008, i, f"{row['C_index']:.3f}", va="center", fontsize=6)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Model"])
        ax.set_xlim(0.45, 0.85)
        ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
        ax.set_xlabel("C-index (bootstrap 95% CI)")
        ax.set_title(set_name)
    fig.tight_layout()
    fig.savefig(OUTDIR / "formal_model_comparison_cindex_ci.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTDIR / "formal_model_comparison_cindex_ci.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_external_subgroups(subgroup_ci: pd.DataFrame) -> None:
    sub = subgroup_ci[subgroup_ci["Model"].isin(["IFM", "CRM"]) & subgroup_ci["C_index"].notna()].copy()
    cohorts = [x for x in sub["Subgroup"].unique()]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y_base = np.arange(len(cohorts))
    offsets = {"IFM": -0.12, "CRM": 0.12}
    for model in ["IFM", "CRM"]:
        g = sub[sub["Model"].eq(model)].set_index("Subgroup").loc[cohorts].reset_index()
        y = y_base + offsets[model]
        xerr = np.vstack([g["C_index"] - g["CI_low"], g["CI_high"] - g["C_index"]])
        ax.errorbar(g["C_index"], y, xerr=xerr, fmt="o", ms=4, lw=1, capsize=2, color=PALETTE[model], label=model)
    ax.set_yticks(y_base)
    ax.set_yticklabels(cohorts)
    ax.set_xlim(0.35, 0.95)
    ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("External subgroup C-index (bootstrap 95% CI)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "external_subgroup_ifm_crm_cindex_ci.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTDIR / "external_subgroup_ifm_crm_cindex_ci.pdf", bbox_inches="tight")
    plt.close(fig)


def external_clinical_subgroup_table(df: pd.DataFrame, model_cols: dict[str, str]) -> pd.DataFrame:
    ext = df[df["Set"].eq("External_Test")].copy()
    age_median = float(ext["Age"].median())
    subgroup_defs: list[tuple[str, str, pd.Series]] = [
        ("Data source", "TCGA", ext["Cohort"].eq("TCGA_External_Test")),
        ("Data source", "non-TCGA", ~ext["Cohort"].eq("TCGA_External_Test")),
        ("Age", f"Age <= {age_median:.0f}", ext["Age"].le(age_median)),
        ("Age", f"Age > {age_median:.0f}", ext["Age"].gt(age_median)),
        ("Sex", "Sex = 1", ext["Sex"].eq(1)),
        ("Sex", "Sex = 2", ext["Sex"].eq(2)),
        ("AFP", "AFP <= 400", pd.to_numeric(ext["AFP"], errors="coerce").le(400)),
        ("AFP", "AFP > 400", pd.to_numeric(ext["AFP"], errors="coerce").gt(400)),
        ("Tumor size", "Tumor size <= 5 cm", pd.to_numeric(ext["Tumor_Size"], errors="coerce").le(5)),
        ("Tumor size", "Tumor size > 5 cm", pd.to_numeric(ext["Tumor_Size"], errors="coerce").gt(5)),
        ("Tumor number", "Single tumor", pd.to_numeric(ext["Tumor_Number"], errors="coerce").eq(1)),
        ("Tumor number", "Multiple tumors", pd.to_numeric(ext["Tumor_Number"], errors="coerce").gt(1)),
        ("BCLC", "BCLC 0-1", pd.to_numeric(ext["BCLC"], errors="coerce").le(1)),
        ("BCLC", "BCLC 2-3", pd.to_numeric(ext["BCLC"], errors="coerce").ge(2)),
    ]
    rows = []
    for family, label, mask in subgroup_defs:
        sub = ext[mask.fillna(False)]
        for model, col in model_cols.items():
            row = {
                "Subgroup_Family": family,
                "Subgroup": label,
                "Model": model,
                "N": len(sub),
                "Events": int(sub[EVENT_COL].sum()),
            }
            if row["Events"] < 5 or len(sub) < 20:
                row.update({"C_index": np.nan, "CI_low": np.nan, "CI_high": np.nan, "N_boot_valid": 0})
            else:
                row.update(bootstrap_cindex(sub, col, seed=stable_seed(family, label, model, "clinical_subgroup")))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_external_clinical_subgroups(clinical_subgroup_ci: pd.DataFrame) -> None:
    sub = clinical_subgroup_ci[
        clinical_subgroup_ci["Model"].isin(["IFM", "CRM"]) & clinical_subgroup_ci["C_index"].notna()
    ].copy()
    sub["Label"] = sub["Subgroup_Family"] + ": " + sub["Subgroup"]
    labels = sub["Label"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.28 * len(labels) + 1.2)))
    y_base = np.arange(len(labels))
    offsets = {"IFM": -0.12, "CRM": 0.12}
    for model in ["IFM", "CRM"]:
        g = sub[sub["Model"].eq(model)].set_index("Label").loc[labels].reset_index()
        y = y_base + offsets[model]
        xerr = np.vstack([g["C_index"] - g["CI_low"], g["CI_high"] - g["C_index"]])
        ax.errorbar(g["C_index"], y, xerr=xerr, fmt="o", ms=4, lw=1, capsize=2, color=PALETTE[model], label=model)
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.35, 0.95)
    ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("External clinical-subgroup C-index (bootstrap 95% CI)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "external_clinical_subgroup_ifm_crm_cindex_ci.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTDIR / "external_clinical_subgroup_ifm_crm_cindex_ci.pdf", bbox_inches="tight")
    plt.close(fig)


def model_ci_tables(df: pd.DataFrame, model_cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    for set_name in ["Train_CV", "External_Test"]:
        sub = df[df["Set"].eq(set_name)]
        for model, col in model_cols.items():
            row = {"Set": set_name, "Model": model, "N": len(sub), "Events": int(sub[EVENT_COL].sum())}
            row.update(bootstrap_cindex(sub, col, seed=stable_seed(set_name, model, "ci")))
            ci_rows.append(row)
    ci = pd.DataFrame(ci_rows)

    diff_rows = []
    for set_name in ["Train_CV", "External_Test"]:
        sub = df[df["Set"].eq(set_name)]
        for comparator in ["IFM", "Clinical-only", "BCLC"]:
            row = {"Set": set_name, "Comparison": f"CRM minus {comparator}"}
            row.update(
                bootstrap_cindex_diff(
                    sub,
                    model_cols["CRM"],
                    model_cols[comparator],
                    seed=stable_seed(set_name, comparator, "diff"),
                )
            )
            diff_rows.append(row)
    diffs = pd.DataFrame(diff_rows)

    ext = df[df["Set"].eq("External_Test")].copy()
    subgroup_defs = {
        "External all": ext.index,
        "TCGA": ext[ext["Cohort"].eq("TCGA_External_Test")].index,
        "non-TCGA": ext[~ext["Cohort"].eq("TCGA_External_Test")].index,
        "TJH 2014-2018": ext[ext["Cohort"].eq("2014-2018")].index,
        "TJH 2018-2021": ext[ext["Cohort"].eq("2018-2021")].index,
        "External Zhongshan": ext[ext["Cohort"].eq("External_Zhongshan")].index,
        "SDPH": ext[ext["Cohort"].eq("SDPH")].index,
    }
    subgroup_rows = []
    for name, idx in subgroup_defs.items():
        sub = ext.loc[idx]
        for model, col in model_cols.items():
            row = {"Subgroup": name, "Model": model, "N": len(sub), "Events": int(sub[EVENT_COL].sum())}
            if row["Events"] < 5 or len(sub) < 10:
                row.update({"C_index": np.nan, "CI_low": np.nan, "CI_high": np.nan, "N_boot_valid": 0})
            else:
                row.update(bootstrap_cindex(sub, col, seed=stable_seed(name, model, "subgroup")))
            subgroup_rows.append(row)
    subgroup = pd.DataFrame(subgroup_rows)
    return ci, diffs, subgroup


def write_readme(ci: pd.DataFrame, diffs: pd.DataFrame, subgroup: pd.DataFrame) -> None:
    ext_ci = ci[ci["Set"].eq("External_Test")].set_index("Model")
    crm = ext_ci.loc["CRM"]
    ifm = ext_ci.loc["IFM"]
    lines = [
        "# Multiphase MRI model statistical validation",
        "",
        "## Analysis scope",
        "",
        "- Endpoint: distant metastasis, using `Time_to_DM_Months` and `Distant_Metastasis`.",
        "- Locked models are not reselected here: IFM uses the locked late-score ExtraST risk; CRM uses the locked Coxnet CRM risk.",
        "- Train_CV uses out-of-fold risks where available; External_Test uses full-fit risks.",
        "- Clinical-only baseline: fixed seven-variable CoxPH model with Age, log1p(AFP), Tumor_Size, and one-hot Sex/Child-Pugh/BCLC/Tumor_Number.",
        "- BCLC baseline: single-variable BCLC CoxPH model.",
        "- Bootstrap CIs use patient-level resampling with 1000 iterations.",
        "- Calibration and DCA use Cox calibration from Train_CV risk scores to 1-, 2-, and 3-year absolute distant-metastasis risks.",
        "",
        "## Main external result",
        "",
        f"- CRM external C-index {crm['C_index']:.3f} (95% CI {crm['CI_low']:.3f}-{crm['CI_high']:.3f}).",
        f"- IFM external C-index {ifm['C_index']:.3f} (95% CI {ifm['CI_low']:.3f}-{ifm['CI_high']:.3f}).",
        "",
        "## Output files",
        "",
        "- `analysis_dataset_with_clinical_baselines.csv`",
        "- `formal_model_comparison_cindex_ci.csv`",
        "- `formal_model_comparison_delta_cindex_bootstrap.csv`",
        "- `external_subgroup_cindex_ci.csv`",
        "- `external_clinical_subgroup_cindex_ci.csv`",
        "- `calibration_table_1y2y3y.csv`",
        "- `decision_curve_table_1y2y3y.csv`",
        "- `absolute_risk_predictions_1y2y3y.csv`",
        "- `calibration_curves_train_external_1y2y3y.png/pdf`",
        "- `decision_curve_external_1y2y3y.png/pdf`",
        "- `formal_model_comparison_cindex_ci.png/pdf`",
        "- `external_subgroup_ifm_crm_cindex_ci.png/pdf`",
        "- `external_clinical_subgroup_ifm_crm_cindex_ci.png/pdf`",
    ]
    (OUTDIR / "STATISTICAL_VALIDATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_analysis_table()
    model_cols = {
        "CRM": "Model_CRM",
        "IFM": "Model_IFM",
        "Clinical-only": "Clinical_only_Risk",
        "BCLC": "BCLC_Risk",
    }

    df.to_csv(OUTDIR / "analysis_dataset_with_clinical_baselines.csv", index=False)
    ci, diffs, subgroup = model_ci_tables(df, model_cols)
    clinical_subgroup = external_clinical_subgroup_table(df, model_cols)
    ci.to_csv(OUTDIR / "formal_model_comparison_cindex_ci.csv", index=False)
    diffs.to_csv(OUTDIR / "formal_model_comparison_delta_cindex_bootstrap.csv", index=False)
    subgroup.to_csv(OUTDIR / "external_subgroup_cindex_ci.csv", index=False)
    clinical_subgroup.to_csv(OUTDIR / "external_clinical_subgroup_cindex_ci.csv", index=False)

    calibration, dca = make_abs_risk_tables(df, model_cols)
    calibration.to_csv(OUTDIR / "calibration_table_1y2y3y.csv", index=False)
    dca.to_csv(OUTDIR / "decision_curve_table_1y2y3y.csv", index=False)

    plot_calibration(calibration)
    plot_dca(dca)
    plot_model_comparison(ci)
    plot_external_subgroups(subgroup)
    plot_external_clinical_subgroups(clinical_subgroup)

    manifest = {
        "root": str(ROOT),
        "prediction_file": str(PRED_PATH),
        "clinical_file": str(CLIN_PATH),
        "split_file": str(SPLIT_PATH),
        "endpoint": {"time": TIME_COL, "event": EVENT_COL},
        "horizons_months": HORIZONS,
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
        "model_columns": model_cols,
    }
    (OUTDIR / "stat_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(ci, diffs, subgroup)

    print("Saved statistical validation outputs to", OUTDIR)
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
