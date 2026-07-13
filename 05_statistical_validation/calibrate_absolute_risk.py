#!/usr/bin/env python3
"""Supplementary CRM absolute-risk calibration at clinical time horizons.

This script does not refit the locked CRM. It fits a one-variable Cox
calibration model in Train_CV to map the locked CRM risk score to absolute
distant-metastasis risks at 6, 12, 24 and 36 months.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
import warnings


warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("MR_VALIDATION_OUT", ROOT / "outputs" / "statistical_validation"))
ANALYSIS_CSV = Path(os.environ.get("MR_VALIDATION_DATASET", OUTDIR / "analysis_dataset.csv"))
SUPP_DIR = Path(os.environ.get("MR_ABSOLUTE_RISK_OUT", OUTDIR / "absolute_risk"))

TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
RISK_COL = "CRM_Risk"
HORIZONS = [6, 12, 24, 36]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def km_risk_at_horizon(df: pd.DataFrame, horizon: int) -> float:
    sub = df[[TIME_COL, EVENT_COL]].dropna()
    if len(sub) < 3 or sub[EVENT_COL].sum() == 0:
        return np.nan
    km = KaplanMeierFitter()
    km.fit(sub[TIME_COL], event_observed=sub[EVENT_COL])
    return float(1.0 - km.predict(horizon))


def fit_crm_calibrator(df: pd.DataFrame) -> CoxPHFitter:
    fit_df = df[df["Set"].eq("Train_CV")][[TIME_COL, EVENT_COL, RISK_COL]].dropna().copy()
    fit_df = fit_df.rename(columns={RISK_COL: "crm_score"})
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
    return cph


def predict_absolute_risks(cph: CoxPHFitter, df: pd.DataFrame) -> pd.DataFrame:
    out = df[["MRI_id", "Set", "Cohort", TIME_COL, EVENT_COL, RISK_COL]].copy()
    x = pd.DataFrame({"crm_score": df[RISK_COL].to_numpy(dtype=float)})
    surv = cph.predict_survival_function(x, times=HORIZONS).T
    for horizon in HORIZONS:
        out[f"CRM_Abs_DM_Risk_{horizon}m"] = np.clip(1.0 - surv[horizon].to_numpy(dtype=float), 0, 1)
    out["CRM_Abs_DM_Risk_Group_36m"] = np.select(
        [
            out["CRM_Abs_DM_Risk_36m"] < 0.10,
            out["CRM_Abs_DM_Risk_36m"] < 0.20,
        ],
        ["Low (<10%)", "Intermediate (10%-20%)"],
        default="High (>=20%)",
    )
    return out


def calibrator_parameter_table(cph: CoxPHFitter) -> pd.DataFrame:
    beta = float(cph.params_["crm_score"])
    rows = [
        {
            "Parameter": "Cox coefficient for locked CRM risk score",
            "Value": beta,
            "Interpretation": "Higher locked CRM risk score implies higher distant-metastasis hazard.",
        },
        {
            "Parameter": "Hazard ratio per one-unit CRM score increase",
            "Value": float(np.exp(beta)),
            "Interpretation": "Relative hazard multiplier for a one-unit increase in the locked CRM score.",
        },
    ]
    x0 = pd.DataFrame({"crm_score": [0.0]})
    surv0 = cph.predict_survival_function(x0, times=HORIZONS).T.iloc[0]
    for horizon in HORIZONS:
        rows.append(
            {
                "Parameter": f"Baseline-calibrated risk at CRM score 0, {horizon} months",
                "Value": float(1.0 - surv0[horizon]),
                "Interpretation": "Reference absolute distant-metastasis risk used by the calibration layer.",
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for set_name, sub in pred.groupby("Set", sort=False):
        for horizon in HORIZONS:
            col = f"CRM_Abs_DM_Risk_{horizon}m"
            values = sub[col].dropna()
            rows.append(
                {
                    "Set": set_name,
                    "Horizon_months": horizon,
                    "N": int(len(values)),
                    "DM_events_total": int(sub[EVENT_COL].sum()),
                    "Observed_KM_DM_risk": km_risk_at_horizon(sub, horizon),
                    "Predicted_mean": float(values.mean()),
                    "Predicted_median": float(values.median()),
                    "Predicted_q25": float(values.quantile(0.25)),
                    "Predicted_q75": float(values.quantile(0.75)),
                    "Predicted_min": float(values.min()),
                    "Predicted_max": float(values.max()),
                    "N_pred_ge_5pct": int((values >= 0.05).sum()),
                    "Pct_pred_ge_5pct": float((values >= 0.05).mean()),
                    "N_pred_ge_10pct": int((values >= 0.10).sum()),
                    "Pct_pred_ge_10pct": float((values >= 0.10).mean()),
                    "N_pred_ge_20pct": int((values >= 0.20).sum()),
                    "Pct_pred_ge_20pct": float((values >= 0.20).mean()),
                }
            )
    return pd.DataFrame(rows)


def calibration_bins(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for set_name, set_df in pred.groupby("Set", sort=False):
        n_bins = 5 if len(set_df) >= 120 else 4
        for horizon in HORIZONS:
            col = f"CRM_Abs_DM_Risk_{horizon}m"
            sub = set_df[[TIME_COL, EVENT_COL, col]].dropna().copy()
            try:
                sub["bin"] = pd.qcut(sub[col], q=n_bins, duplicates="drop")
            except ValueError:
                sub["bin"] = pd.cut(sub[col], bins=n_bins)
            for idx, (_, g) in enumerate(sub.groupby("bin", observed=False), 1):
                rows.append(
                    {
                        "Set": set_name,
                        "Horizon_months": horizon,
                        "Bin": idx,
                        "N": int(len(g)),
                        "Predicted_risk_mean": float(g[col].mean()),
                        "Predicted_risk_median": float(g[col].median()),
                        "Observed_KM_DM_risk": km_risk_at_horizon(g, horizon),
                    }
                )
    return pd.DataFrame(rows)


def plot_calibration(calib: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 4.9), sharex=True, sharey=True)
    colors = {"Train_CV": "#2A6FBB", "External_Test": "#C5793D"}
    titles = {6: "6 months", 12: "1 year", 24: "2 years", 36: "3 years"}
    for ax, horizon in zip(axes.ravel(), HORIZONS):
        ax.plot([0, 0.55], [0, 0.55], color="#777777", lw=0.8, ls="--", zorder=1)
        for set_name in ["Train_CV", "External_Test"]:
            g = calib[(calib["Set"].eq(set_name)) & (calib["Horizon_months"].eq(horizon))].sort_values(
                "Predicted_risk_mean"
            )
            ax.plot(
                g["Predicted_risk_mean"],
                g["Observed_KM_DM_risk"],
                marker="o",
                ms=3.2,
                lw=1.2,
                color=colors[set_name],
                label=set_name.replace("_", " "),
            )
            for _, row in g.iterrows():
                ax.text(
                    row["Predicted_risk_mean"],
                    row["Observed_KM_DM_risk"],
                    str(int(row["N"])),
                    fontsize=5.5,
                    color=colors[set_name],
                    ha="center",
                    va="bottom",
                )
        ax.set_title(titles[horizon])
        ax.set_xlim(0, 0.55)
        ax.set_ylim(0, 0.55)
        ax.grid(axis="both", color="#e6e6e6", lw=0.5)
    axes[0, 0].set_ylabel("Observed KM DM risk")
    axes[1, 0].set_ylabel("Observed KM DM risk")
    axes[1, 0].set_xlabel("Predicted absolute DM risk")
    axes[1, 1].set_xlabel("Predicted absolute DM risk")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("CRM absolute-risk calibration at clinically relevant horizons", x=0.01, ha="left", fontweight="bold")
    fig.text(
        0.01,
        0.025,
        "Numbers denote cases per risk bin. Absolute risks were derived from Train_CV Cox calibration of the locked CRM score.",
        fontsize=6.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.18, hspace=0.32)
    fig.savefig(SUPP_DIR / "supplement_crm_absolute_risk_calibration_6m1y2y3y.png", dpi=600)
    fig.savefig(SUPP_DIR / "supplement_crm_absolute_risk_calibration_6m1y2y3y.pdf")
    plt.close(fig)


def plot_distribution(pred: pd.DataFrame) -> None:
    plot = pred[pred["Set"].isin(["Train_CV", "External_Test"])].copy()
    labels = {6: "6m", 12: "1y", 24: "2y", 36: "3y"}
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35), sharey=True)
    rng = np.random.default_rng(20260626)
    for ax, horizon in zip(axes, HORIZONS):
        col = f"CRM_Abs_DM_Risk_{horizon}m"
        for xpos, set_name, color in [(0, "Train_CV", "#2A6FBB"), (1, "External_Test", "#C5793D")]:
            vals = plot.loc[plot["Set"].eq(set_name), col].dropna().to_numpy()
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(np.full(len(vals), xpos) + jitter, vals * 100, s=6, alpha=0.18, color=color, linewidths=0)
            q25, med, q75 = np.percentile(vals * 100, [25, 50, 75])
            ax.plot([xpos - 0.18, xpos + 0.18], [med, med], color="#111111", lw=1.0)
            ax.plot([xpos, xpos], [q25, q75], color="#111111", lw=1.0)
        if horizon == 36:
            ax.axhline(20, color="#B13A3A", lw=0.9, ls="--")
            ax.text(1.05, 20.8, "20%", color="#B13A3A", fontsize=6.2)
        ax.set_title(labels[horizon])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Train", "External"], rotation=25, ha="right")
        ax.grid(axis="y", color="#e8e8e8", lw=0.5)
    axes[0].set_ylabel("Predicted absolute DM risk (%)")
    fig.suptitle("Distribution of CRM-predicted absolute distant-metastasis risk", x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.78, wspace=0.18)
    fig.savefig(SUPP_DIR / "supplement_crm_absolute_risk_distribution_6m1y2y3y.png", dpi=600)
    fig.savefig(SUPP_DIR / "supplement_crm_absolute_risk_distribution_6m1y2y3y.pdf")
    plt.close(fig)


def write_readme(param: pd.DataFrame, summary: pd.DataFrame) -> None:
    crm_beta = param.loc[param["Parameter"].str.startswith("Cox coefficient"), "Value"].iloc[0]
    train_6 = summary[(summary["Set"].eq("Train_CV")) & (summary["Horizon_months"].eq(6))].iloc[0]
    ext_6 = summary[(summary["Set"].eq("External_Test")) & (summary["Horizon_months"].eq(6))].iloc[0]
    text = f"""# CRM absolute distant-metastasis risk supplement (2026-06-26)

## Purpose

This supplement makes the locked CRM risk score clinically interpretable by
mapping it to predicted absolute distant-metastasis risks at 6, 12, 24 and 36
months. It is a calibration layer only; it does not refit or reselect the CRM.

## Method

The calibration model was fitted in Train_CV using one covariate only: the
locked CRM risk score.

Formula:

`S(t | score) = S0(t) ^ exp(beta * score)`

`Predicted absolute DM risk at t = 1 - S(t | score)`

Here `S0(t)` and `beta` are estimated only from Train_CV DMFS outcomes. The
same fitted calibration model is then applied unchanged to External_Test.

## Key parameter

- Cox coefficient beta for CRM score: {crm_beta:.6f}

## Six-month predicted risk summary

- Train_CV mean predicted 6-month DM risk: {100 * train_6['Predicted_mean']:.2f}%
- External_Test mean predicted 6-month DM risk: {100 * ext_6['Predicted_mean']:.2f}%

## Outputs

- `crm_absolute_risk_predictions_6m1y2y3y.csv`
- `crm_absolute_risk_distribution_summary_6m1y2y3y.csv`
- `crm_absolute_risk_calibration_bins_6m1y2y3y.csv`
- `crm_absolute_risk_calibrator_parameters.csv`
- `supplement_crm_absolute_risk_calibration_6m1y2y3y.png/pdf`
- `supplement_crm_absolute_risk_distribution_6m1y2y3y.png/pdf`

## Interpretation boundary

Use these predicted probabilities as clinically interpretable risk estimates.
Do not describe the 20% threshold as an optimized cutoff. The main performance
evidence should remain continuous C-index, calibration, decision curve analysis
and bootstrap comparisons.
"""
    (SUPP_DIR / "CRM_ABSOLUTE_RISK_HORIZON_SUPPLEMENT_ZH.md").write_text(text, encoding="utf-8")


def main() -> None:
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ANALYSIS_CSV)
    cph = fit_crm_calibrator(df)
    pred = predict_absolute_risks(cph, df)
    param = calibrator_parameter_table(cph)
    summary = distribution_summary(pred)
    calib = calibration_bins(pred)

    pred.to_csv(SUPP_DIR / "crm_absolute_risk_predictions_6m1y2y3y.csv", index=False)
    summary.to_csv(SUPP_DIR / "crm_absolute_risk_distribution_summary_6m1y2y3y.csv", index=False)
    calib.to_csv(SUPP_DIR / "crm_absolute_risk_calibration_bins_6m1y2y3y.csv", index=False)
    param.to_csv(SUPP_DIR / "crm_absolute_risk_calibrator_parameters.csv", index=False)
    plot_calibration(calib)
    plot_distribution(pred)
    write_readme(param, summary)
    manifest = {
        "analysis_dataset": str(ANALYSIS_CSV),
        "output_dir": str(SUPP_DIR),
        "calibration_policy": "Train_CV one-variable Cox calibration using locked CRM risk score only",
        "horizons_months": HORIZONS,
        "risk_formula": "absolute risk at t = 1 - S0(t) ** exp(beta * CRM score)",
    }
    (SUPP_DIR / "crm_absolute_risk_horizon_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
