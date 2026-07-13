#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("MR_VALIDATION_OUT", ROOT / "outputs" / "statistical_validation"))
ABS_RISK_PATH = Path(os.environ.get("MR_ABSOLUTE_RISK_CSV", OUTDIR / "absolute_risk_predictions.csv"))
TIME_COL = "Time_to_DM_Months"
EVENT_COL = "Distant_Metastasis"
HORIZONS = [12, 24, 36]
MODELS = ["CRM", "IFM", "Clinical_only", "BCLC"]


def censor_survival_function(times: np.ndarray, events: np.ndarray):
    km = KaplanMeierFitter()
    km.fit(times, event_observed=1 - events.astype(int))

    def g(t: float) -> float:
        return max(float(km.predict(max(t, 1e-6))), 1e-6)

    return g


def ipcw_brier_score(df: pd.DataFrame, pred_col: str, horizon: int) -> tuple[float, int, int, int]:
    sub = df[[TIME_COL, EVENT_COL, pred_col]].dropna().copy()
    times = sub[TIME_COL].to_numpy(dtype=float)
    events = sub[EVENT_COL].to_numpy(dtype=int)
    pred = sub[pred_col].to_numpy(dtype=float)
    g = censor_survival_function(times, events)

    weights = np.zeros(len(sub), dtype=float)
    y = np.zeros(len(sub), dtype=float)
    event_by_horizon = 0
    control_after_horizon = 0
    censored_before_horizon = 0

    for i, (time, event) in enumerate(zip(times, events)):
        if event == 1 and time <= horizon:
            y[i] = 1.0
            weights[i] = 1.0 / g(time)
            event_by_horizon += 1
        elif time > horizon:
            y[i] = 0.0
            weights[i] = 1.0 / g(horizon)
            control_after_horizon += 1
        else:
            censored_before_horizon += 1

    score = float(np.mean(weights * (y - pred) ** 2))
    return score, event_by_horizon, control_after_horizon, censored_before_horizon


def main() -> None:
    df = pd.read_csv(ABS_RISK_PATH)
    rows = []
    for set_name, sub in df.groupby("Set"):
        for model in MODELS:
            model_scores = []
            for horizon in HORIZONS:
                pred_col = f"{model}_Risk_{horizon}m"
                score, n_event, n_control, n_censored = ipcw_brier_score(sub, pred_col, horizon)
                model_scores.append(score)
                rows.append(
                    {
                        "Set": set_name,
                        "Model": model.replace("_", "-"),
                        "Horizon_months": horizon,
                        "IPCW_Brier": score,
                        "N": len(sub),
                        "Events_by_horizon": n_event,
                        "Controls_followed_beyond_horizon": n_control,
                        "Censored_before_horizon": n_censored,
                    }
                )
            rows.append(
                {
                    "Set": set_name,
                    "Model": model.replace("_", "-"),
                    "Horizon_months": "Mean_12_24_36",
                    "IPCW_Brier": float(np.mean(model_scores)),
                    "N": len(sub),
                    "Events_by_horizon": "",
                    "Controls_followed_beyond_horizon": "",
                    "Censored_before_horizon": "",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "ipcw_brier_scores_1y2y3y.csv", index=False)
    out.to_markdown(OUTDIR / "ipcw_brier_scores_1y2y3y.md", index=False)

    readme = OUTDIR / "IPCW_BRIER_SCORE_README_ZH.md"
    readme.write_text(
        "\n".join(
            [
                "# IPCW Brier scores for absolute distant-metastasis risk",
                "",
                "## Input",
                "",
                "- `absolute_risk_predictions_1y2y3y.csv`",
                "",
                "## Method",
                "",
                "For each 1-, 2- and 3-year horizon, this script computes an inverse-probability-of-censoring-weighted Brier score.",
                "Patients censored before the evaluated horizon contribute zero weight at that horizon.",
                "The `Mean_12_24_36` row is a simple average across the three reported horizons and should not be described as a densely integrated Brier score.",
                "",
                "## Output",
                "",
                "- `ipcw_brier_scores_1y2y3y.csv`",
                "- `ipcw_brier_scores_1y2y3y.md`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
