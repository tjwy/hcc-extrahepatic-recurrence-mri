"""Evaluate frozen scores using explicit adjudicated competing-event records.

Legacy DM column names are retained for compatibility; the endpoint is EHR.
This script does not train or select the image, radiomics, IFM, or CRM models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from competing_risk_metrics import (
    censoring_at, censoring_survival, competing_risk_brier_curve,
    fit_cause_specific_models, integrated_brier, predict_cif,
)

MODEL_COLUMNS = {
    "CRM": "Model_CRM", "IFM": "Model_IFM",
    "Clinical-only": "Clinical_only_Risk", "BCLC": "BCLC_Risk",
}
GRID = np.arange(1., 37.)
HORIZONS = (12., 24., 36., 60.)


def validate_data(data: pd.DataFrame) -> None:
    required = ["MRI_id", "Set", "analysis_time_months", "event_type", *MODEL_COLUMNS.values()]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data["MRI_id"].isna().any() or data["MRI_id"].duplicated().any():
        raise ValueError("Patient identifiers must be present and unique")
    if not data["Set"].isin(["Train_CV", "External_Test"]).all():
        raise ValueError("Set must be Train_CV or External_Test")
    if set(data["Set"]) != {"Train_CV", "External_Test"}:
        raise ValueError("Both development and external partitions are required")
    numeric = data[["analysis_time_months", "event_type", *MODEL_COLUMNS.values()]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Event records and model scores must be finite")
    if (data["analysis_time_months"] <= 0).any() or not data["event_type"].isin([0, 1, 2]).all():
        raise ValueError("Times must be positive and event_type must be 0, 1, or 2")


def development_cutoff(scores, quantile=0.86):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or len(scores) == 0:
        raise ValueError("Development scores must be finite and nonempty")
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between zero and one")
    return float(np.quantile(scores, quantile, method="linear"))


def weighted_auc(times, event_types, scores, horizon):
    """Cumulative/dynamic AUC; competing deaths before the horizon are controls."""
    times = np.asarray(times, dtype=float)
    status = np.asarray(event_types, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cases = (status == 1) & (times <= horizon)
    controls = (times > horizon) | ((status == 2) & (times <= horizon))
    if not cases.any() or not controls.any():
        return np.nan
    left, right = censoring_survival(times, status)
    event_weights = np.array([1. / max(left[t], 1e-12) for t in times[cases]])
    control_weights = np.array([
        1. / max(left[t], 1e-12) if t <= horizon else
        1. / max(censoring_at(right, horizon), 1e-12)
        for t in times[controls]
    ])
    difference = scores[cases, None] - scores[None, controls]
    comparisons = (difference > 0).astype(float) + 0.5 * (difference == 0)
    weights = event_weights[:, None] * control_weights[None, :]
    return float(np.sum(comparisons * weights) / np.sum(weights))


def observed_state_probabilities(times, status, horizons):
    """Aalen-Johansen cause-1 and event-free probabilities, including tied times."""
    times = np.asarray(times, dtype=float)
    status = np.asarray(status, dtype=int)
    horizons = np.asarray(horizons, dtype=float)
    survival, cif = 1., 0.
    records = [(0., cif, survival)]
    for t in np.unique(times):
        at_risk = int(np.sum(times >= t))
        ehr = int(np.sum((times == t) & (status == 1)))
        deaths = int(np.sum((times == t) & (status == 2)))
        cif += survival * ehr / at_risk
        survival *= 1. - (ehr + deaths) / at_risk
        records.append((t, cif, survival))
    records = np.asarray(records)
    index = np.searchsorted(records[:, 0], horizons, side="right") - 1
    return records[index, 1:]


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return tuple(np.percentile(values, [2.5, 97.5])) if len(values) else (np.nan, np.nan)


def run(data: pd.DataFrame, output: Path, bootstrap=2000, seed=20260827):
    validate_data(data)
    if bootstrap < 0:
        raise ValueError("Bootstrap count must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    development = data[data["Set"].eq("Train_CV")]
    summary, brier_rows, group_rows, gray_inputs, cutoffs = [], [], [], [], {}
    for model, column in MODEL_COLUMNS.items():
        fitted, mean, sd = fit_cause_specific_models(development, column)
        cutoffs[model] = development_cutoff(development[column])
        for set_name, cohort in data.groupby("Set", sort=False):
            times = cohort["analysis_time_months"].to_numpy(dtype=float)
            status = cohort["event_type"].to_numpy(dtype=int)
            scores = cohort[column].to_numpy(dtype=float)
            predicted = predict_cif(fitted, (scores - mean) / sd, GRID)
            curve = competing_risk_brier_curve(times, status, predicted, GRID)
            aucs = [weighted_auc(times, status, scores, h) for h in HORIZONS]
            boot_ibs, boot_auc = [], []
            # Identical resampling indices across models give paired comparisons.
            rng = np.random.default_rng(seed + (1 if set_name == "External_Test" else 0))
            for _ in range(bootstrap):
                idx = rng.integers(0, len(cohort), len(cohort))
                sample_curve = competing_risk_brier_curve(times[idx], status[idx], predicted[idx], GRID)
                boot_ibs.append(integrated_brier(sample_curve, GRID))
                boot_auc.append([weighted_auc(times[idx], status[idx], scores[idx], h) for h in HORIZONS])
            low, high = confidence_interval(boot_ibs)
            summary.append({"Set": set_name, "Model": model, "Metric": "IBS_1_36_months",
                            "Estimate": integrated_brier(curve, GRID), "CI_low": low, "CI_high": high})
            for j, horizon in enumerate(HORIZONS):
                low, high = confidence_interval([row[j] for row in boot_auc])
                summary.append({"Set": set_name, "Model": model, "Metric": f"AUC_{horizon:g}m",
                                "Estimate": aucs[j], "CI_low": low, "CI_high": high})
            brier_rows.extend({"Set": set_name, "Model": model, "Month": h, "Brier": b}
                              for h, b in zip(GRID, curve))
            groups = np.where(scores >= cutoffs[model], "High", "Low")
            gray = cohort[["MRI_id", "analysis_time_months", "event_type"]].copy()
            gray["Subset"] = model + "_" + set_name
            gray["Risk_Group"] = groups
            gray_inputs.append(gray)
            for group in ("Low", "High"):
                mask = groups == group
                if not mask.any():
                    continue
                state = observed_state_probabilities(times[mask], status[mask], [0, 12, 24, 36])
                for j, horizon in enumerate((12, 24, 36), 1):
                    conditional = ((state[j, 0] - state[j - 1, 0]) / state[j - 1, 1]
                                   if state[j - 1, 1] > 0 else np.nan)
                    group_rows.append({"Set": set_name, "Model": model, "Risk_Group": group,
                                       "N": int(mask.sum()), "Month": horizon, "Observed_CIF": state[j, 0],
                                       "Conditional_EHR_probability": conditional})
    pd.DataFrame(summary).to_csv(output / "competing_risk_metrics.csv", index=False)
    pd.DataFrame(brier_rows).to_csv(output / "monthly_brier_scores.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "risk_group_incidence.csv", index=False)
    pd.concat(gray_inputs).to_csv(output / "competing_risk_input.csv", index=False)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "endpoint": "extrahepatic recurrence", "event_coding": {"0": "censoring", "1": "EHR", "2": "death before EHR"},
        "risk_cutoff_policy": "development OOF 86th percentile, applied unchanged externally",
        "cutoffs": cutoffs, "bootstrap": bootstrap, "seed": seed,
        "probability_layer": "cause-specific Cox mappings fitted in development and applied externally",
        "development_probability_estimates_fully_cross_fitted": False,
        "bootstrap_conditional_on_fitted_probability_layer": True,
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Analysis dataset with frozen scores")
    parser.add_argument("--events-csv", type=Path, help="Adjudicated event times and 0/1/2 event codes")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype={"MRI_id": str})
    if args.events_csv:
        events = pd.read_csv(args.events_csv, dtype={"MRI_id": str})
        if set(data["MRI_id"]) != set(events["MRI_id"]):
            raise ValueError("Event records must match the analysis patients exactly")
        data = data.drop(columns=["analysis_time_months", "event_type"], errors="ignore").merge(
            events[["MRI_id", "analysis_time_months", "event_type"]], on="MRI_id", validate="one_to_one")
    run(data, args.output, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
