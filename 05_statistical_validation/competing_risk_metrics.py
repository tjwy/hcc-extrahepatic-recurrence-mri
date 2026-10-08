"""Cause-specific Cox CIF mapping and IPCW Brier functions used in the study.

Event coding: 0=censoring, 1=extrahepatic recurrence, 2=death before recurrence.
Input data and trained models are not distributed.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

@dataclass
class CauseSpecificModel:
    cause: int
    fitter: CoxPHFitter


def fit_cause_specific_models(
    development: pd.DataFrame, score_column: str
) -> tuple[dict[int, CauseSpecificModel], float, float]:
    score_mean = float(development[score_column].mean())
    score_sd = float(development[score_column].std(ddof=1))
    if score_sd <= 0:
        raise ValueError(f"Zero variance for {score_column}")
    score_z = (
        development[score_column].to_numpy(dtype=float) - score_mean
    ) / score_sd
    models: dict[int, CauseSpecificModel] = {}
    for cause in (1, 2):
        fit_data = pd.DataFrame(
            {
                "time": development["analysis_time_months"].to_numpy(dtype=float),
                "event": development["event_type"].eq(cause).to_numpy(dtype=int),
                "score_z": score_z,
            }
        )
        fitter = CoxPHFitter()
        fitter.fit(fit_data, duration_col="time", event_col="event")
        models[cause] = CauseSpecificModel(cause=cause, fitter=fitter)
    return models, score_mean, score_sd


def predict_cif(
    models: dict[int, CauseSpecificModel],
    score_z: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    score_z = np.asarray(score_z, dtype=float)
    horizons = np.asarray(horizons, dtype=float)
    if np.any(np.diff(horizons) < 0):
        raise ValueError("Horizons must be sorted")

    new_data = pd.DataFrame({"score_z": score_z})
    partial_hazards = {
        cause: models[cause]
        .fitter.predict_partial_hazard(new_data)
        .to_numpy(dtype=float)
        for cause in (1, 2)
    }
    baseline = {}
    for cause in (1, 2):
        table = models[cause].fitter.baseline_hazard_
        baseline[cause] = pd.Series(
            table.iloc[:, 0].to_numpy(dtype=float),
            index=table.index.to_numpy(dtype=float),
        )
    event_times = np.unique(
        np.concatenate(
            [baseline[1].index.to_numpy(), baseline[2].index.to_numpy()]
        )
    )
    increments = {
        cause: baseline[cause].reindex(event_times, fill_value=0.0).to_numpy()
        for cause in (1, 2)
    }

    survival = np.ones(len(score_z), dtype=float)
    cif1 = np.zeros(len(score_z), dtype=float)
    output = np.zeros((len(score_z), len(horizons)), dtype=float)
    horizon_index = 0
    for event_time, base_dh1, base_dh2 in zip(
        event_times, increments[1], increments[2]
    ):
        while horizon_index < len(horizons) and horizons[horizon_index] < event_time:
            output[:, horizon_index] = cif1
            horizon_index += 1
        dh1 = base_dh1 * partial_hazards[1]
        dh2 = base_dh2 * partial_hazards[2]
        dh_total = dh1 + dh2
        event_probability = -np.expm1(-dh_total)
        cause_fraction = np.divide(
            dh1,
            dh_total,
            out=np.zeros_like(dh1),
            where=dh_total > 0,
        )
        cif1 += survival * event_probability * cause_fraction
        survival *= np.exp(-dh_total)
    while horizon_index < len(horizons):
        output[:, horizon_index] = cif1
        horizon_index += 1
    return output


def censoring_survival(
    times: np.ndarray, event_types: np.ndarray
) -> tuple[dict[float, float], dict[float, float]]:
    times = np.asarray(times, dtype=float)
    event_types = np.asarray(event_types, dtype=int)
    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    sorted_types = event_types[order]
    survival = 1.0
    left: dict[float, float] = {}
    right: dict[float, float] = {}
    index = 0
    while index < len(times):
        event_time = sorted_times[index]
        end = index
        while end < len(times) and sorted_times[end] == event_time:
            end += 1
        left[event_time] = survival
        at_risk = len(times) - index
        censoring_events = int(np.sum(sorted_types[index:end] == 0))
        survival *= 1.0 - censoring_events / at_risk
        right[event_time] = survival
        index = end
    return left, right


def censoring_at(
    right: dict[float, float], horizon: float
) -> float:
    value = 1.0
    for time in sorted(right):
        if time <= horizon:
            value = right[time]
        else:
            break
    return value


def competing_risk_brier_curve(
    times: np.ndarray,
    event_types: np.ndarray,
    predictions: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """IPCW Brier score for the cause-1 cumulative incidence function."""
    times = np.asarray(times, dtype=float)
    event_types = np.asarray(event_types, dtype=int)
    predictions = np.asarray(predictions, dtype=float)
    horizons = np.asarray(horizons, dtype=float)
    if predictions.shape != (len(times), len(horizons)):
        raise ValueError("Prediction matrix shape does not match data and horizons")

    left, right = censoring_survival(times, event_types)
    n = len(times)
    result = np.empty(len(horizons), dtype=float)
    for column, horizon in enumerate(horizons):
        known_event = (event_types > 0) & (times <= horizon)
        at_risk_after_horizon = times > horizon
        outcome = ((event_types == 1) & (times <= horizon)).astype(float)
        weights = np.zeros(n, dtype=float)
        for index in np.flatnonzero(known_event):
            weights[index] = 1.0 / max(left[times[index]], 1e-12)
        g_horizon = max(censoring_at(right, float(horizon)), 1e-12)
        weights[at_risk_after_horizon] = 1.0 / g_horizon
        squared_error = (outcome - predictions[:, column]) ** 2
        result[column] = float(np.sum(weights * squared_error) / n)
    return result


def integrated_brier(curve: np.ndarray, horizons: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=float)
    horizons = np.asarray(horizons, dtype=float)
    if len(horizons) < 2:
        raise ValueError("At least two horizons are required for integration")
    return float(np.trapz(curve, horizons) / (horizons[-1] - horizons[0]))
