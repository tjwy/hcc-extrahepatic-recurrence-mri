from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "05_statistical_validation"))
from competing_risk_metrics import competing_risk_brier_curve, integrated_brier, predict_cif
from evaluate_competing_risk_metrics import (
    MODEL_COLUMNS, development_cutoff, observed_state_probabilities, run, validate_data, weighted_auc,
)


def synthetic_scores(n=120):
    rng = np.random.default_rng(12)
    score = rng.normal(size=n)
    event_time = rng.exponential(24 / np.exp(0.4 * score)) + 0.5
    death_time = rng.exponential(42, size=n) + 0.5
    censor_time = rng.uniform(12, 72, n)
    data = pd.DataFrame({
        "MRI_id": [f"SYN-{i:03d}" for i in range(n)],
        "Set": ["Train_CV"] * (n * 2 // 3) + ["External_Test"] * (n - n * 2 // 3),
        "analysis_time_months": np.minimum.reduce([event_time, death_time, censor_time]),
        "event_type": np.where((event_time < death_time) & (event_time < censor_time), 1,
                               np.where(death_time < censor_time, 2, 0)),
    })
    for column in MODEL_COLUMNS.values():
        data[column] = score + rng.normal(scale=0.2, size=n)
    return data


def test_brier_competing_deaths_are_observed_controls():
    predictions = np.array([[.8], [.2], [.4], [.1]])
    result = competing_risk_brier_curve([2, 3, 10, 12], [1, 2, 0, 0], predictions, np.array([5.]))
    expected = np.mean((np.array([1, 0, 0, 0]) - predictions[:, 0]) ** 2)
    assert result[0] == pytest.approx(expected)


def test_ibs_is_normalized_trapezoidal_integral_not_horizon_mean():
    assert integrated_brier([.1, .1, .1], [1, 12, 36]) == pytest.approx(.1)
    curve, grid = np.array([.01, .2, .4]), np.array([1, 12, 36])
    assert integrated_brier(curve, grid) == pytest.approx(np.trapz(curve, grid) / 35)
    assert integrated_brier(curve, grid) != pytest.approx(np.mean(curve))


def test_auc_handles_ties_and_competing_deaths():
    assert weighted_auc([2, 3, 10, 12], [1, 2, 0, 0], [4, 3, 2, 1], 5) == 1.
    assert weighted_auc([2, 3, 10, 12], [1, 2, 0, 0], [1, 2, 3, 4], 5) == 0.
    assert weighted_auc([2, 3, 10, 12], [1, 2, 0, 0], [1, 1, 1, 1], 5) == .5


def test_auc_returns_nan_when_no_cases_exist():
    assert np.isnan(weighted_auc([2, 3, 10], [0, 2, 0], [3, 2, 1], 5))


def test_ipcw_weights_are_correct_when_censoring_precedes_competing_death():
    times, status = [1, 2, 4, 10], [1, 0, 2, 0]
    assert weighted_auc(times, status, [2, 99, 2, 1], 5) == pytest.approx(.75)
    result = competing_risk_brier_curve(times, status, np.array([[.6], [.9], [.3], [.2]]), np.array([5.]))
    assert result[0] == pytest.approx((.4 ** 2 + 1.5 * .3 ** 2 + 1.5 * .2 ** 2) / 4)


def test_aj_tied_events_and_conditional_probability():
    state = observed_state_probabilities([2, 2, 3, 10], [1, 2, 1, 0], [0, 2, 3])
    assert state.tolist() == [[0, 1], [.25, .5], [.5, .25]]
    assert (state[2, 0] - state[1, 0]) / state[1, 1] == .5


def test_cutoff_is_development_only():
    train = np.arange(100, dtype=float)
    cutoff = development_cutoff(train)
    assert cutoff == pytest.approx(85.14)
    assert (np.array([10, 86, 1000]) >= cutoff).tolist() == [False, True, True]


def test_cif_is_monotone_without_posthoc_accumulate():
    class Fitter:
        def __init__(self, values):
            self.baseline_hazard_ = pd.DataFrame(values, index=[2., 4.])
        def predict_partial_hazard(self, x):
            return pd.Series(np.ones(len(x)))
    fitted = {1: SimpleNamespace(fitter=Fitter([.1, 0.])),
              2: SimpleNamespace(fitter=Fitter([0., .2]))}
    result = predict_cif(fitted, np.array([0., 1.]), np.array([1., 2., 3., 4., 5.]))
    assert np.all(np.diff(result, axis=1) >= 0)
    assert result[:, 0].tolist() == [0., 0.]
    assert result[0, 1:] == pytest.approx(np.repeat(1 - np.exp(-.1), 4))
    with pytest.raises(ValueError, match="sorted"):
        predict_cif(fitted, np.array([0.]), np.array([3., 1.]))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bad_status", "negative_time"])
def test_input_validation_fails_instead_of_silently_dropping_cases(mutation):
    data = synthetic_scores()
    if mutation == "duplicate":
        data.loc[1, "MRI_id"] = data.loc[0, "MRI_id"]
    elif mutation == "missing":
        data.loc[0, "Model_CRM"] = np.nan
    elif mutation == "bad_status":
        data.loc[0, "event_type"] = 3
    else:
        data.loc[0, "analysis_time_months"] = -1
    with pytest.raises(ValueError):
        validate_data(data)


def test_primary_analysis_runs_on_synthetic_data(tmp_path):
    run(synthetic_scores(), tmp_path, bootstrap=3)
    metrics = pd.read_csv(tmp_path / "competing_risk_metrics.csv")
    assert len(metrics) == 40
    assert metrics["Estimate"].between(0, 1).all()
    brier = pd.read_csv(tmp_path / "monthly_brier_scores.csv")
    assert len(brier) == 288
    gray = pd.read_csv(tmp_path / "competing_risk_input.csv")
    assert len(gray) == 480
    assert set(gray["Risk_Group"]) == {"Low", "High"}
