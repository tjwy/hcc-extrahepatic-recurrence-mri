from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "03_imaging_fusion" / "compare_early_fusion.py"
spec = importlib.util.spec_from_file_location("compare_early_fusion", SCRIPT)
assert spec is not None and spec.loader is not None
comparator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparator
spec.loader.exec_module(comparator)


def test_load_locked_branches_requires_exact_corrected_models(tmp_path: Path) -> None:
    rows = []
    for family, backbone in comparator.LOCKED_MODELS.items():
        checkpoint_dir = tmp_path / f"{family}_{backbone}"
        checkpoint_dir.mkdir()
        for fold in range(5):
            (checkpoint_dir / f"{family}_{backbone}_fold{fold}_final_epoch.pth").touch()
        rows.append(
            {
                "Family": family,
                "Backbone": backbone,
                "Checkpoint_Dir": str(checkpoint_dir),
                "Config": '{"dropout": 0.0, "batch_size": 2}',
            }
        )
    selected = tmp_path / "selected.csv"
    pd.DataFrame(rows).to_csv(selected, index=False)

    locked = comparator.load_locked_branches(selected)

    assert [(x.family, x.backbone) for x in locked] == list(comparator.LOCKED_MODELS.items())
    assert all(len(x.checkpoint_paths) == 5 for x in locked)


def test_validate_oof_coverage_rejects_duplicate_or_missing_cases() -> None:
    expected = pd.DataFrame({"MRI_id": ["a", "b", "c"], "Fold": [0, 1, 2]})
    observed = pd.DataFrame({"MRI_id": ["a", "a", "c"], "Source_Fold": [0, 0, 2]})

    with pytest.raises(ValueError, match="OOF coverage mismatch"):
        comparator.validate_oof_coverage(expected, observed)


def test_validate_oof_coverage_rejects_wrong_checkpoint_fold() -> None:
    expected = pd.DataFrame({"MRI_id": ["a", "b"], "Fold": [0, 1]})
    observed = pd.DataFrame({"MRI_id": ["a", "b"], "Source_Fold": [1, 1]})

    with pytest.raises(ValueError, match="checkpoint fold mismatch"):
        comparator.validate_oof_coverage(expected, observed)


def test_select_early_candidate_uses_pooled_oof_then_fixed_model_order() -> None:
    grid = pd.DataFrame(
        {
            "Model": ["RSF", "CoxPH", "ExtraST"],
            "Pooled_OOF_C": [0.70, 0.70, 0.69],
        }
    )

    selected = comparator.select_early_candidate(grid)

    assert selected["Model"] == "CoxPH"


def test_fold_preprocessor_never_exceeds_available_rank() -> None:
    import numpy as np

    train = np.arange(24, dtype=float).reshape(6, 4)
    pred = np.arange(8, dtype=float).reshape(2, 4)

    train_t, pred_t, n_components = comparator.fit_fold_preprocessor(train, pred, requested_components=50)

    assert train_t.shape == (6, 4)
    assert pred_t.shape == (2, 4)
    assert n_components == 4


def test_validate_fold_feature_provenance_requires_same_outer_fold_checkpoint() -> None:
    expected_ids = ["a", "b", "c"]
    features = pd.DataFrame(
        {
            "MRI_id": expected_ids,
            "Source_Fold": [2, 2, 1],
            "DeepFeat_0000": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="source checkpoint mismatch"):
        comparator.validate_fold_feature_provenance(features, expected_ids, outer_fold=2)


def test_merge_fold_features_rejects_branch_case_mismatch() -> None:
    left = pd.DataFrame({"MRI_id": ["a", "b"], "A_0000": [1.0, 2.0]})
    right = pd.DataFrame({"MRI_id": ["a", "c"], "B_0000": [3.0, 4.0]})

    with pytest.raises(ValueError, match="branch MRI_id mismatch"):
        comparator.merge_fold_features([left, right])


def test_c_index_safe_uses_higher_score_as_higher_risk() -> None:
    value = comparator.c_index_safe(
        times=[1.0, 2.0, 3.0, 4.0],
        risk=[4.0, 3.0, 2.0, 1.0],
        events=[1, 1, 1, 1],
    )

    assert value == pytest.approx(1.0)


@pytest.mark.parametrize("name", comparator.MODEL_ORDER)
def test_make_survival_model_supports_locked_comparator_space(name: str) -> None:
    model = comparator.make_survival_model(name)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_paired_bootstrap_delta_is_deterministic_and_uses_early_minus_late() -> None:
    result_1 = comparator.paired_bootstrap_delta(
        times=[1, 2, 3, 4, 5, 6],
        events=[1, 1, 1, 1, 1, 1],
        early_risk=[6, 5, 4, 3, 2, 1],
        late_risk=[1, 2, 3, 4, 5, 6],
        n_bootstrap=100,
        seed=7,
    )
    result_2 = comparator.paired_bootstrap_delta(
        times=[1, 2, 3, 4, 5, 6],
        events=[1, 1, 1, 1, 1, 1],
        early_risk=[6, 5, 4, 3, 2, 1],
        late_risk=[1, 2, 3, 4, 5, 6],
        n_bootstrap=100,
        seed=7,
    )

    assert result_1 == result_2
    assert result_1["Delta_C_Early_Minus_Late"] > 0
