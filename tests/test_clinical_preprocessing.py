import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_clinical_only_imputation_and_scaling_are_training_only():
    stats = module("05_statistical_validation/evaluate_model_performance.py", "stats_test")
    train = pd.DataFrame({"Age": [40., 50., 60.], "AFP": [0., 10., 20.], "Tumor_Size": [2., 4., 6.],
                          "Sex": [0, 1, 1], "Child_Pugh_Score": [1, 1, 2], "BCLC": [0, 1, 1],
                          "Tumor_Number": [1, 1, 2]})
    prep = stats.ClinicalPreprocessor().fit(train)
    pred = train.iloc[:1].copy()
    pred[["Age", "AFP", "Tumor_Size", "Child_Pugh_Score"]] = np.nan
    out = prep.transform(pred)
    assert out["Age"].iloc[0] == pytest.approx(0.)
    assert out["Tumor_Size"].iloc[0] == pytest.approx(0.)
    assert prep.cont_median["AFP_log1p"] == pytest.approx(np.log1p(10))
    extreme = train.copy()
    extreme["AFP"] = 1e9
    prep.transform(extreme)
    assert prep.cont_median["AFP_log1p"] == pytest.approx(np.log1p(10))
    grade_b = train.iloc[[2]].copy()
    assert prep.transform(grade_b)["Child_Pugh_Score_2"].iloc[0] == 1.


def test_crm_numeric_coding_preserves_training_median_imputation():
    crm = module("04_clinical_radiologic_model/train_clinical_radiologic_model.py", "crm_preproc_test")
    train = pd.DataFrame({"Final_Imaging_Score": [1., 2., 3.], "Age": [40., 50., 60.],
                          "AFP": [0., 10., 20.], "Tumor_Size": [2., 4., 6.], "Sex": [0, 1, 1],
                          "Child_Pugh_Score": [1, 1, 2], "BCLC": [0, 1, 1], "Tumor_Number": [1, 1, 2]})
    prep = crm.CRMPreprocessor(crm.PREPROCESS_SPECS[0]).fit(train)
    external = train.iloc[:1].copy()
    external["AFP"] = np.nan
    assert prep.transform(external)["AFP"].iloc[0] == 10.


def test_late_fusion_keeps_clinical_missingness_for_downstream_imputation(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "03_imaging_fusion"))
    late = module("03_imaging_fusion/train_late_score_fusion.py", "late_preproc_test")
    clinical = pd.DataFrame({"MRI_id": ["SYN-A", "SYN-B"], "Time_to_DM_Months": [10., 20.],
                             "Distant_Metastasis": [1, 0], "Age": [50, 60], "Sex": [1, 0],
                             "Child_Pugh_Score": [1, 1], "AFP": [10., np.nan], "BCLC": [1, 1],
                             "Tumor_Size": [3., 4.], "Tumor_Number": [1, 1]})
    path = tmp_path / "synthetic_clinical.xlsx"
    clinical.to_excel(path, index=False)
    split = tmp_path / "synthetic_split.csv"
    pd.DataFrame({"MRI_id": clinical.MRI_id, "Set": ["Train_CV", "External_Test"]}).to_csv(split, index=False)
    monkeypatch.setattr(late, "CLINICAL_EXCEL", path)
    monkeypatch.setattr(late, "SPLIT_CSV", split)
    result = late.load_base_dataframe()
    assert np.isnan(result.loc[result.MRI_id.eq("SYN-B"), "AFP"].iloc[0])
