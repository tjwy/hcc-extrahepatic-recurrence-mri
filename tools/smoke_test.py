"""Run a small synthetic score -> IFM -> CRM -> competing-risk workflow.

This checks file interfaces and execution, not manuscript numerical reproduction.
No clinical data or image/weight downloads are used. Temporary outputs are removed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def smoke(work: Path):
    rng = np.random.default_rng(31)
    n, n_train = 120, 80
    latent = rng.normal(size=n)
    ehr_time = rng.exponential(24 / np.exp(.6 * latent)) + .5
    death_time = rng.exponential(48, n) + .5
    censor = rng.uniform(24, 72, n)
    duration = np.minimum.reduce([ehr_time, death_time, censor])
    status = np.where((ehr_time < death_time) & (ehr_time < censor), 1,
                      np.where(death_time < censor, 2, 0))
    clinical = pd.DataFrame({
        "MRI_id": [f"SYN-{i:03d}" for i in range(n)],
        "Time_to_DM_Months": duration, "Distant_Metastasis": (status == 1).astype(int),
        "Age": rng.uniform(35, 75, n), "Sex": rng.integers(0, 2, n),
        "Child_Pugh_Score": rng.choice([1, 2], n, p=[.8, .2]),
        "AFP": np.exp(3 + latent), "BCLC": rng.integers(0, 3, n),
        "Tumor_Size": np.exp(1 + .3 * latent), "Tumor_Number": rng.integers(1, 4, n),
        "OS_Status": (status > 0).astype(int), "OS_Months": np.where(status == 1, duration + 12, duration),
        "Cohort": "Synthetic",
    })
    clinical.loc[[3, 90], "AFP"] = np.nan
    clinical_path = work / "synthetic_clinical.xlsx"
    clinical.to_excel(clinical_path, index=False)
    splits = pd.DataFrame({"MRI_id": clinical.MRI_id,
                           "Set": ["Train_CV"] * n_train + ["External_Test"] * (n - n_train),
                           "Fold": list(np.arange(n_train) % 5) + [np.nan] * (n - n_train)})
    split_path = work / "synthetic_splits.csv"
    splits.to_csv(split_path, index=False)
    selected_path = work / "synthetic_selected.csv"
    rad_dir = work / "radiomics"
    rad_dir.mkdir()
    radiomics_summary = rad_dir / "radiomics_original107x3_model_summary_synthetic.csv"
    pd.DataFrame({"Model": ["RSF"], "Val_C": [.7], "External_C": [.7]}).to_csv(radiomics_summary, index=False)
    pd.DataFrame({"MRI_id": clinical.MRI_id[:n_train], "Rad_Score": latent[:n_train]}).to_csv(rad_dir / "radiomics_val_predictions.csv", index=False)
    pd.DataFrame({"MRI_id": clinical.MRI_id[n_train:], "Rad_Score": latent[n_train:]}).to_csv(rad_dir / "radiomics_test_predictions.csv", index=False)
    selected = []
    for family, backbone in [("2d", "vgg19_bn"), ("2p5d", "mobilenet_v2"), ("3d", "resnet10_3d")]:
        score = latent + rng.normal(scale=.3, size=n)
        paths = []
        for part, mask in [("oof", np.arange(n) < n_train), ("external", np.arange(n) >= n_train)]:
            path = work / f"{family}_{part}.csv"
            pd.DataFrame({"MRI_id": clinical.MRI_id[mask], "Risk_Score": score[mask]}).to_csv(path, index=False)
            paths.append(path)
        selected.append({"Family": family, "Backbone": backbone, "Mean_FinalEpoch_Val_C": .7,
                         "Mean_BestEpoch_Val_C_Audit_Not_Used": .7,
                         "External_C_Diagnostic_Not_For_Selection": .7, "Pooled_OOF_C": .7,
                         "OOF_CSV": str(paths[0]), "External_CSV": str(paths[1])})
    pd.DataFrame(selected).to_csv(selected_path, index=False)
    os.environ.update({
        "MR_PROJECT_ROOT": str(ROOT), "MR_CLINICAL_EXCEL": str(clinical_path),
        "MR_SPLIT_CSV": str(split_path), "MR_IMAGE_SELECTED_CSV": str(selected_path),
        "MR_RADIOMICS_OUT_ROOT": str(rad_dir), "MR_RADIOMICS_SUMMARY_CSV": str(radiomics_summary),
        "MR_LATE_FUSION_OUT": str(work / "late"), "MR_CRM_OUT": str(work / "crm"),
        "MR_VALIDATION_OUT": str(work / "stats"),
    })
    sys.path.insert(0, str(ROOT / "03_imaging_fusion"))
    late = load("03_imaging_fusion/train_late_score_fusion.py", "smoke_late")
    # One real learner is sufficient to exercise the complete file interface.
    late.LFM_MODEL_NAMES = ["CoxPH"]
    late_dir = late.run(SimpleNamespace(image_selected_csv=selected_path, radiomics_dir=rad_dir,
                                       radiomics_summary_csv=radiomics_summary, radiomics_val_csv=None,
                                       radiomics_test_csv=None, seed=2026, repeats=2, splits=5,
                                       tag="synthetic", allow_external_missing=False))
    crm = load("04_clinical_radiologic_model/train_clinical_radiologic_model.py", "smoke_crm")
    crm.MODEL_PRESETS["synthetic"] = ["Coxnet_l10.5_a0.1"]
    crm.PREPROCESS_PRESETS["synthetic"] = ["numeric_raw"]
    crm_dir = crm.run(SimpleNamespace(input_dir=late_dir, seed=2026, repeats=2, splits=5,
                                    tag="synthetic", model_preset="synthetic", preprocess_preset="synthetic"))
    os.environ["MR_PREDICTIONS_CSV"] = str(crm_dir / "final_predictions.csv")
    stats = load("05_statistical_validation/evaluate_model_performance.py", "smoke_stats")
    stats.OUTDIR.mkdir(parents=True)
    analysis = stats.load_analysis_table()
    analysis["analysis_time_months"] = duration
    analysis["event_type"] = status
    sys.path.insert(0, str(ROOT / "05_statistical_validation"))
    primary = load("05_statistical_validation/evaluate_competing_risk_metrics.py", "smoke_primary")
    primary.run(analysis, work / "primary", bootstrap=3)
    metrics = pd.read_csv(work / "primary" / "competing_risk_metrics.csv")
    if len(analysis) != n or len(metrics) != 40 or not metrics.Estimate.between(0, 1).all():
        raise AssertionError("Synthetic workflow output checks failed")
    return {"status": "passed", "synthetic_patients": n, "metric_rows": len(metrics),
            "private_inputs_used": False, "image_models_trained": False,
            "manuscript_numerical_reproduction": False}


def main():
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="hcc_ehr_synthetic_") as folder:
        result = smoke(Path(folder))
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
