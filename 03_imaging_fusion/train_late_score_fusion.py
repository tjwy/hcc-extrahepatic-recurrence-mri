#!/usr/bin/env python3
"""Fit the four-input late-score imaging fusion model.

The selected image table and radiomics predictions are explicit inputs. This
script does not use external outcomes to reselect single-modality branches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
import fusion_models as fusion


SPLIT_CSV = Path(os.environ.get("MR_SPLIT_CSV", PROJECT_ROOT / "data" / "splits.csv"))
EXPECTED_SPLIT_MD5 = os.environ.get("MR_EXPECTED_SPLIT_MD5", "")
CLINICAL_EXCEL = Path(os.environ.get("MR_CLINICAL_EXCEL", PROJECT_ROOT / "data" / "clinical.xlsx"))
EXPECTED_CLINICAL_MD5 = os.environ.get("MR_EXPECTED_CLINICAL_MD5", "")

IMAGE_SELECTED_CSV = Path(os.environ.get("MR_IMAGE_SELECTED_CSV", PROJECT_ROOT / "outputs" / "single_modality_models" / "selected" / "selected_models.csv"))
RAD_DIR = Path(os.environ.get("MR_RADIOMICS_OUT_ROOT", PROJECT_ROOT / "outputs" / "single_modality_models" / "radiomics"))
RAD_SUMMARY_CSV = Path(os.environ.get("MR_RADIOMICS_SUMMARY_CSV", RAD_DIR / "radiomics_model_summary.csv"))
RAD_VAL_CSV = RAD_DIR / "radiomics_val_predictions.csv"
RAD_EXT_CSV = RAD_DIR / "radiomics_test_predictions.csv"

OUTPUT_ROOT = Path(os.environ.get("MR_LATE_FUSION_OUT", PROJECT_ROOT / "outputs" / "imaging_fusion" / "late"))

ID_COL = fusion.ID_COL
TIME_COL = fusion.TIME_COL
EVENT_COL = fusion.EVENT_COL
EXTERNAL_SET = fusion.EXTERNAL_SET
FOUR_INPUT_FEATURES = fusion.FOUR_INPUT_FEATURES
CLINICAL_FIXED_7 = fusion.CLINICAL_FIXED_7
LFM_MODEL_NAMES = fusion.LFM_MODEL_NAMES
ALLOWED_FUSION_MODELS = {"CoxPH", "RSF", "SVM", "GB", "ExtraST"}

bad_lfm_models = sorted(set(LFM_MODEL_NAMES) - ALLOWED_FUSION_MODELS)
if bad_lfm_models:
    raise RuntimeError(
        f"Disallowed imaging-fusion models detected: {bad_lfm_models}. "
        "Only CoxPH, RSF, SVM, GB, and ExtraST are allowed."
    )


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ID_COL] = out[ID_COL].astype(str).str.strip()
    return out


def latest_required(path_dir: Path, pattern: str) -> Path:
    paths = sorted(path_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No files match {path_dir / pattern}")
    return paths[0]


def load_base_dataframe() -> pd.DataFrame:
    clinical = normalize_id(pd.read_excel(CLINICAL_EXCEL))
    clinical = clinical.dropna(subset=[TIME_COL, EVENT_COL])
    split = normalize_id(pd.read_csv(SPLIT_CSV))
    base = clinical.merge(split[[ID_COL, "Set"]], on=ID_COL, how="inner")
    for col in CLINICAL_FIXED_7:
        base[col] = pd.to_numeric(base[col], errors="coerce")
        # Preserve missingness for downstream training-fold-only imputation.
    return base


def load_locked_branches() -> pd.DataFrame:
    selected = pd.read_csv(IMAGE_SELECTED_CSV)
    risk_names = {
        "2d": "Risk_2D",
        "2p5d": "Risk_2p5D",
        "3d": "Risk_3D",
    }
    rows = []
    for family, risk_name in risk_names.items():
        row = selected[selected["Family"].astype(str) == family].copy()
        if len(row) != 1:
            raise RuntimeError(f"Expected one locked image row for {family}, got {len(row)}")
        row = row.iloc[0].copy()
        row["Risk_Name"] = risk_name
        row["Selection_Metric"] = "final_epoch_Pooled_OOF_C"
        row["FinalEpoch_Val_C"] = float(row["Mean_FinalEpoch_Val_C"])
        row["BestEpoch_Val_C_Audit_Not_Used"] = float(row["Mean_BestEpoch_Val_C_Audit_Not_Used"])
        row["External_C_Diagnostic"] = float(row["External_C_Diagnostic_Not_For_Selection"])
        rows.append(row)

    rad_summary = pd.read_csv(RAD_SUMMARY_CSV)
    if rad_summary.empty:
        raise RuntimeError("Radiomics summary is empty")
    rad_summary["Val_C"] = pd.to_numeric(rad_summary["Val_C"], errors="raise")
    selected_rad = rad_summary.sort_values(["Val_C", "Model"], ascending=[False, True]).iloc[0]
    rows.append(
        pd.Series(
            {
                "ROI_Mode": "tumor_plus_5mm",
                "Family": "Radiomics",
                "Backbone": str(selected_rad["Model"]),
                "Risk_Name": "Risk_Rad",
                "Selection_Metric": "fixed_original107x3_Pooled_OOF_C",
                "FinalEpoch_Val_C": np.nan,
                "BestEpoch_Val_C_Audit_Not_Used": np.nan,
                "Pooled_OOF_C": float(selected_rad["Val_C"]),
                "External_C_Diagnostic": float(selected_rad["External_C"]),
                "OOF_CSV": str(RAD_VAL_CSV),
                "External_CSV": str(RAD_EXT_CSV),
            }
        )
    )
    return pd.DataFrame(rows)


def load_image_candidate(row: pd.Series, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fusion.load_image_candidate(row, base, str(row["Risk_Name"]))


def load_radiomics_candidate(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = normalize_id(pd.read_csv(RAD_VAL_CSV))
    ext = normalize_id(pd.read_csv(RAD_EXT_CSV))
    if "Rad_Score" not in val.columns or "Rad_Score" not in ext.columns:
        raise RuntimeError("Expected Rad_Score in radiomics_val_predictions/test_predictions")
    train_base = base[base["Set"] == "Train_CV"][[ID_COL, TIME_COL, EVENT_COL]].copy()
    ext_base = base[base["Set"] == EXTERNAL_SET][[ID_COL, TIME_COL, EVENT_COL]].copy()
    train = train_base.merge(val[[ID_COL, "Rad_Score"]], on=ID_COL, how="inner")
    external = ext_base.merge(ext[[ID_COL, "Rad_Score"]], on=ID_COL, how="inner")
    return train.rename(columns={"Rad_Score": "Risk_Rad"}), external.rename(columns={"Rad_Score": "Risk_Rad"})


def validate_feature_table(
    df: pd.DataFrame,
    name: str,
    expected_ids: set[str],
    allow_missing: bool = False,
) -> pd.DataFrame:
    observed_ids = set(df[ID_COL].astype(str).str.strip())
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    if (missing_ids or extra_ids) and not allow_missing:
        raise RuntimeError(
            f"{name} ID mismatch: observed={len(observed_ids)} expected={len(expected_ids)} "
            f"missing={missing_ids[:20]} extra={extra_ids[:20]}"
        )
    missing = df[FOUR_INPUT_FEATURES].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(f"{name} missing feature values: {missing.to_dict()}")
    rows = [{"MRI_id": mid, "Issue": "missing_risk"} for mid in missing_ids]
    rows.extend({"MRI_id": mid, "Issue": "extra_risk"} for mid in extra_ids)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    global IMAGE_SELECTED_CSV, RAD_DIR, RAD_SUMMARY_CSV, RAD_VAL_CSV, RAD_EXT_CSV
    IMAGE_SELECTED_CSV = args.image_selected_csv
    RAD_DIR = args.radiomics_dir
    RAD_SUMMARY_CSV = args.radiomics_summary_csv or latest_required(RAD_DIR, "radiomics_original107x3_model_summary_*.csv")
    RAD_VAL_CSV = args.radiomics_val_csv or RAD_DIR / "radiomics_val_predictions.csv"
    RAD_EXT_CSV = args.radiomics_test_csv or RAD_DIR / "radiomics_test_predictions.csv"

    split_md5 = file_md5(SPLIT_CSV)
    if EXPECTED_SPLIT_MD5 and split_md5 != EXPECTED_SPLIT_MD5:
        raise RuntimeError(f"Split MD5 mismatch: got {split_md5}, expected {EXPECTED_SPLIT_MD5}")
    clinical_md5 = file_md5(CLINICAL_EXCEL)
    if EXPECTED_CLINICAL_MD5 and clinical_md5 != EXPECTED_CLINICAL_MD5:
        raise RuntimeError(f"Clinical MD5 mismatch: got {clinical_md5}, expected {EXPECTED_CLINICAL_MD5}")

    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_dataframe()
    train_base = base[base["Set"] == "Train_CV"]
    external_base = base[base["Set"] == EXTERNAL_SET]
    locked = load_locked_branches()
    locked.to_csv(output_dir / "locked_4input_inputs.csv", index=False)

    train_tables = []
    external_tables = []
    for _, row in locked.iterrows():
        if row["Family"] == "Radiomics":
            train, external = load_radiomics_candidate(base)
        else:
            train, external = load_image_candidate(row, base)
        train_tables.append(train)
        external_tables.append(external)

    ifm_train = fusion.merge_feature_tables(train_tables)
    ifm_external = fusion.merge_feature_tables(external_tables)
    train_expected_ids = set(train_base[ID_COL].astype(str).str.strip())
    external_expected_ids = set(external_base[ID_COL].astype(str).str.strip())
    train_missing = validate_feature_table(ifm_train, "IFM Train_CV", train_expected_ids)
    if len(train_missing) != 0:
        raise RuntimeError("Train_CV should not have missing locked risks")
    external_missing = validate_feature_table(
        ifm_external,
        "IFM External_Test",
        external_expected_ids,
        allow_missing=args.allow_external_missing,
    )
    external_missing.to_csv(output_dir / "late_fusion_external_missing_ids.csv", index=False)
    ifm_train.to_csv(output_dir / "late_fusion_ifm_train_features.csv", index=False)
    ifm_external.to_csv(output_dir / "late_fusion_ifm_external_features.csv", index=False)

    ifm_repeat, ifm_grid, ifm_locked = fusion.summarize_repeated_cv(
        ifm_train,
        ifm_external,
        FOUR_INPUT_FEATURES,
        LFM_MODEL_NAMES,
        args.seed,
        args.repeats,
        args.splits,
    )
    ifm_repeat.to_csv(output_dir / "late_fusion_ifm_repeat_selection.csv", index=False)
    ifm_grid.to_csv(output_dir / "late_fusion_ifm_grid.csv", index=False)

    pred_train = pd.DataFrame(
        {
            ID_COL: ifm_train[ID_COL].values,
            "Set": "Train_CV",
            TIME_COL: ifm_train[TIME_COL].values,
            EVENT_COL: ifm_train[EVENT_COL].values,
            "Feature_Set_Label": "all4",
            "IFM_Risk": ifm_locked["OOF_Pred"],
        }
    )
    pred_external = pd.DataFrame(
        {
            ID_COL: ifm_external[ID_COL].values,
            "Set": EXTERNAL_SET,
            TIME_COL: ifm_external[TIME_COL].values,
            EVENT_COL: ifm_external[EVENT_COL].values,
            "Feature_Set_Label": "all4",
            "IFM_Risk": ifm_locked["External_Pred"],
        }
    )
    pd.concat([pred_train, pred_external], ignore_index=True).to_csv(
        output_dir / "all_selected_predictions.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Model": ifm_locked["Model"],
                "Feature_Set": ifm_locked["Feature_Set"],
                "Mean_Val_C": ifm_locked["Mean_Val_C"],
                "Std_Val_C": ifm_locked["Std_Val_C"],
                "OOF_C": ifm_locked["OOF_C_base_seed"],
                "Train_C_full": ifm_locked["Train_C_full"],
                "External_C_Diagnostic": ifm_locked["External_C"],
            }
        ]
    ).to_csv(output_dir / "late_fusion_summary.csv", index=False)
    manifest = {
        "run_tag": tag,
        "checkpoint_policy": "final_epoch_only_for_image_inputs",
        "selection_metric": "mean validation C-index across repeated development CV",
        "feature_set": FOUR_INPUT_FEATURES,
        "candidate_learners": LFM_MODEL_NAMES,
        "selected_learner": ifm_locked["Model"],
        "external_outcomes_used_for_selection": False,
        "split_csv_md5": split_md5,
        "clinical_excel_md5": clinical_md5,
        "train_cv_n": int(len(ifm_train)),
        "external_n": int(len(ifm_external)),
    }
    (output_dir / "late_fusion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-input late-score imaging fusion.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--tag", default="")
    parser.add_argument("--image-selected-csv", type=Path, default=IMAGE_SELECTED_CSV)
    parser.add_argument("--radiomics-dir", type=Path, default=RAD_DIR)
    parser.add_argument("--radiomics-summary-csv", type=Path, default=None)
    parser.add_argument("--radiomics-val-csv", type=Path, default=None)
    parser.add_argument("--radiomics-test-csv", type=Path, default=None)
    parser.add_argument(
        "--allow-external-missing",
        action="store_true",
        help="Allow v20 external cases without locked risk scores; metrics become overlap diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    output_dir = run(parse_args())
    print(f"Saved imaging-fusion outputs to: {output_dir}")


if __name__ == "__main__":
    main()
