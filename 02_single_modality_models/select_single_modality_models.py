from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
FINAL_DIR = Path(os.environ.get("MR_IMAGE_MODEL_OUT", ROOT / "outputs" / "single_modality_models" / "image"))
RAD_DIR = Path(os.environ.get("MR_RADIOMICS_OUT_ROOT", ROOT / "outputs" / "single_modality_models" / "radiomics"))
OUT_DIR = Path(os.environ.get("MR_SINGLE_MODALITY_SELECTION_OUT", ROOT / "outputs" / "single_modality_models" / "selected"))


def latest_selected_path() -> Path:
    candidates = sorted(FINAL_DIR.glob("tumor_plus_5mm_image_finalepoch_fixedfold_selected_by_pooled_oof_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No final-epoch selected image table found. Run "
            "screen_image_models.py first."
        )
    return candidates[-1]


def latest_radiomics_summary_path() -> Path:
    candidates = sorted(RAD_DIR.glob("radiomics_original107x3_model_summary_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No fixed-feature original107x3 radiomics summary found. Run "
            "screen_radiomics_models.py first."
        )
    return candidates[-1]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = latest_selected_path()
    rad_summary_path = latest_radiomics_summary_path()
    img = pd.read_csv(selected_path)
    required = {"2d": "Risk_2D", "2p5d": "Risk_2p5D", "3d": "Risk_3D"}
    missing = sorted(set(required) - set(img["Family"].astype(str)))
    if missing:
        raise RuntimeError(f"Final-epoch selected image table is missing families: {missing}")

    selected_images = (
        img[img["Family"].astype(str).isin(required)]
        .sort_values("Family")
        .reset_index(drop=True)
    )
    selected_images.to_csv(OUT_DIR / "selected_models.csv", index=False)

    rows = []
    for family, risk_name in required.items():
        row = img[img["Family"].astype(str).eq(family)].iloc[0]
        rows.append(
            {
                "Risk_Name": risk_name,
                "Modality": family,
                "Model": row["Backbone"],
                "ROI_Mode": row["ROI_Mode"],
                "Selection_Metric": "Pooled_OOF_C",
                "Checkpoint_Policy": "final_epoch_only",
                "Fold_Policy": "fixed_split_fold_column",
                "Pooled_OOF_C": row["Pooled_OOF_C"],
                "Mean_FinalEpoch_Val_C": row["Mean_FinalEpoch_Val_C"],
                "External_C_Diagnostic_Not_For_Selection": row["External_C_Diagnostic_Not_For_Selection"],
                "Source_CSV": str(selected_path),
            }
        )

    rad = pd.read_csv(rad_summary_path).sort_values("Val_C", ascending=False).iloc[0]
    rows.append(
        {
            "Risk_Name": "Risk_Rad",
            "Modality": "Radiomics",
            "Model": rad["Model"],
            "ROI_Mode": "tumor_plus_5mm",
            "Selection_Metric": "Pooled_OOF_C",
            "Checkpoint_Policy": "not_applicable",
            "Fold_Policy": "fixed_original107x3_no_outcome_feature_screening",
            "Pooled_OOF_C": rad["Val_C"],
            "Mean_FinalEpoch_Val_C": rad["Val_C"],
            "External_C_Diagnostic_Not_For_Selection": rad["External_C"],
            "Source_CSV": str(rad_summary_path),
        }
    )

    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "single_modality_selection_summary.csv"
    md_path = OUT_DIR / "single_modality_selection_summary.md"
    out.to_csv(csv_path, index=False)
    md = [
        "# Final Single-Modality Lock: Final-Epoch Fixed-Fold Route",
        "",
        f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "- ROI: `tumor_plus_5mm`",
        "- Image checkpoint policy: final epoch only",
        "- Image fold policy: existing split `Fold` column",
        "- Radiomics feature policy: fixed AP/PVP/DP original PyRadiomics features, 107 per phase, no univariate/LASSO screening",
        "- Selection metric: `Pooled_OOF_C`",
        "- External C is diagnostic only and not used for selection.",
        "",
        out.to_markdown(index=False),
        "",
        f"Image selected source: `{selected_path}`",
        f"Radiomics source: `{rad_summary_path}`",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
