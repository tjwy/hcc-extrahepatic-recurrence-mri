from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_single_modality_selection_writes_canonical_image_table(tmp_path: Path) -> None:
    selector = load_module(
        "02_single_modality_models/select_single_modality_models.py",
        "public_single_modality_selector",
    )
    image_dir = tmp_path / "image"
    radiomics_dir = tmp_path / "radiomics"
    output_dir = tmp_path / "selected"
    image_dir.mkdir()
    radiomics_dir.mkdir()

    image_rows = []
    for family, backbone in [("2d", "resnet34"), ("2p5d", "mobilenet_v2"), ("3d", "resnet10_3d")]:
        image_rows.append(
            {
                "Family": family,
                "Backbone": backbone,
                "ROI_Mode": "tumor_plus_5mm",
                "Pooled_OOF_C": 0.7,
                "Mean_FinalEpoch_Val_C": 0.7,
                "External_C_Diagnostic_Not_For_Selection": 0.7,
                "Checkpoint_Dir": str(tmp_path / family),
                "Config": "{}",
                "OOF_CSV": str(tmp_path / f"{family}_oof.csv"),
                "External_CSV": str(tmp_path / f"{family}_external.csv"),
            }
        )
    pd.DataFrame(image_rows).to_csv(
        image_dir / "tumor_plus_5mm_image_finalepoch_fixedfold_selected_by_pooled_oof_20260713.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"Model": "RSF", "Val_C": 0.71, "External_C": 0.70},
            {"Model": "CoxPH", "Val_C": 0.69, "External_C": 0.68},
        ]
    ).to_csv(radiomics_dir / "radiomics_original107x3_model_summary_20260713.csv", index=False)

    selector.FINAL_DIR = image_dir
    selector.RAD_DIR = radiomics_dir
    selector.OUT_DIR = output_dir
    selector.main()

    canonical = pd.read_csv(output_dir / "selected_models.csv")
    assert canonical["Family"].tolist() == ["2d", "2p5d", "3d"]
    assert "Checkpoint_Dir" in canonical.columns
    summary = pd.read_csv(output_dir / "single_modality_selection_summary.csv")
    assert summary["Modality"].tolist() == ["2d", "2p5d", "3d", "Radiomics"]


def test_release_audit_ignores_transient_cache_directories() -> None:
    audit = load_module("tools/audit_release.py", "public_release_audit")
    relative_paths = {path.relative_to(ROOT).as_posix() for path in audit.release_files()}
    assert not any(path.startswith(".git/") for path in relative_paths)
    assert not any(path.startswith(".pytest_cache/") for path in relative_paths)
    assert not any("/__pycache__/" in f"/{path}/" for path in relative_paths)


def test_late_fusion_uses_radiomics_oof_winner(tmp_path: Path) -> None:
    stage_dir = ROOT / "03_imaging_fusion"
    sys.path.insert(0, str(stage_dir))
    try:
        late = load_module(
            "03_imaging_fusion/train_late_score_fusion.py",
            "public_late_fusion",
        )
    finally:
        sys.path.remove(str(stage_dir))

    selected = pd.DataFrame(
        [
            {
                "Family": family,
                "Backbone": backbone,
                "Mean_FinalEpoch_Val_C": 0.7,
                "Mean_BestEpoch_Val_C_Audit_Not_Used": 0.72,
                "External_C_Diagnostic_Not_For_Selection": 0.69,
                "Pooled_OOF_C": 0.7,
                "OOF_CSV": str(tmp_path / f"{family}_oof.csv"),
                "External_CSV": str(tmp_path / f"{family}_external.csv"),
            }
            for family, backbone in [("2d", "a"), ("2p5d", "b"), ("3d", "c")]
        ]
    )
    selected_path = tmp_path / "selected_models.csv"
    selected.to_csv(selected_path, index=False)
    radiomics_path = tmp_path / "radiomics_summary.csv"
    pd.DataFrame(
        [
            {"Model": "RSF", "Val_C": 0.68, "External_C": 0.67},
            {"Model": "CoxPH", "Val_C": 0.72, "External_C": 0.66},
        ]
    ).to_csv(radiomics_path, index=False)

    late.IMAGE_SELECTED_CSV = selected_path
    late.RAD_SUMMARY_CSV = radiomics_path
    locked = late.load_locked_branches()
    radiomics = locked.loc[locked["Family"].eq("Radiomics")].iloc[0]
    assert radiomics["Backbone"] == "CoxPH"
    assert radiomics["Pooled_OOF_C"] == 0.72


def test_crm_candidate_selection_uses_internal_oof_only() -> None:
    crm = load_module(
        "04_clinical_radiologic_model/train_clinical_radiologic_model.py",
        "public_crm",
    )
    summary = pd.DataFrame(
        [
            {
                "Preprocess": "numeric_raw",
                "Model": "Coxnet",
                "OOF_C_base_seed": 0.74,
                "Mean_Val_C": 0.72,
                "External_C_Diagnostic": 0.60,
            },
            {
                "Preprocess": "categorical",
                "Model": "RSF",
                "OOF_C_base_seed": 0.73,
                "Mean_Val_C": 0.75,
                "External_C_Diagnostic": 0.90,
            },
        ]
    )
    selected = crm.select_candidate(summary)
    assert selected["Model"] == "Coxnet"
    assert selected["Preprocess"] == "numeric_raw"
