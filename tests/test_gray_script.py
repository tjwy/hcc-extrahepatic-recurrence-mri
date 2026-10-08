import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

RSCRIPT = os.environ.get("RSCRIPT_EXECUTABLE") or shutil.which("Rscript")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not RSCRIPT, reason="Rscript is an optional dependency")
def test_gray_script_handles_two_groups_and_single_group(tmp_path):
    frame = pd.DataFrame({"Subset": ["both"] * 8 + ["single"] * 4,
                          "Risk_Group": ["Low"] * 4 + ["High"] * 4 + ["Low"] * 4,
                          "analysis_time_months": [2, 8, 12, 14, 1, 3, 9, 13, 4, 8, 12, 16],
                          "event_type": [1, 2, 0, 1, 1, 1, 2, 0, 1, 0, 2, 0]})
    source, output = tmp_path / "synthetic.csv", tmp_path / "out"
    frame.to_csv(source, index=False)
    process = subprocess.run([RSCRIPT, "--vanilla", str(ROOT / "05_statistical_validation/evaluate_competing_risks.R"),
                              str(source), str(output)], capture_output=True, text=True,
                             env={**os.environ, "LC_ALL": "C"}, timeout=60)
    assert process.returncode == 0, process.stderr
    result = pd.read_csv(output / "competing_risk_summary.csv")
    assert result.N.tolist() == [8, 4]
    assert result.EHR_Events.tolist() == [4, 1]
    assert result.Gray_P.iloc[0] >= 0 and result.Gray_P.iloc[0] <= 1
    assert pd.isna(result.Gray_P.iloc[1])
