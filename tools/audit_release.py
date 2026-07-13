#!/usr/bin/env python3
"""Audit the public release candidate without reading restricted data."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_FILE_MANIFEST.csv"
AUDITOR = Path(__file__).resolve()
TRANSIENT_DIRS = {"__pycache__", ".pytest_cache", ".mplconfig"}
TEXT_SUFFIXES = {".py", ".R", ".md", ".txt", ".csv", ".yaml", ".yml", ".env"}
FORBIDDEN_SUFFIXES = {
    ".dcm", ".nii", ".gz", ".xlsx", ".xls", ".npy", ".npz",
    ".pth", ".pt", ".pkl", ".joblib", ".h5", ".hdf5",
}
FORBIDDEN_PATH_TERMS = {
    "Analysis_Phase", "GUT", "gut_style", "resplit", "method_lock",
    "methodlock", "mainline_subpanels", "figure", "plot",
}
FORBIDDEN_TEXT = {
    "/media/": "local absolute path",
    "/home/": "local absolute path",
    "Extreme SSD": "local volume name",
    "final_model_data_v": "restricted source filename",
    "@tjh.tjmu.edu.cn": "personal or institutional email",
    "TJ-IRB": "nonblinded ethics identifier",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not TRANSIENT_DIRS.intersection(path.parts)
        and path != MANIFEST
    )


def main() -> int:
    blockers: list[str] = []
    files = release_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            blockers.append(f"forbidden binary file: {relative}")
        for term in FORBIDDEN_PATH_TERMS:
            if term.lower() in relative.lower():
                blockers.append(f"historical or plotting path term '{term}': {relative}")
        if path.suffix in TEXT_SUFFIXES and path != AUDITOR:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle, label in FORBIDDEN_TEXT.items():
                if needle in text:
                    blockers.append(f"{label} in {relative}")

    if not MANIFEST.exists():
        blockers.append("missing RELEASE_FILE_MANIFEST.csv")
    else:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
        observed = {row["release_relative_path"]: row["sha256"] for row in rows}
        if expected != observed:
            blockers.append("release manifest does not match current files or hashes")

    print(f"Audited {len(files)} files")
    for blocker in blockers:
        print(f"BLOCKER: {blocker}")
    print(f"BLOCKER={len(blockers)}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
