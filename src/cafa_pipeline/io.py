"""Explicit TSV schemas shared by every stage."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

KEYS = ["protein_id", "go_id"]
PREDICTION_COLUMNS = ["protein_id", "go_id", "aspect", "score"]
HIT_COLUMNS = ["query", "target", "evalue", "bits", "alntmscore", "qcov", "tcov"]
ASPECTS = {"P": "BP", "F": "MF", "C": "CC", "BP": "BP", "MF": "MF", "CC": "CC"}


def normalize_id(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"^\d{6}_", "", value)
    value = re.sub(r"\.(?:cif|pdb)(?:\.gz)?$", "", value, flags=re.IGNORECASE)
    if value.startswith("AF-") and "-F" in value:
        value = value[3:].split("-F", 1)[0]
    return value


def read_predictions(path: Path) -> pd.DataFrame:
    """Accept documented headered predictions or legacy 3/4-column TSVs."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Prediction file missing or empty: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        first = stream.readline().strip().split("\t")
    has_header = first[0].lower() in {"protein_id", "id", "query", "protein", "entryid"}
    frame = pd.read_csv(path, sep="\t", header=0 if has_header else None, dtype=str)
    if not has_header:
        if len(frame.columns) == 3:
            frame.columns = ["protein_id", "go_id", "score"]
        elif len(frame.columns) == 4:
            frame.columns = PREDICTION_COLUMNS
        else:
            raise ValueError(f"Expected 3 or 4 columns: {path}")
    else:
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        frame = frame.rename(columns={
            "id": "protein_id", "query": "protein_id", "protein": "protein_id",
            "entryid": "protein_id", "term": "go_id", "go": "go_id",
        })
        if "score_norm" in frame:
            frame["score"] = frame["score_norm"]
        elif "score" not in frame:
            for alias in ("prob", "confidence", "probability"):
                if alias in frame:
                    frame["score"] = frame[alias]
                    break
    if not set(KEYS + ["score"]).issubset(frame.columns):
        raise ValueError(f"Missing protein_id, go_id, or score: {path}")
    if "aspect" not in frame:
        frame["aspect"] = ""
    frame = frame[PREDICTION_COLUMNS].copy()
    if frame[KEYS].isna().any().any():
        raise ValueError(f"Missing prediction identifiers: {path}")
    frame["protein_id"] = frame["protein_id"].map(normalize_id)
    frame["go_id"] = frame["go_id"].str.strip()
    frame["aspect"] = frame["aspect"].fillna("").map(lambda a: ASPECTS.get(str(a).strip(), ""))
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    if not np.isfinite(frame["score"]).all() or (frame["score"] < 0).any():
        raise ValueError(f"Scores must be finite and nonnegative: {path}")
    if (frame["protein_id"] == "").any() or not frame["go_id"].str.fullmatch(r"GO:\d{7}").all():
        raise ValueError(f"Invalid protein or GO identifier: {path}")
    return frame


def write_predictions(frame: pd.DataFrame, path: Path, submission: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = KEYS + ["score"] if submission else PREDICTION_COLUMNS
    frame[columns].to_csv(path, sep="\t", index=False, header=not submission, float_format="%.6g")


def read_hits(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=HIT_COLUMNS)
    if frame.empty:
        raise ValueError(f"No Foldseek hits: {path}")
    for column in ("query", "target"):
        frame[column] = frame[column].map(normalize_id)
    frame["alntmscore"] = pd.to_numeric(frame["alntmscore"], errors="raise")
    if not np.isfinite(frame["alntmscore"]).all() or (frame["alntmscore"] < 0).any():
        raise ValueError("alntmscore must be finite and nonnegative")
    return frame
