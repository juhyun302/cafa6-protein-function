"""Preserve the original score patch and global rank averaging strategies."""

import numpy as np
import pandas as pd

from .io import KEYS, PREDICTION_COLUMNS


def deduplicate(frame):
    return frame.groupby(KEYS, as_index=False)["score"].max()


def patch_scores(base: pd.DataFrame, patch: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    base, patch = deduplicate(base), deduplicate(patch)
    patch = patch[patch["score"] >= threshold]
    merged = base.merge(patch, on=KEYS, how="outer", suffixes=("_base", "_patch"))
    merged["score"] = merged["score_patch"].combine_first(merged["score_base"])
    return merged[KEYS + ["score"]].assign(aspect="")[PREDICTION_COLUMNS]


def rank_ensemble(base: pd.DataFrame, other: pd.DataFrame, patch: pd.DataFrame,
                  threshold: float = 0.99) -> pd.DataFrame:
    """Patch the second input, then average global percentile ranks.

    The historical script applies this to all GO rows, not only BP.
    """
    other = patch_scores(other, patch, threshold)
    merged = deduplicate(base).merge(deduplicate(other), on=KEYS, how="outer", suffixes=("_base", "_other"))
    merged[["score_base", "score_other"]] = merged[["score_base", "score_other"]].fillna(0.0)
    merged["score"] = (
        merged["score_base"].rank(pct=True) + merged["score_other"].rank(pct=True)
    ) / 2
    return merged[KEYS + ["score"]].assign(aspect="")[PREDICTION_COLUMNS]


def validate_submission(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Submission is empty")
    if frame[KEYS].isna().any().any() or frame["protein_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Submission identifiers must not be empty")
    if not frame["go_id"].astype(str).str.fullmatch(r"GO:\d{7}").all():
        raise ValueError("Submission contains invalid GO identifiers")
    if frame.duplicated(KEYS).any():
        raise ValueError("Submission contains duplicate protein/GO pairs")
    if not np.isfinite(frame["score"]).all() or not frame["score"].between(0, 1).all():
        raise ValueError("Submission scores must be finite and between 0 and 1")
    return {
        "proteins": int(frame["protein_id"].nunique()),
        "rows": len(frame),
        "max_terms_per_protein": int(frame.groupby("protein_id").size().max()),
        "score_min": float(frame["score"].min()),
        "score_max": float(frame["score"].max()),
    }
