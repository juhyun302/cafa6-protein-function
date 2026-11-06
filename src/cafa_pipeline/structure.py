"""GO label transfer, score normalization, max propagation, and submission."""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from .io import ASPECTS, HIT_COLUMNS, KEYS, PREDICTION_COLUMNS, normalize_id, read_hits, read_predictions, write_predictions


def extract_accessions(fasta: Path, output: Path) -> int:
    identifiers = set()
    with Path(fasta).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split("|")
                identifier = parts[1] if len(parts) >= 3 else header.split()[0]
                identifiers.add(identifier)
    if not identifiers:
        raise ValueError("No FASTA identifiers found")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sorted(identifiers)) + "\n", encoding="utf-8")
    return len(identifiers)


def filter_accessions(source: Path, exclude: Path, output: Path) -> int:
    excluded = set(Path(exclude).read_text(encoding="utf-8").split())
    identifiers = list(dict.fromkeys(Path(source).read_text(encoding="utf-8").split()))
    kept = [identifier for identifier in identifiers if identifier not in excluded]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept)


def merge_hits(input_dir: Path, output: Path) -> pd.DataFrame:
    paths = sorted(Path(input_dir).glob("results_batch_*.tsv"))
    if not paths:
        raise ValueError("No results_batch_*.tsv files found")
    frames = [read_hits(path) for path in paths if path.stat().st_size]
    if not frames:
        raise ValueError("All hit files are empty")
    frame = pd.concat(frames, ignore_index=True).sort_values(["query", "evalue"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame[HIT_COLUMNS].to_csv(output, sep="\t", header=False, index=False)
    return frame


def transfer_go(hits_path: Path, annotations_path: Path) -> pd.DataFrame:
    hits = read_hits(hits_path)
    hits = hits[hits["query"] != hits["target"]].copy()
    hits["weight"] = hits["alntmscore"].clip(upper=1.0)
    terms = pd.read_csv(annotations_path, sep="\t", dtype=str)
    terms = terms.rename(columns={"EntryID": "target", "protein_id": "target", "term": "go_id"})
    if not {"target", "go_id", "aspect"}.issubset(terms):
        raise ValueError("Annotations require EntryID (or protein_id), term (or go_id), aspect")
    terms["target"] = terms["target"].map(normalize_id)
    terms["aspect"] = terms["aspect"].map(ASPECTS)
    if terms["aspect"].isna().any() or not terms["go_id"].str.fullmatch(r"GO:\d{7}", na=False).all():
        raise ValueError("Annotations contain unknown aspects or invalid GO identifiers")
    terms = terms[["target", "go_id", "aspect"]].drop_duplicates()
    joined = hits.merge(terms, on="target", how="inner")
    if joined.empty:
        raise ValueError("No hit targets matched the annotation identifiers")
    result = joined.groupby(["query", "go_id", "aspect"], as_index=False)["weight"].sum()
    return result.rename(columns={"query": "protein_id", "weight": "score"})[PREDICTION_COLUMNS]


def normalize_scores(frame: pd.DataFrame, top_k: int = 50, min_score: float = 0.2) -> pd.DataFrame:
    if top_k < 1 or not 0 <= min_score <= 1:
        raise ValueError("top_k must be positive and min_score must be between 0 and 1")
    if frame["aspect"].eq("").any():
        raise ValueError("Normalization requires a known aspect for every GO term")
    frame = frame.groupby(KEYS + ["aspect"], as_index=False)["score"].sum()
    maxima = frame.groupby("protein_id")["score"].transform("max")
    frame["score"] = np.divide(frame["score"], maxima, out=np.zeros(len(frame)), where=maxima > 0)
    frame = frame.sort_values(["protein_id", "score", "go_id"], ascending=[True, False, True])
    frame = frame.groupby("protein_id", sort=False).head(top_k)
    return frame[frame["score"] >= min_score].reset_index(drop=True)


def split_aspects(frame: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    for aspect in ("BP", "MF", "CC"):
        write_predictions(frame[frame["aspect"] == aspect], Path(output_dir) / f"pred_{aspect.lower()}{suffix}.tsv")


def load_ancestors(parents: Path | None = None, mapping_pkl: Path | None = None,
                   ancestor_npz: Path | None = None) -> dict[str, tuple[str, ...]]:
    if parents:
        table = pd.read_csv(parents, sep="\t", dtype=str)
        if not {"child", "parent"}.issubset(table):
            raise ValueError("Parent TSV requires child and parent columns")
        direct = table.groupby("child")["parent"].agg(list).to_dict()
        visiting = set()

        @lru_cache(maxsize=None)
        def visit(term):
            if term in visiting:
                raise ValueError("GO parent graph contains a cycle")
            visiting.add(term)
            result = set()
            for parent in direct.get(term, []):
                result.add(parent)
                result.update(visit(parent))
            visiting.remove(term)
            return tuple(sorted(result))

        return {term: visit(term) for term in direct}
    if not mapping_pkl or not ancestor_npz:
        raise ValueError("Provide --parents or both --mapping-pkl and --ancestor-npz")
    with Path(mapping_pkl).open("rb") as stream:
        obj = pickle.load(stream)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        go_to_idx, idx_to_go = obj[:2]
    elif isinstance(obj, dict):
        pair = next(((a, b) for a, b in [
            ("go_to_idx", "idx_to_go"), ("go2idx", "idx2go"),
            ("term2idx", "idx2term"), ("term_to_idx", "idx_to_term"),
        ] if a in obj and b in obj), None)
        if pair is None:
            raise ValueError("Unsupported GO mapping keys")
        go_to_idx, idx_to_go = obj[pair[0]], obj[pair[1]]
    else:
        raise ValueError("Unsupported GO mapping format")
    matrix = load_npz(ancestor_npz).tocsr()
    if matrix.shape != (len(go_to_idx), len(go_to_idx)):
        raise ValueError("GO mapping and ancestor matrix dimensions disagree")
    return {
        go: tuple(str(idx_to_go[int(index)]) for index in matrix.getrow(int(row)).indices)
        for go, row in go_to_idx.items()
    }


_ANCESTORS = {}


def _initialize_worker(ancestors):
    global _ANCESTORS
    _ANCESTORS = ancestors


def _propagate_chunk(rows):
    output = {}
    for protein, term, aspect, score in rows:
        for target in (term, *_ANCESTORS.get(term, ())):
            key = (protein, target, aspect)
            output[key] = max(output.get(key, 0.0), float(score))
    return [(*key, score) for key, score in output.items()]


def propagate(frame: pd.DataFrame, ancestors: dict, workers: int = 1) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("workers must be positive")
    if frame.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    rows = list(frame[PREDICTION_COLUMNS].itertuples(index=False, name=None))
    if workers == 1:
        _initialize_worker(ancestors)
        result = _propagate_chunk(rows)
    else:
        chunks = [rows[i:i + 10000] for i in range(0, len(rows), 10000)]
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker, initargs=(ancestors,)) as pool:
            result = [row for chunk in pool.map(_propagate_chunk, chunks) for row in chunk]
    frame = pd.DataFrame(result, columns=PREDICTION_COLUMNS)
    return frame.groupby(KEYS + ["aspect"], as_index=False)["score"].max()[PREDICTION_COLUMNS]


def make_submission(input_dir: Path, top_n: int = 500) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    frames = []
    for aspect in ("bp", "mf", "cc"):
        path = Path(input_dir) / f"pred_{aspect}_propagated.tsv"
        if not path.exists():
            raise ValueError(f"Required aspect file missing: {path}")
        # A header-only aspect file is valid when that namespace has no predictions.
        frame = read_predictions(path)
        frame = frame.groupby(KEYS + ["aspect"], as_index=False)["score"].max()
        frame = frame.sort_values(["protein_id", "score", "go_id"], ascending=[True, False, True])
        frames.append(frame.groupby("protein_id", sort=False).head(top_n))
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError("No predictions to submit")
    return frame.groupby(KEYS, as_index=False)["score"].max().assign(aspect="")[PREDICTION_COLUMNS]
