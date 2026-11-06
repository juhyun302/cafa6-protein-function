"""One entry point for the preserved structure/ensemble workflow."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .download import download_structures
from .ensemble import patch_scores, rank_ensemble, validate_submission
from .io import read_predictions, write_predictions
from .search import search_structures
from .structure import (
    extract_accessions, filter_accessions, merge_hits, transfer_go, normalize_scores,
    split_aspects, load_ancestors, propagate, make_submission,
)


def run_demo(examples: Path, output: Path, workers: int = 1) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    extract_accessions(examples / "query.fasta", output / "accessions.txt")
    raw = transfer_go(examples / "foldseek-hits.tsv", examples / "annotations.tsv")
    write_predictions(raw, output / "raw-go-scores.tsv")
    normalized = normalize_scores(raw)
    split_aspects(normalized, output / "normalized")
    ancestors = load_ancestors(parents=examples / "go-parents.tsv")
    for aspect in ("bp", "mf", "cc"):
        predictions = read_predictions(output / "normalized" / f"pred_{aspect}.tsv")
        write_predictions(propagate(predictions, ancestors, workers),
                          output / "propagated" / f"pred_{aspect}_propagated.tsv")
    structure = make_submission(output / "propagated")
    validate_submission(structure)
    write_predictions(structure, output / "structure-submission.tsv", submission=True)
    mounted = patch_scores(read_predictions(examples / "baseline.tsv"), structure)
    write_predictions(mounted, output / "patched.tsv", submission=True)
    final = rank_ensemble(mounted, read_predictions(examples / "other-model.tsv"), structure)
    report = validate_submission(final)
    write_predictions(final, output / "submission.tsv", submission=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parser():
    result = argparse.ArgumentParser(description="CAFA-6 structure transfer and ensemble pipeline")
    sub = result.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Run synthetic examples without network or cloud")
    p.add_argument("--examples-dir", type=Path, default=Path("examples"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("extract-accessions")
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("filter-accessions")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--exclude", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("download")
    p.add_argument("--accessions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, default=Path("artifacts/download"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("search")
    p.add_argument("--structures-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--foldseek-bin", default="foldseek")
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--max-seqs", type=int, default=20)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--work-dir", type=Path)

    p = sub.add_parser("merge-hits")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("transfer")
    p.add_argument("--hits", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("normalize")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.2)

    p = sub.add_parser("propagate")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--parents", type=Path)
    p.add_argument("--mapping-pkl", type=Path)
    p.add_argument("--ancestor-npz", type=Path)
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("make-submission")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=500)

    for name in ("patch", "rank-ensemble"):
        p = sub.add_parser(name)
        p.add_argument("--base", type=Path, required=True)
        if name == "rank-ensemble":
            p.add_argument("--other", type=Path, required=True)
        p.add_argument("--patch", type=Path, required=True)
        p.add_argument("--threshold", type=float, default=0.99)
        p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("validate")
    p.add_argument("--input", type=Path, required=True)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        command = args.command
        if command == "demo":
            result = run_demo(args.examples_dir, args.output_dir, args.workers)
        elif command == "extract-accessions":
            result = {"accessions": extract_accessions(args.fasta, args.output)}
        elif command == "filter-accessions":
            result = {"kept": filter_accessions(args.input, args.exclude, args.output)}
        elif command == "download":
            result = download_structures(args.accessions, args.output_dir, args.log_dir, args.workers, args.limit)
            print(json.dumps(result))
            if result["FAIL"]:
                arguments.exit(1, "Some downloads failed; inspect download-status.tsv before filtering IDs.\n")
            return
        elif command == "search":
            search_structures(args.structures_dir, args.output_dir, args.reference_dir, args.foldseek_bin,
                              args.batch_size, args.max_seqs, args.threads, args.work_dir)
            result = {"output_dir": str(args.output_dir)}
        elif command == "merge-hits":
            result = {"hits": len(merge_hits(args.input_dir, args.output))}
        elif command == "transfer":
            frame = transfer_go(args.hits, args.annotations)
            write_predictions(frame, args.output)
            result = {"rows": len(frame)}
        elif command == "normalize":
            frame = normalize_scores(read_predictions(args.input), args.top_k, args.min_score)
            split_aspects(frame, args.output_dir)
            result = {"rows": len(frame)}
        elif command == "propagate":
            ancestors = load_ancestors(args.parents, args.mapping_pkl, args.ancestor_npz)
            rows = 0
            for aspect in ("bp", "mf", "cc"):
                frame = read_predictions(args.input_dir / f"pred_{aspect}.tsv")
                frame = propagate(frame, ancestors, args.workers)
                rows += len(frame)
                write_predictions(frame, args.output_dir / f"pred_{aspect}_propagated.tsv")
            result = {"rows": rows}
        elif command == "make-submission":
            frame = make_submission(args.input_dir, args.top_n)
            result = validate_submission(frame)
            write_predictions(frame, args.output, submission=True)
        elif command in ("patch", "rank-ensemble"):
            base, patch = read_predictions(args.base), read_predictions(args.patch)
            if command == "patch":
                frame = patch_scores(base, patch, args.threshold)
            else:
                frame = rank_ensemble(base, read_predictions(args.other), patch, args.threshold)
            result = validate_submission(frame)
            write_predictions(frame, args.output, submission=True)
        else:
            result = validate_submission(read_predictions(args.input))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        arguments.exit(2, f"Error: {exc}\n")
