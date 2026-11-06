"""Batched Foldseek search using an explicitly installed executable."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def search_structures(structures_dir: Path, output_dir: Path, reference_dir: Path | None = None,
                      foldseek_bin: str = "foldseek", batch_size: int = 5000,
                      max_seqs: int = 20, threads: int = 1, work_dir: Path | None = None) -> None:
    if min(batch_size, max_seqs, threads) < 1:
        raise ValueError("batch_size, max_seqs, and threads must be positive")
    executable = shutil.which(foldseek_bin)
    if executable is None:
        raise ValueError("Foldseek executable not found; install it and pass --foldseek-bin")
    structures_dir = Path(structures_dir).resolve()
    reference_dir = Path(reference_dir or structures_dir).resolve()
    files = sorted(structures_dir.rglob("*.cif"))
    if not files or not reference_dir.is_dir():
        raise ValueError("Expected query CIF files and an existing reference directory")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("results_batch_*.tsv")):
        raise ValueError("Output already contains batch results; use a clean output directory")
    if work_dir:
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cafa-foldseek-", dir=work_dir) as temporary:
        temporary = Path(temporary)
        target_db = temporary / "targetDB"

        def run(*args):
            subprocess.run([executable, *map(str, args)], check=True)

        run("createdb", reference_dir, target_db, "--threads", threads)
        for start in range(0, len(files), batch_size):
            batch_number = start // batch_size + 1
            with tempfile.TemporaryDirectory(prefix="batch-", dir=temporary) as batch_work:
                batch_work = Path(batch_work)
                batch_dir = batch_work / "queries"
                batch_dir.mkdir()
                for index, source in enumerate(files[start:start + batch_size]):
                    try:
                        os.symlink(source, batch_dir / f"{index:06d}_{source.name}")
                    except OSError as exc:
                        raise ValueError("Foldseek batching requires symlink support; use Linux/WSL or Azure Linux compute") from exc
                query_db, result_db = batch_work / "queryDB", batch_work / "resultDB"
                run("createdb", batch_dir, query_db, "--threads", threads)
                run("search", query_db, target_db, result_db, batch_work / "tmp",
                    "--threads", threads, "--max-seqs", max_seqs, "-a")
                run("convertalis", query_db, target_db, result_db,
                    output_dir / f"results_batch_{batch_number}.tsv",
                    "--format-output", "query,target,evalue,bits,alntmscore,qcov,tcov", "--threads", threads)
