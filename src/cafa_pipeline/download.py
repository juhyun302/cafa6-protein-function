"""Download existing AlphaFold DB structures with resume and explicit failure states."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import time
import urllib.error
import urllib.request


def valid_cif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1200:
        return False
    with path.open(encoding="utf-8", errors="ignore") as stream:
        head = stream.read(300).lower()
    return head.lstrip().startswith("data_") and "<error>" not in head


def download_one(accession: str, output_dir: Path, retries: int = 3, timeout: float = 30) -> tuple[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", accession):
        raise ValueError(f"Invalid accession: {accession}")
    output = Path(output_dir) / f"{accession}.cif"
    if valid_cif(output):
        return accession, "SKIP"
    temporary = output.with_suffix(".cif.part")
    for attempt in range(retries):
        try:
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    records = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return accession, "NO_STRUCTURE"
                raise
            if not records or not records[0].get("cifUrl"):
                return accession, "NO_STRUCTURE"
            with urllib.request.urlopen(records[0]["cifUrl"], timeout=timeout) as response:
                temporary.write_bytes(response.read())
            if not valid_cif(temporary):
                raise ValueError("Response is not a valid mmCIF structure")
            temporary.replace(output)
            return accession, "OK"
        except (OSError, ValueError, KeyError, IndexError):
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    return accession, "FAIL"


def download_structures(accessions: Path, output_dir: Path, log_dir: Path,
                        workers: int = 8, limit: int | None = None) -> dict:
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    identifiers = list(dict.fromkeys(Path(accessions).read_text(encoding="utf-8").split()))
    if limit is not None:
        identifiers = identifiers[:limit]
    if not identifiers:
        raise ValueError("Accession list is empty")
    output_dir, log_dir = Path(output_dir), Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(["OK", "SKIP", "NO_STRUCTURE", "FAIL"], 0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda accession: download_one(accession, output_dir), identifiers)
        with (log_dir / "download-status.tsv").open("w", encoding="utf-8") as log:
            log.write("accession\tstatus\n")
            for accession, status in results:
                counts[status] += 1
                log.write(f"{accession}\t{status}\n")
    return counts
