#!/usr/bin/env python
"""
download_alphafold_structure.py

Goal:
1) Read accession list from file
2) Check if structure already exists (resume capability)
3) Download missing structures from AlphaFold DB asynchronously
4) Log successes/failures
5) Provide ETA and progress bar

Usage:
    python download_alphafold_structure.py --accession_list <file> --output_dir <dir> --workers <n>
"""

import asyncio
import aiohttp
import argparse
import sys
from pathlib import Path
from tqdm.asyncio import tqdm
import json
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MIN_BYTES = 1200  # Minimum size to consider valid (avoid error XMLs)
MAX_RETRIES = 3   # Max retries for download
TIMEOUT_SEC = 30  # Timeout for requests

def is_valid_cif(path: Path) -> bool:
    """Simple validation for mmCIF file content."""
    try:
        if path.stat().st_size < MIN_BYTES:
            return False
        
        # Read beginning of file to check for error tags or correct format
        # Async file read is not strictly necessary for header check on small/header portion, 
        # but to be safe and simple we use standard I/O here as validation happens typically after download 
        # or before deciding to skip. For high performance validation of *existing* files, 
        # we might want to skip reading if we trust size + existence, but here we prioritize correctness.
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(300).lower()
        
        if "<error>" in head or "nosuchkey" in head:
            return False
        if "data_" in head[:20]:
            return True
        return False
    except Exception:
        return False

async def get_cif_url(session: aiohttp.ClientSession, acc: str) -> str | None:
    """Fetch cifUrl from AlphaFold API."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    try:
        async with session.get(api_url, timeout=TIMEOUT_SEC) as response:
            if response.status != 200:
                return None
            data = await response.json()
            if not data:
                return None
            return data[0].get("cifUrl")
    except Exception:
        return None

async def download_one(
    session: aiohttp.ClientSession, 
    acc: str, 
    output_dir: Path, 
    semaphore: asyncio.Semaphore,
    logs: dict
) -> str:
    """
    Downloads structural data for one accession.
    Returns status string: "OK", "SKIP", "NO_STRUCTURE", "FAIL"
    """
    out_file = output_dir / f"{acc}.cif"

    # Check existence
    if out_file.exists() and is_valid_cif(out_file):
        return "SKIP"

    async with semaphore:
        # Get URL
        cif_url = await get_cif_url(session, acc)
        if not cif_url:
            logs["missing"].write(f"{acc}\n")
            return "NO_STRUCTURE"

        # Download with retries
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(cif_url, timeout=TIMEOUT_SEC) as response:
                    if response.status == 200:
                        content = await response.read()
                        with out_file.open("wb") as f:
                            f.write(content)
                        
                        # Validate
                        if is_valid_cif(out_file):
                            logs["ok"].write(f"{acc}\n")
                            return "OK"
                        else:
                            # If invalid, remove and retry
                            out_file.unlink(missing_ok=True)
            except Exception as e:
                # Retry on connection error
                if attempt == MAX_RETRIES - 1:
                    logs["fail"].write(f"{acc} (error: {str(e)})\n")
                    return "FAIL"
                await asyncio.sleep(0.5 * (attempt + 1))
        
        # If exhausted retries (should be caught above, but safety net)
        logs["fail"].write(f"{acc} (max retries)\n")
        return "FAIL"

async def main_async(args):
    accession_list_path = Path(args.accession_list)
    out_dir = Path(args.output_dir)
    workers = args.workers

    if not accession_list_path.exists():
        logger.error(f"Accession list not found: {accession_list_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read accessions
    accessions = [line.strip() for line in accession_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        accessions = accessions[:int(args.limit)]
    
    total = len(accessions)
    logger.info(f"Target Accessions: {total}")
    logger.info(f"Output Directory: {out_dir}")
    logger.info(f"Max Workers: {workers}")

    # Open log files
    # We'll use simple file handles for logging success/fail
    # Note: In true async high-concurrency, writing to shared file handles relies on thread-safety of write, 
    # but here we are in a single thread event loop. Standard file I/O is blocking but minimal overhead for logs.
    # To be perfectly safe/fast, we could collect results and write in chunks, but direct write is simpler for real-time logs.
    
    ok_log = open("structure_download_ok.txt", "a", encoding="utf-8", buffering=1)
    missing_log = open("missing_af_structure.txt", "a", encoding="utf-8", buffering=1)
    fail_log = open("download_fail_af_structure.txt", "a", encoding="utf-8", buffering=1)

    logs = {"ok": ok_log, "missing": missing_log, "fail": fail_log}

    semaphore = asyncio.Semaphore(workers)
    
    # Session
    conn = aiohttp.TCPConnector(limit=None) # limit handled by semaphore
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            download_one(session, acc, out_dir, semaphore, logs)
            for acc in accessions
        ]
        
        results = {
            "OK": 0,
            "SKIP": 0,
            "NO_STRUCTURE": 0,
            "FAIL": 0
        }
        
        # Use tqdm for progress bar
        for f in tqdm(asyncio.as_completed(tasks), total=total, desc="Downloading", unit="pdb"):
            status = await f
            results[status] += 1

    # Close logs
    ok_log.close()
    missing_log.close()
    fail_log.close()

    logger.info("===== Final Summary =====")
    logger.info(f"OK: {results['OK']}")
    logger.info(f"SKIP (Already existed): {results['SKIP']}")
    logger.info(f"NO_STRUCTURE: {results['NO_STRUCTURE']}")
    logger.info(f"FAIL: {results['FAIL']}")
    logger.info("=========================")

def main():
    parser = argparse.ArgumentParser(description="Async AlphaFold Structure Downloader")
    parser.add_argument("--accession_list", required=True, help="Path to text file with one accession per line")
    parser.add_argument("--output_dir", required=True, help="Directory to save CIF files")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent downloads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of accessions to process (for testing)")
    
    args = parser.parse_args()
    
    # Run async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopped manually.")

if __name__ == "__main__":
    main()
