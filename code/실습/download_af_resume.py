"""
download_alphafold_structure.py

목표:
1) uniprot_accessions.txt에서 앞 N개만(예: 10000개) 먼저 다운로드
2) 나중에 N을 늘리거나(예: 224309) 혹은 N=None(전체)로 다시 실행하면
   이미 받은 파일은 자동으로 SKIP하고 나머지만 이어받기(resume)

출력:
- structures_af/<ACC>.cif  (다운로드된 구조 파일)
- download_ok.txt          (성공한 accession 로그)
- missing_af.txt           (AlphaFold DB에 구조 없는 accession 로그)
- download_fail.txt        (다운로드 실패/오류 accession 로그)
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# ✅ 설정 (여기만 바꾸면 됨)
# =========================
ACCESSION_LIST = Path("fasta_accessions.txt")  # accession 목록 파일
OUT_DIR = Path("structures_alphafold")                  # 구조 저장 폴더(계속 재사용)
N_LIMIT = None                                  # ✅ 이번에 받을 개수 (1만 먼저)
# N_LIMIT = 80000                                # 나중에 8만 이어받기
# N_LIMIT = None                                 # 전체 이어받기

MAX_WORKERS = 8                                  # 병렬 다운로드 개수(8 추천)
SLEEP_SEC = 0.05                                 # 서버 예의(너무 빠르면 차단 가능)
MIN_BYTES = 1200                                 # 너무 작은 파일(에러 XML 등) 방지 기준
# =========================

OUT_DIR.mkdir(parents=True, exist_ok=True)

OK_LOG = Path("download_ok.txt")
MISSING_LOG = Path("missing_af.txt")
FAIL_LOG = Path("download_fail.txt")


def is_valid_cif(path: Path) -> bool:
    """mmCIF 파일인지 간단 검증 (NoSuchKey 같은 XML 저장 방지)"""
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:300].lower()
        if "<error>" in head or "nosuchkey" in head:
            return False
        # 보통 mmCIF는 data_ 로 시작
        if "data_" in head[:20]:
            return True
        return False
    except Exception:
        return False


def get_cif_url(acc: str) -> str | None:
    """AlphaFold API에서 cifUrl 얻기. 없으면 None."""
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    try:
        with urllib.request.urlopen(api, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        if not data:
            return None
        return data[0].get("cifUrl")
    except Exception:
        return None


def download_one(acc: str) -> tuple[str, str]:
    """
    returns: (acc, status)
    status in {"SKIP", "OK", "NO_STRUCTURE", "FAIL"}
    """
    out_file = OUT_DIR / f"{acc}.cif"

    # ✅ 이미 받아둔 건 스킵 (이어받기의 핵심)
    if out_file.exists() and out_file.stat().st_size >= MIN_BYTES and is_valid_cif(out_file):
        return acc, "SKIP"

    cif_url = get_cif_url(acc)
    if not cif_url:
        return acc, "NO_STRUCTURE"

    try:
        urllib.request.urlretrieve(cif_url, out_file)
        time.sleep(SLEEP_SEC)

        # 검증 실패면 삭제(오염 방지)
        if (not out_file.exists()) or out_file.stat().st_size < MIN_BYTES or (not is_valid_cif(out_file)):
            try:
                if out_file.exists():
                    out_file.unlink()
            except Exception:
                pass
            return acc, "FAIL"

        return acc, "OK"

    except Exception:
        try:
            if out_file.exists():
                out_file.unlink()
        except Exception:
            pass
        return acc, "FAIL"


def load_accessions(path: Path, n_limit: int | None) -> list[str]:
    accs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if n_limit is not None:
        accs = accs[:n_limit]
    return accs


def append_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    accs = load_accessions(ACCESSION_LIST, N_LIMIT)
    total = len(accs)
    print(f"대상 accession 수: {total} (N_LIMIT={N_LIMIT})")
    print(f"저장 폴더: {OUT_DIR.resolve()}")
    print(f"병렬 workers: {MAX_WORKERS}")

    stats = {"OK": 0, "SKIP": 0, "NO_STRUCTURE": 0, "FAIL": 0}

    # 로그 파일은 "append"로 누적 (여러 번 실행해도 이어서 기록)
    OK_LOG.touch(exist_ok=True)
    MISSING_LOG.touch(exist_ok=True)
    FAIL_LOG.touch(exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(download_one, acc) for acc in accs]

        for i, fut in enumerate(as_completed(futures), 1):
            acc, status = fut.result()
            stats[status] += 1

            if status == "OK":
                append_line(OK_LOG, acc)
            elif status == "NO_STRUCTURE":
                append_line(MISSING_LOG, acc)
            elif status == "FAIL":
                append_line(FAIL_LOG, acc)

            if i % 200 == 0 or i == total:
                print(f"[{i}/{total}] OK={stats['OK']} SKIP={stats['SKIP']} "
                      f"NO_STRUCTURE={stats['NO_STRUCTURE']} FAIL={stats['FAIL']}")

    print("\n===== 최종 요약 =====")
    print(f"OK(다운로드 성공): {stats['OK']}")
    print(f"SKIP(이미 존재): {stats['SKIP']}")
    print(f"NO_STRUCTURE(구조 없음): {stats['NO_STRUCTURE']}")
    print(f"FAIL(오류): {stats['FAIL']}")
    print("=====================")


if __name__ == "__main__":
    main()
