# Step A-2 (TEST): AlphaFold DB 구조 존재 여부를 100개 accession으로만 점검
# 목적:
# 1) uniprot_accessions.txt에서 앞 100개만 사용
# 2) AlphaFold DB에 구조가 있는지(cifUrl 존재 여부) 확인
# 3) 있는 것 / 없는 것 / 오류를 명확히 카운트
# 4) 실제 파일은 다운로드하되, 테스트 규모로만 수행

from pathlib import Path
import json
import time
import urllib.request
import urllib.error

# ===== 설정 =====
ACCESSION_LIST = Path("uniprot_accessions.txt")  # accession 목록 파일
OUT_DIR = Path("structures_test_100")           # 테스트용 구조 저장 폴더
N_TEST = 100                                      # 테스트 개수
SLEEP_SEC = 0.1                                   # 서버 예의용 대기
# ==================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 결과 기록 파일
ok_path = Path("test_ok.txt")
missing_path = Path("test_missing.txt")
error_path = Path("test_error.txt")


def get_cif_url(acc: str):
    """AlphaFold API에서 cifUrl 가져오기. 없으면 None 반환."""
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    try:
        with urllib.request.urlopen(api, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        if not data:
            return None
        return data[0].get("cifUrl")
    except Exception:
        return None


def download(url: str, out_file: Path) -> bool:
    """cif 파일 다운로드"""
    try:
        urllib.request.urlretrieve(url, out_file)
        return True
    except Exception:
        return False


# accession 앞 100개만 읽기
accs = [line.strip() for line in ACCESSION_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
accs = accs[:N_TEST]

print(f"테스트 대상 accession 수: {len(accs)}")

ok = 0
missing = 0
error = 0

with ok_path.open("w", encoding="utf-8") as ok_f, \
     missing_path.open("w", encoding="utf-8") as miss_f, \
     error_path.open("w", encoding="utf-8") as err_f:

    for i, acc in enumerate(accs, 1):
        print(f"[{i}/{len(accs)}] {acc} 확인 중...")

        try:
            cif_url = get_cif_url(acc)
            if not cif_url:
                missing += 1
                miss_f.write(acc + "\n")
                print(f"  -> AlphaFold DB에 구조 없음")
                continue

            out_file = OUT_DIR / f"{acc}.cif"
            success = download(cif_url, out_file)

            if success and out_file.exists() and out_file.stat().st_size > 1000:
                ok += 1
                ok_f.write(acc + "\n")
                print(f"  -> 구조 다운로드 성공")
            else:
                error += 1
                err_f.write(acc + "\n")
                print(f"  -> 다운로드 오류")
                if out_file.exists():
                    out_file.unlink()

        except Exception as e:
            error += 1
            err_f.write(acc + "\n")
            print(f"  -> 예외 발생: {e}")

        time.sleep(SLEEP_SEC)

print("\n===== 테스트 결과 요약 =====")
print(f"성공(구조 존재): {ok}")
print(f"구조 없음: {missing}")
print(f"오류: {error}")
print("==========================")