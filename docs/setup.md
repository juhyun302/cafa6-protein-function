# 설치와 전체 실행 순서

## 실행 범위

기본 CLI는 구조 기반 기능 전이와 앙상블을 다룹니다. Python 3.10 이상과 NumPy·pandas·SciPy가 필요합니다.

~~~bash
python -m venv .venv
~~~

가상환경 활성화:
- Windows PowerShell: .\.venv\Scripts\Activate.ps1
- Linux/WSL: source .venv/bin/activate

~~~bash
python -m pip install -r requirements.txt
python scripts/cafa.py --help
python scripts/cafa.py demo
~~~

패키지 설치를 선호하면 python -m pip install -e . 이후 cafa-pipeline 명령도 사용할 수 있습니다.

## 입력 준비

실제 대회 데이터는 data/raw/에 둡니다. 아래 명령은 저장소 루트 기준 한 줄 명령입니다.

- data/raw/query.fasta: 예측할 단백질 서열
- data/raw/annotations.tsv: 참조 단백질의 EntryID, term, aspect
- data/raw/go-parents.tsv: child, parent 컬럼의 직접 GO 부모 관계
- 또는 go_term_mappings.pkl, go_ancestor_matrix.npz: 기존 전처리 산출물
- data/raw/baseline.tsv, other-model.tsv: 별도로 생성한 다른 모델의 예측

입력 형식은 [데이터 안내](data-and-glossary.md)를 참고합니다. 작은 examples/ 파일은 가상 데이터입니다.

## 1. ID 추출과 구조 다운로드

~~~bash
python scripts/cafa.py extract-accessions --fasta data/raw/query.fasta --output data/processed/accessions.txt
python scripts/cafa.py download --accessions data/processed/accessions.txt --output-dir data/processed/structures --log-dir artifacts/download --workers 8 --limit 100
~~~

전체 다운로드는 --limit 100을 제거합니다. 이미 유효한 CIF 파일이 있으면 SKIP합니다. 로그는 artifacts/download/download-status.tsv입니다.

- OK: 다운로드·파일 검증 성공
- SKIP: 이미 유효한 구조가 있음
- NO_STRUCTURE: API가 404 또는 구조가 없는 정상 응답을 반환
- FAIL: 재시도 후에도 조회·다운로드 실패

FAIL이 있으면 CLI는 비정상 종료 코드를 반환합니다. 통신 실패를 구조 부재로 처리하지 않습니다. 일시적 서버 응답의 의미는 실제 데이터 준비 시 다시 확인할 수 있습니다.

기존 제외 목록을 재사용하는 경우:
~~~bash
python scripts/cafa.py filter-accessions --input data/processed/accessions.txt --exclude data/raw/exclude-accessions.txt --output data/processed/filtered-accessions.txt
~~~

## 2. Foldseek 구조 검색

Foldseek 바이너리는 별도로 설치합니다. 사용한 릴리스·버전, 참조 구조 데이터의 기준일을 실험 기록에 남깁니다. 코드가 실행 중 임의의 최신 바이너리를 내려받지는 않습니다.

Linux/WSL 또는 Azure Linux에서:
~~~bash
python scripts/cafa.py search --structures-dir data/processed/structures --reference-dir data/raw/reference-structures --output-dir artifacts/hits --threads 32 --batch-size 5000 --max-seqs 20 --foldseek-bin foldseek
python scripts/cafa.py merge-hits --input-dir artifacts/hits --output artifacts/foldseek-hits.tsv
~~~

--reference-dir를 생략하면 query 구조 폴더를 참조 DB로도 사용합니다. 기능 전이에는 참조 단백질과 일치하는 annotation이 필요합니다. 자기 자신 hit는 ID 정리 후 제거합니다.

배치에는 심볼릭 링크를 사용합니다. CIF 폴더 입력을 받으며 ZIP은 먼저 압축 해제해야 합니다. 기존 batch 결과가 있는 출력 폴더는 덮어쓰지 않습니다. 작업용 임시 DB는 작업마다 새 임시 폴더에서 생성·정리합니다.

## 3. GO 기능 전이와 정규화

~~~bash
python scripts/cafa.py transfer --hits artifacts/foldseek-hits.tsv --annotations data/raw/annotations.tsv --output artifacts/raw-go-scores.tsv
python scripts/cafa.py normalize --input artifacts/raw-go-scores.tsv --output-dir artifacts/normalized --top-k 50 --min-score 0.2
~~~

전이 점수는 유사도 가중치의 합입니다. 정규화는 단백질별 최대값으로 나누며, Top-K는 namespace 분리 전 단백질의 전체 GO 후보에 적용합니다. 이 상대 점수는 보정된 정답 확률이 아닙니다.

정규화 출력: pred_bp.tsv, pred_mf.tsv, pred_cc.tsv. 모든 중간 TSV에는 명시적인 헤더가 있습니다.

## 4. GO 상향 전파

작은 직접 부모 관계 파일:
~~~bash
python scripts/cafa.py propagate --input-dir artifacts/normalized --output-dir artifacts/propagated --parents data/raw/go-parents.tsv --workers 1
~~~

기존 pickle·희소 행렬:
~~~bash
python scripts/cafa.py propagate --input-dir artifacts/normalized --output-dir artifacts/propagated --mapping-pkl data/raw/go_term_mappings.pkl --ancestor-npz data/raw/go_ancestor_matrix.npz --workers 32
~~~

행렬은 행=자식, 열=조상이며 GO 매핑 인덱스와 정확히 일치해야 합니다. --workers 1은 로컬 확인용입니다. 병렬 프로세스마다 매핑을 준비하므로 실제 머신의 메모리에 맞춰 worker 수를 조정합니다.

## 5. 구조 예측 제출 형식 생성

~~~bash
python scripts/cafa.py make-submission --input-dir artifacts/propagated --output artifacts/structure-submission.tsv --top-n 500
~~~

각 BP/MF/CC에서 단백질당 Top-N을 고른 후 합칩니다. 따라서 단백질의 전체 GO 후보가 500개로 제한되는 것은 아닙니다. 최종 파일은 헤더 없는 3열입니다. 실제 대회 규칙에 맞춘 추가 제한은 제출 전에 별도로 확인합니다.

## 6. 다른 모델과 앙상블

~~~bash
python scripts/cafa.py patch --base data/raw/baseline.tsv --patch artifacts/structure-submission.tsv --threshold 0.99 --output artifacts/patched.tsv
python scripts/cafa.py rank-ensemble --base artifacts/patched.tsv --other data/raw/other-model.tsv --patch artifacts/structure-submission.tsv --threshold 0.99 --output artifacts/submission.tsv
python scripts/cafa.py validate --input artifacts/submission.tsv
~~~

0.99는 선택 임계값입니다. 순위 평균은 기존 구현처럼 전체 입력 행의 전역 백분위 순위를 사용하며 BP로 한정하지 않습니다. 임계값과 결합 전략의 효과는 별도 검증셋으로 평가해야 합니다.

## 검사와 실행 한계

~~~bash
python -m unittest discover -s tests -v
~~~

가상 예제의 직렬·병렬 결과, 첫 행 보존, 전파, 필터, 앙상블, 제출 형식 등을 검사합니다. 실제 Foldseek 검색·대규모 데이터·GPU 학습·Azure 원격 Job의 수행 결과는 이 검사에 포함되지 않습니다.

정리 시 로컬 검증 환경: NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3. 의존성 파일의 하한 조건은 설치 호환 범위이며 과거 실험 환경의 완전한 lockfile이 아닙니다.
