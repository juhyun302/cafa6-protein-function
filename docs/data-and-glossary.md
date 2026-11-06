# 데이터와 용어

## 처음 보는 용어

| 용어 | 의미 |
| --- | --- |
| FASTA | >로 시작하는 단백질 이름과 아미노산 서열을 담는 텍스트 형식 |
| Accession | UniProt 등에서 단백질을 구분하는 ID |
| AlphaFold DB | 미리 예측된 단백질 구조를 조회하는 공개 데이터베이스 |
| CIF/mmCIF | 3D 구조 파일 형식 |
| Foldseek | 구조 유사성을 검색하는 외부 실행 도구 |
| GO term | 단백질의 기능·과정·위치를 표현하는 분류 번호 |
| BP | Biological Process: 어떤 생물학적 과정에 참여하는지 |
| MF | Molecular Function: 어떤 분자 기능을 수행하는지 |
| CC | Cellular Component: 어떤 세포 구성요소·위치와 관련되는지 |
| GO DAG | 기능 간 상하위 관계를 담는 유향 비순환 그래프 |
| 앙상블 | 여러 방법의 예측을 결합하는 과정 |

## 실제 데이터의 위치

원본 FASTA·annotation·GO 관계·임베딩·가중치는 Git에 포함하지 않습니다. 로컬 data/raw/에 준비하고, 생성 목록은 data/processed/, 결과와 로그는 artifacts/에 둡니다.

기존 fasta_accessions.txt 네 사본은 같은 전체 목록이었습니다. 원천 FASTA에서 extract-accessions 명령으로 재생성합니다. train과 test의 목록을 섞지 않도록 입력 파일명·분할·버전을 별도로 기록합니다.

examples/real-accessions-100.txt는 과거의 작은 실제 ID 목록입니다. 데모는 이 파일을 사용하지 않으며, 이를 download 명령에 넘기면 외부 API를 호출합니다.

## 파일 스키마

### Foldseek hits

헤더 없는 7열:
query, target, evalue, bits, alntmscore, qcov, tcov

배치의 000000_ 접두사, .cif/.pdb 확장자, 표준 AF-...-F1-model_v... 파일명을 ID 정리 단계에서 처리합니다. 다른 파일명 규칙은 사전에 ID 매핑이 필요합니다.

### 참조 annotation

~~~text
EntryID    term          aspect
REF_A      GO:9000001    P
~~~

실제 파일은 탭으로 구분합니다. EntryID/term 또는 protein_id/go_id 이름을 허용합니다. aspect는 P/F/C 또는 BP/MF/CC입니다. 예제 GO:9xxxxxx와 DEMO_*는 합성 데이터이며 실제 생물학적 관계를 뜻하지 않습니다.

### 중간 예측

헤더 있는 4열:
protein_id, go_id, aspect, score

과거의 헤더 없는 3열과 score_norm 컬럼도 읽을 수 있습니다. score와 score_norm이 함께 있으면 정규화한 score_norm을 우선 사용합니다. 새 중간 파일은 공통 헤더를 사용합니다.

### 최종 제출

헤더 없는 3열:
protein_id, go_id, score

score는 유한한 0~1 값이며 같은 단백질·GO 쌍의 중복이 없어야 합니다. validate는 이 기본 형식을 검사하며 대회 서버의 전체 평가를 대신하지 않습니다.

### GO 부모 관계와 기존 행렬

작은 부모 TSV: child, parent 헤더.

희소 행렬: 행 i가 자식 term, 열 j가 그 조상을 나타내는 CSR 행렬.
pickle 매핑은 go_to_idx/idx_to_go, go2idx/idx2go, term2idx/idx2term, term_to_idx/idx_to_term 또는 두 매핑의 tuple/list를 지원합니다. pickle은 본인이 준비한 전처리 산출물로 사용합니다.

## 과거 탐색 메모

원본 메모에는 학습 단백질 82,404개, annotation 537,027행, GO term 26,125종 등이 기록되어 있었습니다. 입력 파일 버전이 현재 저장소에 남아 있지 않아 재집계하지 않았으며 전체 대회 데이터의 고정 규모로 주장하지 않습니다.

서열 길이·희귀 라벨·불균형은 notebooks/fasta-exploration.ipynb와 당시 학습 메모의 분석 주제입니다. 통계 수치를 새 데이터셋에 적용할 때는 다시 집계해야 합니다.
