# 파일 정리 이력

아래 표는 정리 전 54개 파일의 이동·통합·생성물 제외 내역입니다. 이주현의 구조 기반 파이프라인·앙상블과 홍지승이 작성한 서열 모델 실험을 구분해 보존했습니다.

## 폐기된 Azure 환경과 설정 기록

당시 Azure 계정과 리소스는 폐기되어 복구할 수 없습니다. 현재 설정 파일은 빈 예제 템플릿만 제공하며, 클라우드 실행에는 본인의 새 환경이 필요합니다.

과거 Git 이력의 실제 subscription_id·resource_group·workspace_name 값은 식별 가능한 원래 문자열 대신 치환값으로 정리했습니다. 코드와 작업 이력은 보존하되, 이력 재작성으로 커밋 ID가 변경되었습니다. 기존 clone에서 작업을 계속할 때는 정리된 원격 저장소를 새로 받아 사용하는 것을 권장합니다.

## 동작을 명확히 한 부분

- 헤더 없는 예측 TSV의 첫 데이터 행을 보존합니다. 새 중간 파일에는 공통 헤더를 씁니다.
- 숫자 prefix·구조 파일명을 정리한 뒤 self-hit를 제외합니다.
- score_norm이 있으면 원시 합산 score보다 우선합니다.
- 높은 점수 패치가 비어 있어도 기본 예측을 반환합니다.
- 구조 없음(404·정상 빈 응답)과 통신 실패를 구분하고 후자를 재시도합니다.
- 전파는 max 방식, 제출 제한은 namespace별 Top-N, 순위 앙상블은 전체 행 대상이라는 기존 의미를 명시했습니다.
- 원본 annotation의 중복 라벨과 제출 쌍의 중복을 정리합니다.
- Foldseek 실행 파일은 자동 다운로드하지 않고 설치된 버전을 명시합니다.
- 기준 실험 코드 11개는 보존하고 이름·import만 정리했습니다. 과거 노트북의 출력은 지웠습니다.

## 생성물과 예제

전체 accession 목록·필터 결과·다운로드 로그는 재생성 가능한 산출물로 추적에서 제외했습니다. 예제 ID 100개는 보존했고, 기본 데모에는 외부 API를 부르지 않는 합성 데이터를 추가했습니다.

## 기존 파일 대응표

| 이전 파일 | 현재 위치 또는 대체 명령 |
| --- | --- |
| # 2차 프로젝트 알아야할 단어들.txt | docs/data-and-glossary.md |
| .gitignore | .gitignore |
| 2차 pj 사진/asdf.ipynb | src/cafa_pipeline/structure.py: filter_accessions |
| 2차 pj 사진/emsemble_mounted.py | src/cafa_pipeline/ensemble.py: patch_scores |
| 2차 pj 사진/emsemble주현아름수아.py | src/cafa_pipeline/ensemble.py: rank_ensemble |
| 2차 pj 사진/fasta_accessions.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| 2차 pj 사진/fasta_accessions_filtered.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| 2차 pj 사진/no_alphafold_structure.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| ESM, 1D CNN(지승님)/ESM2_chunk_tokenizer.py | models/sequence_baselines/esm2_3b_chunk_embeddings.py |
| ESM, 1D CNN(지승님)/ESM2_tokenizer.py | models/sequence_baselines/esm2_650m_embeddings.py |
| ESM, 1D CNN(지승님)/ancestral_esonance.py | models/sequence_baselines/ancestral_resonance.py |
| ESM, 1D CNN(지승님)/attention_mil_model.py | models/sequence_baselines/attention_mil_model.py |
| ESM, 1D CNN(지승님)/extract_chunk_embeddings.py | models/sequence_baselines/extract_chunk_embeddings.py |
| ESM, 1D CNN(지승님)/h5_data_loder.py | models/sequence_baselines/h5_data_loader.py |
| ESM, 1D CNN(지승님)/model.py | models/sequence_baselines/hierarchical_cnn.py |
| ESM, 1D CNN(지승님)/predict_submission.py | models/sequence_baselines/predict_submission.py |
| ESM, 1D CNN(지승님)/train.py | models/sequence_baselines/train_cnn.py |
| ESM, 1D CNN(지승님)/train_attention_mil.py | models/sequence_baselines/train_attention_mil.py |
| ESM, 1D CNN(지승님)/validation.py | models/sequence_baselines/evaluate.py |
| README.md | README.md |
| SeekFold azureml로 돌리기/azure_ml_alphafold_download_submit.ipynb | scripts/cafa.py + src/cafa_pipeline/ + azureml/submit.py |
| SeekFold azureml로 돌리기/config.json | configs/azureml.example.json |
| SeekFold azureml로 돌리기/env/conda.yaml | environments/structure.yml |
| SeekFold azureml로 돌리기/fasta_accessions.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| SeekFold azureml로 돌리기/no_alphafold_structure.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| SeekFold azureml로 돌리기/src/foldseek_driver.py | src/cafa_pipeline/search.py |
| SeekFold azureml로 돌리기/src/make_submission.py | src/cafa_pipeline/structure.py: make_submission |
| SeekFold azureml로 돌리기/src/propagate_parallel.py | src/cafa_pipeline/structure.py: propagate |
| azure 최종파일/download_alphafold_structure.py | src/cafa_pipeline/download.py |
| azure 최종파일/fasta_accessions.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| azureml 사용법__alphafold_db_search/azure_ml_alphafold_download_submit.ipynb | azureml/submit.py --stage download |
| azureml 사용법__alphafold_db_search/azure_ml_fc_mil_submit.ipynb | azureml/experimental/submit-fc-mil.ipynb |
| azureml 사용법__alphafold_db_search/config.json | configs/azureml.example.json |
| azureml 사용법__alphafold_db_search/env/conda.yaml | environments/structure.yml |
| azureml 사용법__alphafold_db_search/src/download_alphafold_structure.py | src/cafa_pipeline/download.py |
| azureml 사용법__alphafold_db_search/src/fasta_accessions.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| code/structure_go_transfer.py | src/cafa_pipeline/structure.py: transfer_go |
| code/structure_go_transfer_test.py | src/cafa_pipeline/structure.py: transfer_go |
| code/실습/FASTA(단백질서열)읽기.ipynb | notebooks/fasta-exploration.ipynb |
| code/실습/Step_A2_Test_100_Accessions.py | scripts/cafa.py download --limit 100 |
| code/실습/download_af_resume.py | src/cafa_pipeline/download.py |
| code/실습/download_ok.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| code/실습/fasta에서 accession뽑기.ipynb | scripts/cafa.py extract-accessions |
| code/실습/missing_af.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| code/실습/test_missing.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| code/실습/train_term해부.ipynb | docs/data-and-glossary.md |
| code/실습/uniprot_accessions.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| 대본.txt | docs/structure-method.md |
| 지승님 DAG 트리구조.ipynb | notebooks/sequence-baseline-analysis.ipynb |
| 최종 실습/100_fasta_accessions.txt | examples/real-accessions-100.txt |
| 최종 실습/FASTA_accession.ipynb | scripts/cafa.py extract-accessions |
| 최종 실습/download_alphafold_structure.py | src/cafa_pipeline/download.py |
| 최종 실습/missing_af_structure.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |
| 최종 실습/structure_download_ok.txt | Git 추적 제외; data/processed/ 또는 artifacts/에서 재생성 |

## 검증 범위

기본 처리 파이프라인의 단위·통합 테스트, 가상 데이터 직렬·병렬 실행, 파일 형식 검사를 추가했습니다. Azure 원격 실행, 실제 Foldseek 검색, GPU 학습, 전체 대회 데이터 성능은 별도로 검증해야 합니다.

발표 PDF 원본은 업로드하지 않았습니다. README와 설명 문서에는 개인 식별자가 포함되지 않은 파이프라인·처리 전략·GO 설명 슬라이드 3개를 선별해 사용했습니다.
