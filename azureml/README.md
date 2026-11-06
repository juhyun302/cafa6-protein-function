# Azure ML 작업 제출

기본 예제는 Azure 없이 실행합니다. 이 폴더는 실제 클라우드 작업이 필요할 때만 사용합니다.

**프로젝트 당시 Azure 계정과 리소스는 폐기되어 복구할 수 없습니다.** 여기에 남은 코드는 제출 방식을 설명하는 참고 구현입니다. 실제 실행에는 본인의 새 Azure 환경을 준비해야 하며, 과거 리소스 이름·식별자를 재사용하지 않습니다.

## 준비

1. python -m pip install -e ".[azure]"로 제출 SDK를 설치합니다.
2. configs/azureml.example.json을 configs/azureml.json으로 복사하고 본인의 subscription_id, resource_group, workspace_name을 입력합니다.
3. DefaultAzureCredential이 사용할 인증을 준비합니다.
4. environments/structure.yml을 바탕으로 Azure 환경을 등록하고 명시적인 버전을 사용합니다.
5. 실제 compute 이름과 데이터 폴더 자산 URI를 지정합니다.

Foldseek 검색 환경에는 해당 실행 파일도 설치되어 있어야 합니다. 구조 처리 YAML 자체는 바이너리를 설치하지 않습니다.

## 제출 전에 명령만 확인

~~~bash
python azureml/submit.py --stage propagate --data azureml:propagation-input:1 --compute my-cpu-cluster --environment azureml:cafa-structure:1 --workers 32 --dry-run
~~~

--dry-run은 Azure SDK를 불러오거나 원격 Job을 만들지 않습니다. 실제 제출은 --dry-run을 제거합니다.

## 지원 단계와 입력 폴더

| stage | 필요한 폴더 내용 |
| --- | --- |
| download | accessions.txt |
| search | CIF 구조 파일들. ZIP은 사전 압축 해제 |
| propagate | pred_bp.tsv, pred_mf.tsv, pred_cc.tsv, go_term_mappings.pkl, go_ancestor_matrix.npz |
| make-submission | pred_bp_propagated.tsv, pred_mf_propagated.tsv, pred_cc_propagated.tsv |

mode=download로 데이터를 가져오고 결과는 명시적인 uri_folder 출력에 upload합니다. 원래 팀의 다른 학습 작업에서 사용한 ro_mount 전략과 구분됩니다. compute·environment·데이터 버전을 코드에 고정하지 않고 인자로 받습니다.

experimental/submit-fc-mil.ipynb는 당시 제출 기록입니다. 필요한 학습 원본 파일이 현재 없으므로 바로 실행 가능한 학습 예제로 안내하지 않습니다.
