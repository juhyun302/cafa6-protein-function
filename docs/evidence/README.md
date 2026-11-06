# 과거 작업의 근거 자료

원래 저장소에 보존된 코드·실행 출력·ID 목록에서 공개 가능한 관찰값만 추렸습니다. Azure 식별자·작업 URL·개인 PC 경로·원본 채팅·개인 사진은 이 폴더에 복사하지 않았습니다.

## 기계 판독 가능한 관찰값

[historical-observations.json](historical-observations.json)

- 원본 commit·파일 경로·blob SHA
- 노트북의 0부터 시작하는 셀 번호
- 저장된 배치 병합·GO 전이·namespace 분리 출력의 수치
- 입력 ID 목록과 성공·미확보 목록의 집합 대조 결과
- 같은 실행으로 연결할 수 없는 별도 Markdown 요약

JSON의 saved_outputs는 당시 노트북에 저장된 출력입니다. 지금 검색·학습을 재실행한 결과가 아닙니다. identifier_list_checks는 Git 이력의 실제 목록을 읽어 다시 계산한 결과입니다. 합성 데모의 결과와도 구분합니다.

## 출처

설정값을 치환한 현재 Git 이력의 초기 commit을 기준으로 합니다. 해당 이력을 다시 정리하면 commit ID·링크가 달라질 수 있습니다.

- [구조 검색 코드](https://github.com/juhyun302/cafa6-protein-function/blob/f6b2606d54afb9ee1aa54dcfc91f1b9103102cd0/SeekFold%20azureml%EB%A1%9C%20%EB%8F%8C%EB%A6%AC%EA%B8%B0/src/foldseek_driver.py)
- [Azure 제출·결과 처리 노트북](https://github.com/juhyun302/cafa6-protein-function/blob/f6b2606d54afb9ee1aa54dcfc91f1b9103102cd0/SeekFold%20azureml%EB%A1%9C%20%EB%8F%8C%EB%A6%AC%EA%B8%B0/azure_ml_alphafold_download_submit.ipynb)
- [원래 GO 기능 전이 코드](https://github.com/juhyun302/cafa6-protein-function/blob/f6b2606d54afb9ee1aa54dcfc91f1b9103102cd0/code/structure_go_transfer.py)

ID 목록별 경로와 blob SHA는 JSON의 sources에 기록했습니다. 기존 파일이 현재 어디로 통합되었는지는 [파일 대응표](../reorganization.md)를 참고하세요.

## 이 자료로 확정하지 않는 내용

- 원본 전체 실험의 완전 재현
- 실측 비용·속도 개선율
- 초기 내부 F-max의 공식 평가 적합성
- 0.99 점수의 확률 보정·최적성
- 구조 branch 단독의 대회 점수 개선 폭

[개인 기여](../contributions.md) · [프로젝트 경과](../project-history.md) · [팀 성과](../experiments.md)
