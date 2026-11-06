# 작은 예제

query.fasta, foldseek-hits.tsv, annotations.tsv, go-parents.tsv, baseline.tsv, other-model.tsv는 처리 흐름 테스트를 위해 만든 합성 데이터입니다. DEMO_* ID와 GO:9xxxxxx 관계는 실제 단백질·기능 관계를 나타내지 않습니다.

python scripts/cafa.py demo는 이 파일들을 읽어 artifacts/demo/에 결과를 만듭니다. 실제 구조 검색·다운로드·학습·클라우드 호출은 수행하지 않습니다.

real-accessions-100.txt는 과거 실습에서 보존한 실제 단백질 ID 목록입니다. 데모에 사용하지 않습니다. download 명령의 입력으로 사용할 경우 외부 API를 호출합니다.
