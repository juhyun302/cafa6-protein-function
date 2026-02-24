import pandas as pd
import numpy as np

# 파일 경로 (실제 경로에 맞게 수정)
FILE_USER = 'submission_top500_1231_1PM.tsv'  # 너의 대용량 파일
FILE_AREUM = 'interpro_submission_8.tsv'      # 아름님 파일

# 1. 아름님 파일 로드 (베이스)
print("1. 아름님 데이터(Base) 로드 중...")
df_areum = pd.read_csv(FILE_AREUM, sep='\t', header=None, names=['id', 'term', 'score'])

# 2. 너의 파일 로드 및 필터링 (패치)
print("2. 내 데이터(Patch) 로드 및 고정밀 구간(>=0.99) 추출 중...")
# 청크 단위로 읽어서 메모리 절약 (파일이 크니까)
chunk_size = 500000
high_conf_list = []

for chunk in pd.read_csv(FILE_USER, sep='\t', header=None, names=['id', 'term', 'score'], chunksize=chunk_size):
    # 점수가 0.99 이상인 것만 '확실한 정보'로 간주하고 추출
    filtered = chunk[chunk['score'] >= 0.99].copy()
    if not filtered.empty:
        high_conf_list.append(filtered)

df_user_high = pd.concat(high_conf_list)
print(f"   -> 추출된 고정밀 예측 개수: {len(df_user_high)}개")

# 3. 데이터 병합 (Outer Join)
print("3. 마운트(Mount) 작업 수행 중...")
# id와 term을 기준으로 병합. suffixes는 (_base, _patch)
merged = pd.merge(df_areum, df_user_high, on=['id', 'term'], how='outer', suffixes=('_base', '_patch'))

# 4. 점수 결정 로직 (Overwrite)
# 내 거(patch)가 있으면 내 점수를 쓰고, 없으면 아름님 거(base)를 씀
merged['final_score'] = np.where(merged['score_patch'].notna(), merged['score_patch'], merged['score_base'])

# 5. 결과 저장
print("4. 결과 파일 저장 중...")
output = merged[['id', 'term', 'final_score']]
# 점수 포맷 깔끔하게 정리 (필요시)
output['final_score'] = output['final_score'].round(3)

output.to_csv('ensemble_mounted.tsv', sep='\t', index=False, header=False)

print("완료! 'ensemble_mounted.tsv' 파일이 생성되었습니다.")