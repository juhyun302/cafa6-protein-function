import pandas as pd
import numpy as np

# ==========================================
# 1. 파일 경로 설정 (파일명 정확한지 확인 필수!)
# ==========================================
FILE_AJ_ENSEMBLE = 'ensemble_mounted.tsv'  # 아름+주현 앙상블 결과
FILE_SUA = '수아제출용final_results.tsv'     # 수아님 예측 파일
FILE_USER = 'submission_top500_1231_1PM.tsv' # 너의 파일 (BP 패치용)

print("1. 데이터 로드 및 전처리 중...")

# 함수: 데이터를 읽고 숫자로 변환하며, 헤더가 섞여 있으면 제거함
def load_clean_data(filepath, name):
    # 일단 header=None으로 다 읽음
    df = pd.read_csv(filepath, sep='\t', header=None, names=['id', 'term', 'score'])
    
    # 'score' 컬럼을 숫자로 강제 변환 (문자는 NaN이 됨)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # NaN이 된 행(헤더였던 것) 제거
    initial_len = len(df)
    df = df.dropna(subset=['score'])
    
    if len(df) < initial_len:
        print(f"   -> [{name}] 헤더 또는 문자열 데이터 {initial_len - len(df)}개 제거됨.")
        
    return df

# 안전하게 로드
df_aj = load_clean_data(FILE_AJ_ENSEMBLE, "아름+주현")
df_sua = load_clean_data(FILE_SUA, "수아")
df_user = load_clean_data(FILE_USER, "주현")

print("2. 주현님의 고정밀 BP 패치 생성 중...")
# 0.99 이상인 것만 추출
user_patch = df_user[df_user['score'] >= 0.99].copy()

print("3. 전략적 병합 (Smart Ensemble: BP Swap + Rank Avg) 수행...")

# (1) 수아(Base) + 주현(Patch) -> BP 최적화 점수(score_bp_opt) 생성
merged_sua = pd.merge(df_sua, user_patch, on=['id', 'term'], how='outer', suffixes=('_sua', '_user'))

# 내 점수가 있으면 내 걸 쓰고, 없으면 수아 점수를 씀
merged_sua['score_bp_opt'] = np.where(merged_sua['score_user'].notna(), 
                                      merged_sua['score_user'], 
                                      merged_sua['score_sua'])
# 필요한 컬럼만 남김
df_bp_opt = merged_sua[['id', 'term', 'score_bp_opt']]

# (2) 기존 1등(아름+주현) + 수아 최적화 버전(df_bp_opt) 합치기
merged_final = pd.merge(df_aj, df_bp_opt, on=['id', 'term'], how='outer', suffixes=('_aj', '_sua'))

# 점수 없는 곳은 0.0으로 채움
merged_final['score'] = merged_final['score'].fillna(0.0)
merged_final['score_bp_opt'] = merged_final['score_bp_opt'].fillna(0.0)

print("4. 랭킹 평균(Rank Averaging) 계산 중...")
# 이제 둘 다 완벽한 float 타입이라 에러 안 날 거야
merged_final['rank_aj'] = merged_final['score'].rank(pct=True)
merged_final['rank_sua'] = merged_final['score_bp_opt'].rank(pct=True)

# 두 모델의 순위 평균
merged_final['final_score'] = (merged_final['rank_aj'] + merged_final['rank_sua']) / 2
merged_final['final_score'] = merged_final['final_score'].round(3)

print("5. 결과 저장 중...")
merged_final[['id', 'term', 'final_score']].to_csv('final_ensemble_v2.tsv', sep='\t', index=False, header=False)

print("완료! 이제 에러 없이 돌아갈 거야.")