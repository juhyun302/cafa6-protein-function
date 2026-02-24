import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="propagated tsv 파일들이 있는 폴더")
    parser.add_argument("--output_dir", type=str, required=True, help="결과 저장 폴더")
    return parser.parse_args()

def load_pred(path: Path):
    """TSV 로드 및 컬럼명 통일"""
    print(f"📖 Loading {path.name}...")
    # 메모리 절약을 위해 필요한 컬럼만 로드하거나 타입을 지정하면 좋음
    df = pd.read_csv(path, sep="\t")
    
    cols = [c.lower() for c in df.columns]
    
    # 컬럼 찾기 함수
    def pick(cands, default_idx):
        for c in cands:
            if c in cols: return df.columns[cols.index(c)]
        return df.columns[default_idx]

    pcol = pick(["protein_id", "query", "protein"], 0)
    gcol = pick(["go_id", "go", "term"], 1)
    scol = pick(["score", "score_norm", "prob", "confidence"], -1)

    # 필요한 컬럼만 뽑아서 이름 통일
    df = df[[pcol, gcol, scol]].copy()
    df.columns = ["protein_id", "go_id", "score"]
    return df

def top_n_per_protein(df: pd.DataFrame, n: int):
    """단백질 별 Top-N 추출"""
    # 점수 높은 순 정렬
    df = df.sort_values(["protein_id", "score"], ascending=[True, False])
    # 상위 N개 자르기
    return df.groupby("protein_id", sort=False).head(n).copy()

def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    TOP_N = 500
    
    # 1. 파일 경로 설정 (이전 단계에서 만든 파일명과 일치해야 함)
    # 파일명이 조금 달라도 찾을 수 있게 glob 사용 추천하지만, 여기선 고정함
    in_bp = input_dir / "pred_bp_test_propagated.tsv"
    in_mf = input_dir / "pred_mf_test_propagated.tsv"
    in_cc = input_dir / "pred_cc_test_propagated.tsv"

    # 2. 로드 및 Top-500 필터링
    # (메모리 관리를 위해 하나 처리하고 변수 덮어쓰기 고려 가능하지만, 32코어 머신 믿고 진행)
    
    print("🚀 Processing BP...")
    if in_bp.exists():
        bp = top_n_per_protein(load_pred(in_bp), TOP_N)
        print(f"   BP: {bp['protein_id'].nunique()} proteins, {len(bp)} rows")
    else:
        print("⚠️ BP file not found!")
        bp = pd.DataFrame()

    print("🚀 Processing MF...")
    if in_mf.exists():
        mf = top_n_per_protein(load_pred(in_mf), TOP_N)
        print(f"   MF: {mf['protein_id'].nunique()} proteins, {len(mf)} rows")
    else:
        print("⚠️ MF file not found!")
        mf = pd.DataFrame()

    print("🚀 Processing CC...")
    if in_cc.exists():
        cc = top_n_per_protein(load_pred(in_cc), TOP_N)
        print(f"   CC: {cc['protein_id'].nunique()} proteins, {len(cc)} rows")
    else:
        print("⚠️ CC file not found!")
        cc = pd.DataFrame()

    # 3. 합치기
    print("🔄 Concatenating...")
    # Rank Score 방식 대신, 원래 Score를 유지하는 것이 일반적임.
    # 만약 Rank Score를 쓰고 싶다면 아래 주석을 해제하고 위를 주석 처리.
    sub = pd.concat([bp, mf, cc], ignore_index=True)

    # 4. 중복 제거 (protein, go가 겹칠 경우 Max score 채택)
    # Aspect가 다르면 GO ID가 안 겹치는 게 정상이지만, 혹시 모르니 안전장치
    sub = sub.groupby(["protein_id", "go_id"], as_index=False)["score"].max()

    # 5. 소수점 정리 (용량 줄이기, 소수점 3자리)
    sub["score"] = sub["score"].round(3)

    # 6. 저장 (헤더 없이)
    out_path = output_dir / "submission_top500.tsv"
    sub.to_csv(out_path, sep="\t", index=False, header=False)
    
    print(f"✅ Final Submission Saved: {out_path}")
    print(f"📊 Total Rows: {len(sub)}")
    print(f"👤 Total Unique Proteins: {sub['protein_id'].nunique()}")

if __name__ == "__main__":
    main()