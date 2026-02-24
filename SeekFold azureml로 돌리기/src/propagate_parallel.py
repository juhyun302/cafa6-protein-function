import os
import pickle
import argparse
import multiprocessing
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# ==========================================
# 1. 유틸리티 함수 (변경 없음)
# ==========================================
def _load_mappings(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if "go_to_idx" in obj and "idx_to_go" in obj: return obj["go_to_idx"], obj["idx_to_go"]
        for k1, k2 in [("go2idx", "idx2go"), ("term2idx", "idx2term")]:
            if k1 in obj and k2 in obj: return obj[k1], obj[k2]
    if isinstance(obj, (tuple, list)) and len(obj) >= 2: return obj[0], obj[1]
    raise TypeError("Unsupported pkl format.")

def _guess_columns(df):
    cols = [c.lower() for c in df.columns]
    if "query" in cols: pcol = df.columns[cols.index("query")]
    elif "protein_id" in cols: pcol = df.columns[cols.index("protein_id")]
    else: pcol = df.columns[0]
    
    if "go_id" in cols: gcol = df.columns[cols.index("go_id")]
    elif "term" in cols: gcol = df.columns[cols.index("term")]
    else: gcol = df.columns[1]
    
    if "score" in cols: scol = df.columns[cols.index("score")]
    elif "prob" in cols: scol = df.columns[cols.index("prob")]
    else: scol = df.columns[-1]
    return pcol, gcol, scol

# ==========================================
# 2. 워커 함수 (병렬 처리용)
# ==========================================
def worker_task(args):
    """하나의 프로세스가 처리할 작업"""
    sub_df, go_to_idx, idx_to_go, ancestor_csr, pcol, gcol, scol = args
    out_rows = []
    
    try:
        # 그룹바이 속도 최적화를 위해 sort=False 사용
        for pid, g in sub_df.groupby(pcol, sort=False):
            go_ids = g[gcol].astype(str).tolist()
            scores = g[scol].astype(float).to_numpy()
            
            # 매핑에 있는 GO Term만 필터링
            valid_indices = []
            valid_scores = []
            for go, sc in zip(go_ids, scores):
                if go in go_to_idx:
                    valid_indices.append(go_to_idx[go])
                    valid_scores.append(sc)
            
            if not valid_indices: continue

            child_indices = np.array(valid_indices, dtype=np.int32)
            child_scores = np.array(valid_scores, dtype=np.float32)
            
            propagated = {}
            # Max Propagation 로직
            for ci, cs in zip(child_indices, child_scores):
                # 조상 전파
                ancestors = ancestor_csr.getrow(ci).indices
                for anc in ancestors:
                    prev = propagated.get(anc)
                    if prev is None or cs > prev: propagated[anc] = cs
                # 자기 자신
                prev = propagated.get(ci)
                if prev is None or cs > prev: propagated[ci] = cs
                
            final_go = [idx_to_go[i] for i in propagated.keys()]
            final_sc = list(propagated.values())
            
            for fg, fs in zip(final_go, final_sc):
                out_rows.append((pid, fg, fs))
                
    except Exception as e:
        print(f"⚠️ Worker Error: {e}")
        return []
        
    return out_rows

# ==========================================
# 3. 메인 함수
# ==========================================
def main():
    print("🚀 Script Started: 32-Core Parallel Propagation")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # [매우 중요] 출력 폴더 강제 생성! (이게 없어서 저장이 안 된 거임)
    print(f"📂 Output Directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드
    print("⏳ Loading mapping files...")
    mapping_pkl = input_dir / "go_term_mappings.pkl"
    ancestor_npz = input_dir / "go_ancestor_matrix.npz"
    go_to_idx, idx_to_go = _load_mappings(mapping_pkl)
    ancestor = load_npz(ancestor_npz).tocsr()

    # CPU 코어 확인
    num_cores = multiprocessing.cpu_count()
    print(f"🔥 Detected CPU Cores: {num_cores}")
    
    target_files = ["pred_bp_test.tsv", "pred_mf_test.tsv", "pred_cc_test.tsv"]

    for fname in target_files:
        in_path = input_dir / fname
        if not in_path.exists():
            print(f"⚠️ {fname} not found. Skipping.")
            continue
            
        print(f"🚀 Processing {fname} with {num_cores} cores...")
        start_time = time.time()
        
        df = pd.read_csv(in_path, sep="\t")
        pcol, gcol, scol = _guess_columns(df)
        
        # 데이터 쪼개기 (코어 수의 4배수로 잘게 쪼개서 로드 밸런싱)
        unique_proteins = df[pcol].unique()
        chunks = np.array_split(unique_proteins, num_cores * 4)
        
        tasks = []
        for chunk_prots in chunks:
            sub_df = df[df[pcol].isin(chunk_prots)].copy()
            if not sub_df.empty:
                tasks.append((sub_df, go_to_idx, idx_to_go, ancestor, pcol, gcol, scol))
        
        # 병렬 실행
        results = []
        with multiprocessing.Pool(processes=num_cores) as pool:
            for res in pool.imap_unordered(worker_task, tasks): # 순서 상관없이 빨리 끝나는 대로 수집
                results.extend(res)
        
        # 저장
        print(f"💾 Saving results for {fname}...")
        out_df = pd.DataFrame(results, columns=[pcol, gcol, scol])
        
        # 최종 중복 제거 (Max)
        out_df = out_df.groupby([pcol, gcol], as_index=False)[scol].max()
        
        out_name = f"{in_path.stem}_propagated.tsv"
        save_path = output_dir / out_name
        
        out_df.to_csv(save_path, sep="\t", index=False)
        print(f"✅ Saved: {save_path}")
        print(f"⏱️ Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()