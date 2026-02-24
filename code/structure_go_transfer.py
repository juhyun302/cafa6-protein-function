import pandas as pd
from collections import defaultdict

# ===== 입력 파일 경로 =====
FOLDSEEK_HITS = "foldseek_hits.tsv"          # 현재 폴더(code)에 있다고 가정
TRAIN_TERMS   = "../Train/train_terms.tsv"    # MS_2nd_Project/Train/train_terms.tsv
OUT_FILE      = "structure_go_scores.tsv"

# ===== 1) Foldseek hits 로드 =====
hits = pd.read_csv(
    FOLDSEEK_HITS,
    sep="\t",
    header=None,
    names=["query", "target", "evalue", "bits", "alntmscore", "qcov", "tcov"]
)

# (중요) 자기 자신과 매칭된 줄 제거 (self-hit)
hits = hits[hits["query"] != hits["target"]].copy()

# (선택) coverage 필터링을 하고 싶으면 주석 해제
# hits = hits[(hits["qcov"] >= 0.3) & (hits["tcov"] >= 0.3)].copy()

# 가중치: alntmscore 사용 (1.0 초과는 1.0으로 고정)
hits["w"] = hits["alntmscore"].clip(upper=1.0)

# ===== 2) train_terms 로드: protein_id -> GO list =====
terms = pd.read_csv(TRAIN_TERMS, sep="\t")   # columns: protein_id, go_id, aspect

prot2go = defaultdict(list)
for r in terms.itertuples(index=False):
    prot2go[r.EntryID].append((r.term, r.aspect))

# ===== 3) query별 GO 점수 누적 =====
records = []

for query, g in hits.groupby("query", sort=False):
    go_scores = defaultdict(float)

    for r in g.itertuples(index=False):
        target = r.target
        w = r.w

        if target not in prot2go:
            continue

        for go_id, aspect in prot2go[target]:
            go_scores[(go_id, aspect)] += w

    for (go_id, aspect), score in go_scores.items():
        records.append((query, go_id, aspect, score))

out = pd.DataFrame(records, columns=["protein_id", "go_id", "aspect", "score"])
out.to_csv(OUT_FILE, sep="\t", index=False)
print("완료:", OUT_FILE, "rows=", len(out))
