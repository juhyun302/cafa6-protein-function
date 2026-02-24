import pandas as pd
from collections import defaultdict
from pathlib import Path

FOLDSEEK_HITS = "foldseek_hits_test.tsv"
TRAIN_TERMS   = "../Train/train_terms.tsv"
OUT_FILE      = "structure_go_scores_test.tsv"

hits = pd.read_csv(
    FOLDSEEK_HITS,
    sep="\t",
    header=None,
    names=["query", "target", "evalue", "bits", "alntmscore", "qcov", "tcov"]
)

# self-hit 제거
hits = hits[hits["query"] != hits["target"]].copy()

# 가중치: alntmscore (1.0 초과는 1.0으로 고정)
hits["w"] = hits["alntmscore"].clip(upper=1.0)

terms = pd.read_csv(TRAIN_TERMS, sep="\t")  # protein_id, go_id, aspect

prot2go = defaultdict(list)
for r in terms.itertuples(index=False):
    prot2go[r.EntryID].append((r.term, r.aspect))

records = []
for query, g in hits.groupby("query", sort=False):
    go_scores = defaultdict(float)
    for r in g.itertuples(index=False):
        if r.target not in prot2go:
            continue
        for go_id, aspect in prot2go[r.target]:
            go_scores[(go_id, aspect)] += r.w

    for (go_id, aspect), score in go_scores.items():
        records.append((query, go_id, aspect, score))

out = pd.DataFrame(records, columns=["protein_id", "go_id", "aspect", "score"])
out.to_csv(OUT_FILE, sep="\t", index=False)

print("완료:", OUT_FILE)
print("query 수:", out["protein_id"].nunique())
print("총 row:", len(out))
