import numpy as np
import json
import pandas as pd

X    = np.load("artifacts/tag_vectors.npy")
meta = pd.read_parquet("artifacts/materials_meta.parquet")
vocab_inv = list(json.load(open("artifacts/tag_vocab.json", encoding="utf-8")).keys())
vocab     = json.load(open("artifacts/tag_vocab.json", encoding="utf-8"))

sim = X @ X.T  # косинус, т.к. L2-нормированы

# Посмотреть топ-5 соседей для 3 случайных материалов
for i in [0, 100, 300]:
    print(f"\n[{i}] {meta.title.iloc[i]!r}")
    print(f"    {' '.join(f for f in vocab if X[i, vocab[f]] > 0)}")
    top = np.argsort(-sim[i])[1:6]
    for j in top:
        common = [f for f in vocab if X[i, vocab[f]] > 0 and X[j, vocab[f]] > 0]
        print(f"  {sim[i,j]:.3f}  {meta.title.iloc[j][:60]!r}")
        print(f"          общее: {common}")