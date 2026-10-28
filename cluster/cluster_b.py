"""Clustering B: 2026 AI users with task answers (70% file, all countries). LCA on tasks, sentiment, trust,
AI capability rating, daily intensity, satisfaction with AI on writing code and debugging, verification habits.
Missing answers are handled inside the model (they add nothing to the likelihood), not imputed.
Usage: python3 -I cluster_b.py <prep_dir>"""
import sys, pickle, numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, sys.argv[1]); import lca
P = sys.argv[1]
E = pd.read_pickle(f"{P}/encoded_long.pkl")
T = [c for c in E.columns if c.startswith("t_")]
F5 = ["sent_score", "trust26_score", "complex_score", "ai_daily_ord", "sat_impl", "sat_debug"]
F2 = ["verify_own", "verify_external"]
F = T + F5 + F2
U = E[(E.imp == 1) & (E.year == 2026) & (E.uses_ai == 1) & E.task_share.notna()].copy()
X = U[F].astype(float).values
levels = [[0, 1]] * len(T) + [[1, 2, 3, 4, 5]] * len(F5) + [[0, 1]] * len(F2)
print("n:", len(U), "| % missing per feature:", (U[F5 + F2].isna().mean() * 100).round(0).to_dict())
for k in range(2, 8):
    m = lca.fit(X, levels, k, starts=8, seed=k)
    print(f"k={k} BIC={m['bic']:.0f} entropy={m['entropy']:.2f} smallest={m['pi'].min():.3f}")
rng = np.random.default_rng(3); idx = rng.permutation(len(X)); h1, h2 = idx[:len(X)//2], idx[len(X)//2:]
for k in [3, 4, 5]:
    a = lca.fit(X[h1], levels, k, starts=8, seed=31); b = lca.fit(X[h2], levels, k, starts=8, seed=32)
    print(f"k={k} ARI={adjusted_rand_score(lca.predict(X, levels, a).argmax(1), lca.predict(X, levels, b).argmax(1)):.2f}")
pickle.dump((U[["year", "id"]], X, levels, F), open(f"{P}/lcab_data.pkl", "wb"))
