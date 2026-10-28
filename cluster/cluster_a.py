"""Clustering A: AI users with task answers, 2023-2026. Latent class analysis on 8 shared tasks (0/1) + sentiment (1-5).
Features are never imputed, so the model uses imputation-1 rows; profiles by investment and security average all 5 imputations.
Usage: python3 -I cluster_a.py <prep_dir>"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, sys.argv[1]); import lca
P = sys.argv[1]
E = pd.read_pickle(f"{P}/encoded_long.pkl")
T = [c for c in E.columns if c.startswith("t_")]; F = T + ["sent_score"]
U1 = E[(E.imp == 1) & (E.uses_ai == 1)].dropna(subset=F).copy()
X = U1[F].astype(int).values; levels = [[0, 1]] * len(T) + [[1, 2, 3, 4, 5]]
rng = np.random.default_rng(7); sub = rng.choice(len(X), size=30000, replace=False)
fits = {}
for k in []:
    m = lca.fit(X[sub], levels, k, starts=6, seed=k); fits[k] = m
    print(f"k={k} BIC={m['bic']:.0f} entropy={m['entropy']:.2f} smallest class={m['pi'].min():.3f}")
print("\n## stability: fit on two random halves, compare assignments on all rows (adjusted Rand index)")
idx = rng.permutation(len(X)); h1, h2 = idx[:len(X)//2], idx[len(X)//2:]
for k in [4, 5, 6]:
    a = lca.fit(X[h1], levels, k, starts=6, seed=11).copy(); b = lca.fit(X[h2], levels, k, starts=6, seed=12)
    ari = adjusted_rand_score(lca.predict(X, levels, a).argmax(1), lca.predict(X, levels, b).argmax(1))
    print(f"k={k} ARI={ari:.2f}")
    full = lca.fit(X, levels, k, starts=6, seed=21)
    prof = pd.DataFrame({t: th[:, 1] for t, th in zip(T, full["theta"][:-1])}).round(2)
    prof["sent_mean"] = (full["theta"][-1] * np.arange(1, 6)).sum(1).round(2); prof["share"] = full["pi"].round(3)
    print(prof.sort_values("share", ascending=False).to_string())
    np.save(f"{P}/lca_post_k{k}.npy", full["post"])
    import pickle; pickle.dump(full, open(f"{P}/lca_k{k}.pkl", "wb"))
U1[["year", "id"]].to_pickle(f"{P}/lca_rows.pkl")
