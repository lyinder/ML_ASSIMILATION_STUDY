# Clustering (exploratory)

Latent class analysis (`lca.py`, EM with multiple starts; missing answers add nothing to the likelihood).
Gaussian mixtures were tried first and rejected: on mostly 0/1 features the BIC never stops improving.

- **A** (`cluster_a.py`, `profile_a.py`): AI users 2023–2026, 8 shared tasks + sentiment. 4 classes (adjusted Rand index between random halves = 0.98). Shares by year, security and investment, pooled over 5 imputations.
- **B** (`cluster_b.py`, `profile_b.py`): 2026 AI users (70% file), tasks + sentiment, trust, AI capability rating, daily use, satisfaction, verification. 5 classes (ARI 0.78).

Clustering uses outcomes only; investment and security are used afterwards to profile the classes. The sealed holdout is not used.
