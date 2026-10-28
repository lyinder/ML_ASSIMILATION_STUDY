"""Step 3: multiple imputation (MICE) of predictors and controls, then standardised features for clustering.
- Sample: professional developers, low-effort responses removed. 2026 = 70% exploration file.
- Imputed: age, education, organisation size, manager, work setting, years coding, years in the profession.
  Investment (theta) is recomputed from the imputed years.
- Outcomes (AI use, sentiment, trust, tasks, threat) are used as predictors in the imputation model
  but are never replaced: their observed values, including missing ones, are kept.
- Country is not imputed (high cardinality); EU and US flags stay missing where country is missing.
- 5 imputations per year, BayesianRidge with posterior sampling. Imputers and scalers are fitted per year on
  exploration data and saved, so the 2026 ones can be applied unchanged to the holdout.
Usage: python3 -I impute.py <prep_dir>"""
import sys, pickle, numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
P = sys.argv[1]; M = 5
A = pd.read_pickle(f"{P}/clean_all.pkl")
A = A[A.prodev & ~A.f_low_effort].reset_index(drop=True)
A["remote_ord"] = A.remote.map({"remote": 0, "hybrid": 1, "in_person": 2}).astype(float)
IMPUTE = {"age_ord": (1, 7, True), "edu_ord": (1, 7, True), "org_ord": (1, 8, True), "manager": (0, 1, True),
          "remote_ord": (0, 2, True), "yearscode": (0, 50, False), "sigma": (0, 50, False)}
AUX = ["uses_ai", "sent_score", "trust_score", "trust26_score", "complex_score", "task_share", "eu", "us"]
out, imputers = [], {}
for y in sorted(A.year.unique()):
    D = A[A.year == y].copy()
    aux = [c for c in AUX if D[c].notna().any()]
    cols = list(IMPUTE) + aux
    X = D[cols].astype(float).values
    imputers[y] = []
    for m in range(1, M + 1):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=15, random_state=1000 * y + m,
                               min_value=[IMPUTE[c][0] if c in IMPUTE else -np.inf for c in cols],
                               max_value=[IMPUTE[c][1] if c in IMPUTE else np.inf for c in cols])
        Xi = imp.fit_transform(X); imputers[y].append((cols, imp))
        E = D.copy(); E["imp"] = m
        for k, c in enumerate(IMPUTE):
            v = Xi[:, k]
            if IMPUTE[c][2]: v = np.rint(v)
            E[c] = np.clip(v, IMPUTE[c][0], IMPUTE[c][1])
            E[f"was_missing_{c}"] = D[c].isna().values
        E["theta"] = np.clip(E.yearscode - E.sigma, 0, None)
        out.append(E)
    print(y, "imputed", len(D), "rows x", M)
I = pd.concat(out, ignore_index=True)
I["remote"] = I.remote_ord.map({0: "remote", 1: "hybrid", 2: "in_person"})
# standardised clustering features: z-scores fitted per year on the exploration data, imputation 1..5 pooled
FEAT = ["age_ord", "edu_ord", "org_ord", "manager", "remote_ord", "yearscode", "sigma", "theta",
        "sent_score", "trust_score", "trust26_score", "complex_score", "task_share"] + \
       [c for c in I.columns if c.startswith("t_")]
scalers = {}
for y in sorted(I.year.unique()):
    J = I[I.year == y]
    mu, sd = J[FEAT].mean(), J[FEAT].std().replace(0, np.nan)
    scalers[y] = pd.DataFrame({"mean": mu, "sd": sd})
    I.loc[I.year == y, [f"z_{c}" for c in FEAT]] = ((J[FEAT] - mu) / sd).values
pd.concat(scalers, names=["year", "feature"]).to_csv(f"{P}/scalers.csv")
I.to_pickle(f"{P}/imputed_long.pkl")
with open(f"{P}/imputers.pkl", "wb") as f: pickle.dump(imputers, f)
print("long file:", I.shape)
