"""Step 2: missing-data analysis for professional developers, 2023-2026 (2026 = 70% exploration file).
Usage: python3 -I missing.py <prep_dir>"""
import sys, math, numpy as np, pandas as pd
P = sys.argv[1]
A = pd.read_pickle(f"{P}/clean_all.pkl")
A = A[A.prodev & ~A.f_low_effort].copy()
VARS = ["age_ord", "edu_ord", "org_ord", "manager", "remote", "country", "yearscode", "sigma", "theta",
        "uses_ai", "sent_score", "trust_score", "trust26_score", "complex_score", "threat", "task_share"]
# 1. share missing by variable and year
m = A.groupby("year")[VARS].apply(lambda g: g.isna().mean() * 100).round(1)
avail = {"trust_score": [2023, 2024, 2025], "trust26_score": [2026], "complex_score": [2024, 2025, 2026], "threat": [2024, 2025]}
for v, yrs in avail.items():
    for y in m.index:
        if y not in yrs: m.loc[y, v] = np.nan        # not asked that year
m.T.to_csv(f"{P}/missing_by_year.csv")
print("## % missing, professional developers (blank = not asked that year)"); print(m.T.to_string())
print("n:", A.groupby("year").size().to_dict())
# 2. task answers among AI users
u = A[A.uses_ai == 1]
print("\n## % of AI users with no task answer:", (u.groupby("year").task_share.apply(lambda s: s.isna().mean() * 100)).round(1).to_dict())

# 3. what predicts missingness: logistic models
def logit(y, X):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); H = X.T@(X*(p*(1-p))[:, None]) + 1e-8*np.eye(len(b))
        st = np.linalg.solve(H, X.T@(y-p)); b += st
        if np.abs(st).max() < 1e-9: break
    p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); Hi = np.linalg.inv(X.T@(X*(p*(1-p))[:, None]))
    s = X*(y-p)[:, None]; V = Hi@(s.T@s)@Hi
    return b, np.sqrt(np.diag(V))
rows = []
for target, preds in [("sigma", ["age_ord", "edu_ord", "uses_ai", "eu", "us"]),
                      ("sent_score", ["age_ord", "edu_ord", "yearscode", "eu", "us"]),
                      ("org_ord", ["age_ord", "edu_ord", "uses_ai", "eu", "us"])]:
    for y in sorted(A.year.unique()):
        D = A[A.year == y].copy(); D["miss"] = D[target].isna().astype(float)
        if D.miss.mean() < 0.005: continue
        D = D.dropna(subset=preds)
        Z = D[preds].astype(float); Z = (Z - Z.mean()) / Z.std()
        X = np.column_stack([np.ones(len(D)), Z.values]); b, se = logit(D.miss, X)
        for k, pr in enumerate(preds, 1):
            z = b[k]/se[k]
            rows.append(dict(missing=target, year=y, predictor=pr, OR_per_SD=round(math.exp(b[k]), 2),
                             p=round(2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2)))), 4), share_missing=round(D.miss.mean()*100, 1), n=len(D)))
R = pd.DataFrame(rows); R.to_csv(f"{P}/missing_predictors.csv", index=False)
print("\n## What predicts missingness (odds ratio per SD)"); print(R.to_string(index=False))
# 4. outcome means: complete vs incomplete experience (MCAR check)
print("\n## AI use and sentiment by whether experience is missing")
print(A.assign(exp_missing=A.sigma.isna()).groupby(["year", "exp_missing"])[["uses_ai", "sent_score", "age_ord"]].mean().round(2).to_string())
