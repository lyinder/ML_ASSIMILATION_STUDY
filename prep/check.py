"""Step 4: checks. (a) imputed vs observed values, (b) outcomes untouched, (c) key models: MICE (Rubin's rules) vs complete cases."""
import sys, math, numpy as np, pandas as pd
P = sys.argv[1]
I = pd.read_pickle(f"{P}/imputed_long.pkl"); C = pd.read_pickle(f"{P}/clean_all.pkl"); C = C[C.prodev & ~C.f_low_effort]
print("## (a) observed vs imputed means (imputation 1)")
rows = []
for c in ["sigma", "yearscode", "theta", "age_ord", "edu_ord", "org_ord", "manager"]:
    for y in sorted(I.year.unique()):
        J = I[(I.year == y) & (I.imp == 1)]; w = J[f"was_missing_{c}"] if f"was_missing_{c}" in J else J.was_missing_sigma | J.was_missing_yearscode
        rows.append(dict(var=c, year=y, pct_imputed=round(w.mean()*100, 1), observed=round(J.loc[~w, c].mean(), 2), imputed=round(J.loc[w, c].mean(), 2) if w.any() else np.nan))
print(pd.DataFrame(rows).to_string(index=False))
print("\n## (b) outcomes untouched: missing counts equal before/after?",
      all((I[I.imp == 1].groupby("year")[v].apply(lambda s: s.isna().sum()) == C.groupby("year")[v].apply(lambda s: s.isna().sum())).all()
          for v in ["uses_ai", "sent_score", "task_share"]))
def logit(y, X, g):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); H = X.T@(X*(p*(1-p))[:, None]) + 1e-8*np.eye(len(b)); st = np.linalg.solve(H, X.T@(y-p)); b += st
        if np.abs(st).max() < 1e-9: break
    p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); Hi = np.linalg.inv(X.T@(X*(p*(1-p))[:, None]))
    s = X*(y-p)[:, None]; V = Hi@(s.T@s)@Hi
    return b, V
def design(D):
    top = D.country.value_counts().index[:8]
    X = pd.DataFrame({"const": 1.0, "theta10": D.theta/10, "sigma10": D.sigma/10, "grad": (D.edu_ord >= 6).astype(float),
                      "large": (D.org_ord >= 6).astype(float), "manager": D.manager.astype(float)}, index=D.index)
    return pd.concat([X, pd.get_dummies(np.where(D.country.isin(top), D.country, "other"), prefix="c", drop_first=True, dtype=float).set_index(D.index)], axis=1)
def rubin(ests):
    b = np.array([e[0] for e in ests]); v = np.array([e[1] for e in ests]); m = len(ests)
    qbar = b.mean(0); W = v.mean(0); B = b.var(0, ddof=1); T = W + (1+1/m)*B
    return qbar, np.sqrt(T)
print("\n## (c) EU, odds ratio per 10 years: MICE (5 imputations, Rubin) vs complete cases")
res = []
for outcome, f in [("contesting", lambda D: 1-D.uses_ai), ("unfavourable", lambda D: (D.sent_score <= 2).astype(float).where(D.sent_score.notna()))]:
    for y in sorted(I.year.unique()):
        ests = []
        for m in range(1, 6):
            D = I[(I.year == y) & (I.imp == m) & (I.eu == 1)].copy(); D["y"] = f(D); D = D.dropna(subset=["y"])
            X = design(D); b, V = logit(D.y, X, None); ests.append((b[1:3], np.diag(V)[1:3]))
        q, se = rubin(ests)
        Dc = C[(C.year == y) & (C.eu == 1)].copy(); Dc["y"] = f(Dc)
        Dc = Dc.dropna(subset=["y", "theta", "sigma", "edu_ord", "org_ord", "manager"])
        bc, Vc = logit(Dc.y, design(Dc), None)
        for k, t in enumerate(["theta10", "sigma10"]):
            res.append(dict(outcome=outcome, year=y, term=t, MICE_OR=round(math.exp(q[k]), 2),
                            MICE_CI=f"{math.exp(q[k]-1.96*se[k]):.2f}-{math.exp(q[k]+1.96*se[k]):.2f}", n_MICE=len(D),
                            CC_OR=round(math.exp(bc[k+1]), 2), CC_CI=f"{math.exp(bc[k+1]-1.96*np.sqrt(Vc[k+1,k+1])):.2f}-{math.exp(bc[k+1]+1.96*np.sqrt(Vc[k+1,k+1])):.2f}", n_CC=len(Dc)))
R = pd.DataFrame(res); R.to_csv(f"{P}/check_mice_vs_cc.csv", index=False); print(R.to_string(index=False))
