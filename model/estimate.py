"""Estimate the identity-assimilation model. Logit with cluster-robust (by person) standard errors.
Usage: python3 -I estimate.py <scratchpad_dir> <EU|ALL>"""
import sys, math, numpy as np, pandas as pd
R, SCOPE = sys.argv[1], sys.argv[2]
P = pd.read_pickle(f"{R}/model/persons.pkl"); TP = pd.read_pickle(f"{R}/model/tasks.pkl"); T26 = pd.read_pickle(f"{R}/model/tasks2026.pkl")
if SCOPE == "EU": P = P[P.eu]
P = P.copy(); P["theta10"] = P.theta / 10; P["sigma10"] = P.sigma / 10
top = P.country.value_counts().index[:8]; P["ctry"] = np.where(P.country.isin(top), P.country, "other")
P = P.reset_index(drop=True)

def logit(y, X, groups):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); W = p*(1-p)
        H = X.T@(X*W[:, None]) + 1e-8*np.eye(len(b)); step = np.linalg.solve(H, X.T@(y-p)); b += step
        if np.abs(step).max() < 1e-8: break
    p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); Hi = np.linalg.inv(X.T@(X*(p*(1-p))[:, None]))
    sc = pd.DataFrame(X*(y-p)[:, None]).groupby(np.asarray(groups)).sum().values
    G = sc.shape[0]; V = Hi@(sc.T@sc)@Hi * G/(G-1)
    return b, np.sqrt(np.diag(V))
def pval(z): return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
CTRL = ["grad", "large_org", "manager"]
def design(D, terms, cat=("ctry",)):
    X = D[terms + CTRL].copy()
    for c in CTRL: X[c] = X[c].fillna(X[c].mean())   # mean-impute controls only
    for c in cat: X = pd.concat([X, pd.get_dummies(D[c], prefix=c, drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0); return X
RES = []
def fit(name, D, yname, terms, groups=None, cat=("ctry",)):
    D = D.dropna(subset=[yname] + terms)
    X = design(D, terms, cat); b, se = logit(D[yname], X, D.pid if groups is None else groups(D))
    for t in terms:
        k = list(X.columns).index(t); RES.append(dict(model=name, term=t, OR=math.exp(b[k]), lo=math.exp(b[k]-1.96*se[k]),
                                                       hi=math.exp(b[k]+1.96*se[k]), p=pval(b[k]/se[k]), n=len(D), people=D.pid.nunique()))
# M1 contesting (does not use AI), per year
for y in [2023, 2024, 2025, 2026]:
    D = P[P.year == y].copy(); D["contest"] = 1 - D.uses_ai
    fit(f"M1 contesting {y}", D, "contest", ["theta10", "sigma10"])
# M2 task delegation, 8 shared tasks, AI users, per year: core x investment, core x security
TT = TP.merge(P[["pid", "theta10", "sigma10", "ctry"] + CTRL], on="pid")
for y in [2023, 2024, 2025, 2026]:
    D = TT[TT.year == y].copy(); D["core_theta"] = D.core*D.theta10; D["core_sigma"] = D.core*D.sigma10
    D = pd.concat([D, pd.get_dummies(D.task, prefix="t", drop_first=True, dtype=float)], axis=1)
    tcols = [c for c in D.columns if c.startswith("t_")]
    fit(f"M2 delegation {y}", D, "use", ["theta10", "sigma10", "core_theta", "core_sigma"] + tcols)
    RES[:] = [r for r in RES if not r["term"].startswith("t_")]
# M3 unfavourable sentiment, per year
for y in [2023, 2024, 2025, 2026]:
    fit(f"M3 unfavourable {y}", P[P.year == y], "unfav", ["theta10", "sigma10"])
# 2026 only
Q = P[P.year == 2026].copy()
for lv in ["medium", "high"]: Q[f"press_{lv}"] = np.where(Q.pressure.isna(), np.nan, (Q.pressure == lv).astype(float))
S = T26.merge(Q[["pid", "theta10", "sigma10", "ctry", "press_medium", "press_high"] + CTRL], on="pid")
S["core_theta"] = S.core*S.theta10; S["core_sigma"] = S.core*S.sigma10
fit("M4 delegation 2026 (gain+pressure)", S, "use", ["useful", "press_medium", "press_high", "theta10", "sigma10", "core"])
fit("M5 felt cost 2026 (AI lowered satisfaction)", S, "sat_down", ["core", "theta10", "sigma10", "core_theta", "core_sigma", "useful"])
fit("M6 avoid for skill loss 2026", Q, "avoid_skill_loss", ["theta10", "sigma10"])
dl = Q[Q.tasks.map(lambda t: bool(t & {"Writing code", "Debugging"}))]
fit("M7 external verification 2026", dl, "verify_external", ["theta10", "sigma10"])
fit("M7 own-expertise verification 2026", dl, "verify_own", ["theta10", "sigma10"])
out = pd.DataFrame(RES); out.to_csv(f"{R}/model/results_{SCOPE}.csv", index=False)
pd.set_option("display.width", 200)
print(out.assign(OR=out.OR.round(2), lo=out.lo.round(2), hi=out.hi.round(2), p=out.p.round(3)).to_string(index=False))
