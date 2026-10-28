"""Profile the 4 latent classes by year, security band, investment band; pooled over 5 imputations.
Non-users are added as a fifth group ("No AI use") so shares cover all professional developers who answered AI use."""
import sys, pickle, numpy as np, pandas as pd
P = sys.argv[1]
E = pd.read_pickle(f"{P}/encoded_long.pkl"); rows = pd.read_pickle(f"{P}/lca_rows.pkl")
post = np.load(f"{P}/lca_post_k4.npy"); mdl = pickle.load(open(f"{P}/lca_k4.pkl", "rb"))
T = [c for c in E.columns if c.startswith("t_")]
prof = pd.DataFrame({t: th[:, 1] for t, th in zip(T, mdl["theta"][:-1])})
# name classes from their profiles
names = {}
for c in range(4):
    p = prof.loc[c]
    names[c] = ("Full delegation" if p.t_deployment_and_monitoring > 0.4 else
                "Peripheral helper" if p.t_writing_code < 0.5 else
                "Broad integrator" if p.t_debugging > 0.6 else "Code writing only")
rows = rows.assign(cls=[names[c] for c in post.argmax(1)])
ORDER = ["No AI use", "Peripheral helper", "Code writing only", "Broad integrator", "Full delegation"]
out = {}
for scope in ["EU", "ALL"]:
    tabs_s, tabs_t, means = [], [], []
    for m in range(1, 6):
        D = E[(E.imp == m) & E.uses_ai.notna()].copy()
        if scope == "EU": D = D[D.eu == 1]
        D = D.merge(rows, on=["year", "id"], how="left")
        D.loc[D.uses_ai == 0, "cls"] = "No AI use"
        D = D[D.cls.notna()]   # AI users without task or sentiment answers are left out
        D["sec"] = pd.cut(D.sigma, [-1, 4, 9, 19, 50], labels=["0-4", "5-9", "10-19", "20+"])
        D["inv"] = pd.cut(D.theta, [-1, 2, 5, 9, 40], labels=["0-2", "3-5", "6-9", "10+"])
        tabs_s.append(pd.crosstab([D.year, D.sec], D.cls, normalize="index")[ORDER])
        tabs_t.append(pd.crosstab([D.year, D.inv], D.cls, normalize="index")[ORDER])
        means.append(D.groupby(["year", "cls"])[["sigma", "theta"]].mean())
        if m == 1: nyear = D.groupby("year").size().to_dict()
    S = (sum(tabs_s) / 5 * 100).round(0); Tt = (sum(tabs_t) / 5 * 100).round(0); Mn = (sum(means) / 5).round(1)
    yr = (S.groupby(level=0).mean()).round(0)
    print(f"\n===== {scope} (n per year: {nyear})")
    print("## class shares by year and years in profession (security), %"); print(S.to_string())
    print("## class shares by year and pre-career coding years (investment), %"); print(Tt.to_string())
    print("## mean years in profession / pre-career coding by class"); print(Mn.unstack("cls")["sigma"][ORDER].to_string()); print(Mn.unstack("cls")["theta"][ORDER].to_string())
    S.to_csv(f"{P}/lca_shares_security_{scope}.csv"); Tt.to_csv(f"{P}/lca_shares_investment_{scope}.csv")
print("\n## class profiles"); prof.index = [names[c] for c in range(4)]
prof["sentiment_mean"] = (mdl["theta"][-1] * np.arange(1, 6)).sum(1); prof["share"] = mdl["pi"]
print(prof.round(2).loc[ORDER[1:]].to_string()); prof.round(3).to_csv(f"{P}/lca_profiles.csv")
