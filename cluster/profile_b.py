import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import lca
P = sys.argv[1]; K = 5
rows, X, levels, F = pickle.load(open(f"{P}/lcab_data.pkl", "rb"))
m = lca.fit(X, levels, K, starts=10, seed=55); pickle.dump(m, open(f"{P}/lcab_k{K}.pkl", "wb"))
prof = {}
for f, th, lv in zip(F, m["theta"], levels):
    prof[f] = th[:, 1] if lv == [0, 1] else (th * np.array(lv)).sum(1)
prof = pd.DataFrame(prof); prof["share"] = m["pi"]
print(prof.round(2).T.to_string())
rows = rows.assign(cls=m["post"].argmax(1))
E = pd.read_pickle(f"{P}/encoded_long.pkl")
res = []
for i in range(1, 6):
    D = E[(E.imp == i) & (E.year == 2026)].merge(rows, on=["year", "id"])
    g = D.groupby("cls")
    res.append(pd.DataFrame({"sigma": g.sigma.mean(), "theta": g.theta.mean(), "pressure_high_%": g.pressure_ord.apply(lambda s: (s == 3).mean() * 100),
        "EU_%": g.eu.mean() * 100, "skill_loss_avoid_%": g.avoid_skill_loss.mean() * 100, "learn_own_only_%": g.learnai_own.mean() * 100,
        "sec_0-4_%": g.sigma.apply(lambda s: (s <= 4).mean() * 100), "sec_20+_%": g.sigma.apply(lambda s: (s >= 20).mean() * 100),
        "inv_10+_%": g.theta.apply(lambda s: (s >= 10).mean() * 100), "n": g.size()}))
R = (sum(res) / 5).round(1); print(R.to_string())
prof.round(3).to_csv(f"{P}/lcab_profiles.csv"); R.to_csv(f"{P}/lcab_class_profiles.csv")
# class shares by security band, 2026
D = E[(E.imp == 1) & (E.year == 2026)].merge(rows, on=["year", "id"])
D["sec"] = pd.cut(D.sigma, [-1, 4, 9, 19, 50], labels=["0-4", "5-9", "10-19", "20+"])
D["inv"] = pd.cut(D.theta, [-1, 2, 5, 9, 40], labels=["0-2", "3-5", "6-9", "10+"])
print((pd.crosstab(D.sec, D.cls, normalize="index") * 100).round(0).to_string())
print((pd.crosstab(D.inv, D.cls, normalize="index") * 100).round(0).to_string())
