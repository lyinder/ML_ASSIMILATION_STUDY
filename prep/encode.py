"""Step 5: encode the remaining unordered categorical variables (one-hot).
Ordered variables are already ordinal scores and yes/no variables already 0/1 (see clean.py).
- Missing stays missing: a respondent with no answer gets NaN in every dummy of that variable, not 0.
- Country becomes region dummies (EU, US, other, unknown) plus the 10 most common countries (rest pooled).
  The country list is fixed on the exploration data and saved, so the holdout gets the same columns.
Writes encoded_long.pkl and encoding_dictionary.csv.
Usage: python3 -I encode.py <prep_dir>"""
import sys, numpy as np, pandas as pd
P = sys.argv[1]
I = pd.read_pickle(f"{P}/imputed_long.pkl")
rows = []
def onehot(df, col, levels, prefix, note):
    for lv in levels:
        name = f"{prefix}_{lv}"
        df[name] = np.where(df[col].isna(), np.nan, (df[col] == lv).astype(float))
        rows.append(dict(variable=name, source=col, level=lv, type="one-hot", note=note))
# work setting: complete after imputation
onehot(I, "remote", ["remote", "hybrid", "in_person"], "work", "imputed (MICE) work setting")
# employment: small categories pooled
I["employment_grp"] = I.employment.replace({"not_employed": "other", "student": "other", "retired": "other"})
onehot(I, "employment_grp", ["employed", "self_employed", "other"], "emp", "other = student, not employed, retired, other")
# perceived job threat (2024-2025 only)
I["threat"] = I.threat.where(I.threat.isin(["yes", "no", "unsure"]))
onehot(I, "threat", ["yes", "unsure", "no"], "threat", "asked 2024-2025 only; missing otherwise")
# where AI tooling was learned (2026 only)
onehot(I, "learn_ai", ["both", "work", "own", "no"], "learnai", "asked 2026 only")
# region and country
I["region"] = np.select([I.eu == 1, I.us == 1, I.country.notna()], ["eu", "us", "other"], "unknown")
onehot(I, "region", ["eu", "us", "other", "unknown"], "region", "unknown = country not reported")
top = I[I.imp == 1].country.value_counts().index[:10].tolist()
pd.Series(top, name="country").to_csv(f"{P}/country_levels.csv", index=False)
I["country_grp"] = np.where(I.country.isna(), "unknown", np.where(I.country.isin(top), I.country, "other"))
short = {c: c.split(" of ")[0].split(",")[0].replace(" ", "_") for c in top}
for c in top + ["other", "unknown"]:
    name = f"ctry_{short.get(c, c)}"
    I[name] = (I.country_grp == c).astype(float)
    rows.append(dict(variable=name, source="country", level=c, type="one-hot", note="top 10 countries fixed on exploration data"))
# document the ordinal and binary encodings made earlier
ORD = {"age_ord": "1 = under 18 ... 7 = 65+", "edu_ord": "1 primary ... 5 bachelor, 6 master, 7 professional/doctorate",
       "org_ord": "1 freelancer, 2 under 20 ... 8 = 10,000+", "remote_ord": "0 remote, 1 hybrid, 2 in person",
       "sent_score": "1 very unfavourable ... 5 very favourable", "trust_score": "1 highly distrust ... 5 highly trust (2023-2025)",
       "trust26_score": "1 do not trust for most tasks ... 5 trust incl. important decisions (2026)",
       "complex_score": "1 very poor ... 5 very well (2024-2026)", "pressure_ord": "1 low, 2 medium, 3 high (2026)",
       "ai_daily_ord": "1 under 1 hour not daily ... 5 = 4+ hours a day (2026)",
       "sat_*": "1 greatly decreased ... 5 greatly improved (2026)", "useful_*": "1 not useful ... 4 very useful (2026)"}
BIN = ["manager", "uses_ai", "plans_ai", "eu", "us", "avoid_none", "avoid_skill_loss", "verify_external", "verify_own"] + \
      [c for c in I.columns if c.startswith("t_")]
for k, v in ORD.items(): rows.append(dict(variable=k, source=k, level="", type="ordinal", note=v))
for k in BIN: rows.append(dict(variable=k, source=k, level="", type="binary 0/1", note=""))
pd.DataFrame(rows).to_csv(f"{P}/encoding_dictionary.csv", index=False)
I.to_pickle(f"{P}/encoded_long.pkl")
new = [r["variable"] for r in rows if r["type"] == "one-hot"]
print("one-hot columns added:", len(new)); print(new)
chk = I[I.imp == 1]
for pre in ["work", "emp", "threat", "learnai", "region"]:
    cols = [c for c in new if c.startswith(pre + "_")]
    s = chk[cols].sum(axis=1, min_count=1)
    print(f"{pre}: rows summing to 1 = {(s == 1).sum()}, all-missing = {s.isna().sum()}, other = {((s != 1) & s.notna()).sum()}")
