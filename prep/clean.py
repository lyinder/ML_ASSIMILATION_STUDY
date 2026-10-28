"""Step 1: clean and harmonise the Stack Overflow survey, 2023-2026, all respondents, all countries.
Writes clean_all.pkl and clean_all.csv.gz (one row per respondent) and cleaning_log.csv.
2026 uses only the 70% exploration file. clean_2026(path) can be applied unchanged to the holdout later.
Usage: python3 -I clean.py <data_dir> <out_dir>
"""
import sys, numpy as np, pandas as pd

EU = ("Austria Belgium Bulgaria Croatia Cyprus Czech Denmark Estonia Finland France Germany Greece Hungary "
      "Ireland Italy Latvia Lithuania Luxembourg Malta Netherlands Poland Portugal Romania Slovakia Slovenia Spain Sweden").split()
TASKS = ["Writing code", "Debugging", "Reviewing code", "Documenting code", "Testing code",
         "Learning a codebase", "Project planning", "Deployment and monitoring"]
OLD = {"Writing code": "Writing code", "Committing and reviewing code": "Reviewing code", "Documenting code": "Documenting code",
       "Testing code": "Testing code", "Learning about a codebase": "Learning a codebase", "Project planning": "Project planning",
       "Deployment and monitoring": "Deployment and monitoring"}
NEW = {"Generating code in a familiar area": "Writing code", "Generating code in an unfamiliar area": "Writing code",
       "Debugging, troubleshooting, or refactoring code": "Debugging",
       "Reviewing code, pull requests, or design decisions": "Reviewing code",
       "Writing documentation, comments, or knowledge base content": "Documenting code",
       "Writing or improving tests": "Testing code", "Understanding or onboarding to a new codebase": "Learning a codebase",
       "Writing tickets, requirements, plans, or project updates": "Project planning",
       "Deploying, operating, or troubleshooting production systems": "Deployment and monitoring"}
AGE = {"Under 18 years old": 1, "18-24 years old": 2, "25-34 years old": 3, "35-44 years old": 4,
       "45-54 years old": 5, "55-64 years old": 6, "65 years or older": 7}
AGE_MAX = {1: 17, 2: 24, 3: 34, 4: 44, 5: 54, 6: 64, 7: 99}   # upper bound of each age group

def edu_ord(s):
    s = s.astype(str)
    out = np.select([s.str.contains("Primary"), s.str.contains("Secondary"), s.str.contains("Some college"),
                     s.str.contains("Associate"), s.str.contains("Bachelor"), s.str.contains("Master"),
                     s.str.contains("Professional degree")], [1, 2, 3, 4, 5, 6, 7], np.nan)
    return out
def org_ord(s):
    s = s.astype(str)
    return np.select([s.str.startswith("Just me"), s.str.startswith("2 to 9") | s.str.startswith("10 to 19") | s.str.startswith("Less than 20"),
                      s.str.startswith("20 to 99"), s.str.startswith("100 to 499"), s.str.startswith("500 to 999"),
                      s.str.startswith("1,000"), s.str.startswith("5,000"), s.str.startswith("10,000")],
                     [1, 2, 3, 4, 5, 6, 7, 8], np.nan)
def remote(s):
    s = s.astype(str)
    return np.select([s.eq("Remote"), s.str.startswith("Hybrid") | s.str.startswith("Your choice"), s.eq("In-person")],
                     ["remote", "hybrid", "in_person"], None)
def employment(s):
    s = s.astype(str)
    return np.select([s.str.contains("Employed"), s.str.contains("Independent contractor"), s.str.contains("Student"),
                      s.str.contains("Not employed"), s.str.contains("Retired")],
                     ["employed", "self_employed", "student", "not_employed", "retired"], "other")
def years(x):
    return pd.to_numeric(pd.Series(x).replace({"Less than 1 year": 0, "More than 50 years": 51}), errors="coerce").values
def tl(x): return set() if pd.isna(x) else {t.strip() for t in str(x).split(";") if t.strip()}
def score(s, mapping):
    return s.map(mapping).astype(float).values
SENT_OLD = {"Very unfavorable": 1, "Unfavorable": 2, "Indifferent": 3, "Favorable": 4, "Very favorable": 5}
SENT_NEW = {"Very unfavorable": 1, "Somewhat unfavorable": 2, "Neutral": 3, "Somewhat favorable": 4, "Very favorable": 5}
TRUST_OLD = {"Highly distrust": 1, "Somewhat distrust": 2, "Neither trust nor distrust": 3, "Somewhat trust": 4, "Highly trust": 5}
TRUST_NEW = {"I do not trust it for most work tasks": 1, "I trust it for low-risk work tasks only": 2,
             "I trust it when I can easily verify the output": 3, "I trust it for many tasks, but not important work decisions": 4,
             "I trust it for many tasks, including important work decisions": 5}
def complex_score(s):
    s = s.astype(str)
    return np.select([s.str.startswith("Very poor"), s.str.startswith("Bad"), s.str.startswith("Neither"),
                      s.str.startswith("Good"), s.str.startswith("Very well")], [1, 2, 3, 4, 5], np.nan)

def finish(o, raw_cols_answered):
    """Shared cleaning rules. Each rule writes a flag; the log counts them."""
    o["eu"] = np.where(o.country.isna(), np.nan, o.country.fillna("").apply(lambda x: any(x.startswith(e) for e in EU)).astype(float))
    o["us"] = np.where(o.country.isna(), np.nan, o.country.fillna("").str.startswith("United States").astype(float))
    # experience: impossible or inconsistent values become missing (then imputed)
    for c in ["workexp", "yearscode", "yearscodepro"]:
        o[f"f_{c}_gt50"] = (o[c] > 50).fillna(False).astype(bool)
        o.loc[o[f"f_{c}_gt50"], c] = np.nan
    amax = o.age_ord.map(AGE_MAX)
    o["f_exp_exceeds_age"] = ((o.workexp > amax - 14) | (o.yearscode > amax - 5)).fillna(False).astype(bool)
    o.loc[o.f_exp_exceeds_age, ["workexp", "yearscode", "yearscodepro"]] = np.nan
    o["f_pro_exceeds_total"] = (o.yearscodepro > o.yearscode).fillna(False).astype(bool)
    o.loc[o.f_pro_exceeds_total, "yearscodepro"] = np.nan
    # low effort: answered under 30% of the core questions
    o["answered_share"] = raw_cols_answered
    o["f_low_effort"] = o.answered_share < 0.30
    o["f_duplicate_id"] = o.duplicated(["year", "id"], keep="first")
    return o

CORE_Q = ["Age", "Employment", "Country", "YearsCode", "WorkExp", "OrgSize", "ICorPM", "AISelect", "AISent"]

def clean_old(path, y):
    d = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    o = pd.DataFrame({"year": y, "id": d.ResponseId.values, "country": d.Country.values})
    o["prodev"] = d.MainBranch.eq("I am a developer by profession").values
    o["age_ord"] = d.Age.map(AGE).values
    o["edu_ord"] = edu_ord(d.EdLevel); o["org_ord"] = org_ord(d.OrgSize)
    o["manager"] = np.where(d.ICorPM.isna(), np.nan, d.ICorPM.eq("People manager").astype(float))
    o["remote"] = remote(d.RemoteWork); o["employment"] = employment(d.Employment)
    o["yearscode"] = years(d.YearsCode); o["workexp"] = years(d.WorkExp)
    o["yearscodepro"] = years(d.YearsCodePro) if "YearsCodePro" in d else np.nan
    o["f_2025_blank_exp_set0"] = False
    if y == 2025:   # the 2025 form told respondents with no work experience to leave the question blank
        blank0 = d.WorkExp.isna() & d.YearsCode.notna() & d.Employment.notna() & d.Age.map(AGE).le(3)   # under 35 only; older blanks are treated as missing
        o.loc[blank0.values, "workexp"] = 0; o["f_2025_blank_exp_set0"] = blank0.values
    use = d.AISelect.astype(str).str.startswith("Yes")
    o["uses_ai"] = np.where(d.AISelect.isna(), np.nan, use.astype(float))
    o["plans_ai"] = np.where(d.AISelect.isna(), np.nan, d.AISelect.astype(str).str.contains("plan to soon").astype(float))
    o["sent_score"] = score(d.AISent, SENT_OLD)
    trust_col = "AIBen" if y == 2023 else "AIAcc"   # 2023 stores trust under AIBen (columns swapped)
    o["trust_score"] = score(d[trust_col], TRUST_OLD)
    o["complex_score"] = complex_score(d.AIComplex) if "AIComplex" in d else np.nan
    o["threat"] = d.AIThreat.map({"Yes": "yes", "No": "no", "I'm not sure": "unsure"}).values if "AIThreat" in d else None
    cur = d["AIToolCurrently Using"].map(tl) if y < 2025 else \
        d["AIToolCurrently mostly AI"].map(tl).combine(d["AIToolCurrently partially AI"].map(tl), lambda a, b: a | b)
    cur = cur.map(lambda s: {OLD.get(t) or ("Debugging" if t.startswith("Debugging") else None) for t in s} - {None})
    has = use & (cur.map(len) > 0)
    for t in TASKS:
        o["t_" + t.lower().replace(" ", "_")] = np.where(has, cur.map(lambda s: float(t in s)), np.nan)
    o["task_share"] = np.where(has, cur.map(len) / len(TASKS), np.nan)
    answered = d[[c for c in CORE_Q if c in d]].notna().mean(axis=1).values
    return finish(o, answered)

def clean_2026(path):
    d = pd.read_csv(path, low_memory=False)
    o = pd.DataFrame({"year": 2026, "id": d.ResponseId.values, "country": d.Country.values})
    o["prodev"] = (d.CodeMain.eq("I create code as a primary part of my job") &
                   d.Employment.isin(["Employed", "Independent contractor, freelancer, or self-employed"])).values
    o["age_ord"] = d.Age.map(AGE).values
    o["edu_ord"] = edu_ord(d.Education); o["org_ord"] = org_ord(d.OrgSize)
    o["manager"] = np.where(d.ICorPM.isna(), np.nan, d.ICorPM.astype(str).str.contains("manager").astype(float))
    o["remote"] = remote(d.WorkSetting); o["employment"] = employment(d.Employment)
    o["yearscode"] = years(d.YearsCode); o["workexp"] = years(d.WorkExp); o["yearscodepro"] = np.nan
    o["f_2025_blank_exp_set0"] = False
    sel = d.AISelect.fillna("")
    o["uses_ai"] = np.where(d.AISelect.isna(), np.nan, (~sel.eq("I don’t use AI tools")).astype(float))
    o["plans_ai"] = np.nan
    o["sent_score"] = score(d.AISentiment, SENT_NEW)
    o["trust_score"] = np.nan                       # 2026 trust question uses a different scale
    o["trust26_score"] = score(d.AITrust, TRUST_NEW)
    o["complex_score"] = complex_score(d.AIComplex)
    o["threat"] = None
    cur = d.AIWorkCurrentlyUse.map(tl).map(lambda s: {NEW[t] for t in s if t in NEW})
    has = (o.uses_ai == 1).values & (cur.map(len) > 0).values
    for t in TASKS:
        o["t_" + t.lower().replace(" ", "_")] = np.where(has, cur.map(lambda s: float(t in s)), np.nan)
    o["task_share"] = np.where(has, cur.map(len) / len(TASKS), np.nan)
    # 2026-only measures
    press = {"AI use is mostly optional and left up to individuals": 1, "My organization does not currently use AI tools or agents for work": 1,
             "My organization provides approved AI tools and guidance for how to use them": 2,
             "Some teams actively encourage AI use, but practices vary across the organization": 2,
             "My organization expects some employees to use AI tools for certain types of work": 3,
             "My organization measures or evaluates AI use as part of employee or team performance": 3}
    o["pressure_ord"] = d.AIUse.map(press).astype(float).values
    o["learn_ai"] = d.LearnAI.map({"Yes, both on my own and at work": "both", "Yes, at work": "work",
                                   "Yes, on my own": "own", "No": "no"}).values
    av = d.AIAvoid
    o["avoid_none"] = np.where(av.isna(), np.nan, av.eq("I do not avoid using AI at work or school").astype(float))
    o["avoid_skill_loss"] = np.where(av.isna(), np.nan, av.fillna("").str.contains("losing job skills").astype(float))
    daily = {"<1 hour and not every day": 1, "<1 hour every day": 2, "1-2 hours/day": 3, "2-4 hours/day": 4, "4+ hours/day": 5}
    o["ai_daily_ord"] = d.AIDaily.map(daily).astype(float).values
    for i, t in {2: "impl", 4: "debug", 3: "review", 6: "arch", 5: "docs", 7: "comm", 1: "design"}.items():
        o[f"sat_{t}"] = d[f"AISat_{i}"].map({"Greatly decreased": 1, "Somewhat decreased": 2, "No change": 3,
                                             "Somewhat improved": 4, "Greatly improved": 5}).astype(float).values
        o[f"useful_{t}"] = d[f"AIImpact_{i}"].map({"More impractical than useful, not useful": 1, "No strong opinion": 2,
                                                   "Useful": 3, "Very useful": 4}).astype(float).values
    nxt = d.AIWorkNext.fillna(""); ok = d.AIWorkNext.notna() & ~nxt.str.contains("I do not use")
    o["verify_external"] = np.where(ok, (nxt.str.contains("Read official docs") | nxt.str.contains("Search Stack Overflow") |
                                         nxt.str.contains("Ask a coworker")).astype(float), np.nan)
    o["verify_own"] = np.where(ok, (nxt.str.contains("Inspect tests") | nxt.str.contains("Compare against codebase")).astype(float), np.nan)
    q = ["Age", "Employment", "Country", "YearsCode", "WorkExp", "OrgSize", "ICorPM", "AISelect", "AISentiment"]
    return finish(o, d[q].notna().mean(axis=1).values)

def derive(o):
    o["sigma"] = np.where(o.year < 2025, o.yearscodepro, o.workexp)       # security: years in the profession
    o["theta"] = np.clip(o.yearscode - o.sigma, 0, None)                   # investment: coding years before it
    return o

if __name__ == "__main__":
    D, OUT = sys.argv[1], sys.argv[2]
    parts = [clean_old(f"{D}/so{y}/results.csv", y) for y in [2023, 2024, 2025]]
    parts.append(clean_2026(f"{D}/split2026/so2026_explore_70.csv"))
    A = derive(pd.concat(parts, ignore_index=True))
    flags = [c for c in A.columns if c.startswith("f_")]
    log = A.groupby("year")[flags].sum().T
    log["all_years"] = log.sum(axis=1)
    log.loc["rows"] = list(A.groupby("year").size()) + [len(A)]
    log.loc["professional developers"] = list(A.groupby("year").prodev.sum()) + [A.prodev.sum()]
    log.to_csv(f"{OUT}/cleaning_log.csv")
    A = A[~A.f_duplicate_id]
    A.to_pickle(f"{OUT}/clean_all.pkl"); A.to_csv(f"{OUT}/clean_all.csv.gz", index=False)
    print(log.to_string())
