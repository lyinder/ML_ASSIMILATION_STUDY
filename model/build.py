"""Build a harmonised person-level and person-task dataset, Stack Overflow survey 2023-2026.
2026 uses only the 70% exploration file; the sealed 30% holdout is never read here.
Usage: python3 -I build.py <scratchpad_dir>
"""
import sys, numpy as np, pandas as pd
R = sys.argv[1]
EU = ("Austria Belgium Bulgaria Croatia Cyprus Czech Denmark Estonia Finland France Germany Greece Hungary "
      "Ireland Italy Latvia Lithuania Luxembourg Malta Netherlands Poland Portugal Romania Slovakia Slovenia Spain Sweden").split()
def is_eu(c): return c.fillna("").apply(lambda x: any(x.startswith(e) for e in EU))
def num(x): return pd.to_numeric(pd.Series(x).replace({"Less than 1 year": 0, "More than 50 years": 51}), errors="coerce").values
def tl(x): return set() if pd.isna(x) else {t.strip() for t in str(x).split(";") if t.strip()}

# Eight tasks asked in every year. CORE = producing and judging code (declared from theory, see pre-registration).
TASKS = ["Writing code", "Debugging", "Reviewing code", "Documenting code", "Testing code",
         "Learning a codebase", "Project planning", "Deployment and monitoring"]
CORE = {"Writing code", "Debugging", "Reviewing code"}
OLD = {"Writing code": "Writing code", "Debugging": "Debugging", "Committing and reviewing code": "Reviewing code",
       "Documenting code": "Documenting code", "Testing code": "Testing code",
       "Learning about a codebase": "Learning a codebase", "Project planning": "Project planning",
       "Deployment and monitoring": "Deployment and monitoring"}
NEW = {"Generating code in a familiar area": "Writing code", "Generating code in an unfamiliar area": "Writing code",
       "Debugging, troubleshooting, or refactoring code": "Debugging",
       "Reviewing code, pull requests, or design decisions": "Reviewing code",
       "Writing documentation, comments, or knowledge base content": "Documenting code",
       "Writing or improving tests": "Testing code", "Understanding or onboarding to a new codebase": "Learning a codebase",
       "Writing tickets, requirements, plans, or project updates": "Project planning",
       "Deploying, operating, or troubleshooting production systems": "Deployment and monitoring"}
def map_old(s): return {OLD.get("Debugging" if t.startswith("Debugging") else t) for t in s} - {None}
def map_new(s): return {NEW[t] for t in s if t in NEW}

def common(df, y, edu_col, sent_col, sent_unfav):
    out = pd.DataFrame({"year": y, "eu": is_eu(df.Country).values,
                        "us": df.Country.fillna("").str.startswith("United States").values,
                        "country": df.Country.values, "age": df.Age.values})
    out["grad"] = df[edu_col].astype(str).str.contains("Master|Professional degree").astype(float).values
    out.loc[df[edu_col].isna().values, "grad"] = np.nan
    org = df.OrgSize.astype(str)
    out["large_org"] = org.str.contains("1,000|5,000|10,000").astype(float).values
    out.loc[(df.OrgSize.isna() | org.str.contains("know")).values, "large_org"] = np.nan
    out["manager"] = df.ICorPM.astype(str).str.contains("manager").astype(float).values
    out.loc[df.ICorPM.isna().values, "manager"] = np.nan
    s = df[sent_col]
    out["unfav"] = s.isin(sent_unfav).astype(float).values
    out.loc[(s.isna() | s.isin(["Unsure", "Not sure"])).values, "unfav"] = np.nan
    return out

frames = []
for y in [2023, 2024, 2025]:
    d = pd.read_csv(f"{R}/so{y}/results.csv", low_memory=False, encoding="utf-8-sig")
    d = d[d.MainBranch.eq("I am a developer by profession")].reset_index(drop=True)
    o = common(d, y, "EdLevel", "AISent", ["Unfavorable", "Very unfavorable"])
    yc = num(d.YearsCode)
    pro = num(d.YearsCodePro) if y < 2025 else num(d.WorkExp)   # security: years in the profession
    o["sigma"] = pro; o["theta"] = np.clip(yc - pro, 0, None)  # investment: coding years before the profession
    use = d.AISelect.astype(str).str.startswith("Yes")
    cur = d["AIToolCurrently Using"].map(tl) if y < 2025 else \
        d["AIToolCurrently mostly AI"].map(tl).combine(d["AIToolCurrently partially AI"].map(tl), lambda a, b: a | b)
    tasks = cur.map(map_old)
    o["uses_ai"] = np.where(d.AISelect.isna(), np.nan, use.astype(float))
    o["tasks"] = [t if u else set() for t, u in zip(tasks, use)]
    o["has_tasks"] = (use & (tasks.map(len) > 0)).values
    o["threat"] = d["AIThreat"].values if "AIThreat" in d else np.nan
    frames.append(o)

e = pd.read_csv(f"{R}/split2026/so2026_explore_70.csv", low_memory=False)
e = e[(e.CodeMain == "I create code as a primary part of my job") &
      e.Employment.isin(["Employed", "Independent contractor, freelancer, or self-employed"])].reset_index(drop=True)
o = common(e, 2026, "Education", "AISentiment", ["Somewhat unfavorable", "Very unfavorable"])
pro = num(e.WorkExp); o["sigma"] = pro; o["theta"] = np.clip(num(e.YearsCode) - pro, 0, None)
sel = e.AISelect
none_only = sel.fillna("").eq("I don’t use AI tools")
o["uses_ai"] = np.where(sel.isna(), np.nan, (~none_only).astype(float))
cur = e.AIWorkCurrentlyUse.map(tl).map(map_new)
o["tasks"] = [t if u == 1 else set() for t, u in zip(cur, o.uses_ai)]
o["has_tasks"] = ((o.uses_ai == 1) & (cur.map(len) > 0)).values
o["threat"] = np.nan
# 2026-only measures
press = {"AI use is mostly optional and left up to individuals": "low",
         "My organization does not currently use AI tools or agents for work": "low",
         "My organization provides approved AI tools and guidance for how to use them": "medium",
         "Some teams actively encourage AI use, but practices vary across the organization": "medium",
         "My organization expects some employees to use AI tools for certain types of work": "high",
         "My organization measures or evaluates AI use as part of employee or team performance": "high"}
o["pressure"] = e.AIUse.map(press).values
nxt = e.AIWorkNext.fillna("")
ok = e.AIWorkNext.notna() & ~nxt.str.contains("I do not use")
o["verify_external"] = np.where(ok, (nxt.str.contains("Read official docs") | nxt.str.contains("Search Stack Overflow") |
                                     nxt.str.contains("Ask a coworker")).astype(float), np.nan)
o["verify_own"] = np.where(ok, (nxt.str.contains("Inspect tests") | nxt.str.contains("Compare against codebase")).astype(float), np.nan)
o["use_as_is"] = np.where(ok, nxt.str.contains("Use as-is").astype(float), np.nan)
o["avoid_skill_loss"] = np.where(e.AIAvoid.isna(), np.nan, e.AIAvoid.fillna("").str.contains("losing job skills").astype(float))
# task-level usefulness and satisfaction for the six tasks with AIImpact/AISat items
IMP = {2: "Writing code", 4: "Debugging", 3: "Reviewing code", 6: "Architecture", 5: "Documenting code", 7: "Communication"}
ITEM = {"Writing code": "Generating code in a familiar area", "Debugging": "Debugging, troubleshooting, or refactoring code",
        "Reviewing code": "Reviewing code, pull requests, or design decisions",
        "Architecture": "Architecture or technical design exploration",
        "Documenting code": "Writing documentation, comments, or knowledge base content",
        "Communication": "Writing emails, messages, summaries, or other communication"}
rows = []
w = e.AIWorkCurrentlyUse.fillna(""); niw = e.AIWorkNotInWorkflow.fillna("")
answered = e.AIWorkCurrentlyUse.notna() | e.AIWorkDoNotPlanToUse.notna()
for i, t in IMP.items():
    imp = e[f"AIImpact_{i}"]; sat = e[f"AISat_{i}"]
    keep = answered & imp.isin(["Very useful", "Useful", "No strong opinion", "More impractical than useful, not useful"]) \
        & ~niw.str.contains(ITEM[t], regex=False)
    rows.append(pd.DataFrame({"pid": np.where(keep)[0], "task": t, "core": float(t in CORE | {"Architecture"}),
                              "useful": imp[keep].isin(["Very useful", "Useful"]).astype(float).values,
                              "use": w[keep].str.contains(ITEM[t], regex=False).astype(float).values,
                              "sat_down": sat[keep].map(lambda v: np.nan if pd.isna(v) else
                                                        float(v in ["Somewhat decreased", "Greatly decreased"])).values}))
T26 = pd.concat(rows, ignore_index=True)
frames.append(o)

P = pd.concat(frames, ignore_index=True)
P = P[P.sigma.between(0, 50) & P.theta.between(0, 40)].copy()
P["pid"] = np.arange(len(P))
P.to_pickle(f"{R}/model/persons.pkl")
# person-task panel, 8 shared tasks, AI users with task answers
TP = []
for _, r in P[P.has_tasks].iterrows():
    for t in TASKS:
        TP.append((r.pid, r.year, t, float(t in CORE), float(t in r.tasks)))
TP = pd.DataFrame(TP, columns=["pid", "year", "task", "core", "use"])
TP.to_pickle(f"{R}/model/tasks.pkl")
# 2026 six-task panel: re-key pid to persons.pkl
p26 = P[P.year == 2026]
o_index = o.index[(o.sigma.between(0, 50)) & (o.theta.between(0, 40))]
key = dict(zip(o_index, p26.pid.values))
T26["pid"] = T26.pid.map(key); T26 = T26.dropna(subset=["pid"])
T26.to_pickle(f"{R}/model/tasks2026.pkl")
print("persons by year:", P.groupby("year").size().to_dict(), "| EU:", P[P.eu].groupby("year").size().to_dict())
print("task rows:", len(TP), "| 2026 six-task rows:", len(T26))
