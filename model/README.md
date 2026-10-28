# Model code

- `build.py` builds a harmonised person-level and person-task dataset from the raw Stack Overflow survey files (2023–2025) and the 2026 70% exploration file. It never reads the sealed 30% holdout.
- `estimate.py` fits the logistic models with standard errors clustered by respondent. Run it with `EU` or `ALL`.
- `results_EU.csv`, `results_ALL.csv`: exploratory estimates (odds ratios per 10 years, 95% CI).
- `results_EU_agecontrol.csv`: the contesting and sentiment models with age group added.

Usage: `python3 -I build.py <data_dir>` then `python3 -I estimate.py <data_dir> EU`. `<data_dir>` holds `so2023/results.csv`, `so2024/results.csv`, `so2025/results.csv` and `split2026/so2026_explore_70.csv`.
