# Preprocessing

Run in order, from a folder holding the raw survey files (`so2023/results.csv` ... `split2026/so2026_explore_70.csv`):

1. `clean.py <data_dir> <out_dir>`: harmonises 2023–2026, recodes non-answers, applies cleaning rules with flags (`cleaning_log.csv`).
2. `missing.py <out_dir>`: missingness by variable and year (`missing_by_year.csv`) and what predicts it (`missing_predictors.csv`).
3. `impute.py <out_dir>`: MICE, 5 imputations per year, predictors and controls only (outcomes never replaced); z-scored clustering features (`scalers.csv`).
4. `check.py <out_dir>`: imputed vs observed values, and key models with MICE vs complete cases (`check_mice_vs_cc.csv`).

The sealed 2026 holdout is never read. `clean_2026()` in `clean.py` and the saved 2026 imputers and scalers are applied to it unchanged later.
5. `encode.py <out_dir>`: one-hot encodes unordered categories (work setting, employment, job threat, where AI was learned, region, top 10 countries); missing stays missing. Every encoded variable is listed in `encoding_dictionary.csv`.
