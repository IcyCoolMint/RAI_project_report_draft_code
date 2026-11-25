import pandas as pd, duckdb
R = r"C:\Users\amanm\OneDrive\Desktop 1\csv_files\rai_audit\outputs"

ov = pd.read_csv(R + r"\profiles\table_overview.csv").set_index("table")
for t, k in {"applprev_1": 2, "applprev_2": 3, "bureau_a_1": 2, "bureau_a_2": 3}.items():
    print(t, "rows", int(ov.loc[t, "rows"]), "data cols", int(ov.loc[t, "columns"]) - k)
print(ov[["dup_key_groups", "dup_key_extra_rows"]].sum())

sc = pd.read_csv(R + r"\profiles\schema.csv")
skip = {"case_id", "num_group1", "num_group2", "WEEK_NUM", "MONTH", "target", "date_decision"}
d = sc[~sc.column.isin(skip)].drop_duplicates(["table", "column"])
print(d.column.str[-1].value_counts())

print(duckdb.connect().execute(
    f"SELECT quantile_cont(age_years,[0.01,0.5,0.99]) FROM '{R}\\features\\demo.parquet'").fetchone()[0])