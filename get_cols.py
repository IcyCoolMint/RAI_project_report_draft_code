import pandas as pd
R = r"C:\Users\amanm\OneDrive\Desktop 1\csv_files\rai_audit\outputs"
ov = pd.read_csv(R + r"\profiles\table_overview.csv")
print(len(ov), "tables;", ov[["dup_key_groups", "dup_key_extra_rows"]].sum().to_dict())
sc = pd.read_csv(R + r"\profiles\schema.csv")
skip = {"case_id", "num_group1", "num_group2", "WEEK_NUM", "MONTH", "target", "date_decision"}
d = sc[~sc.column.isin(skip)].drop_duplicates(["table", "column"])
print(d.column.str[-1].value_counts())