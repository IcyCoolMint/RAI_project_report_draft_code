import pathlib

HERE = pathlib.Path(__file__).resolve().parent      # ...\csv_files\code
BASE = HERE.parent                                   # ...\csv_files

# Small outputs only (profiles, aggregates, figures); data stays where it is.
ROOT = BASE / "rai_audit"
OUT = ROOT / "outputs"
DB = ROOT / "eda.duckdb"
TMP = pathlib.Path.home() / "duck_tmp"               # spill dir, outside OneDrive

MEMORY_LIMIT = "12GB"  # keep below physical RAM
THREADS = 8

# Logical table -> source stems (shards of one table are listed together)
TABLES = {
    "base":       ["train_base"],
    "applprev_1": [f"train_applprev_1_{i}" for i in range(2)],
    "applprev_2": ["train_applprev_2"],
    "bureau_a_1": [f"train_credit_bureau_a_1_{i}" for i in range(4)],
    "bureau_a_2": [f"train_credit_bureau_a_2_{i}" for i in range(11)],
    "static_0":   [f"train_static_0_{i}" for i in range(2)],
    "static_cb":  ["train_static_cb_0"],
    "person_1":   ["train_person_1"],
    "person_2":   ["train_person_2"],
    "tax_a":      ["train_tax_registry_a_1"],
    "tax_b":      ["train_tax_registry_b_1"],
    "tax_c":      ["train_tax_registry_c_1"],
    "bureau_b_1": ["train_credit_bureau_b_1"],
    "bureau_b_2": ["train_credit_bureau_b_2"],
    "debitcard_1": ["train_debitcard_1"],
    "deposit_1":  ["train_deposit_1"],
    "other_1":    ["train_other_1"],
}

INT_COLS = {"case_id", "num_group1", "num_group2", "MONTH", "WEEK_NUM", "target"}

# stem -> path for every train_*.parquet under csv_files (any subfolder depth)
PARQUET = {p.stem: p for p in BASE.rglob("train_*.parquet") if ROOT not in p.parents}

for p in (TMP, OUT / "profiles", OUT / "missingness", OUT / "features",
          OUT / "drift", OUT / "fairness", OUT / "figures"):
    p.mkdir(parents=True, exist_ok=True)

print(f"[config] ROOT={ROOT}\n[config] found {len(PARQUET)} train parquet files")