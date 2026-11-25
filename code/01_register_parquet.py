"""
01_register_parquet.py

Register the existing train Parquet files as one DuckDB view per logical table
and audit their schemas. Nothing is converted or rewritten.

Checks
- which expected files exist (per TABLES in config.py)
- row count per shard and per table
- column dtype per shard; conflicts between shards of one table
- observed dtype vs the dtype implied by the column-name suffix
    (D -> date, P/A/L/T -> numeric, M -> string, INT_COLS -> integer)

Outputs (outputs/profiles/)
  schema.csv         table, shard, column, dtype, expected, flag
  shard_rows.csv     table, shard, rows, columns
  schema_conflicts.csv  columns whose dtype differs between shards
"""

import duckdb
import pandas as pd

from config import DB, INT_COLS, MEMORY_LIMIT, OUT, PARQUET, TABLES, THREADS, TMP


def sql_path(p):
    # DuckDB string literals expect forward slashes and doubled single quotes.
    return str(p).replace("\\", "/").replace("'", "''")


def expected(col):
    if col in INT_COLS:
        return "numeric"
    if col == "date_decision" or col.endswith("D"):
        return "date"
    if col.endswith("M"):
        return "string"
    if col and col[-1] in "PALT":
        return "numeric"
    return "string"


def observed(dtype):
    d = dtype.upper()
    if d.startswith(("DATE", "TIMESTAMP")):
        return "date"
    if d.startswith(("VARCHAR", "STRING")):
        return "string"
    return "numeric"


def main():
    con = duckdb.connect(str(DB))
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA temp_directory='{sql_path(TMP)}'")

    profile_dir = OUT / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    schema = []
    shards = []
    conflicts = []

    for tbl, stems in TABLES.items():
        have = [s for s in stems if s in PARQUET]
        miss = [s for s in stems if s not in PARQUET]
        if miss:
            print(f"[warn] {tbl}: missing {miss}")
        if not have:
            continue

        # Track {col_name: {shard_name: dtype}} so we can spot cross-shard drift.
        types = {}

        for s in have:
            f = sql_path(PARQUET[s])
            desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").fetchall()
            n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{f}')").fetchone()[0]

            shards.append({
                "table": tbl,
                "shard": s,
                "rows": n,
                "columns": len(desc),
            })

            for name, dtype, *_ in desc:
                exp = expected(name)
                obs = observed(dtype)
                flag = "" if exp == obs else f"expected {exp}"
                schema.append({
                    "table": tbl,
                    "shard": s,
                    "column": name,
                    "dtype": dtype,
                    "expected": exp,
                    "flag": flag,
                })
                types.setdefault(name, {})[s] = dtype

        for col, d in types.items():
            if len(set(d.values())) > 1:
                conflicts.append({"table": tbl, "column": col, **d})

        plist = ", ".join(f"'{sql_path(PARQUET[s])}'" for s in have)
        con.execute(
            f"CREATE OR REPLACE VIEW {tbl} AS "
            f"SELECT * FROM read_parquet([{plist}], union_by_name=true)"
        )

        total = sum(r["rows"] for r in shards if r["table"] == tbl)
        print(f"[view] {tbl}: {len(have)}/{len(stems)} shards, {total:,} rows")

    pd.DataFrame(schema).to_csv(profile_dir / "schema.csv", index=False)
    pd.DataFrame(shards).to_csv(profile_dir / "shard_rows.csv", index=False)
    pd.DataFrame(conflicts).to_csv(profile_dir / "schema_conflicts.csv", index=False)

    s = pd.DataFrame(schema)
    flagged = s[s["flag"] != ""]
    print(
        f"\n[schema] {len(s):,} column entries, {len(flagged):,} flagged vs suffix, "
        f"{len(conflicts)} cross-shard dtype conflicts"
    )
    if not flagged.empty:
        summary = flagged.groupby(["table", "dtype", "flag"]).size().rename("n_columns")
        print(summary.to_string())

    if "base" in TABLES and any(r["table"] == "base" for r in shards):
        print("\n[base]")
        base_stats = con.execute("""
            SELECT
                COUNT(*) AS n,
                COUNT(DISTINCT case_id) AS n_cases,
                AVG(target) AS default_rate,
                MIN(date_decision) AS first_decision,
                MAX(date_decision) AS last_decision
            FROM base
        """).df()
        print(base_stats.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()