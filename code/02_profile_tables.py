"""
02_profile_tables.py

Profile each logical table (DuckDB views from 01_register_parquet.py).

Usage
  python 02_profile_tables.py                 # all tables, smallest first

Per table, written to outputs/profiles/
  summ_<t>.parquet         DuckDB SUMMARIZE (min, max, avg, std, q25/50/75, approx_unique, null %)
  colstats_<t>.parquet     exact null counts; numeric: skew, kurtosis, zero / -1 / sentinel / inf counts;
                           date: min, max, unparsed, <1900, >2100, share after date_decision;
                           string: empty count, approx distinct, top value
  flags_<t>.csv            one row per column with rule-based flags (see FLAGS below)
  overview_<t>.csv         rows, distinct cases, rows per case, base cases with no rows,
                           orphan cases, duplicate keys, shard coverage
  rows_per_case_<t>.csv    distribution of rows per case_id
  dup_keys_<t>.csv         sample of duplicated key tuples (only if any)
Combined at the end: table_overview.csv, column_flags.csv

FLAGS
  all_null, null>=95%, constant, iqr_zero (q25 == q75, so about half the rows share one value),
  high_cardinality (M columns with more than 50 distinct values), sentinel>=5% (-1 or 9999999-type
  values), zero_heavy (>=50% zeros), has_inf, unparsed_dates, impossible_dates (<1900 or >2100),
  dates_after_decision>0 (informational; due dates can legitimately be later)
"""

import sys
import time

import duckdb
import pandas as pd

from config import DB, INT_COLS, MEMORY_LIMIT, OUT, PARQUET, TABLES, THREADS, TMP

PROF = OUT / "profiles"
BATCH = 50
SENTINELS = "9999999, 99999999, 999999999, -9999999, -99999999, -999999999"
ORDER = [
    "base", "static_cb", "static_0", "person_1", "person_2",
    "tax_a", "tax_b", "tax_c", "bureau_b_1", "bureau_b_2",
    "debitcard_1", "deposit_1", "other_1", "applprev_1", "applprev_2",
    "bureau_a_1", "bureau_a_2",
]


def q(name):
    return '"' + name.replace('"', '""') + '"'


def sql_path(p):
    return str(p).replace("\\", "/").replace("'", "''")


def is_date_name(c):
    return c == "date_decision" or c.endswith("D")


def category(col, dtype):
    d = dtype.upper()
    if d.startswith(("DATE", "TIMESTAMP")):
        return "date"
    if d.startswith(("VARCHAR", "STRING")):
        return "date" if is_date_name(col) else "string"
    return "numeric"


def batches(xs, n=BATCH):
    # Wide tables choke DuckDB if we pack hundreds of expressions into one query.
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def column_stats(con, tbl, cols, n_rows):
    """cols: list of (name, dtype, category). Returns a DataFrame, one row per column."""
    recs = []
    for batch in batches(cols):
        parts = []
        meta = []
        for i, (c, dtype, cat) in enumerate(batch):
            x = q(c)
            parts.append(f"COUNT(*) - COUNT({x}) AS n_null_{i}")

            if cat == "numeric":
                v = f"CAST({x} AS DOUBLE)"
                parts += [
                    f"skewness({v}) AS skew_{i}",
                    f"kurtosis({v}) AS kurt_{i}",
                    f"COUNT(*) FILTER (WHERE {v} = 0) AS n_zero_{i}",
                    f"COUNT(*) FILTER (WHERE {v} = -1) AS n_neg1_{i}",
                    f"COUNT(*) FILTER (WHERE {v} IN ({SENTINELS})) AS n_sentinel_{i}",
                    f"COUNT(*) FILTER (WHERE isinf({v})) AS n_inf_{i}",
                ]
            elif cat == "date":
                d = f"TRY_CAST({x} AS DATE)"
                parts += [
                    f"MIN({d}) AS dmin_{i}",
                    f"MAX({d}) AS dmax_{i}",
                    f"COUNT({x}) - COUNT({d}) AS n_unparsed_{i}",
                    f"COUNT(*) FILTER (WHERE {d} < DATE '1900-01-01') AS n_lt1900_{i}",
                    f"COUNT(*) FILTER (WHERE {d} > DATE '2100-01-01') AS n_gt2100_{i}",
                ]
            else:
                parts += [
                    f"COUNT(*) FILTER (WHERE TRIM({x}) = '') AS n_empty_{i}",
                    f"approx_count_distinct({x}) AS n_distinct_approx_{i}",
                    f"approx_top_k({x}, 1)[1] AS top_value_{i}",
                ]

            meta.append((i, c, dtype, cat))

        cur = con.execute(f"SELECT {', '.join(parts)} FROM {tbl}")
        names = [d[0] for d in cur.description]
        row = dict(zip(names, cur.fetchone()))

        for i, c, dtype, cat in meta:
            r = {"table": tbl, "column": c, "dtype": dtype, "category": cat, "rows": n_rows}
            for k, v in row.items():
                metric, idx = k.rsplit("_", 1)
                if idx == str(i):
                    r[metric] = v
            recs.append(r)

        print(f"    stats {len(recs)}/{len(cols)} columns")

    df = pd.DataFrame(recs)
    df["null_pct"] = df["n_null"] / n_rows * 100 if n_rows else 0.0
    return df


def date_vs_decision(con, tbl, date_cols):
    """Share of non-null dates later than the case's date_decision (informational)."""
    out = {}
    for batch in batches(date_cols):
        parts = []
        for i, c in enumerate(batch):
            d = f"TRY_CAST(t.{q(c)} AS DATE)"
            parts.append(f"COUNT({d}) AS nn_{i}")
            parts.append(
                f"COUNT(*) FILTER (WHERE {d} > TRY_CAST(b.date_decision AS DATE)) AS after_{i}"
            )
        cur = con.execute(f"SELECT {', '.join(parts)} FROM {tbl} t JOIN base b USING (case_id)")
        names = [d[0] for d in cur.description]
        row = dict(zip(names, cur.fetchone()))
        for i, c in enumerate(batch):
            out[c] = (row[f"nn_{i}"], row[f"after_{i}"])
    return out


def build_flags(stats, summ):
    s = stats.merge(
        summ[["column_name", "approx_unique", "q25", "q75"]],
        left_on="column",
        right_on="column_name",
        how="left",
    )
    rows = []
    for _, r in s.iterrows():
        f = []
        n = r["rows"]

        if r["null_pct"] >= 100:
            f.append("all_null")
        elif r["null_pct"] >= 95:
            f.append("null>=95%")

        if r["null_pct"] < 100:
            if pd.notna(r["approx_unique"]) and r["approx_unique"] <= 1:
                f.append("constant")
            elif pd.notna(r["q25"]) and r["q25"] == r["q75"]:
                f.append("iqr_zero")

        if (
            r["column"].endswith("M")
            and r["column"] not in INT_COLS
            and pd.notna(r["approx_unique"])
            and r["approx_unique"] > 50
        ):
            f.append("high_cardinality")

        if r["category"] == "numeric" and n:
            if (r["n_neg1"] + r["n_sentinel"]) / n >= 0.05:
                f.append("sentinel>=5%")
            if r["n_zero"] / n >= 0.5:
                f.append("zero_heavy")
            if r["n_inf"] > 0:
                f.append("has_inf")

        if r["category"] == "date":
            if r["n_unparsed"] > 0:
                f.append("unparsed_dates")
            if r["n_lt1900"] > 0 or r["n_gt2100"] > 0:
                f.append("impossible_dates")
            if pd.notna(r.get("n_after_decision")) and r["n_after_decision"] > 0:
                f.append("dates_after_decision>0")

        rows.append({
            "table": r["table"],
            "column": r["column"],
            "dtype": r["dtype"],
            "null_pct": round(r["null_pct"], 3),
            "approx_unique": r["approx_unique"],
            "flags": ";".join(f),
        })

    return pd.DataFrame(rows)


def profile(con, tbl):
    t0 = time.time()
    desc = con.execute(f"DESCRIBE SELECT * FROM {tbl}").fetchall()
    cols = [(n, d, category(n, d)) for n, d, *_ in desc]
    names = [c[0] for c in cols]
    n_rows = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"[{tbl}] {n_rows:,} rows, {len(cols)} columns")

    summ = pd.concat(
        [
            con.execute(f"SUMMARIZE SELECT {', '.join(q(c) for c in b)} FROM {tbl}").df()
            for b in batches(names)
        ],
        ignore_index=True,
    )
    summ.to_parquet(PROF / f"summ_{tbl}.parquet")

    stats = column_stats(con, tbl, cols, n_rows)
    date_cols = [c for c, _, cat in cols if cat == "date" and c != "date_decision"]
    if tbl != "base" and "case_id" in names and date_cols:
        dv = date_vs_decision(con, tbl, date_cols)
        stats["n_date_nonnull"] = stats["column"].map(lambda c: dv.get(c, (None, None))[0])
        stats["n_after_decision"] = stats["column"].map(lambda c: dv.get(c, (None, None))[1])
    stats.to_parquet(PROF / f"colstats_{tbl}.parquet")

    build_flags(stats, summ).to_csv(PROF / f"flags_{tbl}.csv", index=False)

    ov = {
        "table": tbl,
        "rows": n_rows,
        "columns": len(cols),
        "shards_found": len([s for s in TABLES[tbl] if s in PARQUET]),
        "shards_expected": len(TABLES[tbl]),
        "distinct_cases": con.execute(f"SELECT COUNT(DISTINCT case_id) FROM {tbl}").fetchone()[0],
    }

    keys = [k for k in ("case_id", "num_group1", "num_group2") if k in names]
    kl = ", ".join(q(k) for k in keys)
    ng, extra = con.execute(f"""
        SELECT COUNT(*), COALESCE(SUM(c - 1), 0)
        FROM (
            SELECT COUNT(*) AS c
            FROM {tbl}
            GROUP BY {kl}
            HAVING COUNT(*) > 1
        )
    """).fetchone()
    ov["dup_key_groups"] = ng
    ov["dup_key_extra_rows"] = extra
    ov["key_columns"] = "+".join(keys)

    if ng:
        dup_df = con.execute(f"""
            SELECT {kl}, COUNT(*) AS n
            FROM {tbl}
            GROUP BY {kl}
            HAVING COUNT(*) > 1
            ORDER BY n DESC
            LIMIT 100
        """).df()
        dup_df.to_csv(PROF / f"dup_keys_{tbl}.csv", index=False)

    con.execute(f"CREATE OR REPLACE TEMP TABLE _rpc AS SELECT COUNT(*) AS c FROM {tbl} GROUP BY case_id")
    p50, p90, p99, mx, mean = con.execute("""
        SELECT
            quantile_cont(c, 0.5),
            quantile_cont(c, 0.9),
            quantile_cont(c, 0.99),
            MAX(c),
            AVG(c)
        FROM _rpc
    """).fetchone()
    ov.update({
        "rows_per_case_mean": mean,
        "rows_per_case_p50": p50,
        "rows_per_case_p90": p90,
        "rows_per_case_p99": p99,
        "rows_per_case_max": mx,
    })

    rpc_dist = con.execute(
        "SELECT c AS rows_per_case, COUNT(*) AS n_cases FROM _rpc GROUP BY c ORDER BY c LIMIT 500"
    ).df()
    rpc_dist.to_csv(PROF / f"rows_per_case_{tbl}.csv", index=False)

    if tbl != "base":
        ov["base_cases_without_rows"] = con.execute(
            f"SELECT COUNT(*) FROM base b ANTI JOIN (SELECT DISTINCT case_id FROM {tbl}) t USING (case_id)"
        ).fetchone()[0]
        ov["orphan_cases"] = con.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT case_id FROM {tbl}) t ANTI JOIN base b USING (case_id)"
        ).fetchone()[0]

    pd.DataFrame([ov]).to_csv(PROF / f"overview_{tbl}.csv", index=False)
    print(f"[{tbl}] done in {time.time() - t0:.0f}s")


def main():
    PROF.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB))
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA temp_directory='{sql_path(TMP)}'")

    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    if "base" not in views:
        sys.exit("View 'base' not found. Run 01_register_parquet.py first.")

    for tbl in ORDER:
        if tbl not in views:
            print(f"[skip] {tbl}: no view")
            continue
        if len([s for s in TABLES[tbl] if s in PARQUET]) < len(TABLES[tbl]):
            print(f"[warn] {tbl}: partial shard set; all results cover found shards only")
        profile(con, tbl)

    for pat, out in (("overview_*.csv", "table_overview.csv"), ("flags_*.csv", "column_flags.csv")):
        fs = sorted(PROF.glob(pat))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(PROF / out, index=False)

    con.close()


if __name__ == "__main__":
    main()