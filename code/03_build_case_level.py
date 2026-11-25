"""
03_build_case_level.py

Build (1) a demographics table and (2) a case-level feature table with one row per case_id.

Usage
  python 03_build_case_level.py
      demographics + aggregates of EVERY column of every depth-1/2 table + all static columns


Outputs (outputs/features/)
  demo.parquet, demo_columns.csv   one row per case: protected-attribute candidates and proxies
  agg_plan.csv                     feature, source table, source column, aggregate
  features_unmatched.csv           model features that could not be built
  case_level_agg\\chunk_*.parquet   one row per case_id, written in row chunks (row count asserted
                                   equal to base); DuckDB view 'case_level' reads all chunks
  parts\\<tag>\\*.parquet            cached aggregate batches (batches from earlier runs with identical
                                   columns are reused, so adding tables does not recompute old ones)
"""

import argparse
import hashlib
import os
import re
import shutil
import time

import duckdb
import pandas as pd

from config import DB, INT_COLS, MEMORY_LIMIT, OUT, THREADS, TMP

FEAT = OUT / "features"
AGG_DEPTH = {
    "applprev_1": 1,
    "applprev_2": 2,
    "bureau_a_1": 1,
    "bureau_a_2": 2,
    "person_1": 1,
    "person_2": 2,
    "tax_a": 1,
    "tax_b": 1,
    "tax_c": 1,
    "bureau_b_1": 1,
    "bureau_b_2": 2,
    "debitcard_1": 1,
    "deposit_1": 1,
    "other_1": 1,
}
STATIC = ["static_0", "static_cb"]
SEX_COLS = ["sex_738L", "gender_992L"]
DOB_COLS = ["birth_259D", "birthdate_574D", "dateofbirth_337D", "dateofbirth_342D"]
PROTECTED_RE = re.compile(r"^(sex|gender|birth|dateofbirth)", re.I)
PROXY_RE = re.compile(
    r"(marital|family|education|zipcode|district|addr|empl_|housetype|"
    r"mainoccupation|relationship|persontype|role)",
    re.I,
)
AGG_NAMES = ("max", "min", "mean", "median", "std", "var", "sum", "last", "first", "n_unique", "nunique")
NUMERIC_ONLY = {"mean", "median", "std", "var", "sum"}
GENERIC = {
    "numeric": ["max", "min", "mean", "last"],
    "date": ["max", "min", "last"],
    "string": ["last", "max"],
}
BATCH = 60


def q(n):
    return '"' + n.replace('"', '""') + '"'


def sql_path(p):
    return str(p).replace("\\", "/").replace("'", "''")


def category(col, dtype):
    d = dtype.upper()
    if d.startswith(("DATE", "TIMESTAMP")):
        return "date"
    if d.startswith(("VARCHAR", "STRING")):
        return "date" if (col == "date_decision" or col.endswith("D")) else "string"
    return "numeric"


def agg_expr(agg, x, order):
    expressions = {
        "max": f"max({x})",
        "min": f"min({x})",
        "mean": f"avg({x})",
        "median": f"median({x})",
        "std": f"stddev_samp({x})",
        "var": f"var_samp({x})",
        "sum": f"sum({x})",
        "last": f"arg_max({x}, {order})",
        "first": f"arg_min({x}, {order})",
        "n_unique": f"count(DISTINCT {x})",
        "nunique": f"count(DISTINCT {x})",
    }
    return expressions[agg]


def src_expr(col, cat, dates, alias="t"):
    x = f"{alias}.{q(col)}"
    if cat == "numeric":
        return f"CAST({x} AS DOUBLE)"
    if cat == "date":
        d = f"TRY_CAST({x} AS DATE)"
        return f"({d} - TRY_CAST(b.date_decision AS DATE))" if dates == "days" else d
    return x


def build_demo(con, schemas):
    if "person_1" not in schemas:
        print("[demo] person_1 view missing; skipped")
        return

    p_cols = [c for c in schemas["person_1"] if c not in ("case_id", "num_group1")]
    c_cols = [c for c in schemas.get("static_cb", {}) if c != "case_id" and c not in p_cols]

    sex = [f"p.{q(c)}" for c in SEX_COLS if c in schemas["person_1"]]
    dobs = [
        (f"p.{q(c)}" if c in schemas["person_1"] else f"c.{q(c)}")
        for c in DOB_COLS
        if c in schemas["person_1"] or c in schemas.get("static_cb", {})
    ]

    sex_sql = f"COALESCE({', '.join(sex)}) AS sex" if sex else "NULL AS sex"
    dob_sql = f"COALESCE({', '.join(f'TRY_CAST({d} AS DATE)' for d in dobs)})" if dobs else "NULL::DATE"

    sel = ", ".join([f"p.{q(c)}" for c in p_cols] + [f"c.{q(c)}" for c in c_cols])
    join_c = "LEFT JOIN static_cb c USING (case_id)" if "static_cb" in schemas else ""

    demo_path = FEAT / "demo.parquet"
    con.execute(f"""
        COPY (
          SELECT
            b.case_id,
            b.date_decision,
            b.MONTH,
            b.WEEK_NUM,
            b.target,
            {sex_sql},
            {dob_sql} AS date_of_birth,
            (TRY_CAST(b.date_decision AS DATE) - {dob_sql}) / 365.25 AS age_years,
            {sel}
          FROM base b
          LEFT JOIN (SELECT * FROM person_1 WHERE num_group1 = 0) p USING (case_id)
          {join_c}
        ) TO '{sql_path(demo_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    demo = f"read_parquet('{sql_path(demo_path)}')"
    n, with_p, sex_nn, age_nn = con.execute(
        f"SELECT COUNT(*), COUNT({q(p_cols[0])}), COUNT(sex), COUNT(age_years) FROM {demo}"
    ).fetchone()
    print(
        f"[demo] {n:,} cases; applicant row: {with_p / n:.1%}; "
        f"sex non-null: {sex_nn / n:.1%}; age non-null: {age_nn / n:.1%}"
    )
    print(
        con.execute(
            f"SELECT sex, COUNT(*) n, AVG(target) AS default_rate FROM {demo} GROUP BY sex ORDER BY n DESC"
        ).df().to_string(index=False)
    )
    print(
        con.execute(
            f"SELECT quantile_cont(age_years, [0, 0.01, 0.5, 0.99, 1]) AS age_quantiles FROM {demo}"
        ).fetchone()[0]
    )

    rows = []
    for c in p_cols + c_cols:
        if PROTECTED_RE.match(c):
            role = "protected_candidate"
        elif PROXY_RE.search(c):
            role = "proxy_candidate"
        else:
            role = "other"
        nn = con.execute(f"SELECT COUNT({q(c)}) FROM {demo}").fetchone()[0]
        rows.append({
            "column": c,
            "source": "person_1" if c in p_cols else "static_cb",
            "null_pct": round((1 - nn / n) * 100, 2),
            "role": role,
        })
    pd.DataFrame(rows).to_csv(FEAT / "demo_columns.csv", index=False)


def make_plan(schemas, args):
    """Return (plan DataFrame, static column dict, derived list, unmatched list)."""
    colmap = {}
    for t, cols in schemas.items():
        if t in AGG_DEPTH or t in STATIC:
            for c in cols:
                if c not in ("case_id", "num_group1", "num_group2"):
                    colmap.setdefault(c, []).append(t)

    plan = []
    unmatched = []
    statics = {t: [] for t in STATIC if t in schemas}
    derived = []
    names = []

    if args.features is not None:
        with open(args.features, encoding="utf-8") as fh:
            names = [line.strip() for line in fh if line.strip()]

        for name in names:
            if name in ("month_decision", "weekday_decision"):
                derived.append(name)
                continue
            if name in INT_COLS or name == "date_decision":
                continue

            hit = colmap.get(name)
            if hit and hit[0] in STATIC:
                statics[hit[0]].append(name)
                continue

            m = re.match(rf"^({'|'.join(AGG_NAMES)})_(.+)$", name)
            if m and m.group(2) in colmap and colmap[m.group(2)][0] in AGG_DEPTH:
                agg, col = m.groups()
                tbl = colmap[col][0]
                cat = category(col, schemas[tbl][col])
                if agg in NUMERIC_ONLY and cat != "numeric":
                    unmatched.append({"feature": name, "reason": f"{agg} needs a numeric column"})
                else:
                    plan.append({"feature": name, "table": tbl, "column": col, "agg": agg, "cat": cat})
            else:
                unmatched.append({"feature": name, "reason": "no matching table/column/aggregate"})
    else:
        drop = set()
        flags_path = OUT / "profiles" / "column_flags.csv"
        if args.drop_flagged and flags_path.exists():
            flags = pd.read_csv(flags_path).fillna("")
            bad = flags[flags["flags"].str.contains("all_null|constant")]
            drop = set(zip(bad.table, bad.column))

        derived = ["month_decision", "weekday_decision"]
        held_out = []
        kept_protected = []

        for t, cols in schemas.items():
            for c, dtype in cols.items():
                if c in ("case_id", "num_group1", "num_group2") or (t, c) in drop:
                    continue
                if PROTECTED_RE.match(c):
                    if args.exclude_protected:
                        held_out.append(f"{t}.{c}")
                        continue
                    kept_protected.append(f"{t}.{c}")

                cat = category(c, dtype)
                if t in STATIC:
                    statics[t].append(c)
                elif t in AGG_DEPTH:
                    for agg in GENERIC[cat]:
                        plan.append({
                            "feature": f"{agg}_{c}",
                            "table": t,
                            "column": c,
                            "agg": agg,
                            "cat": cat,
                        })

        if held_out:
            print(
                f"[plan] protected-attribute candidates held out of the feature table "
                f"(see demo.parquet): {', '.join(held_out)}"
            )
        if kept_protected:
            print(
                f"[plan] feature table INCLUDES protected-attribute candidates: "
                f"{', '.join(kept_protected)}"
            )

    if args.features is not None:
        strip = re.compile(rf"^({'|'.join(AGG_NAMES)})_")
        used = sorted({n for n in names if PROTECTED_RE.match(strip.sub("", n))})
        if used:
            print(f"[plan] WARNING: the model uses protected-attribute candidates directly: {', '.join(used)}")

    plan_df = pd.DataFrame(plan, columns=["feature", "table", "column", "agg", "cat"])
    return plan_df, statics, derived, unmatched


def reuse_cached(con, tbl, batch_idx, expected, dest, parts_root):
    """Reuse an aggregate batch from an earlier run when its columns match exactly."""
    for cand in parts_root.glob(f"*/{tbl}_{batch_idx:03d}.parquet"):
        if cand.parent == dest.parent:
            continue
        names = [
            r[0]
            for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql_path(cand)}')").fetchall()
        ]
        if names == expected:
            try:
                os.link(cand, dest)
            except OSError:
                shutil.copy2(cand, dest)
            return True
    return False


def build_features(con, schemas, args):
    plan, statics, derived, unmatched = make_plan(schemas, args)
    plan.drop_duplicates("feature", inplace=True)
    plan.to_csv(FEAT / "agg_plan.csv", index=False)
    pd.DataFrame(unmatched, columns=["feature", "reason"]).to_csv(FEAT / "features_unmatched.csv", index=False)

    n_tables = plan.table.nunique() if len(plan) else 0
    n_static = sum(map(len, statics.values()))
    print(f"[plan] {len(plan):,} aggregate features from {n_tables} tables, {n_static} static columns, {len(unmatched)} unmatched")

    tag = hashlib.md5((args.dates + "|" + ",".join(plan.feature)).encode()).hexdigest()[:10]
    parts_dir = FEAT / "parts" / tag
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []

    for tbl, grp in plan.groupby("table", sort=False):
        cols = schemas[tbl]
        order = "t.num_group1 * 100000 + t.num_group2" if "num_group2" in cols else "t.num_group1"
        need_base = args.dates == "days" and (grp.cat == "date").any()
        join = "JOIN base b USING (case_id)" if need_base else ""
        rows = grp.to_dict("records")

        for k in range(0, len(rows), BATCH):
            chunk = rows[k : k + BATCH]
            batch_num = k // BATCH
            path = parts_dir / f"{tbl}_{batch_num:03d}.parquet"
            parts.append(path)

            if path.exists() and not args.force:
                continue

            expected = ["case_id"] + (["n_rows_" + tbl] if k == 0 else []) + [r["feature"] for r in chunk]
            if not args.force and args.dates == "days" and reuse_cached(con, tbl, batch_num, expected, path, FEAT / "parts"):
                print(f"  {tbl} batch {batch_num}: reused from an earlier run")
                continue

            t0 = time.time()
            exprs = [
                f"{agg_expr(r['agg'], src_expr(r['column'], r['cat'], args.dates), order)} AS {q(r['feature'])}"
                for r in chunk
            ]
            if k == 0:
                exprs.insert(0, f"COUNT(*) AS {q('n_rows_' + tbl)}")

            tmp = path.with_suffix(".tmp")
            con.execute(f"""
                COPY (
                  SELECT t.case_id, {', '.join(exprs)}
                  FROM {tbl} t {join}
                  GROUP BY t.case_id
                ) TO '{sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            tmp.replace(path)
            print(f"  {tbl} batch {batch_num}: {len(chunk)} features, {time.time() - t0:.0f}s")

    # Assemble in case_id chunks so peak memory stays predictable on wide feature sets.
    sel = ["b.case_id", "b.date_decision", "b.MONTH", "b.WEEK_NUM", "b.target"]
    static_joins = []
    if "month_decision" in derived:
        sel.append("month(TRY_CAST(b.date_decision AS DATE)) AS month_decision")
    if "weekday_decision" in derived:
        sel.append("isodow(TRY_CAST(b.date_decision AS DATE)) AS weekday_decision")

    for i, (t, cols) in enumerate(statics.items()):
        if not cols:
            continue
        n, nd = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT case_id) FROM {t}").fetchone()
        if n != nd:
            raise SystemExit(f"{t}: case_id not unique ({n:,} rows, {nd:,} cases); it is not depth 0")
        static_joins.append((t, f"s{i}"))
        for c in dict.fromkeys(cols):
            sel.append(f"{src_expr(c, category(c, schemas[t][c]), args.dates, f's{i}')} AS {q(c)}")

    sorted_dir = parts_dir / "sorted"
    sorted_dir.mkdir(exist_ok=True)
    sparts = []
    for path in parts:
        sp = sorted_dir / path.name
        if args.force or not sp.exists():
            tmp = sp.with_suffix(".tmp")
            con.execute(f"""
                COPY (SELECT * FROM read_parquet('{sql_path(path)}') ORDER BY case_id)
                TO '{sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
            """)
            tmp.replace(sp)
        sparts.append(sp)
    print(f"[assemble] {len(sparts)} sorted batches")

    out_dir = FEAT / "case_level_agg"
    out_dir.mkdir(exist_ok=True)
    tag_file = out_dir / "_tag.txt"
    expected_tag = f"{tag}|{args.chunk_rows}"
    if args.force or not tag_file.exists() or tag_file.read_text() != expected_tag:
        for f in out_dir.glob("chunk_*.parquet"):
            f.unlink()
        tag_file.write_text(expected_tag)

    ids = [r[0] for r in con.execute("SELECT case_id FROM base ORDER BY case_id").fetchall()]
    bounds = [
        (ids[i], ids[min(i + args.chunk_rows, len(ids)) - 1])
        for i in range(0, len(ids), args.chunk_rows)
    ]

    for k, (lo, hi) in enumerate(bounds):
        path = out_dir / f"chunk_{k:03d}.parquet"
        if path.exists():
            continue

        t0 = time.time()
        rng = f"case_id BETWEEN {lo} AND {hi}"
        joins = [f"LEFT JOIN (SELECT * FROM {t} WHERE {rng}) {a} USING (case_id)" for t, a in static_joins]
        chunk_sel = list(sel)
        for j, sp in enumerate(sparts):
            joins.append(
                f"LEFT JOIN (SELECT * FROM read_parquet('{sql_path(sp)}') WHERE {rng}) a{j} USING (case_id)"
            )
            chunk_sel.append(f"a{j}.* EXCLUDE (case_id)")

        tmp = path.with_suffix(".tmp")
        con.execute(f"""
            COPY (
              SELECT {', '.join(chunk_sel)}
              FROM (SELECT * FROM base WHERE {rng}) b
              {' '.join(joins)}
              ORDER BY b.case_id
            ) TO '{sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        tmp.replace(path)
        print(f"  chunk {k + 1}/{len(bounds)}: case_id {lo}..{hi}, {time.time() - t0:.0f}s")

    glob = f"{sql_path(out_dir)}/chunk_*.parquet"
    con.execute(f"CREATE OR REPLACE VIEW case_level AS SELECT * FROM read_parquet('{glob}')")
    n_base = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
    n_out = con.execute("SELECT COUNT(*) FROM case_level").fetchone()[0]
    n_col = len(con.execute("DESCRIBE SELECT * FROM case_level").fetchall())
    assert n_out == n_base, f"row count {n_out} != base {n_base}"
    print(
        f"[done] {out_dir}: {len(bounds)} chunks, {n_out:,} rows x {n_col:,} columns "
        f"(row count equals base; DuckDB view 'case_level' registered)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="?", const=str(FEAT / "lgbm_feature_names.txt"), default=None)
    ap.add_argument("--dates", choices=["days", "raw"], default="days")
    ap.add_argument("--demo-only", action="store_true")
    ap.add_argument("--drop-flagged", action="store_true")
    ap.add_argument("--exclude-protected", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=50000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{sql_path(TMP)}'")

    views = [r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()]
    if "base" not in views:
        raise SystemExit("View 'base' not found. Run 01_register_parquet.py first.")

    # Ignore our own output view since its underlying files may still be building.
    views = [v for v in views if v != "case_level"]
    schemas = {
        v: {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM {v}").fetchall()}
        for v in views
    }

    build_demo(con, schemas)
    if not args.demo_only:
        build_features(con, schemas, args)
    con.close()


if __name__ == "__main__":
    main()