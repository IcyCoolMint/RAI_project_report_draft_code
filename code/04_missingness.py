"""
04_missingness.py

Missingness analysis of the case-level table (DuckDB view 'case_level', built by 03_build_case_level.py).

Usage
  python 04_missingness.py

Method
  One scan per batch of columns, using GROUPING SETS, gives null counts overall and by MONTH, target,
  sex and age band at once. Groups come from demo.parquet (written by 03 --demo-only); without it
  only MONTH and target are used. Float NaN counts as missing.

Outputs (outputs/missingness/)
  feature_missingness.csv    one row per feature: null rate; nulls explained by "no rows in the source
                             table" vs null despite having rows; default rate when null vs not null
                             (rate difference, ratio, indicator AUC, chi-square p, BH q); drift in the
                             null rate over MONTH; null-rate gap across sex and age bands; mechanism
                             label and suggested treatment
  null_by_month.parquet      feature x MONTH: n, n_null, null_pct
  null_by_group.parquet      feature x (target | sex | age_band): n, n_null
  table_coverage.csv         share of cases with at least one row in each source table, by group
  row_missingness.csv        distribution of missing features per case; default rate by decile
  row_missingness_by_group.csv
  missingness_blocks.csv     groups of features whose null indicators are (almost) identical
  missingness_pairs.csv      feature pairs with |phi| >= 0.8 (sample)

Mechanism labels are hypotheses to test, not conclusions:
  structural   nearly all nulls come from cases with no rows in the source table
  informative  default rate differs materially between null and non-null cases (MNAR-like)
  mar_candidate / mcar_like  neither of the above; MAR vs MCAR is not testable from these data alone
Imputation suggestions are hypotheses for later modeling work; LightGBM itself handles NaN natively.
"""

import argparse
import math
import time
import warnings

import duckdb
import numpy as np
import pandas as pd

from config import DB, MEMORY_LIMIT, OUT, THREADS, TMP

MISS = OUT / "missingness"
FEAT = OUT / "features"
KEYS = {"case_id", "target", "MONTH", "WEEK_NUM", "date_decision"}

DIMS = {7: "month", 11: "sex", 13: "age_band", 14: "target", 15: "all"}

AGE_SQL = """CASE
    WHEN d.age_years IS NULL THEN 'unknown'
    WHEN d.age_years < 25 THEN '<25'
    WHEN d.age_years < 35 THEN '25-34'
    WHEN d.age_years < 45 THEN '35-44'
    WHEN d.age_years < 55 THEN '45-54'
    WHEN d.age_years < 65 THEN '55-64'
    ELSE '65+'
END"""

MIN_MONTH_N = 500
STEP_JUMP = 0.20      
INFORMATIVE_DIFF = 0.01 
INFORMATIVE_RR = 1.25
MIN_NULL_DEFAULTS = 30
Q_THRESH = 1e-3
GROUP_GAP = 0.02
STRUCTURAL = 0.90
MIN_GROUP_N = 1000


def q(n):
    return '"' + n.replace('"', '""') + '"'


def sql_path(p):
    return str(p).replace("\\", "/").replace("'", "''")


def miss(col, dtype):
    x = f"cl.{q(col)}"
    if dtype.upper() in ("FLOAT", "DOUBLE"):
        return f"({x} IS NULL OR isnan({x}))"
    return f"({x} IS NULL)"


def chi2_sf(stat, df):
    if not np.isfinite(stat):
        return np.nan
    try:
        from scipy.stats import chi2
        return float(chi2.sf(stat, df))
    except ImportError:
        # Wilson-Hilferty approximation when scipy is not installed.
        if df == 1:
            return math.erfc(math.sqrt(stat / 2))
        z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_2xk(events, ns):
    """Chi-square test of equal event rates across k groups."""
    events = np.asarray(events, float)
    ns = np.asarray(ns, float)
    tot = ns.sum()
    ev = events.sum()
    if tot == 0 or ev == 0 or ev == tot or len(ns) < 2:
        return np.nan

    p = ev / tot
    exp_ev = ns * p
    exp_non = ns * (1 - p)
    stat = (((events - exp_ev) ** 2) / exp_ev + (((ns - events) - exp_non) ** 2) / exp_non).sum()
    return chi2_sf(stat, len(ns) - 1)


def bh(p):
    """Benjamini-Hochberg false discovery rate adjustment."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]

    res = np.empty_like(pv)
    res[order] = np.minimum(ranked, 1)
    out[ok] = res
    return out


def scan_columns(con, cols, dtypes, tbl_of, batch):
    """Null counts by grouping set for every column. Returns long DataFrame."""
    parts = []
    for b in range(0, len(cols), batch):
        chunk = cols[b : b + batch]
        exprs = []
        for i, c in enumerate(chunk):
            m = miss(c, dtypes[c])
            exprs.append(f"COUNT(*) FILTER (WHERE {m}) AS nn_{i}")
            t = tbl_of.get(c)
            if t and f"n_rows_{t}" in dtypes and c != f"n_rows_{t}":
                exprs.append(
                    f"COUNT(*) FILTER (WHERE {m} AND cl.{q('n_rows_' + t)} IS NULL) AS nr_{i}"
                )

        t0 = time.time()
        raw = con.execute(f"""
            SELECT
                GROUPING(m.MONTH, m.sex, m.age_band, m.target) AS gid,
                m.MONTH AS g_month,
                m.sex AS g_sex,
                m.age_band AS g_age,
                m.target AS g_target,
                COUNT(*) AS n,
                {', '.join(exprs)}
            FROM case_level cl
            JOIN meta m USING (case_id)
            GROUP BY GROUPING SETS ((m.MONTH), (m.sex), (m.age_band), (m.target), ())
        """).df()

        raw["dim"] = raw["gid"].map(DIMS)
        raw["level"] = np.select(
            [
                raw.dim == "month",
                raw.dim == "sex",
                raw.dim == "age_band",
                raw.dim == "target",
            ],
            [
                raw.g_month.astype("Int64").astype(str),
                raw.g_sex.astype(str),
                raw.g_age.astype(str),
                raw.g_target.astype("Int64").astype(str),
            ],
            default="all",
        )

        for i, c in enumerate(chunk):
            d = raw[["dim", "level", "n", f"nn_{i}"]].rename(columns={f"nn_{i}": "n_null"})
            d["feature"] = c
            d["n_null_no_rows"] = raw[f"nr_{i}"] if f"nr_{i}" in raw else np.nan
            parts.append(d)

        print(f"  scanned {min(b + batch, len(cols))}/{len(cols)} columns ({time.time() - t0:.0f}s)")

    return pd.concat(parts, ignore_index=True)


def group_stats(long, dim, min_n):
    """Per feature: gap in null rate across levels of dim (levels with n >= min_n, excluding 'unknown')."""
    g = long[(long.dim == dim) & (long.level != "unknown") & (long.n >= min_n)]
    rows = {}
    for f, d in g.groupby("feature"):
        if len(d) < 2:
            continue
        rate = d.n_null / d.n
        hi = rate.idxmax()
        lo = rate.idxmin()
        rows[f] = (
            rate.max() - rate.min(),
            d.loc[hi, "level"],
            d.loc[lo, "level"],
            chi2_2xk(d.n_null.values, d.n.values),
        )
    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=[f"{dim}_gap", f"{dim}_max_level", f"{dim}_min_level", f"{dim}_p"],
    )


def month_stats(long):
    m = long[long.dim == "month"].copy()
    m["month"] = m.level.astype(int)
    m = m[m.n >= MIN_MONTH_N]
    rate = (m.n_null / m.n).groupby([m.feature, m.month]).first().unstack(0).sort_index()

    out = pd.DataFrame(index=rate.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["null_pct_first3_months"] = rate.head(3).mean() * 100
        out["null_pct_last3_months"] = rate.tail(3).mean() * 100
        jumps = rate.diff().abs()
        out["max_month_jump"] = jumps.max()
        out["max_jump_month"] = jumps.idxmax().where(jumps.notna().any())
    return out


def build_flags(feat):
    f = feat
    f["target_diff"] = f.default_rate_null - f.default_rate_nonnull
    f["target_rr"] = f.default_rate_null / f.default_rate_nonnull

    informative = (
        (f.target_q < Q_THRESH)
        & (f.n_null_defaults >= MIN_NULL_DEFAULTS)
        & (
            (f.target_diff.abs() >= INFORMATIVE_DIFF)
            | (f.target_rr >= INFORMATIVE_RR)
            | (f.target_rr <= 1 / INFORMATIVE_RR)
        )
    )
    f["flag_informative"] = informative
    f["flag_step_change"] = f.max_month_jump >= STEP_JUMP
    f["flag_group_linked"] = (
        ((f.sex_gap >= GROUP_GAP) & (f.sex_p < Q_THRESH))
        | ((f.age_band_gap >= GROUP_GAP) & (f.age_band_p < Q_THRESH))
    )
    return f


def label(r):
    notes = []
    if r.flag_step_change:
        notes.append(
            f"null rate shifts over time (max month jump {r.max_month_jump:.2f} around {r.max_jump_month:.0f}); "
            f"check for a collection change, impute within period"
        )
    if r.flag_group_linked:
        notes.append("null rate differs by sex or age band")

    if r.null_pct == 0:
        return "none", "no action", ""
    if r.null_pct >= 100:
        return "all_null", "drop from modeling; keep in the column inventory", ""

    if pd.notna(r.structural_share) and r.structural_share >= STRUCTURAL:
        count_like = r["agg"] in ("n_unique", "nunique", "sum") or r.feature.startswith("n_rows_")
        if count_like:
            treatment = f"fill 0 for a count or sum, plus has_{r['table']} flag"
        else:
            treatment = (
                f"keep NaN (meaningful: no rows in {r['table']}); "
                f"add has_{r['table']} flag; no statistical imputation"
            )
        return "structural", treatment, "; ".join(notes)

    if r.flag_informative:
        return (
            "informative",
            "keep NaN and add an is_missing indicator; avoid median/mean imputation without it",
            "; ".join(notes),
        )

    if r.null_pct < 5:
        return "mcar_like", "median (numeric) or mode / '__MISSING__' (categorical)", "; ".join(notes)

    return (
        "mar_candidate",
        "group median (by table, month) or iterative/KNN on a sample; compare against raw NaN",
        "; ".join(notes),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--block-threshold", type=float, default=0.95)
    args = ap.parse_args()

    MISS.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB))
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{sql_path(TMP)}'")

    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    if "case_level" not in views:
        raise SystemExit("View 'case_level' not found. Run 03_build_case_level.py first.")

    dtypes = {r[0]: r[1] for r in con.execute("DESCRIBE SELECT * FROM case_level").fetchall()}
    cols = [c for c in dtypes if c not in KEYS]

    plan_path = FEAT / "agg_plan.csv"
    if plan_path.exists():
        plan = pd.read_csv(plan_path)
    else:
        plan = pd.DataFrame(columns=["feature", "table", "agg"])

    tbl_of = dict(zip(plan["feature"], plan["table"]))
    agg_of = dict(zip(plan["feature"], plan["agg"]))
    for c in cols:
        if c.startswith("n_rows_"):
            tbl_of[c] = c[len("n_rows_") :]

    demo = FEAT / "demo.parquet"
    if demo.exists():
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE meta AS
            SELECT
                cl.case_id,
                cl.MONTH,
                cl.target,
                COALESCE(CAST(d.sex AS VARCHAR), 'unknown') AS sex,
                {AGE_SQL} AS age_band
            FROM case_level cl
            LEFT JOIN read_parquet('{sql_path(demo)}') d USING (case_id)
        """)
    else:
        print("[warn] demo.parquet not found; sex and age analyses skipped (run 03 --demo-only)")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE meta AS
            SELECT case_id, MONTH, target, 'unknown' AS sex, 'unknown' AS age_band
            FROM case_level
        """)

    n_cases = con.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    print(f"[missingness] {n_cases:,} cases, {len(cols)} features")

    long = scan_columns(con, cols, dtypes, tbl_of, args.batch)

    group_rows = long.loc[
        long.dim.isin(["target", "sex", "age_band", "all"]),
        ["feature", "dim", "level", "n", "n_null", "n_null_no_rows"],
    ]
    group_rows.to_parquet(MISS / "null_by_group.parquet")

    month_rows = (
        long[long.dim == "month"][["feature", "level", "n", "n_null"]]
        .rename(columns={"level": "MONTH"})
        .assign(null_pct=lambda d: d.n_null / d.n * 100)
    )
    month_rows.to_parquet(MISS / "null_by_month.parquet")

    # Per-feature summary table combining coverage, target lift, time drift, and group gaps.
    a = long[long.dim == "all"].set_index("feature")
    feat = pd.DataFrame({
        "feature": a.index,
        "dtype": [dtypes[c] for c in a.index],
        "table": [tbl_of.get(c, "static/base") for c in a.index],
        "agg": [agg_of.get(c, "") for c in a.index],
        "n": a.n.values,
        "n_null": a.n_null.values,
    }).set_index("feature")

    feat["null_pct"] = feat.n_null / feat.n * 100
    feat["n_null_no_rows"] = a.n_null_no_rows
    is_count = [c.startswith("n_rows_") for c in feat.index]
    feat.loc[is_count, "n_null_no_rows"] = feat.loc[is_count, "n_null"]
    feat["n_null_with_rows"] = feat.n_null - feat.n_null_no_rows
    feat["structural_share"] = feat.n_null_no_rows / feat.n_null.where(feat.n_null > 0)

    t = long[long.dim == "target"].pivot(index="feature", columns="level", values=["n", "n_null"])
    n0, n1, k0, k1 = t[("n", "0")], t[("n", "1")], t[("n_null", "0")], t[("n_null", "1")]
    feat["n_null_defaults"] = k1
    feat["default_rate_null"] = k1 / (k0 + k1).where((k0 + k1) > 0)
    nn_total = (n0 - k0) + (n1 - k1)
    feat["default_rate_nonnull"] = (n1 - k1) / nn_total.where(nn_total > 0)
    feat["missing_indicator_auc"] = 0.5 * (k1 / n1 + (n0 - k0) / n0)

    p_vals = []
    for f in feat.index:
        tot_null = k0[f] + k1[f]
        tot_all = n0[f] + n1[f]
        if 0 < tot_null < tot_all:
            p_vals.append(chi2_2xk([k1[f], n1[f] - k1[f]], [tot_null, tot_all - tot_null]))
        else:
            p_vals.append(np.nan)
    feat["target_p"] = p_vals
    feat["target_q"] = bh(feat.target_p.values)

    feat = feat.join(month_stats(long))
    for dim in ("sex", "age_band"):
        gs = group_stats(long, dim, MIN_GROUP_N)
        if len(gs):
            feat = feat.join(gs)
        else:
            feat = feat.assign(**{
                f"{dim}_gap": np.nan,
                f"{dim}_max_level": None,
                f"{dim}_min_level": None,
                f"{dim}_p": np.nan,
            })

    feat = build_flags(feat.reset_index())
    lab = feat.apply(label, axis=1, result_type="expand")
    feat["mechanism"], feat["suggested_treatment"], feat["notes"] = lab[0], lab[1], lab[2]
    feat.to_csv(MISS / "feature_missingness.csv", index=False)

    print("\n[mechanism counts]")
    print(feat.mechanism.value_counts().to_string())
    print(f"step changes: {int(feat.flag_step_change.sum())}; group-linked: {int(feat.flag_group_linked.sum())}")

    # Source