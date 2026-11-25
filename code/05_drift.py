"""
05_drift.py

Temporal drift of the default rate and of every feature in the case-level table (view 'case_level').

Usage
  python 05_drift.py

Method
  Pass 1  decile edges (numeric) or top-10 levels (string) per feature, over all cases.
  Pass 2  one scan per batch of features with GROUPING SETS: counts and default counts per (MONTH, bin).
          Missing values are their own bin, so drift in missingness is included in PSI.
  Pass 3  mean, median and non-null count per MONTH for numeric features.
  PSI, approximate KS and Information Value are all computed from these binned counts.

Periods
  early / late   first / last third of cases by MONTH (months whose cumulative share before them is
                 below 1/3, or at least 2/3)
  regimes        split at --breaks; PSI between consecutive regimes
  months         PSI of each month vs the previous month and vs the early reference

PSI bands: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 major (conventional rules of thumb).
KS is approximate (resolution of the bins) and numeric only.

Pattern label per feature
  stable          early-late PSI < 0.10 and no month-to-month PSI >= 0.10
  step_at_break   largest month-to-month PSI >= 0.10 and falls on a regime start
  step_other      largest month-to-month PSI >= 0.10 in another month
  gradual         early-late PSI >= 0.10 without a single large monthly step

Outputs (outputs/drift/)
  target_by_month.csv, target_by_week.csv   n, defaults, rate, Wilson 95% CI
  target_by_regime.csv                      same, per regime
  feature_drift.csv                         one row per feature (PSI, KS, IV by period, pattern)
  psi_by_month.parquet                      feature x MONTH: psi_vs_prev, psi_vs_early
  numeric_by_month.parquet                  feature x MONTH: n_nonnull, mean, median
  bin_definitions.csv, feature_bins_by_month.parquet   reusable for later steps
"""

import argparse
import math
import time

import duckdb
import numpy as np
import pandas as pd

from config import DB, MEMORY_LIMIT, OUT, THREADS, TMP

DRIFT = OUT / "drift"
MISS = OUT / "missingness"
KEYS = {"case_id", "target", "MONTH", "WEEK_NUM", "date_decision"}
EPS = 1e-4
PSI_MOD, PSI_MAJOR = 0.10, 0.25


def q(n):
    return '"' + n.replace('"', '""') + '"'


def sql_path(p):
    return str(p).replace("\\", "/").replace("'", "''")


def lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def kind_of(dtype):
    d = dtype.upper()
    return "string" if d.startswith(("VARCHAR", "STRING")) else "numeric"


def numeric_expr(col, dtype):
    x = f"cl.{q(col)}"
    if dtype.upper().startswith(("DATE", "TIMESTAMP")):
        return f"CAST(date_diff('day', DATE '1970-01-01', CAST({x} AS DATE)) AS DOUBLE)"
    return f"NULLIF(CAST({x} AS DOUBLE), 'NaN'::DOUBLE)"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - margin, center + margin


def psi(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    p = np.maximum(a / a.sum(), EPS)
    r = np.maximum(b / b.sum(), EPS)
    return float(((p - r) * np.log(p / r)).sum())


def ks(a, b, na_idx):
    keep = np.arange(len(a)) != na_idx
    a = np.asarray(a, float)[keep]
    b = np.asarray(b, float)[keep]
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    return float(np.abs(np.cumsum(a) / a.sum() - np.cumsum(b) / b.sum()).max())


def iv(n, d):
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    g = n - d
    if d.sum() == 0 or g.sum() == 0:
        return np.nan
    # Floor bin counts at 0.5 to avoid log(0) or division by zero in sparse bins.
    dg = np.maximum(g, 0.5) / g.sum()
    db = np.maximum(d, 0.5) / d.sum()
    return float(((dg - db) * np.log(dg / db)).sum())


def find_edges(con, cols, dtypes, batch):
    """Return {col: (kind, edges_or_levels)}."""
    out = {}
    for b in range(0, len(cols), batch):
        chunk = cols[b : b + batch]
        ex = []
        for i, c in enumerate(chunk):
            if kind_of(dtypes[c]) == "string":
                ex.append(f"approx_top_k(cl.{q(c)}, 10) AS e{i}")
            else:
                num = numeric_expr(c, dtypes[c])
                ex.append(f"approx_quantile({num}, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) AS e{i}")

        row = con.execute(f"SELECT {', '.join(ex)} FROM case_level cl").fetchone()
        for i, c in enumerate(chunk):
            v = row[i]
            if kind_of(dtypes[c]) == "string":
                out[c] = ("string", list(v) if v is not None else [])
            else:
                vals = sorted({float(x) for x in (v or []) if x is not None and np.isfinite(x)})
                out[c] = ("numeric", vals)

        print(f"  edges {min(b + batch, len(cols))}/{len(cols)}")

    return out


def bin_sql(col, dtypes, kind, spec):
    if kind == "string":
        x = f"cl.{q(col)}"
        whens = " ".join(f"WHEN {x} = {lit(v)} THEN {j}" for j, v in enumerate(spec))
        return f"CASE WHEN {x} IS NULL THEN -1 {whens} ELSE {len(spec)} END"

    x = numeric_expr(col, dtypes[col])
    if not spec:
        return f"CASE WHEN {x} IS NULL THEN -1 ELSE 0 END"
    s = " + ".join(f"CAST(({x} > {e!r}) AS INTEGER)" for e in spec)
    return f"CASE WHEN {x} IS NULL THEN -1 ELSE ({s}) END"


def bin_labels(kind, spec):
    labels = {-1: "missing"}
    if kind == "string":
        labels.update({j: str(v) for j, v in enumerate(spec)})
        labels[len(spec)] = "other"
    else:
        edges = [-np.inf] + list(spec) + [np.inf]
        for j in range(len(edges) - 1):
            labels[j] = f"({edges[j]:g}, {edges[j + 1]:g}]"
    return labels


def scan_bins(con, cols, dtypes, defs, batch):
    parts = []
    for b in range(0, len(cols), batch):
        chunk = cols[b : b + batch]
        sel = ", ".join(f"{bin_sql(c, dtypes, *defs[c])} AS b{i}" for i, c in enumerate(chunk))
        sets = ", ".join(f"(month, b{i})" for i in range(len(chunk)))
        bcols = ", ".join(f"b{i}" for i in range(len(chunk)))