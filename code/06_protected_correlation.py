"""
06_protected_correlation.py

Proxy search for the protected attributes. For every case-level feature, measures how strongly it is
associated with each protected attribute, lists the strong ones as proxy candidates, and draws
correlation maps. Also measures whether a feature being NULL is associated with a protected attribute.

Run order: 01 -> 02 -> 03 (needs demo.parquet and the case_level view) -> 04 -> this script.
Run it from the code folder, next to config.py:
  python 06_protected_correlation.py

Data used
  protected attributes   outputs/features/demo.parquet (applicant row of person_1 + static_cb, written by 03):
                           sex             derived column: COALESCE(sex_738L, gender_992L)
                           age             age_years (decision date minus date of birth)
                           marital_status  first of maritalst_385M, maritalst_893M, maritalst_703L that is
                                           at least half filled
                           family_state    familystate_447L
                           disability_income  flag: incometype_1044T matches HANDICAP|DISAB
  features               every column of the DuckDB view case_level except keys and the protected columns
                         and their aggregates (max_sex_738L, last_birth_259D, ...)
  feature -> table       outputs/features/agg_plan.csv (static columns count as table 'static/base')

Measures (all 0-1 except Spearman, which is signed; thresholds and heatmaps use the absolute value)
  numeric vs numeric         Spearman rank correlation
  numeric vs categorical     correlation ratio eta
  categorical vs categorical Cramer's V, bias-corrected
Numbers with at most 10 distinct values are treated as codes (categorical). Categories beyond
30 are lumped into one level. A 0-1 eta or V and a Spearman rho are comparable, not identical.
The placeholder a55475b1 is a normal category here, as in 04_missingness.py.

Outputs (outputs/correlation/)
  associations_long.csv       feature x protected attribute x (value | missingness): measure, assoc, n
  associations_wide.csv       |assoc| of the best aggregate per source column, one column per attribute
  proxy_candidates.csv        best aggregate per source column with |assoc| >= 0.30 (tier: strong >= 0.50)
  missingness_proxies.csv     features whose NULL pattern is associated with a protected attribute
  heatmap_protected.png       protected attributes against each other
  heatmap_proxies_overview.png        top features x protected attributes
  heatmap_<attribute>.png             attribute plus its top proxies, all pairwise
  heatmap_missingness_overview.png    top null-pattern proxies x protected attributes
"""

import re
import sys
import time
import warnings

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from config import DB, MEMORY_LIMIT, OUT, THREADS, TMP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

FEAT = OUT / "features"
CORR = OUT / "correlation"
KEYS = {"case_id", "target", "MONTH", "WEEK_NUM", "date_decision"}

SAMPLE_SIZE = 200_000
SEED = 42
THRESHOLD = 0.30
STRONG = 0.50
TOP_OVERVIEW = 30
TOP_MATRIX = 20
MIN_N = 5000
LOW_CARD = 10
MAX_LEVELS = 30
BATCH = 60

PROTECTED = {
    "sex": ["sex", "sex_738L", "gender_992L"],
    "age": ["age_years"],
    "marital_status": ["maritalst_385M", "maritalst_893M", "maritalst_703L"],
    "family_state": ["familystate_447L"],
}
DISABILITY_SOURCE = "incometype_1044T"
DISABILITY_PATTERN = "HANDICAP|DISAB"
MIN_FILLED = 0.50

EXCLUDE_RE = re.compile(
    r"(sex_738L|gender_992L|birth_259D|birthdate_|dateofbirth_|maritalst_|"
    r"familystate_|incometype_1044T)"
)


def q(n):
    return '"' + n.replace('"', '""') + '"'


def sql_path(p):
    return str(p).replace("\\", "/").replace("'", "''")


def collapse(cat, k):
    """Keep the k-1 most frequent levels, lump the rest."""
    vc = cat.value_counts()
    if len(vc) <= k:
        return cat
    top_levels = set(vc.index[: k - 1])
    return cat.where(cat.isin(top_levels), "__OTHER__")


def eta(cat, num):
    """Correlation ratio: share of numeric variance explained by categories."""
    g = num.groupby(cat)
    n = g.count()
    mean_g = g.mean()
    grand = num.mean()
    ssb = float((n * (mean_g - grand) ** 2).sum())
    sst = float(((num - grand) ** 2).sum())
    return float(np.sqrt(ssb / sst)) if sst > 0 else np.nan


def cramers_v(a, b):
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    ct = pd.crosstab(a, b).values.astype(float)
    n = ct.sum()
    r, k = ct.shape
    if n < 2 or r < 2 or k < 2:
        return np.nan

    expected = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = float(((ct - expected) ** 2 / expected).sum())
    phi2c = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else np.nan


def assoc(x, kx, y, ky):
    """Association on jointly non-null rows. Returns (value, measure, n)."""
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < MIN_N:
        return np.nan, "", n

    x = x[m]
    y = y[m]

    if kx == "num" and ky == "num":
        if x.nunique() < 2 or y.nunique() < 2:
            return np.nan, "spearman", n
        return float(x.rank().corr(y.rank())), "spearman", n

    if kx == "cat" and ky == "cat":
        return cramers_v(collapse(x, MAX_LEVELS), collapse(y, MAX_LEVELS)), "cramers_v", n

    cat, num = (x, y) if kx == "cat" else (y, x)
    return eta(collapse(cat, MAX_LEVELS), num), "eta", n


def prepare(s):
    """Return (series, kind) with kind 'num', 'cat', or None (constant or unusable)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        s = (s - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    elif pd.api.types.is_bool_dtype(s):
        s = s.astype(float)

    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).where(s.notna(), np.nan)
        return (s, "cat") if s.nunique() >= 2 else (s, None)

    s = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    nun = s.nunique(dropna=True)
    if nun < 2:
        return s, None
    if nun <= LOW_CARD:
        return s.astype(str).where(s.notna(), np.nan), "cat"
    return s, "num"


def heatmap(M, rows, cols, title, path, cbar="|association|"):
    M = np.asarray(M, float)
    w = max(4.5, 0.9 * len(cols) + 0.22 * max(len(r) for r in rows) + 1.5)
    h = max(3.0, 0.28 * len(rows) + 1.8)

    fig, ax = plt.subplots(figsize=(min(w, 22), min(h, 26)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)

    if M.size <= 1200:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    text_color = "white" if M[i, j] > 0.6 else "black"
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pair_matrix(df, kinds):
    cols = list(df.columns)
    M = np.full((len(cols), len(cols)), np.nan)
    for i, a in enumerate(cols):
        M[i, i] = 1.0
        for j in range(i + 1, len(cols)):
            b = cols[j]
            v, _, _ = assoc(df[a], kinds[a], df[b], kinds[b])
            val = abs(v) if pd.notna(v) else np.nan
            M[i, j] = val
            M[j, i] = val
    return M


def main():
    CORR.mkdir(parents=True, exist_ok=True)

    demo_path = FEAT / "demo.parquet"
    if not demo_path.exists():
        sys.exit(f"{demo_path} not found. Run: python 03_build_case_level.py --demo-only")

    con = duckdb.connect(str(DB))
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{sql_path(TMP)}'")

    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    if "case_level" not in views:
        sys.exit("View 'case_level' not found. Run 03_build_case_level.py first.")

    con.execute(f"CREATE OR REPLACE TEMP VIEW demo_cases AS SELECT * FROM read_parquet('{sql_path(demo_path)}')")

    def cols_of(view):
        return {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()}

    cl_cols = cols_of("case_level")
    demo_cols = cols_of("demo_cases")

    n_total = con.execute("SELECT COUNT(*) FROM case_level").fetchone()[0]
    n_s = min(SAMPLE_SIZE, n_total)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sample_ids AS
        SELECT case_id FROM case_level
        USING SAMPLE reservoir({n_s} ROWS) REPEATABLE ({SEED})
    """)
    sample_ids = con.execute("SELECT case_id FROM sample_ids").df()["case_id"]
    index = pd.Index(sorted(sample_ids), name="case_id")
    print(f"[sample] {len(index):,} of {n_total:,} applications (seed {SEED})")

    def fetch(view, cols):
        sel = ", ".join(q(c) for c in cols)
        df = con.execute(f"""
            SELECT case_id, {sel}
            FROM {view}
            WHERE case_id IN (SELECT case_id FROM sample_ids)
        """).df()
        return df.drop_duplicates("case_id").set_index("case_id").reindex(index)

    def locate(col):
        if col in demo_cols:
            return "demo_cases"
        if col in cl_cols:
            return "case_level"
        return None

    prot = {}
    kinds = {}

    def add(label, col, series=None):
        view = locate(col)
        s = series if series is not None else fetch(view, [col])[col]
        s, k = prepare(s)
        if k is None:
            print(f"[warn] {label} ({col}) is constant or empty; skipped")
            return False
        prot[label] = s
        kinds[label] = k
        levels_info = s.nunique() if k == "cat" else "-"
        print(
            f"[protected] {label:18s} <- {view}.{col:24s} "
            f"kind={k}  null={s.isna().mean():.1%}  levels={levels_info}"
        )
        return True

    for label, cands in PROTECTED.items():
        present = [c for c in cands if locate(c)]
        if not present:
            print(f"[warn] {label}: none of {cands} found; skipped")
            continue

        chosen = None
        for c in present:
            if fetch(locate(c), [c])[c].notna().mean() >= MIN_FILLED:
                chosen = c
                break
        if chosen is None:
            chosen = present[0]
            print(f"[warn] {label}: every candidate is under {MIN_FILLED:.0%} filled; using {chosen}")
        add(label, chosen)

    if locate(DISABILITY_SOURCE):
        raw = fetch(locate(DISABILITY_SOURCE), [DISABILITY_SOURCE])[DISABILITY_SOURCE]
        print(f"\n[disability] levels of {DISABILITY_SOURCE} in the sample:")
        print(raw.value_counts(dropna=False).head(15).to_string())

        hit = raw.astype(str).str.contains(DISABILITY_PATTERN, case=False, regex=True, na=False)
        if hit.sum() > 0:
            flag = hit.astype(int).astype(str).where(raw.notna(), np.nan)
            add("disability_income", DISABILITY_SOURCE, flag)
            print(f"[disability] flagged by /{DISABILITY_PATTERN}/: {int(hit.sum()):,} rows.\n")
        else:
            print(f"[warn] no label matches /{DISABILITY_PATTERN}/; disability skipped\n")

    if not prot:
        sys.exit("[error] no protected attributes available")

    prot = pd.DataFrame(prot, index=index)
    plabels = list(prot.columns)

    feats = [c for c in cl_cols if c not in KEYS and not EXCLUDE_RE.search(c)]
    print(
        f"[features] {len(feats)} candidates ({len(cl_cols) - len(feats)} columns excluded: "
        f"keys, protected columns and their aggregates)"
    )

    meta = pd.DataFrame({"feature": feats})
    plan_f = FEAT / "agg_plan.csv"
    plan = pd.read_csv(plan_f) if plan_f.exists() else pd.DataFrame(columns=["feature", "table", "column", "agg"])
    tbl_of = dict(zip(plan.feature, plan.table))
    col_of = dict(zip(plan.feature, plan.column))

    meta["table"] = [
        tbl_of.get(f, f[len("n_rows_") :] if f.startswith("n_rows_") else "static/base")
        for f in feats
    ]
    meta["base"] = [
        col_of.get(f, "n_rows" if f.startswith("n_rows_") else f)
        for f in feats
    ]

    mf = FEAT.parent / "missingness" / "feature_missingness.csv"
    if mf.exists():
        meta = meta.merge(pd.read_csv(mf, usecols=["feature", "mechanism"]), on="feature", how="left")

    recs = []
    t0 = time.time()
    for b0 in range(0, len(feats), BATCH):
        batch = feats[b0 : b0 + BATCH]
        df = fetch("case_level", batch)
        for f in batch:
            raw = df[f]
            null_pct = float(raw.isna().mean())
            x, kx = prepare(raw)
            if kx is not None:
                for p in plabels:
                    v, meas, n = assoc(x, kx, prot[p], kinds[p])
                    recs.append((f, p, "value", meas, v, n, null_pct))

            if 0.01 <= null_pct <= 0.99:
                ind = raw.isna().astype(int).astype(str)
                for p in plabels:
                    v, meas, n = assoc(ind, "cat", prot[p], kinds[p])
                    recs.append((f, p, "missingness", meas, v, n, null_pct))

        done = min(b0 + BATCH, len(feats))
        print(f"[assoc] {done}/{len(feats)} features ({time.time() - t0:.0f}s)")

    long = pd.DataFrame(
        recs,
        columns=["feature", "protected", "target", "measure", "assoc", "n", "null_pct"],
    )
    long["abs_assoc"] = long["assoc"].abs()
    long = long.merge(meta, on="feature", how="left")
    long.to_csv(CORR / "associations_long.csv", index=False)

    def best_per_source(d):
        """Strongest aggregate (max / min / last ...) per source column and protected attribute."""
        d = d.dropna(subset=["abs_assoc"]).copy()
        d["_k"] = d["table"].fillna("") + "|" + d["base"]
        return (
            d.sort_values("abs_assoc", ascending=False)
            .drop_duplicates(["protected", "_k"])
            .drop(columns="_k")
        )

    vals = best_per_source(long[long.target == "value"])
    prox = vals[vals.abs_assoc >= THRESHOLD].copy()
    prox["tier"] = np.where(prox.abs_assoc >= STRONG, "strong", "high")
    prox.sort_values(["protected", "abs_assoc"], ascending=[True, False]).to_csv(
        CORR / "proxy_candidates.csv", index=False
    )

    wide = vals.pivot_table(index="feature", columns="protected", values="abs_assoc")
    wide["max_assoc"] = wide.max(axis=1)
    wide.sort_values("max_assoc", ascending=False).to_csv(CORR / "associations_wide.csv")

    miss = best_per_source(long[long.target == "missingness"])
    if len(miss):
        miss.sort_values("abs_assoc", ascending=False).to_csv(CORR / "missingness_proxies.csv", index=False)

    heatmap(
        pair_matrix(prot, kinds),
        plabels,
        plabels,
        "Association between protected attributes",
        CORR / "heatmap_protected.png",
    )

    def overview(d, title, name):
        if d.empty:
            return
        top = (
            d.groupby("feature")
            .abs_assoc.max()
            .sort_values(ascending=False)
            .head(TOP_OVERVIEW)
            .index.tolist()
        )
        w = d[d.feature.isin(top)].pivot_table(index="feature", columns="protected", values="abs_assoc")
        w = w.reindex(top)[[p for p in plabels if p in w.columns]]
        heatmap(w.values, list(w.index), list(w.columns), title, CORR / name)

    overview(
        vals,
        f"Top {TOP_OVERVIEW} proxy candidates by association with protected attributes",
        "heatmap_proxies_overview.png",
    )
    overview(
        miss,
        f"Top {TOP_OVERVIEW} features whose NULL pattern is associated with a protected attribute",
        "heatmap_missingness_overview.png",
    )

    for p in plabels:
        sub = vals[vals.protected == p].sort_values("abs_assoc", ascending=False).head(TOP_MATRIX)
        if sub.empty:
            continue
        names = sub.feature.tolist()
        df = fetch("case_level", names)
        data = {p: prot[p]}
        k2 = {p: kinds[p]}
        for f in names:
            s, k = prepare(df[f])
            if k is not None:
                data[f] = s
                k2[f] = k
        M = pair_matrix(pd.DataFrame(data), k2)
        labs = list(data)
        heatmap(M, labs, labs, f"Correlation map: {p} and its top proxy candidates", CORR / f"heatmap_{p}.png")

    print(
        f"\n[summary] proxy candidates, one aggregate per source column "
        f"(>= {THRESHOLD:.2f} high, >= {STRONG:.2f} strong)"
    )
    for p in plabels:
        d = prox[prox.protected == p].sort_values("abs_assoc", ascending=False)
        strong_cnt = int((d.tier == "strong").sum())
        print(f"  {p:20s} candidates {len(d):4d}   strong {strong_cnt:4d}")
        for _, r in d.head(5).iterrows():
            print(f"      {r.feature:48s} {r.abs_assoc:.3f} ({r.measure}, {r.table})")

    con.close()
    print(f"\n[done] outputs in {CORR}")


if __name__ == "__main__":
    main()