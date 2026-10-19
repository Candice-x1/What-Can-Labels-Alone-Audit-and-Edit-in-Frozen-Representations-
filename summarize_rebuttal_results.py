import os
import glob
import argparse
import warnings

import numpy as np
import pandas as pd


# ============================================================
# General helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_paths(values):
    """
    Supports either:
      --core_roots path1 path2
    or a single comma/semicolon-separated string.
    """
    out = []
    if values is None:
        return out

    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue

        # Semicolon is safest on Windows. Comma is also accepted.
        pieces = [value]
        if ";" in value:
            pieces = value.split(";")
        elif "," in value:
            pieces = value.split(",")

        for p in pieces:
            p = p.strip().strip('"').strip("'")
            if p:
                out.append(os.path.abspath(p))
    return out


def discover_all_records(paths):
    """
    Given files or directories, locate all_rebuttal_records.csv.

    If a directory directly contains the file, use that file.
    Otherwise search recursively.
    """
    found = []

    for path in paths:
        if os.path.isfile(path):
            if os.path.basename(path).lower() == "all_rebuttal_records.csv":
                found.append(os.path.abspath(path))
            else:
                warnings.warn(f"Skipping non-target file: {path}")
            continue

        if not os.path.isdir(path):
            warnings.warn(f"Path does not exist: {path}")
            continue

        direct = os.path.join(path, "all_rebuttal_records.csv")
        if os.path.isfile(direct):
            found.append(os.path.abspath(direct))
            continue

        matches = glob.glob(
            os.path.join(path, "**", "all_rebuttal_records.csv"),
            recursive=True,
        )
        found.extend(os.path.abspath(x) for x in matches)

    return sorted(set(found))


def load_record_files(files, source_group):
    frames = []

    for f in files:
        print(f"[load:{source_group}] {f}", flush=True)
        df = pd.read_csv(f, low_memory=False)
        df["source_group"] = source_group
        df["source_file"] = f
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Normalize common categorical columns.
    for c in [
        "experiment", "dataset", "backbone", "method",
        "representation", "probe", "metric_name", "budget_type",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    # Normalize numeric columns used for filtering.
    for c in [
        "seed", "q", "r", "alpha", "budget_tol", "feasible",
        "value",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def validate_unique_records(df, label):
    """
    Detect accidental duplicate experiment records, e.g. when a root
    directory contains two complete runs with the same seeds.
    """
    if len(df) == 0:
        return

    candidate_keys = [
        "experiment",
        "dataset",
        "backbone",
        "seed",
        "pretrained",
        "method",
        "representation",
        "probe",
        "metric_name",
        "budget_type",
        "budget_tol",
        "q",
        "r",
        "alpha",
    ]
    keys = [c for c in candidate_keys if c in df.columns]

    duplicated = df.duplicated(subset=keys, keep=False)

    if duplicated.any():
        bad_cols = keys + [
            c for c in ["source_file"]
            if c in df.columns
        ]
        bad = (
            df.loc[duplicated, bad_cols]
            .sort_values(keys)
            .head(50)
        )

        print(
            f"\n[ERROR] Duplicate records detected in {label}.",
            flush=True,
        )
        print(
            "This usually means that a root directory contains "
            "multiple runs with the same seeds.",
            flush=True,
        )
        print(bad.to_string(index=False), flush=True)

        raise ValueError(
            f"Duplicate experiment records detected in {label}. "
            "Pass the exact intended all_rebuttal_records.csv files "
            "or remove duplicate run directories."
        )

def finite_values(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def mean_std(series):
    x = finite_values(series)
    if len(x) == 0:
        return np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), 0.0, 1
    return float(np.mean(x)), float(np.std(x, ddof=1)), int(len(x))


def fmt_pm(mean, std, digits=4):
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_num(x, digits=4):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def save_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, float_format="%.8f")
    print(f"[save] {path} ({len(df)} rows)", flush=True)


def markdown_table(df, max_rows=None, digits=4):
    if df is None or len(df) == 0:
        return "_No records available._\n"

    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)

    cols = list(show.columns)

    def clean(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return fmt_num(v, digits)
        s = str(v)
        return s.replace("|", "\\|").replace("\n", " ")

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in show.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")

    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n_Only the first {max_rows} of {len(df)} rows are shown._")

    return "\n".join(lines) + "\n"


def group_metric_summary(df, group_cols, metrics):
    """
    Return one row per group with metric_mean, metric_std, metric_n.
    """
    rows = []

    if len(df) == 0:
        return pd.DataFrame()

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        if "seed" in g.columns:
            row["num_seeds"] = int(g["seed"].nunique())

        for m in metrics:
            if m not in g.columns:
                continue
            mu, sd, n = mean_std(g[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
            row[f"{m}_n"] = n

        rows.append(row)

    return pd.DataFrame(rows)


def isclose_series(series, value, atol=1e-10):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.isclose(x, float(value), atol=atol, rtol=0.0)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# 00: Coverage
# ============================================================

def summarize_coverage(df):
    if len(df) == 0:
        return pd.DataFrame()

    rows = []
    for keys, g in df.groupby(
        ["source_group", "dataset", "backbone", "experiment"],
        dropna=False,
    ):
        source_group, dataset, backbone, experiment = keys
        rows.append({
            "source_group": source_group,
            "dataset": dataset,
            "backbone": backbone,
            "experiment": experiment,
            "num_seeds": int(g["seed"].nunique()) if "seed" in g.columns else 0,
            "num_rows": int(len(g)),
        })

    return pd.DataFrame(rows).sort_values(
        ["source_group", "backbone", "dataset", "experiment"]
    )

def warn_unexpected_seed_counts(coverage):
    """
    Warn when a summarized experiment does not contain the expected
    number of outer seeds.
    """
    if coverage is None or len(coverage) == 0:
        return

    for _, row in coverage.iterrows():
        source_group = str(row["source_group"])
        expected = 10 if source_group == "core" else 3
        observed = int(row["num_seeds"])

        if observed != expected:
            print(
                f"[WARNING] {source_group}: "
                f"{row['dataset']}/{row['backbone']}/"
                f"{row['experiment']} has {observed} seeds; "
                f"expected {expected}.",
                flush=True,
            )

# ============================================================
# 01: Stage-one equivalence
# ============================================================

def summarize_stage1_equivalence(core):
    x = core[core["experiment"] == "stage1_equivalence"].copy()
    if len(x) == 0:
        return pd.DataFrame()

    x["stage1_key"] = x["method"].astype(str)

    # metric_name is needed for whitening and commutator records.
    mask = x["metric_name"].astype(str) != "subspace_overlap"
    x.loc[mask, "stage1_key"] = (
        x.loc[mask, "method"].astype(str)
        + "__"
        + x.loc[mask, "metric_name"].astype(str)
    )

    keys = [
        "dmd_vs_lda",
        "dmd_vs_between_cov",
        "lda_vs_between_cov",
        "pls_vs_dmd",
        "whitening_check__total_cov_fro_error_from_identity",
        "lda_whitened_geometry__between_within_commutator_norm",
    ]

    rows = []
    for (dataset, backbone), g in x.groupby(["dataset", "backbone"]):
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "num_seeds": int(g["seed"].nunique()),
        }

        for key in keys:
            s = g[g["stage1_key"] == key]["value"]
            mu, sd, n = mean_std(s)
            row[f"{key}_mean"] = mu
            row[f"{key}_std"] = sd
            row[f"{key}_n"] = n

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["backbone", "dataset"])


# ============================================================
# 02: Safe edit, q=8, overlap and preservation
# ============================================================

def summarize_safe_q8(core, q_eval=8):
    x = core[
        (core["experiment"] == "safe_edit_raw_overlap")
        & isclose_series(core["q"], q_eval)
    ].copy()

    if len(x) == 0:
        return pd.DataFrame()

    metrics = [
        "quotient_preservation_error_train",
        "quotient_preservation_error_test",
        "removed_subspace_overlap_with_U",
        "label_change_vs_full",
        "label_acc_drop_vs_full",
        "label_nll_change_vs_full",
        "env_leak_reduction_vs_full_quadratic",
    ]

    out = group_metric_summary(
        x,
        ["dataset", "backbone", "method", "q"],
        metrics,
    )
    return out.sort_values(["backbone", "dataset", "method"])


def summarize_safe_edit_key(core, q_eval=8):
    edit = core[core["experiment"] == "safe_edit_raw_overlap"].copy()
    corr = core[
        core["experiment"] == "safe_edit_raw_overlap_summary"
    ].copy()

    if len(edit) == 0:
        return pd.DataFrame()

    rows = []

    for (dataset, backbone), g in edit.groupby(["dataset", "backbone"]):
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "num_seeds": int(g["seed"].nunique()),
        }

        dmd = g[g["method"] == "dmd"]
        raw8 = g[
            (g["method"] == "raw_cov")
            & isclose_series(g["q"], q_eval)
        ]
        dmd8 = g[
            (g["method"] == "dmd")
            & isclose_series(g["q"], q_eval)
        ]

        if len(dmd) > 0:
            vals = finite_values(dmd["quotient_preservation_error_test"])
            row["dmd_preservation_error_test_max_all_q"] = (
                float(np.max(vals)) if len(vals) else np.nan
            )

            vals = finite_values(dmd["removed_subspace_overlap_with_U"])
            row["dmd_overlap_with_U_max_all_q"] = (
                float(np.max(vals)) if len(vals) else np.nan
            )

        for prefix, z in [("dmd_q8", dmd8), ("raw_q8", raw8)]:
            for metric in [
                "removed_subspace_overlap_with_U",
                "label_acc_drop_vs_full",
                "label_nll_change_vs_full",
                "env_leak_reduction_vs_full_quadratic",
            ]:
                if metric in z.columns:
                    mu, sd, _ = mean_std(z[metric])
                    row[f"{prefix}_{metric}_mean"] = mu
                    row[f"{prefix}_{metric}_std"] = sd

        c = corr[
            (corr["dataset"] == dataset)
            & (corr["backbone"] == backbone)
            & (corr["method"] == "raw_cov")
        ]

        for metric_name, short in [
            (
                "corr_overlap_with_U_vs_label_acc_drop",
                "raw_within_seed_corr_overlap_vs_acc_drop",
            ),
            (
                "corr_overlap_with_U_vs_nll_deterioration",
                "raw_within_seed_corr_overlap_vs_nll",
            ),
        ]:
            vals = c[c["metric_name"] == metric_name]["value"]
            mu, sd, _ = mean_std(vals)
            row[f"{short}_mean"] = mu
            row[f"{short}_std"] = sd

        # Also compute one pooled descriptive correlation for reference.
        raw_all = g[g["method"] == "raw_cov"]
        row["raw_pooled_corr_overlap_vs_acc_drop"] = safe_corr(
            raw_all["removed_subspace_overlap_with_U"],
            raw_all["label_acc_drop_vs_full"],
        )
        row["raw_pooled_corr_overlap_vs_nll"] = safe_corr(
            raw_all["removed_subspace_overlap_with_U"],
            raw_all["label_nll_change_vs_full"],
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["backbone", "dataset"])


# ============================================================
# 03: Full q-frontiers and constrained summaries
# ============================================================

def summarize_frontier_curve(core):
    x = core[core["experiment"] == "frontier_budget_curve"].copy()
    if len(x) == 0:
        return pd.DataFrame()

    metrics = [
        "env_leak_reduction_vs_full",
        "label_change_vs_full",
        "label_acc_drop_vs_full",
        "label_nll_change_vs_full",
        "label_acc_after",
        "label_nll_after",
    ]

    out = group_metric_summary(
        x,
        ["dataset", "backbone", "method", "q"],
        metrics,
    )
    return out.sort_values(["backbone", "dataset", "method", "q"])


def summarize_frontier_budget(core):
    """
    Summarize constrained frontiers.

    Important: in the original experiment, when no q satisfies a budget,
    feasible=0 and best_* stores an unconstrained fallback point. Those
    fallback values must NOT be included in the constrained-performance
    averages below.
    """
    x = core[core["experiment"] == "frontier_budget_summary"].copy()
    if len(x) == 0:
        return pd.DataFrame()

    group_cols = [
        "dataset", "backbone", "method",
        "budget_type", "budget_tol",
    ]

    best_metrics = [
        "best_q",
        "best_env_leak_reduction",
        "best_label_change",
        "best_label_acc_drop",
        "best_nll_change",
    ]

    rows = []

    for keys, g in x.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["num_seeds"] = int(g["seed"].nunique())

        feasible_values = pd.to_numeric(
            g["feasible"], errors="coerce"
        )
        feasible_mask = feasible_values >= 0.5
        feasible_g = g.loc[feasible_mask].copy()

        feasible_rate = float(np.mean(feasible_mask))
        feasible_std = (
            float(np.std(feasible_mask.astype(float), ddof=1))
            if len(feasible_mask) > 1 else 0.0
        )

        row["feasible_mean"] = feasible_rate
        row["feasible_std"] = feasible_std
        row["feasible_n"] = int(len(g))
        row["num_feasible_seeds"] = int(
            feasible_g["seed"].nunique()
        )

        # Summarize best_* only over seeds for which the constraint
        # was actually feasible.
        for metric in best_metrics:
            if metric not in feasible_g.columns:
                continue

            mu, sd, n = mean_std(feasible_g[metric])
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
            row[f"{metric}_n"] = n

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "budget_type", "budget_tol",
            "backbone", "dataset", "method",
        ]
    )


def paired_budget_comparison(core, comparator):
    """
    Compare DMD against comparator using per-seed constrained optima.
    Only seeds where both methods are feasible are used.
    """
    x = core[core["experiment"] == "frontier_budget_summary"].copy()
    if len(x) == 0:
        return pd.DataFrame()

    dmd = x[x["method"] == "dmd"].copy()
    cmp_df = x[x["method"] == comparator].copy()

    keys = [
        "dataset", "backbone", "seed",
        "budget_type", "budget_tol",
    ]

    keep = keys + [
        "feasible", "best_env_leak_reduction",
        "best_q", "best_label_acc_drop", "best_nll_change",
    ]

    dmd = dmd[keep].rename(columns={
        "feasible": "dmd_feasible",
        "best_env_leak_reduction": "dmd_best_leak",
        "best_q": "dmd_best_q",
        "best_label_acc_drop": "dmd_acc_drop",
        "best_nll_change": "dmd_nll_change",
    })

    cmp_df = cmp_df[keep].rename(columns={
        "feasible": "cmp_feasible",
        "best_env_leak_reduction": "cmp_best_leak",
        "best_q": "cmp_best_q",
        "best_label_acc_drop": "cmp_acc_drop",
        "best_nll_change": "cmp_nll_change",
    })

    m = dmd.merge(cmp_df, on=keys, how="inner")
    if len(m) == 0:
        return pd.DataFrame()

    rows = []
    for group_keys, g in m.groupby(
        ["dataset", "backbone", "budget_type", "budget_tol"]
    ):
        dataset, backbone, budget_type, budget_tol = group_keys

        both = g[
            (g["dmd_feasible"] >= 0.5)
            & (g["cmp_feasible"] >= 0.5)
        ].copy()

        row = {
            "dataset": dataset,
            "backbone": backbone,
            "budget_type": budget_type,
            "budget_tol": budget_tol,
            "comparator": comparator,
            "num_paired_seeds_total": int(len(g)),
            "num_both_feasible": int(len(both)),
            "dmd_feasible_rate": float(np.mean(g["dmd_feasible"] >= 0.5)),
            "comparator_feasible_rate": float(np.mean(g["cmp_feasible"] >= 0.5)),
        }

        if len(both) > 0:
            diff = (
                both["dmd_best_leak"].to_numpy(dtype=float)
                - both["cmp_best_leak"].to_numpy(dtype=float)
            )
            row["dmd_minus_comparator_leak_mean"] = float(np.mean(diff))
            row["dmd_minus_comparator_leak_std"] = (
                float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
            )
            row["dmd_win_or_tie_rate"] = float(np.mean(diff >= -1e-12))
            row["dmd_strict_win_rate"] = float(np.mean(diff > 1e-12))
            row["dmd_best_leak_mean"] = float(
                np.mean(both["dmd_best_leak"])
            )
            row["comparator_best_leak_mean"] = float(
                np.mean(both["cmp_best_leak"])
            )
        else:
            row["dmd_minus_comparator_leak_mean"] = np.nan
            row["dmd_minus_comparator_leak_std"] = np.nan
            row["dmd_win_or_tie_rate"] = np.nan
            row["dmd_strict_win_rate"] = np.nan
            row["dmd_best_leak_mean"] = np.nan
            row["comparator_best_leak_mean"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["budget_type", "budget_tol", "backbone", "dataset"]
    )


# ============================================================
# 04: Quotient-visible / residual leakage decomposition
# ============================================================

def summarize_quotient_representations(df):
    x = df[
        df["experiment"] == "quotient_leakage_decomposition"
    ].copy()
    if len(x) == 0:
        return pd.DataFrame()

    candidate_metrics = [
        "label_acc",
        "label_nll",
        "env_acc_linear",
        "env_acc_quadratic",
        "env_acc_mlp",
        "env_acc_rff",
        "env_acc_max",
    ]
    metrics = [m for m in candidate_metrics if m in x.columns]

    out = group_metric_summary(
        x,
        ["dataset", "backbone", "representation", "q"],
        metrics,
    )
    return out.sort_values(["backbone", "dataset", "representation"])


def summarize_quotient_ratios(df):
    x = df[df["experiment"] == "quotient_leakage_fraction"].copy()
    if len(x) == 0:
        return pd.DataFrame()

    candidate_metrics = [
        "majority_baseline_env_acc",
        "full_env_acc",
        "quotient_env_acc",
        "residual_env_acc",
        "quotient_visible_leakage_ratio",
        "residual_leakage_ratio",
        "quotient_visible_leakage_fraction_capped",
        "residual_leakage_fraction_capped",
    ]
    metrics = [m for m in candidate_metrics if m in x.columns]

    out = group_metric_summary(
        x,
        ["dataset", "backbone", "probe", "q"],
        metrics,
    )
    return out.sort_values(["backbone", "dataset", "probe"])


# ============================================================
# 05: Alternative auditor probes
# ============================================================

def summarize_alternative_probes(df, include_only=None):
    x = df[df["experiment"] == "alternative_probes"].copy()
    if len(x) == 0:
        return pd.DataFrame()

    candidate_metrics = [
        "label_acc_after",
        "label_nll_after",
        "label_change_vs_full",
        "label_nll_change_vs_full",
        "env_acc_linear",
        "env_acc_quadratic",
        "env_acc_mlp",
        "env_acc_rff",
        "env_acc_max",
        "env_leak_reduction_linear",
        "env_leak_reduction_quadratic",
        "env_leak_reduction_mlp",
        "env_leak_reduction_rff",
        "env_leak_reduction_max",
    ]

    if include_only is not None:
        candidate_metrics = [
            m for m in candidate_metrics
            if any(tag in m for tag in include_only)
            or m in [
                "label_acc_after",
                "label_nll_after",
                "label_change_vs_full",
                "label_nll_change_vs_full",
            ]
        ]

    metrics = [m for m in candidate_metrics if m in x.columns]

    out = group_metric_summary(
        x,
        ["dataset", "backbone", "method", "q"],
        metrics,
    )
    return out.sort_values(["backbone", "dataset", "method"])


# ============================================================
# 06: r / alpha sensitivity
# ============================================================

def default_r_for_dataset(dataset):
    dataset = str(dataset).lower()
    if dataset in ["celeba", "waterbirds"]:
        return 1
    if dataset == "pacs":
        return 4
    return None


def summarize_sensitivity_compact(core):
    if core is None or len(core) == 0:
        return pd.DataFrame()

    if "experiment" not in core.columns:
        return pd.DataFrame()

    x = core[
        core["experiment"] == "r_alpha_sensitivity"
    ].copy()

    if len(x) == 0:
        return pd.DataFrame()

    required_columns = {"r", "alpha", "method"}

    if not required_columns.issubset(set(x.columns)):
        return pd.DataFrame()

    metrics = [
        "env_leak_reduction_vs_full",
        "label_change_vs_full",
        "label_acc_drop_vs_full",
        "label_nll_change_vs_full",
    ]

    # First average each valid hyperparameter configuration over seeds.
    config = group_metric_summary(
        x,
        ["dataset", "backbone", "method", "r", "alpha"],
        metrics,
    )

    rows = []

    for (dataset, backbone, method), g in config.groupby(
        ["dataset", "backbone", "method"]
    ):
        raw_obs = x[
            (x["dataset"] == dataset)
            & (x["backbone"] == backbone)
            & (x["method"] == method)
        ]

        row = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "num_configs": int(len(g)),
            "num_seeds": int(raw_obs["seed"].nunique()),
        }

        for metric in metrics:
            col = f"{metric}_mean"
            vals = (
                finite_values(g[col])
                if col in g.columns
                else np.array([])
            )

            if len(vals):
                row[f"{metric}_config_mean"] = float(np.mean(vals))
                row[f"{metric}_config_std"] = (
                    float(np.std(vals, ddof=1))
                    if len(vals) > 1 else 0.0
                )
                row[f"{metric}_config_min"] = float(np.min(vals))
                row[f"{metric}_config_max"] = float(np.max(vals))

        if "label_acc_drop_vs_full" in raw_obs.columns:
            acc_drop = pd.to_numeric(
                raw_obs["label_acc_drop_vs_full"],
                errors="coerce",
            )
            valid_acc = acc_drop.notna()

            row["fraction_runs_acc_drop_le_0.005"] = (
                float(np.mean(acc_drop[valid_acc] <= 0.005))
                if valid_acc.any() else np.nan
            )
            row["fraction_runs_acc_drop_le_0.010"] = (
                float(np.mean(acc_drop[valid_acc] <= 0.010))
                if valid_acc.any() else np.nan
            )

        if "label_nll_change_vs_full" in raw_obs.columns:
            nll = pd.to_numeric(
                raw_obs["label_nll_change_vs_full"],
                errors="coerce",
            )
            valid_nll = nll.notna()

            row["fraction_runs_nll_increase_le_0.010"] = (
                float(np.mean(nll[valid_nll] <= 0.010))
                if valid_nll.any() else np.nan
            )
            row["fraction_runs_nll_increase_le_0.020"] = (
                float(np.mean(nll[valid_nll] <= 0.020))
                if valid_nll.any() else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["backbone", "dataset", "method"]
    )

def summarize_sensitivity_alpha(core):
    """
    Alpha sweep at the paper's default r.

    Returns an empty DataFrame when sensitivity experiments were
    not run.
    """
    if core is None or len(core) == 0:
        return pd.DataFrame()

    if "experiment" not in core.columns:
        return pd.DataFrame()

    x = core[
        core["experiment"] == "r_alpha_sensitivity"
    ].copy()

    if len(x) == 0:
        return pd.DataFrame()

    required_columns = {"r", "alpha", "method"}

    if not required_columns.issubset(set(x.columns)):
        return pd.DataFrame()


def summarize_sensitivity_r(core, alpha_eval=0.10):
    """
    DMD quotient-rank r sweep at fixed alpha.

    Raw covariance is excluded because it has no quotient-rank
    hyperparameter. If sensitivity experiments were not run, an
    empty DataFrame is returned.
    """
    if core is None or len(core) == 0:
        return pd.DataFrame()

    if "experiment" not in core.columns:
        return pd.DataFrame()

    # Filter by experiment before accessing sensitivity-specific
    # columns. Text modality checks intentionally skip sensitivity.
    x = core[
        core["experiment"] == "r_alpha_sensitivity"
    ].copy()

    if len(x) == 0:
        return pd.DataFrame()

    required_columns = {"alpha", "method", "r"}

    if not required_columns.issubset(set(x.columns)):
        return pd.DataFrame()

    x = x[
        isclose_series(x["alpha"], alpha_eval)
        & (x["method"] == "dmd")
    ].copy()

    if len(x) == 0:
        return pd.DataFrame()

    metrics = [
        "env_leak_reduction_vs_full",
        "label_change_vs_full",
        "label_acc_drop_vs_full",
        "label_nll_change_vs_full",
    ]

    out = group_metric_summary(
        x,
        ["dataset", "backbone", "method", "r", "alpha"],
        metrics,
    )

    if out is None or len(out) == 0:
        return pd.DataFrame()

    return out.sort_values(
        ["backbone", "dataset", "method", "r"]
    ).reset_index(drop=True)

# ============================================================
# Compact report
# ============================================================

def stage1_report_view(df):
    if len(df) == 0:
        return df

    cols = [
        "dataset", "backbone", "num_seeds",
        "dmd_vs_lda_mean",
        "dmd_vs_between_cov_mean",
        "lda_vs_between_cov_mean",
        "pls_vs_dmd_mean",
        "whitening_check__total_cov_fro_error_from_identity_mean",
        "lda_whitened_geometry__between_within_commutator_norm_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    out = out.rename(columns={
        "dmd_vs_lda_mean": "DMD/LDA overlap",
        "dmd_vs_between_cov_mean": "DMD/between overlap",
        "lda_vs_between_cov_mean": "LDA/between overlap",
        "pls_vs_dmd_mean": "PLS/DMD overlap",
        "whitening_check__total_cov_fro_error_from_identity_mean":
            "whitening error",
        "lda_whitened_geometry__between_within_commutator_norm_mean":
            "between/within commutator",
    })
    return out


def safe_key_report_view(df):
    if len(df) == 0:
        return df

    cols = [
        "dataset", "backbone", "num_seeds",
        "dmd_preservation_error_test_max_all_q",
        "dmd_overlap_with_U_max_all_q",
        "raw_q8_removed_subspace_overlap_with_U_mean",
        "dmd_q8_label_acc_drop_vs_full_mean",
        "raw_q8_label_acc_drop_vs_full_mean",
        "dmd_q8_env_leak_reduction_vs_full_quadratic_mean",
        "raw_q8_env_leak_reduction_vs_full_quadratic_mean",
        "raw_within_seed_corr_overlap_vs_acc_drop_mean",
        "raw_within_seed_corr_overlap_vs_nll_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy().rename(columns={
        "dmd_preservation_error_test_max_all_q":
            "max DMD surrogate-preservation error",
        "dmd_overlap_with_U_max_all_q":
            "max DMD/U overlap",
        "raw_q8_removed_subspace_overlap_with_U_mean":
            "raw/U overlap q8",
        "dmd_q8_label_acc_drop_vs_full_mean":
            "DMD acc drop q8",
        "raw_q8_label_acc_drop_vs_full_mean":
            "raw acc drop q8",
        "dmd_q8_env_leak_reduction_vs_full_quadratic_mean":
            "DMD leak reduction q8",
        "raw_q8_env_leak_reduction_vs_full_quadratic_mean":
            "raw leak reduction q8",
        "raw_within_seed_corr_overlap_vs_acc_drop_mean":
            "raw overlap/acc-drop corr",
        "raw_within_seed_corr_overlap_vs_nll_mean":
            "raw overlap/NLL corr",
    })


def frontier_report_view(df, budget_type, tol, methods=None):
    if len(df) == 0:
        return df

    z = df[
        (df["budget_type"] == budget_type)
        & np.isclose(
            pd.to_numeric(df["budget_tol"], errors="coerce"),
            float(tol),
        )
    ].copy()

    if methods is not None:
        z = z[z["method"].isin(methods)]

    cols = [
        "dataset", "backbone", "method",
        "num_seeds",
        "feasible_mean",
        "best_q_mean",
        "best_env_leak_reduction_mean",
        "best_env_leak_reduction_std",
        "best_label_acc_drop_mean",
        "best_nll_change_mean",
    ]
    cols = [c for c in cols if c in z.columns]

    return z[cols].copy().rename(columns={
        # "feasible_mean": "feasible rate",
        "feasible_mean": "positive-rank feasible rate",
        "best_q_mean": "mean selected q",
        "best_env_leak_reduction_mean": "best leak reduction",
        "best_env_leak_reduction_std": "leak reduction std",
        "best_label_acc_drop_mean": "acc drop",
        "best_nll_change_mean": "NLL increase",
    })


def quotient_report_view(df, probe="quadratic"):
    if len(df) == 0:
        return df

    env_col = f"env_acc_{probe}_mean"
    cols = [
        "dataset", "backbone", "representation",
        "num_seeds", "label_acc_mean", "label_nll_mean",
        env_col,
    ]
    cols = [c for c in cols if c in df.columns]

    return df[cols].copy().rename(columns={
        "label_acc_mean": "label accuracy",
        "label_nll_mean": "label NLL",
        env_col: f"{probe} env accuracy",
    })


def alt_probe_report_view(df, probe, methods=None):
    if len(df) == 0:
        return df

    if methods is not None:
        df = df[df["method"].isin(methods)].copy()

    cols = [
        "dataset", "backbone", "method", "q", "num_seeds",
        "label_change_vs_full_mean",
        "label_nll_change_vs_full_mean",
        f"env_acc_{probe}_mean",
        f"env_leak_reduction_{probe}_mean",
    ]
    cols = [c for c in cols if c in df.columns]

    return df[cols].copy().rename(columns={
        "label_change_vs_full_mean": "label change",
        "label_nll_change_vs_full_mean": "NLL change",
        f"env_acc_{probe}_mean": f"{probe} env accuracy",
        f"env_leak_reduction_{probe}_mean":
            f"{probe} leak reduction",
    })


def sensitivity_report_view(df):
    if len(df) == 0:
        return df

    z = df[df["method"].isin(["dmd", "raw_cov"])].copy()
    cols = [
        "dataset", "backbone", "method",
        "num_configs", "num_seeds",
        "env_leak_reduction_vs_full_config_mean",
        "env_leak_reduction_vs_full_config_min",
        "env_leak_reduction_vs_full_config_max",
        "label_acc_drop_vs_full_config_mean",
        "label_acc_drop_vs_full_config_min",
        "label_acc_drop_vs_full_config_max",
        "label_nll_change_vs_full_config_mean",
        "label_nll_change_vs_full_config_min",
        "label_nll_change_vs_full_config_max",
        "fraction_runs_acc_drop_le_0.005",
        "fraction_runs_nll_increase_le_0.010",
    ]
    cols = [c for c in cols if c in z.columns]
    return z[cols]


def generate_key_facts(
    stage1,
    safe_key,
    pair_raw,
    pair_pca,
    out_path,
):
    lines = []
    lines.append("Automatically extracted descriptive facts")
    lines.append("=========================================")
    lines.append("")

    if len(stage1) > 0 and "dmd_vs_lda_mean" in stage1.columns:
        vals = finite_values(stage1["dmd_vs_lda_mean"])
        if len(vals):
            lines.append(
                "Stage-one DMD/LDA mean subspace overlap across settings: "
                f"min={np.min(vals):.6f}, max={np.max(vals):.6f}, "
                f"mean={np.mean(vals):.6f}."
            )

    if len(stage1) > 0 and "dmd_vs_between_cov_mean" in stage1.columns:
        vals = finite_values(stage1["dmd_vs_between_cov_mean"])
        if len(vals):
            lines.append(
                "DMD/between-class-covariance mean overlap across settings: "
                f"min={np.min(vals):.6f}, max={np.max(vals):.6f}, "
                f"mean={np.mean(vals):.6f}."
            )

    if (
        len(safe_key) > 0
        and "dmd_preservation_error_test_max_all_q" in safe_key.columns
    ):
        vals = finite_values(
            safe_key["dmd_preservation_error_test_max_all_q"]
        )
        if len(vals):
            lines.append(
            "Largest observed DMD stage-one surrogate-preservation error "
            f"across settings and tested q values: {np.max(vals):.6e}."
        )

    if (
        len(safe_key) > 0
        and "dmd_overlap_with_U_max_all_q" in safe_key.columns
    ):
        vals = finite_values(safe_key["dmd_overlap_with_U_max_all_q"])
        if len(vals):
            lines.append(
                "Largest observed overlap between a DMD removed subspace "
                f"and the DMD stage-one surrogate subspace: "
                f"{np.max(vals):.6e}."
            )

    if (
        len(safe_key) > 0
        and "raw_within_seed_corr_overlap_vs_acc_drop_mean"
        in safe_key.columns
    ):
        vals = finite_values(
            safe_key["raw_within_seed_corr_overlap_vs_acc_drop_mean"]
        )
        if len(vals):
            lines.append(
                "Mean within-seed raw-overlap/accuracy-drop correlations "
                "across settings: "
                f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, "
                f"mean={np.mean(vals):.4f}."
            )

    for name, pair in [
        ("raw_cov", pair_raw),
        ("pca_first", pair_pca),
    ]:
        if len(pair) == 0:
            continue

        for budget_type, tol in [
            ("acc_drop", 0.005),
            ("nll_increase", 0.01),
        ]:
            z = pair[
                (pair["budget_type"] == budget_type)
                & np.isclose(
                    pd.to_numeric(pair["budget_tol"], errors="coerce"),
                    tol,
                )
            ]
            vals = finite_values(z["dmd_minus_comparator_leak_mean"])
            if len(vals):
                n_nonnegative = int(np.sum(vals >= -1e-12))
                lines.append(
                    f"Under {budget_type}<={tol}, and among outer seeds where "
                    f"both methods have a feasible tested positive-rank edit, "
                    f"DMD's mean constrained leakage reduction is at least "
                    f"{name}'s in {n_nonnegative}/{len(vals)} "
                    f"dataset/backbone settings; setting-level mean difference "
                    f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]."
                )

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_markdown_report(
    path,
    coverage,
    stage1,
    safe_key,
    budget,
    pair_raw,
    pair_pca,
    quotient_core,
    quotient_ratio_core,
    alt_core,
    sensitivity,
    quotient_mlp=None,
    alt_mlp=None,
):
    methods_short = [
        "dmd", "raw_cov", "pca_first", "lda_first", "pls_first"
    ]

    sections = []
    sections.append("# Compact Rebuttal Experiment Report\n")
    sections.append(
        "All values are aggregated over the available outer seeds. "
        "Environment annotations are used only for post-hoc evaluation.\n"
    )

    sections.append("## 1. Run coverage\n")
    sections.append(markdown_table(coverage))

    sections.append("## 2. Stage-one whitening/equivalence diagnostics\n")
    sections.append(markdown_table(stage1_report_view(stage1)))

    sections.append(
        "## 3. Quotient-surrogate preservation and "
        "raw-covariance overlap\n"
    )
    sections.append(markdown_table(safe_key_report_view(safe_key)))

    sections.append(
        "_The following constrained-frontier summaries are retrospective "
        "evaluation envelopes over the tested positive-rank q grid. They use "
        "held-out task and environment metrics only to summarize the frontier; "
        "they are not proposed as a label-only deployment-time rule for "
        "selecting q._\n"
    )

    sections.append(
        "## 4. Best leakage reduction among feasible positive-rank edits "
        "under accuracy-drop budget 0.005\n"
    )

    sections.append(markdown_table(
        frontier_report_view(
            budget,
            budget_type="acc_drop",
            tol=0.005,
            methods=methods_short,
        )
    ))

    sections.append(
        "## 5. Best leakage reduction among feasible positive-rank edits "
        "under NLL-increase budget 0.01\n"
    )
    sections.append(markdown_table(
        frontier_report_view(
            budget,
            budget_type="nll_increase",
            tol=0.01,
            methods=methods_short,
        )
    ))

    sections.append(
        "_Paired constrained comparisons below use only outer seeds for "
        "which both methods have at least one tested positive-rank edit "
        "satisfying the stated label-loss budget. The separate feasibility "
        "rates show how often such a nonzero edit exists._\n"
    )
    sections.append(
        "## 6. Paired constrained-frontier comparison: DMD vs raw covariance\n"
    )
    pair_raw_view = pair_raw[
        (
            (pair_raw["budget_type"] == "acc_drop")
            & np.isclose(
                pd.to_numeric(pair_raw["budget_tol"], errors="coerce"),
                0.005,
            )
        )
        |
        (
            (pair_raw["budget_type"] == "nll_increase")
            & np.isclose(
                pd.to_numeric(pair_raw["budget_tol"], errors="coerce"),
                0.01,
            )
        )
    ].copy() if len(pair_raw) else pair_raw
    sections.append(markdown_table(pair_raw_view))

    sections.append(
        "## 7. Paired constrained-frontier comparison: DMD vs PCA-first\n"
    )
    pair_pca_view = pair_pca[
        (
            (pair_pca["budget_type"] == "acc_drop")
            & np.isclose(
                pd.to_numeric(pair_pca["budget_tol"], errors="coerce"),
                0.005,
            )
        )
        |
        (
            (pair_pca["budget_type"] == "nll_increase")
            & np.isclose(
                pd.to_numeric(pair_pca["budget_tol"], errors="coerce"),
                0.01,
            )
        )
    ].copy() if len(pair_pca) else pair_pca
    sections.append(markdown_table(pair_pca_view))

    sections.append(
        "## 8. Auditor performance on full, quotient-surrogate, "
        "residual, and edited representations "
        "(quadratic auditor)\n"
    )
    
    sections.append(markdown_table(
        quotient_report_view(quotient_core, probe="quadratic")
    ))

    sections.append(
        "## 9. Auditor performance on full, quotient-surrogate, "
        "residual, and edited representations "
        "(linear auditor)\n"
    )

    sections.append(markdown_table(
        quotient_report_view(quotient_core, probe="linear")
    ))

    # sections.append(
    #     "## 9. Alternative linear auditor at q=8\n"
    # )
    # sections.append(markdown_table(
    #     alt_probe_report_view(
    #         alt_core,
    #         probe="linear",
    #         methods=["full", "dmd", "raw_cov", "pca_first"],
    #     )
    # ))

    # sections.append(
    #     "## 10. Alternative quadratic auditor at q=8\n"
    # )
    # sections.append(markdown_table(
    #     alt_probe_report_view(
    #         alt_core,
    #         probe="quadratic",
    #         methods=["full", "dmd", "raw_cov", "pca_first"],
    #     )
    # ))

    if quotient_mlp is not None and len(quotient_mlp) > 0:
        sections.append(
            "## 10. Auditor performance on full, quotient-surrogate, "
            "residual, and edited representations "
            "(MLP auditor; three-seed supplementary run)\n"
        )
        sections.append(markdown_table(
            quotient_report_view(quotient_mlp, probe="mlp")
        ))

    # if alt_mlp is not None and len(alt_mlp) > 0:
    #     sections.append(
    #         "## 12. MLP auditor at q=8 "
    #         "(three-seed supplementary run)\n"
    #     )
    #     sections.append(markdown_table(
    #         alt_probe_report_view(
    #             alt_mlp,
    #             probe="mlp",
    #             methods=["full", "dmd", "raw_cov", "pca_first"],
    #         )
    #     ))
    sections.append(
        "_For sensitivity, each valid hyperparameter configuration is first "
        "averaged over outer seeds; the reported mean/min/max then summarize "
        "these configuration-level means. DMD is summarized over all tested "
        "(r, alpha) pairs, whereas raw covariance is summarized only over "
        "alpha at the default r because raw covariance has no quotient-rank "
        "hyperparameter._\n"
    )

    sections.append(
        "## 11. Compact r/alpha sensitivity summary\n"
    )
    sections.append(markdown_table(
        sensitivity_report_view(sensitivity)
    ))

    sections.append(
        "\n## Interpretation warning\n\n"
        "Auditor accuracies on the full, quotient-surrogate, and residual "
        "representations are parallel audits, not an additive decomposition "
        "of information. Derived leakage ratios can exceed one when a reduced "
        "representation is easier for a selected auditor than the full "
        "representation, or when full leakage is close to the majority "
        "baseline. We therefore emphasize direct held-out auditor accuracies "
        "and leakage reductions. No probe result is interpreted as proof of "
        "complete nuisance removal.\n"
    )

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"[save] {path}", flush=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--core_roots",
        nargs="+",
        required=True,
        help=(
            "Directories or all_rebuttal_records.csv files for the "
            "10-seed ResNet/CLIP core runs."
        ),
    )
    parser.add_argument(
        "--mlp_roots",
        nargs="*",
        default=[],
        help=(
            "Optional directories/files for the three-seed MLP runs. "
            "They are summarized separately to avoid duplicating the "
            "linear/quadratic core observations."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="rebuttal_compact_summary",
    )
    parser.add_argument("--q_eval", type=int, default=8)

    args = parser.parse_args()
    ensure_dir(args.outdir)

    core_paths = split_paths(args.core_roots)
    mlp_paths = split_paths(args.mlp_roots)

    core_files = discover_all_records(core_paths)
    mlp_files = discover_all_records(mlp_paths)

    if not core_files:
        raise FileNotFoundError(
            "No core all_rebuttal_records.csv files were found."
        )

    print("====================================================")
    print("Core files:")
    for f in core_files:
        print(f"  {f}")
    print("MLP files:")
    for f in mlp_files:
        print(f"  {f}")
    print("====================================================")

    core = load_record_files(core_files, "core")
    mlp = (
        load_record_files(mlp_files, "mlp")
        if mlp_files else pd.DataFrame()
    )
    
    validate_unique_records(core, "core runs")
    if len(mlp):
        validate_unique_records(mlp, "MLP runs")

    # Coverage
    coverage_parts = [summarize_coverage(core)]
    if len(mlp):
        coverage_parts.append(summarize_coverage(mlp))
    coverage = pd.concat(
    coverage_parts, ignore_index=True
    ) if coverage_parts else pd.DataFrame()

    warn_unexpected_seed_counts(coverage)

    # Core summaries
    stage1 = summarize_stage1_equivalence(core)
    safe_q8 = summarize_safe_q8(core, q_eval=args.q_eval)
    safe_key = summarize_safe_edit_key(core, q_eval=args.q_eval)

    frontier_curve = summarize_frontier_curve(core)
    frontier_budget = summarize_frontier_budget(core)
    pair_raw = paired_budget_comparison(core, "raw_cov")
    pair_pca = paired_budget_comparison(core, "pca_first")
    pair_lda = paired_budget_comparison(core, "lda_first")
    pair_pls = paired_budget_comparison(core, "pls_first")

    quotient_core = summarize_quotient_representations(core)
    quotient_ratio_core = summarize_quotient_ratios(core)
    alt_core = summarize_alternative_probes(
        core,
        include_only=["linear", "quadratic", "max"],
    )

    sensitivity_compact = summarize_sensitivity_compact(core)
    sensitivity_alpha = summarize_sensitivity_alpha(core)
    sensitivity_r = summarize_sensitivity_r(
        core,
        alpha_eval=0.10,
    )

    if (
        len(sensitivity_compact) == 0
        and "r_alpha_sensitivity"
        not in set(core["experiment"].astype(str))
    ):
        print(
            "[info] No r/alpha sensitivity records were found. "
            "This is expected for the frozen-language modality check.",
            flush=True,
        )




    # MLP-only summaries. Do not merge linear/quadratic columns from
    # these three-seed reruns into the 10-seed core summaries.
    if len(mlp):
        quotient_mlp = summarize_quotient_representations(mlp)
        quotient_ratio_mlp = summarize_quotient_ratios(mlp)
        alt_mlp = summarize_alternative_probes(
            mlp,
            include_only=["mlp"],
        )
    else:
        quotient_mlp = pd.DataFrame()
        quotient_ratio_mlp = pd.DataFrame()
        alt_mlp = pd.DataFrame()

    # Save compact and detailed outputs.
    outputs = {
        "00_coverage.csv": coverage,
        "01_stage1_equivalence.csv": stage1,
        "02_safe_edit_q8.csv": safe_q8,
        "03_safe_edit_key_summary.csv": safe_key,
        "04_full_q_frontier_aggregated.csv": frontier_curve,
        "05_frontier_budget_summary.csv": frontier_budget,
        "06a_paired_dmd_vs_raw.csv": pair_raw,
        "06b_paired_dmd_vs_pca.csv": pair_pca,
        "06c_paired_dmd_vs_lda.csv": pair_lda,
        "06d_paired_dmd_vs_pls.csv": pair_pls,
        "07_quotient_residual_decomposition_core.csv":
            quotient_core,
        "08_quotient_leakage_ratios_core.csv":
            quotient_ratio_core,
        "09_alternative_probes_core.csv": alt_core,
        "10_sensitivity_compact.csv": sensitivity_compact,
        "11_sensitivity_alpha_default_r.csv": sensitivity_alpha,
        "12_sensitivity_r_alpha_010.csv": sensitivity_r,
        "13_quotient_residual_decomposition_mlp.csv":
            quotient_mlp,
        "14_quotient_leakage_ratios_mlp.csv":
            quotient_ratio_mlp,
        "15_alternative_probe_mlp.csv": alt_mlp,
    }

    for name, df in outputs.items():
        save_csv(df, os.path.join(args.outdir, name))

    generate_key_facts(
        stage1=stage1,
        safe_key=safe_key,
        pair_raw=pair_raw,
        pair_pca=pair_pca,
        out_path=os.path.join(args.outdir, "KEY_FACTS.txt"),
    )

    generate_markdown_report(
        path=os.path.join(
            args.outdir,
            "REBUTTAL_COMPACT_REPORT.md",
        ),
        coverage=coverage,
        stage1=stage1,
        safe_key=safe_key,
        budget=frontier_budget,
        pair_raw=pair_raw,
        pair_pca=pair_pca,
        quotient_core=quotient_core,
        quotient_ratio_core=quotient_ratio_core,
        alt_core=alt_core,
        sensitivity=sensitivity_compact,
        quotient_mlp=quotient_mlp,
        alt_mlp=alt_mlp,
    )

    print("====================================================")
    print("Summary complete.")
    print(
        "Please provide these files for rebuttal drafting:"
    )
    print(
        os.path.join(
            args.outdir,
            "REBUTTAL_COMPACT_REPORT.md",
        )
    )
    print(os.path.join(args.outdir, "KEY_FACTS.txt"))
    print(os.path.join(args.outdir, "00_coverage.csv"))
    print("====================================================")


if __name__ == "__main__":
    main()