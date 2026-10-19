import os
import json
import argparse
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    log_loss,
)
from sklearn.exceptions import ConvergenceWarning

import run_labelonly_rebuttal_experiments as rb

base = rb.base


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def parse_int_list(value):
    return [
        int(x.strip())
        for x in str(value).split(",")
        if x.strip()
    ]


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable."
    )


def safe_auroc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(y_true, scores)
    )


def safe_auprc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        average_precision_score(y_true, scores)
    )


def binary_metrics(y_true, pred, prob_pos):
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=float)

    return {
        "accuracy": float(
            accuracy_score(y_true, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "auroc": safe_auroc(
            y_true,
            prob_pos,
        ),
        "auprc": safe_auprc(
            y_true,
            prob_pos,
        ),
        "positive_recall": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "negative_recall": float(
            recall_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0,
            )
        ),
    }


def joint_stratified_bootstrap_indices(
    y,
    e,
    rng,
):
    """
    Resample test examples with replacement separately within each
    observed (Y,E) cell. This preserves task and audit classes in
    every bootstrap replicate.
    """
    y = np.asarray(y)
    e = np.asarray(e)

    sampled_parts = []

    cells = sorted(
        set(
            (int(yy), int(ee))
            for yy, ee in zip(y, e)
        )
    )

    for yy, ee in cells:
        idx = np.flatnonzero(
            (y == yy) & (e == ee)
        )

        if len(idx) == 0:
            continue

        sampled = rng.choice(
            idx,
            size=len(idx),
            replace=True,
        )
        sampled_parts.append(sampled)

    if len(sampled_parts) == 0:
        raise ValueError(
            "No nonempty (Y,E) cells for bootstrap."
        )

    indices = np.concatenate(sampled_parts)
    rng.shuffle(indices)

    return indices.astype(np.int64)


# ============================================================
# Probe fitting
# ============================================================

def fit_mlp_auditor(
    Xtr,
    etr,
    Xte,
    seed,
    hidden_dim=64,
):
    """
    Same MLP family as the supplementary auditor:
    one hidden layer, early stopping, no explicit class weighting.
    Balanced accuracy/AUROC/AUPRC are used for evaluation.
    """
    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=ConvergenceWarning,
        )

        clf = MLPClassifier(
            hidden_layer_sizes=(int(hidden_dim),),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            random_state=int(seed),
        )

        clf.fit(Xtr_s, etr)

    pred = clf.predict(Xte_s)
    prob = clf.predict_proba(Xte_s)

    classes = list(clf.classes_)

    if 1 not in classes:
        raise ValueError(
            "MLP auditor training did not contain class 1."
        )

    positive_index = classes.index(1)

    return {
        "pred": np.asarray(
            pred,
            dtype=np.int64,
        ),
        "prob_pos": np.asarray(
            prob[:, positive_index],
            dtype=float,
        ),
        "classes": np.asarray(
            clf.classes_,
            dtype=np.int64,
        ),
    }


def fit_task_probe(
    Xtr,
    ytr,
    Xte,
    yte,
    seed,
):
    result = base.fit_logreg_representation(
        Xtr,
        ytr,
        Xte,
        yte,
        seed=int(seed),
    )

    classes = np.asarray(result["classes"])

    if 1 not in classes.tolist():
        raise ValueError(
            "Task-probe training did not contain class 1."
        )

    positive_index = classes.tolist().index(1)

    prob = np.asarray(result["prob"])
    pred = np.asarray(result["pred"])

    metrics = binary_metrics(
        y_true=yte,
        pred=pred,
        prob_pos=prob[:, positive_index],
    )

    metrics["nll"] = float(
        log_loss(
            yte,
            prob,
            labels=classes,
        )
    )

    return {
        "pred": pred.astype(np.int64),
        "prob": prob.astype(float),
        "prob_pos": prob[:, positive_index].astype(float),
        "classes": classes.astype(np.int64),
        "metrics": metrics,
    }


# ============================================================
# Representations and metrics
# ============================================================

def construct_representations(
    Xtr,
    Xte,
    U,
    V_dmd,
    V_raw,
):
    return {
        "full": (
            Xtr,
            Xte,
        ),
        "quotient": (
            Xtr @ U,
            Xte @ U,
        ),
        "residual": (
            base.residualize_subspace(Xtr, U),
            base.residualize_subspace(Xte, U),
        ),
        "dmd": (
            base.residualize_subspace(Xtr, V_dmd),
            base.residualize_subspace(Xte, V_dmd),
        ),
        "raw": (
            base.residualize_subspace(Xtr, V_raw),
            base.residualize_subspace(Xte, V_raw),
        ),
    }


def evaluate_task_outputs(
    y,
    output,
    indices,
):
    y_boot = y[indices]
    pred_boot = output["pred"][indices]
    prob_boot = output["prob"][indices]
    prob_pos_boot = output["prob_pos"][indices]

    metrics = binary_metrics(
        y_boot,
        pred_boot,
        prob_pos_boot,
    )

    metrics["nll"] = float(
        log_loss(
            y_boot,
            prob_boot,
            labels=output["classes"],
        )
    )

    return metrics


def evaluate_env_outputs(
    e,
    output,
    indices,
):
    return binary_metrics(
        e[indices],
        output["pred"][indices],
        output["prob_pos"][indices],
    )


def summarize_per_seed_rows(rows):
    df = pd.DataFrame(rows)

    id_columns = {
        "seed",
        "representation",
    }

    metric_columns = [
        c for c in df.columns
        if c not in id_columns
    ]

    summary_rows = []

    for representation, group in df.groupby(
        "representation"
    ):
        row = {
            "representation": representation,
            "num_seeds": int(
                group["seed"].nunique()
            ),
        }

        for metric in metric_columns:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            row[f"{metric}_mean"] = float(
                values.mean()
            )
            row[f"{metric}_std"] = float(
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ============================================================
# Hierarchical paired bootstrap
# ============================================================

def bootstrap_civilcomments(
    prediction_store,
    y,
    e,
    num_bootstrap,
    bootstrap_seed,
):
    """
    Two-level paired bootstrap:
      1. fitting/optimization seeds are sampled with replacement;
      2. test examples are sampled with replacement within joint
         (Y,E) cells.

    The result reflects algorithmic-seed and held-out test-sample
    variability conditional on the fixed training cohort.
    """
    rng = np.random.default_rng(
        int(bootstrap_seed)
    )

    fitting_seeds = sorted(
        prediction_store.keys()
    )

    task_utility_metrics = [
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
    ]

    task_damage_metrics = [
        "nll",
    ]

    env_metrics = [
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "positive_recall",
        "negative_recall",
    ]

    bootstrap_values = {}

    def append_value(name, value):
        bootstrap_values.setdefault(
            name,
            [],
        ).append(float(value))

    for bootstrap_id in range(
        int(num_bootstrap)
    ):
        indices = joint_stratified_bootstrap_indices(
            y,
            e,
            rng,
        )

        sampled_seed_positions = rng.integers(
            0,
            len(fitting_seeds),
            size=len(fitting_seeds),
        )

        sampled_seeds = [
            fitting_seeds[position]
            for position in sampled_seed_positions
        ]

        per_seed = []

        for seed in sampled_seeds:
            outputs = prediction_store[seed]

            record = {}

            for representation in [
                "full",
                "dmd",
                "raw",
            ]:
                task_metrics = evaluate_task_outputs(
                    y,
                    outputs[representation]["task"],
                    indices,
                )

                audit_metrics = evaluate_env_outputs(
                    e,
                    outputs[representation]["env"],
                    indices,
                )

                record[representation] = {
                    "task": task_metrics,
                    "env": audit_metrics,
                }

            per_seed.append(record)

        for edited in ["dmd", "raw"]:
            # Task metrics: edited minus full.
            for metric in task_utility_metrics:
                values = [
                    rec[edited]["task"][metric]
                    - rec["full"]["task"][metric]
                    for rec in per_seed
                ]

                append_value(
                    f"{edited}:task_delta_{metric}",
                    np.mean(values),
                )

            for metric in task_damage_metrics:
                values = [
                    rec[edited]["task"][metric]
                    - rec["full"]["task"][metric]
                    for rec in per_seed
                ]

                append_value(
                    f"{edited}:task_delta_{metric}",
                    np.mean(values),
                )

            # Environment metrics: full minus edited.
            # Positive means lower auditor performance after edit.
            for metric in env_metrics:
                values = [
                    rec["full"]["env"][metric]
                    - rec[edited]["env"][metric]
                    for rec in per_seed
                ]

                append_value(
                    f"{edited}:env_reduction_{metric}",
                    np.mean(values),
                )

        # Direct DMD-versus-raw task comparison.
        for metric in task_utility_metrics:
            values = [
                rec["dmd"]["task"][metric]
                - rec["raw"]["task"][metric]
                for rec in per_seed
            ]

            append_value(
                f"dmd_vs_raw:task_utility_{metric}",
                np.mean(values),
            )

        # Positive means DMD has lower NLL than raw.
        values = [
            rec["raw"]["task"]["nll"]
            - rec["dmd"]["task"]["nll"]
            for rec in per_seed
        ]

        append_value(
            "dmd_vs_raw:task_nll_advantage",
            np.mean(values),
        )

        # Positive means DMD leaves lower auditor performance.
        for metric in env_metrics:
            values = [
                rec["raw"]["env"][metric]
                - rec["dmd"]["env"][metric]
                for rec in per_seed
            ]

            append_value(
                f"dmd_vs_raw:env_advantage_{metric}",
                np.mean(values),
            )

    rows = []

    for key, values in sorted(
        bootstrap_values.items()
    ):
        values = np.asarray(
            values,
            dtype=float,
        )

        comparison, metric = key.split(
            ":",
            1,
        )

        rows.append({
            "comparison": comparison,
            "metric": metric,
            "mean": float(
                np.nanmean(values)
            ),
            "ci_lower_95": float(
                np.nanquantile(values, 0.025)
            ),
            "ci_upper_95": float(
                np.nanquantile(values, 0.975)
            ),
            "num_bootstrap": int(
                num_bootstrap
            ),
            "num_fitting_seeds": int(
                len(fitting_seeds)
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="all_mpnet_base_v2",
    )
    parser.add_argument(
        "--seed_list",
        type=str,
        default="42,123,3407,2024,2025",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_avg",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_bootstrap",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="civilcomments_balanced_audit",
    )

    args = parser.parse_args()
    ensure_dir(args.outdir)

    seeds = parse_int_list(
        args.seed_list
    )

    if len(seeds) == 0:
        raise ValueError(
            "No fitting seeds were provided."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Duplicate fitting seeds were provided."
        )

    if args.q <= 0:
        raise ValueError(
            "--q must be positive."
        )

    if args.num_avg <= 0:
        raise ValueError(
            "--num_avg must be positive."
        )

    if args.num_bootstrap <= 0:
        raise ValueError(
            "--num_bootstrap must be positive."
        )

    if args.hidden_dim <= 0:
        raise ValueError(
            "--hidden_dim must be positive."
        )


    (
        Xtr,
        ytr,
        etr,
        Xte,
        yte,
        ete,
        source_meta,
    ) = rb.load_preprocessed(
        data_root=args.data_root,
        dataset_name="civilcomments",
        backbone_name=args.backbone,
        pretrained=1,
        max_train=0,
        max_test=0,
        seed=0,
    )

    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)
    etr = np.asarray(etr, dtype=np.int64)
    ete = np.asarray(ete, dtype=np.int64)

    if set(np.unique(ytr).tolist()) != {0, 1}:
        raise ValueError(
            "Training task label must be binary."
        )

    if set(np.unique(yte).tolist()) != {0, 1}:
        raise ValueError(
            "Test task label must be binary."
        )

    if set(np.unique(etr).tolist()) != {0, 1}:
        raise ValueError(
            "Training audit attribute must be binary."
        )

    if set(np.unique(ete).tolist()) != {0, 1}:
        raise ValueError(
            "Test audit attribute must be binary."
        )

    rows = []
    prediction_store = {}

    for seed in seeds:
        print(
            f"[fitting seed] {seed}",
            flush=True,
        )

        set_seed(seed)

        methods, artifacts = rb.fit_stage2_methods(
            dataset_name="civilcomments",
            Xtr=Xtr,
            ytr=ytr,
            etr=etr,
            q_max=args.q,
            num_avg=args.num_avg,
            seed=seed + 20,
        )

        U = artifacts["U_dmd"]
        V_dmd = methods["dmd"]["V"][
            :, :args.q
        ]
        V_raw = methods["raw_cov"]["V"][
            :, :args.q
        ]

        representations = construct_representations(
            Xtr,
            Xte,
            U,
            V_dmd,
            V_raw,
        )

        prediction_store[seed] = {}

        # Same MLP seed across representations within fitting seed.
        mlp_seed = int(seed + 500)

        for name, (
            Xtr_rep,
            Xte_rep,
        ) in representations.items():
            task = fit_task_probe(
                Xtr_rep,
                ytr,
                Xte_rep,
                yte,
                seed=seed + 100,
            )

            env = fit_mlp_auditor(
                Xtr_rep,
                etr,
                Xte_rep,
                seed=mlp_seed,
                hidden_dim=args.hidden_dim,
            )

            env_metrics = binary_metrics(
                ete,
                env["pred"],
                env["prob_pos"],
            )

            prediction_store[seed][name] = {
                "task": task,
                "env": env,
            }

            row = {
                "seed": int(seed),
                "representation": name,

                "task_accuracy":
                    task["metrics"]["accuracy"],
                "task_balanced_accuracy":
                    task["metrics"][
                        "balanced_accuracy"
                    ],
                "task_auroc":
                    task["metrics"]["auroc"],
                "task_auprc":
                    task["metrics"]["auprc"],
                "task_positive_recall":
                    task["metrics"][
                        "positive_recall"
                    ],
                "task_negative_recall":
                    task["metrics"][
                        "negative_recall"
                    ],
                "task_nll":
                    task["metrics"]["nll"],
            }

            for metric, value in (
                env_metrics.items()
            ):
                row[f"env_{metric}"] = value

            rows.append(row)

    per_seed_df = pd.DataFrame(rows)

    per_seed_path = os.path.join(
        args.outdir,
        "civilcomments_balanced_audit_per_seed.csv",
    )
    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )

    summary_df = summarize_per_seed_rows(
        rows
    )

    summary_path = os.path.join(
        args.outdir,
        "civilcomments_balanced_audit_summary.csv",
    )
    summary_df.to_csv(
        summary_path,
        index=False,
    )

    bootstrap_df = bootstrap_civilcomments(
        prediction_store=prediction_store,
        y=yte,
        e=ete,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=271828,
    )

    bootstrap_path = os.path.join(
        args.outdir,
        "civilcomments_paired_bootstrap_ci.csv",
    )
    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    metadata = {
        "dataset": "civilcomments",
        "backbone": args.backbone,
        "q": int(args.q),
        "fitting_seeds": seeds,
        "num_fitting_seeds": int(len(seeds)),
        "num_bootstrap": int(
            args.num_bootstrap
        ),
        "hidden_dim": int(
            args.hidden_dim
        ),
        "train_task_positive_prevalence":
            float(np.mean(ytr == 1)),
        "test_task_positive_prevalence":
            float(np.mean(yte == 1)),
        "train_attribute_positive_prevalence":
            float(np.mean(etr == 1)),
        "test_attribute_positive_prevalence":
            float(np.mean(ete == 1)),
        "attribute_majority_accuracy":
            float(
                max(
                    np.mean(ete == 0),
                    np.mean(ete == 1),
                )
            ),
        "attribute_auprc_random_baseline":
            float(np.mean(ete == 1)),
        "bootstrap_scope": (
            "Two-level paired bootstrap over algorithmic "
            "fitting seeds and held-out test examples. "
            "Test examples are resampled within joint (Y,E) "
            "cells. The interval is conditional on the fixed "
            "training cohort and does not include training-"
            "sample uncertainty."
        ),
        "auditor": {
            "family": "one-hidden-layer MLP",
            "hidden_dim": int(args.hidden_dim),
            "activation": "relu",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "early_stopping": True,
            "class_weight": (
                "not used; imbalance-robust evaluation metrics "
                "are reported"
            ),
        },
        "source_meta": source_meta,
        "output_files": {
            "per_seed": os.path.abspath(
                per_seed_path
            ),
            "summary": os.path.abspath(
                summary_path
            ),
            "bootstrap": os.path.abspath(
                bootstrap_path
            ),
        },
    }

    metadata_path = os.path.join(
        args.outdir,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=json_default,
        )

    print("=====================================================")
    print("CivilComments balanced-audit summary")
    print("=====================================================")
    print(summary_df.to_string(index=False))
    print("=====================================================")
    print("Paired bootstrap intervals")
    print("=====================================================")
    print(bootstrap_df.to_string(index=False))
    print(f"[save] {per_seed_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)
    print(f"[save] {bootstrap_path}", flush=True)
    print(f"[save] {metadata_path}", flush=True)
    print(f"[done] {args.outdir}", flush=True)


if __name__ == "__main__":
    main()