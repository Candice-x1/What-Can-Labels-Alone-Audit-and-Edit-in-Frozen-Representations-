import os
import json
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
)

import run_labelonly_rebuttal_experiments as rb

base = rb.base


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def parse_list(value, cast=str):
    return [
        cast(x.strip())
        for x in str(value).split(",")
        if x.strip()
    ]


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


def joint_stratified_bootstrap_indices(
    y,
    e,
    rng,
):
    """
    Resample held-out examples separately within each observed
    (Y,E) cell. This preserves every observed task/environment
    cell in each bootstrap replicate.
    """
    y = np.asarray(y)
    e = np.asarray(e)

    cells = sorted(
        set(
            (int(yy), int(ee))
            for yy, ee in zip(y, e)
        )
    )

    sampled_parts = []

    for yy, ee in cells:
        indices = np.flatnonzero(
            (y == yy) & (e == ee)
        )

        if len(indices) == 0:
            continue

        sampled = rng.choice(
            indices,
            size=len(indices),
            replace=True,
        )
        sampled_parts.append(sampled)

    if len(sampled_parts) == 0:
        raise ValueError(
            "No observed (Y,E) cells for bootstrap."
        )

    output = np.concatenate(sampled_parts)
    rng.shuffle(output)

    return output.astype(np.int64)


def task_metrics(
    y_true,
    pred,
    prob,
    classes,
    indices,
):
    y_boot = y_true[indices]
    pred_boot = pred[indices]
    prob_boot = prob[indices]

    return {
        "accuracy": float(
            accuracy_score(
                y_boot,
                pred_boot,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_boot,
                pred_boot,
            )
        ),
        "nll": float(
            log_loss(
                y_boot,
                prob_boot,
                labels=classes,
            )
        ),
    }


def environment_metrics(
    e_true,
    pred,
    indices,
):
    e_boot = e_true[indices]
    pred_boot = pred[indices]

    return {
        "accuracy": float(
            accuracy_score(
                e_boot,
                pred_boot,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                e_boot,
                pred_boot,
            )
        ),
    }


# ============================================================
# Fit fixed-q representations and probes
# ============================================================

def fit_setting_seed(
    Xtr,
    ytr,
    etr,
    Xte,
    yte,
    ete,
    dataset,
    seed,
    q,
    num_avg,
):
    """
    Fit DMD/raw at prespecified q and fit matched task/environment
    probes. The fitting-seed offset matches the safe-edit experiment.
    """
    methods, _ = rb.fit_stage2_methods(
        dataset_name=dataset,
        Xtr=Xtr,
        ytr=ytr,
        etr=etr,
        q_max=q,
        num_avg=num_avg,
        seed=seed + 20,
    )

    V_dmd = methods["dmd"]["V"][:, :q]
    V_raw = methods["raw_cov"]["V"][:, :q]

    representations = {
        "full": (
            Xtr,
            Xte,
        ),
        "dmd": (
            base.residualize_subspace(
                Xtr,
                V_dmd,
            ),
            base.residualize_subspace(
                Xte,
                V_dmd,
            ),
        ),
        "raw": (
            base.residualize_subspace(
                Xtr,
                V_raw,
            ),
            base.residualize_subspace(
                Xte,
                V_raw,
            ),
        ),
    }

    outputs = {}

    # These probe seeds are shared across representations.
    # LBFGS is deterministic on fixed inputs, but sharing the seed
    # preserves a paired protocol.
    task_probe_seed = seed + 100
    env_probe_seed = seed + 200

    for name, (
        Xtr_rep,
        Xte_rep,
    ) in representations.items():
        task = base.fit_logreg_representation(
            Xtr_rep,
            ytr,
            Xte_rep,
            yte,
            seed=task_probe_seed,
        )

        env = base.fit_env_probe_quadratic(
            Xtr_rep,
            etr,
            Xte_rep,
            ete,
            seed=env_probe_seed,
        )

        outputs[name] = {
            "task_pred": np.asarray(
                task["pred"]
            ),
            "task_prob": np.asarray(
                task["prob"],
                dtype=float,
            ),
            "task_classes": np.asarray(
                task["classes"]
            ),
            "env_pred": np.asarray(
                env["pred"]
            ),
        }

    return outputs


# ============================================================
# Point estimates
# ============================================================

def compute_seed_point_estimates(
    outputs,
    yte,
    ete,
):
    indices = np.arange(
        len(yte),
        dtype=np.int64,
    )

    full_task = task_metrics(
        y_true=yte,
        pred=outputs["full"]["task_pred"],
        prob=outputs["full"]["task_prob"],
        classes=outputs["full"]["task_classes"],
        indices=indices,
    )

    full_env = environment_metrics(
        e_true=ete,
        pred=outputs["full"]["env_pred"],
        indices=indices,
    )

    rows = []

    edited_metrics = {}

    for method in ["dmd", "raw"]:
        method_task = task_metrics(
            y_true=yte,
            pred=outputs[method]["task_pred"],
            prob=outputs[method]["task_prob"],
            classes=outputs[method]["task_classes"],
            indices=indices,
        )

        method_env = environment_metrics(
            e_true=ete,
            pred=outputs[method]["env_pred"],
            indices=indices,
        )

        edited_metrics[method] = {
            "task": method_task,
            "env": method_env,
        }

        rows.append({
            "comparison": method,

            # Absolute task metrics
            "full_task_accuracy":
                full_task["accuracy"],
            "edited_task_accuracy":
                method_task["accuracy"],
            "full_task_balanced_accuracy":
                full_task["balanced_accuracy"],
            "edited_task_balanced_accuracy":
                method_task["balanced_accuracy"],
            "full_task_nll":
                full_task["nll"],
            "edited_task_nll":
                method_task["nll"],

            # Absolute environment-auditor metrics
            "full_env_accuracy":
                full_env["accuracy"],
            "edited_env_accuracy":
                method_env["accuracy"],
            "full_env_balanced_accuracy":
                full_env["balanced_accuracy"],
            "edited_env_balanced_accuracy":
                method_env["balanced_accuracy"],

            # Paired changes relative to full
            "task_delta_accuracy": (
                method_task["accuracy"]
                - full_task["accuracy"]
            ),
            "task_delta_balanced_accuracy": (
                method_task["balanced_accuracy"]
                - full_task["balanced_accuracy"]
            ),
            "task_delta_nll": (
                method_task["nll"]
                - full_task["nll"]
            ),
            "env_reduction_accuracy": (
                full_env["accuracy"]
                - method_env["accuracy"]
            ),
            "env_reduction_balanced_accuracy": (
                full_env["balanced_accuracy"]
                - method_env["balanced_accuracy"]
            ),
        })

    rows.append({
        "comparison": "dmd_minus_raw",
        "task_accuracy_advantage": (
            edited_metrics["dmd"]["task"]["accuracy"]
            - edited_metrics["raw"]["task"]["accuracy"]
        ),
        "task_balanced_accuracy_advantage": (
            edited_metrics["dmd"]["task"]["balanced_accuracy"]
            - edited_metrics["raw"]["task"]["balanced_accuracy"]
        ),
        # Positive means DMD has lower NLL than raw.
        "task_nll_advantage": (
            edited_metrics["raw"]["task"]["nll"]
            - edited_metrics["dmd"]["task"]["nll"]
        ),
        # Positive means DMD leaves lower auditor accuracy.
        "env_accuracy_advantage": (
            edited_metrics["raw"]["env"]["accuracy"]
            - edited_metrics["dmd"]["env"]["accuracy"]
        ),
        "env_balanced_accuracy_advantage": (
            edited_metrics["raw"]["env"]["balanced_accuracy"]
            - edited_metrics["dmd"]["env"]["balanced_accuracy"]
        ),
    })

    return rows


# ============================================================
# Hierarchical paired bootstrap
# ============================================================

def bootstrap_setting(
    seed_outputs,
    yte,
    ete,
    num_bootstrap,
    bootstrap_seed,
):
    """
    Two-level paired bootstrap:
      1. algorithmic fitting seeds are sampled with replacement;
      2. held-out examples are sampled within observed (Y,E) cells.

    Intervals are conditional on the fixed training split and do
    not include training-sample uncertainty.
    """
    rng = np.random.default_rng(
        int(bootstrap_seed)
    )

    fitting_seeds = sorted(
        seed_outputs.keys()
    )

    if len(fitting_seeds) == 0:
        raise ValueError(
            "No fitting-seed outputs were supplied."
        )

    values = {}

    def append_value(key, value):
        values.setdefault(
            key,
            [],
        ).append(float(value))

    for _ in range(int(num_bootstrap)):
        indices = joint_stratified_bootstrap_indices(
            yte,
            ete,
            rng,
        )

        sampled_positions = rng.integers(
            0,
            len(fitting_seeds),
            size=len(fitting_seeds),
        )

        sampled_seeds = [
            fitting_seeds[position]
            for position in sampled_positions
        ]

        per_seed = []

        for seed in sampled_seeds:
            outputs = seed_outputs[seed]

            record = {}

            for representation in [
                "full",
                "dmd",
                "raw",
            ]:
                record[representation] = {
                    "task": task_metrics(
                        y_true=yte,
                        pred=outputs[
                            representation
                        ]["task_pred"],
                        prob=outputs[
                            representation
                        ]["task_prob"],
                        classes=outputs[
                            representation
                        ]["task_classes"],
                        indices=indices,
                    ),
                    "env": environment_metrics(
                        e_true=ete,
                        pred=outputs[
                            representation
                        ]["env_pred"],
                        indices=indices,
                    ),
                }

            per_seed.append(record)

        for method in ["dmd", "raw"]:
            append_value(
                f"{method}:task_delta_accuracy",
                np.mean([
                    record[method]["task"]["accuracy"]
                    - record["full"]["task"]["accuracy"]
                    for record in per_seed
                ]),
            )

            append_value(
                (
                    f"{method}:"
                    "task_delta_balanced_accuracy"
                ),
                np.mean([
                    record[method]["task"][
                        "balanced_accuracy"
                    ]
                    - record["full"]["task"][
                        "balanced_accuracy"
                    ]
                    for record in per_seed
                ]),
            )

            append_value(
                f"{method}:task_delta_nll",
                np.mean([
                    record[method]["task"]["nll"]
                    - record["full"]["task"]["nll"]
                    for record in per_seed
                ]),
            )

            append_value(
                f"{method}:env_reduction_accuracy",
                np.mean([
                    record["full"]["env"]["accuracy"]
                    - record[method]["env"]["accuracy"]
                    for record in per_seed
                ]),
            )

            append_value(
                (
                    f"{method}:"
                    "env_reduction_balanced_accuracy"
                ),
                np.mean([
                    record["full"]["env"][
                        "balanced_accuracy"
                    ]
                    - record[method]["env"][
                        "balanced_accuracy"
                    ]
                    for record in per_seed
                ]),
            )

        append_value(
            "dmd_minus_raw:task_accuracy_advantage",
            np.mean([
                record["dmd"]["task"]["accuracy"]
                - record["raw"]["task"]["accuracy"]
                for record in per_seed
            ]),
        )

        append_value(
            (
                "dmd_minus_raw:"
                "task_balanced_accuracy_advantage"
            ),
            np.mean([
                record["dmd"]["task"][
                    "balanced_accuracy"
                ]
                - record["raw"]["task"][
                    "balanced_accuracy"
                ]
                for record in per_seed
            ]),
        )

        # Positive means DMD has lower NLL.
        append_value(
            "dmd_minus_raw:task_nll_advantage",
            np.mean([
                record["raw"]["task"]["nll"]
                - record["dmd"]["task"]["nll"]
                for record in per_seed
            ]),
        )

        # Positive means DMD leaves lower auditor accuracy.
        append_value(
            "dmd_minus_raw:env_accuracy_advantage",
            np.mean([
                record["raw"]["env"]["accuracy"]
                - record["dmd"]["env"]["accuracy"]
                for record in per_seed
            ]),
        )

        append_value(
            (
                "dmd_minus_raw:"
                "env_balanced_accuracy_advantage"
            ),
            np.mean([
                record["raw"]["env"][
                    "balanced_accuracy"
                ]
                - record["dmd"]["env"][
                    "balanced_accuracy"
                ]
                for record in per_seed
            ]),
        )

    rows = []

    for key, samples in sorted(
        values.items()
    ):
        samples = np.asarray(
            samples,
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
                np.mean(samples)
            ),
            "ci_lower_95": float(
                np.quantile(samples, 0.025)
            ),
            "ci_upper_95": float(
                np.quantile(samples, 0.975)
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
        "--datasets",
        type=str,
        default="celeba,waterbirds,pacs",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="resnet18,clip_vit_b32",
    )
    parser.add_argument(
        "--seed_list",
        type=str,
        default=(
            "42,123,3407,2024,2025,"
            "7,17,29,314,2718"
        ),
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
        "--bootstrap_seed",
        type=int,
        default=271828,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="q8_paired_bootstrap",
    )

    args = parser.parse_args()
    ensure_dir(args.outdir)

    datasets = parse_list(
        args.datasets,
        str,
    )
    backbones = parse_list(
        args.backbones,
        str,
    )
    fitting_seeds = parse_list(
        args.seed_list,
        int,
    )

    if len(fitting_seeds) == 0:
        raise ValueError(
            "No fitting seeds were provided."
        )

    if len(set(fitting_seeds)) != len(fitting_seeds):
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

    all_point_rows = []
    all_ci_frames = []
    setting_metadata = []

    setting_index = 0

    for backbone in backbones:
        for dataset in datasets:
            print(
                f"[setting] {dataset}/{backbone}",
                flush=True,
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
                dataset_name=dataset,
                backbone_name=backbone,
                pretrained=1,
                max_train=0,
                max_test=0,
                seed=0,
            )

            ytr = np.asarray(ytr)
            etr = np.asarray(etr)
            yte = np.asarray(yte)
            ete = np.asarray(ete)

            seed_outputs = {}

            for seed in fitting_seeds:
                print(
                    f"  [fitting seed] {seed}",
                    flush=True,
                )

                outputs = fit_setting_seed(
                    Xtr=Xtr,
                    ytr=ytr,
                    etr=etr,
                    Xte=Xte,
                    yte=yte,
                    ete=ete,
                    dataset=dataset,
                    seed=seed,
                    q=args.q,
                    num_avg=args.num_avg,
                )

                seed_outputs[seed] = outputs

                point_rows = (
                    compute_seed_point_estimates(
                        outputs=outputs,
                        yte=yte,
                        ete=ete,
                    )
                )

                for row in point_rows:
                    row["dataset"] = dataset
                    row["backbone"] = backbone
                    row["seed"] = int(seed)
                    row["q"] = int(args.q)
                    all_point_rows.append(row)

            setting_bootstrap_seed = (
                int(args.bootstrap_seed)
                + 1009 * setting_index
            )

            ci_df = bootstrap_setting(
                seed_outputs=seed_outputs,
                yte=yte,
                ete=ete,
                num_bootstrap=args.num_bootstrap,
                bootstrap_seed=setting_bootstrap_seed,
            )

            ci_df["dataset"] = dataset
            ci_df["backbone"] = backbone
            ci_df["q"] = int(args.q)
            ci_df["bootstrap_seed"] = int(
                setting_bootstrap_seed
            )

            all_ci_frames.append(ci_df)

            setting_metadata.append({
                "dataset": dataset,
                "backbone": backbone,
                "num_train": int(len(ytr)),
                "num_test": int(len(yte)),
                "num_task_classes": int(
                    len(np.unique(ytr))
                ),
                "num_environment_classes": int(
                    len(np.unique(etr))
                ),
                "q": int(args.q),
                "bootstrap_seed": int(
                    setting_bootstrap_seed
                ),
                "source_meta": source_meta,
            })

            setting_index += 1

    point_df = pd.DataFrame(
        all_point_rows
    )

    ci_df = pd.concat(
        all_ci_frames,
        ignore_index=True,
    )

    point_path = os.path.join(
        args.outdir,
        "q8_per_seed_point_estimates.csv",
    )
    ci_path = os.path.join(
        args.outdir,
        "q8_paired_bootstrap_ci.csv",
    )

    point_df.to_csv(
        point_path,
        index=False,
    )
    ci_df.to_csv(
        ci_path,
        index=False,
    )

    metadata = {
        "q": int(args.q),
        "datasets": datasets,
        "backbones": backbones,
        "fitting_seeds": fitting_seeds,
        "num_fitting_seeds": int(
            len(fitting_seeds)
        ),
        "num_bootstrap": int(
            args.num_bootstrap
        ),
        "bootstrap_type": (
            "two-level paired bootstrap: algorithmic fitting "
            "seeds are resampled with replacement and held-out "
            "test examples are resampled within observed (Y,E) "
            "cells"
        ),
        "scope": (
            "Fixed prespecified q=8. Intervals are conditional "
            "on the fixed training split and include algorithmic-"
            "seed and held-out test-sample variability, but not "
            "training-sample uncertainty. These intervals do not "
            "apply to the test-selected constrained frontier."
        ),
        "environment_auditor": (
            "train-fitted PCA to at most 16 dimensions, all "
            "linear and upper-triangular quadratic features, "
            "training-statistics standardization, and balanced "
            "L2 logistic regression with LBFGS"
        ),
        "settings": setting_metadata,
        "output_files": {
            "point_estimates": os.path.abspath(
                point_path
            ),
            "bootstrap_ci": os.path.abspath(
                ci_path
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
    print("q=8 paired bootstrap intervals")
    print("=====================================================")
    print(ci_df.to_string(index=False))
    print(f"[save] {point_path}", flush=True)
    print(f"[save] {ci_path}", flush=True)
    print(f"[save] {metadata_path}", flush=True)
    print(f"[done] {args.outdir}", flush=True)


if __name__ == "__main__":
    main()