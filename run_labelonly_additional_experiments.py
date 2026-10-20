import os
import json
import argparse
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

try:
    import run_labelonly_workflow_full_v2_appendix as base
except Exception:
    import run_labelonly_full as base


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def parse_csv_list(s, cast=str):
    if s is None:
        return []
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()]


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def round_recursive(obj, ndigits=5):
    if isinstance(obj, dict):
        return {k: round_recursive(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_recursive(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return [round_recursive(v, ndigits) for v in obj]
    if isinstance(obj, np.ndarray):
        return round_recursive(obj.tolist(), ndigits)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_json(path, obj, ndigits=5):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(round_recursive(jsonable(obj), ndigits), f, indent=2, sort_keys=True)


def save_records_csv(path, records, ndigits=5):
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(records)
    if len(df) == 0:
        df.to_csv(path, index=False)
        return df
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(ndigits)
    df.to_csv(path, index=False)
    return df


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def projector_overlap(A, B):
    """
    trace(P_A P_B) / min(dim A, dim B).
    """
    if A is None or B is None:
        return 0.0
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    PA = A @ A.T
    PB = B @ B.T
    return float(np.trace(PA @ PB) / max(1, min(A.shape[1], B.shape[1])))


def orthogonalize_against(U, V, tol=1e-10):
    """
    Orthonormal basis for span((I-UU^T)V).
    This makes quotient-first residual edits explicitly satisfy P_V U = 0.
    """
    if V is None or V.shape[1] == 0:
        p = U.shape[0] if U is not None and U.shape[0] > 0 else 0
        return np.zeros((p, 0), dtype=float)

    Vres = V.copy()
    if U is not None and U.shape[1] > 0:
        Vres = Vres - U @ (U.T @ Vres)

    Q, R = np.linalg.qr(Vres)
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.array([])
    keep = np.where(diag > tol)[0]

    if keep.size == 0:
        return np.zeros((V.shape[0], 0), dtype=float)

    return Q[:, keep]


def rel_fro_norm(A, B=None, eps=1e-12):
    if B is None:
        return float(np.linalg.norm(A, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(B, ord="fro") + eps))


def quotient_preservation_error(X, U, V):
    """
    || U^T (I-P_V)X - U^T X ||_F / ||U^T X||_F.
    """
    if U.shape[1] == 0 or V.shape[1] == 0:
        return 0.0
    before = X @ U
    Xedit = base.residualize_subspace(X, V)
    after = Xedit @ U
    return rel_fro_norm(after, before)


def majority_baseline_accuracy(y_train, y_test):
    vals, counts = np.unique(y_train, return_counts=True)
    if len(vals) == 0:
        return 0.0
    maj = vals[np.argmax(counts)]
    return float(np.mean(np.asarray(y_test) == maj))


# ============================================================
# Loading and fitting
# ============================================================

def load_preprocessed(data_root, dataset_name, backbone_name, pretrained, max_train, max_test, seed):
    Xtr_raw, ytr, etr, train_path = base.load_cached_features(
        data_root=data_root,
        dataset_name=dataset_name,
        split="train",
        backbone_name=backbone_name,
        pretrained_flag=bool(pretrained),
    )
    Xte_raw, yte, ete, test_path = base.load_cached_features(
        data_root=data_root,
        dataset_name=dataset_name,
        split="test",
        backbone_name=backbone_name,
        pretrained_flag=bool(pretrained),
    )

    if max_train > 0:
        Xtr_raw, ytr, etr = base.subsample_if_needed(
            Xtr_raw, ytr, etr, max_n=max_train, seed=seed + 11
        )
    if max_test > 0:
        Xte_raw, yte, ete = base.subsample_if_needed(
            Xte_raw, yte, ete, max_n=max_test, seed=seed + 17
        )

    pca_dim = base.choose_pca_dim(dataset_name)
    Xtr, Xte = base.preprocess_real_features(
        Xtr_raw, Xte_raw, pca_dim=pca_dim, whiten=True, l2norm=True
    )

    meta = {
        "train_path": train_path,
        "test_path": test_path,
        "num_train": int(len(ytr)),
        "num_test": int(len(yte)),
        "num_classes": int(base.unique_count(ytr)),
        "num_envs": int(base.unique_count(etr)),
        "feature_dim": int(Xtr.shape[1]),
        "pca_dim": int(pca_dim),
    }
    return Xtr, ytr, etr, Xte, yte, ete, meta


def fit_stage2_methods(
    dataset_name,
    Xtr,
    ytr,
    etr,
    q_max,
    num_avg=8,
    seed=0,
    shrink_alpha=0.10,
    r_override=None,
):
    if r_override is None:
        r, _ = base.choose_r_q(dataset_name, ytr, etr)
    else:
        r = int(r_override)

    q_max = min(int(q_max), max(1, Xtr.shape[1] - 1))

    fit_dmd = base.fit_real_dmd_bagged_refit(
        Xtr,
        ytr,
        r=r,
        q=q_max,
        split_seed=seed + 10,
        num_avg=num_avg,
        shrink_alpha=shrink_alpha,
    )
    U_dmd = fit_dmd["Uhat"]

    U_pca = base.top_principal_subspace(Xtr, r)
    U_lda = base.lda_subspace(Xtr, ytr, r)
    U_pls = base.pls_subspace(Xtr, ytr, r)
    U_rand = base.random_subspace(Xtr, r, seed=seed + 20)

    fit_pca = base.fit_second_stage_given_U(Xtr, ytr, U_pca, q_max, shrink_alpha=shrink_alpha)
    fit_lda = base.fit_second_stage_given_U(Xtr, ytr, U_lda, q_max, shrink_alpha=shrink_alpha)
    fit_rand = base.fit_second_stage_given_U(Xtr, ytr, U_rand, q_max, shrink_alpha=shrink_alpha)

    if U_pls.shape[1] > 0:
        fit_pls = base.fit_second_stage_given_U(
            Xtr, ytr, U_pls, q_max, shrink_alpha=shrink_alpha
        )
        V_pls_raw = fit_pls["Vhat"]
        T_pls = fit_pls["That"]
    else:
        V_pls_raw = np.zeros((Xtr.shape[1], 0), dtype=float)
        T_pls = np.zeros((Xtr.shape[1], Xtr.shape[1]), dtype=float)

    V_dmd = orthogonalize_against(U_dmd, fit_dmd["Vhat"])
    V_pca = orthogonalize_against(U_pca, fit_pca["Vhat"])
    V_lda = orthogonalize_against(U_lda, fit_lda["Vhat"])
    V_pls = orthogonalize_against(U_pls, V_pls_raw)
    V_rand = orthogonalize_against(U_rand, fit_rand["Vhat"])

    methods = {
        "dmd": {
            "U": U_dmd,
            "V": V_dmd,
            "T": fit_dmd["That"],
        },
        "raw_cov": {
            "U": np.zeros((Xtr.shape[1], 0), dtype=float),
            "V": fit_dmd["Vhat_raw"],
            "T": fit_dmd["That_raw"],
        },
        "pca_first": {
            "U": U_pca,
            "V": V_pca,
            "T": fit_pca["That"],
        },
        "lda_first": {
            "U": U_lda,
            "V": V_lda,
            "T": fit_lda["That"],
        },
        "pls_first": {
            "U": U_pls,
            "V": V_pls,
            "T": T_pls,
        },
        "rand_first": {
            "U": U_rand,
            "V": V_rand,
            "T": fit_rand["That"],
        },
    }

    artifacts = {
        "r": int(r),
        "q_max": int(q_max),
        "U_dmd": U_dmd,
        "U_lda": U_lda,
        "U_pls": U_pls,
        "U_pca": U_pca,
        "fit_dmd": fit_dmd,
    }
    return methods, artifacts


# ============================================================
# Environment probes
# ============================================================

def fit_env_probe_linear(Xtr, etr, Xte, ete, seed=0):
    ev = base.fit_logreg_representation(Xtr, etr, Xte, ete, seed=seed)
    return {"acc": float(ev["acc"])}


def fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=0):
    ev = base.fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed)
    return {"acc": float(ev["acc"])}


def fit_env_probe_mlp(Xtr, etr, Xte, ete, seed=0, hidden_dim=64):
    etr = np.asarray(etr)
    ete = np.asarray(ete)
    env_classes = np.unique(etr)
    env_to_idx = {c: i for i, c in enumerate(env_classes)}

    if Xtr.shape[1] == 0 or len(env_classes) < 2:
        return {"acc": majority_baseline_accuracy(etr, ete)}

    etr_enc = np.array([env_to_idx[c] for c in etr], dtype=np.int64)
    ete_enc = np.array([env_to_idx[c] for c in ete], dtype=np.int64)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
        clf.fit(Xtr_s, etr_enc)

    pred = clf.predict(Xte_s)
    return {"acc": float(np.mean(pred == ete_enc))}


def fit_env_probe_rff(Xtr, etr, Xte, ete, seed=0, n_components=256):
    etr = np.asarray(etr)
    ete = np.asarray(ete)
    env_classes = np.unique(etr)
    env_to_idx = {c: i for i, c in enumerate(env_classes)}

    if Xtr.shape[1] == 0 or len(env_classes) < 2:
        return {"acc": majority_baseline_accuracy(etr, ete)}

    etr_enc = np.array([env_to_idx[c] for c in etr], dtype=np.int64)
    ete_enc = np.array([env_to_idx[c] for c in ete], dtype=np.int64)

    gamma = 1.0 / max(1, Xtr.shape[1])
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RBFSampler(
            gamma=gamma,
            n_components=int(n_components),
            random_state=seed,
        ),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )
    pipe.fit(Xtr, etr_enc)
    pred = pipe.predict(Xte)
    return {"acc": float(np.mean(pred == ete_enc))}


def eval_env_probe_family(Xtr, etr, Xte, ete, probes, seed=0):
    out = {}
    for p in probes:
        p = p.strip().lower()
        if p == "linear":
            out[p] = fit_env_probe_linear(Xtr, etr, Xte, ete, seed=seed + 1)["acc"]
        elif p == "quadratic":
            out[p] = fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 2)["acc"]
        elif p == "mlp":
            out[p] = fit_env_probe_mlp(Xtr, etr, Xte, ete, seed=seed + 3)["acc"]
        elif p == "rff":
            out[p] = fit_env_probe_rff(Xtr, etr, Xte, ete, seed=seed + 4)["acc"]
        else:
            raise ValueError(f"Unknown probe: {p}")
    out["max"] = float(max(out.values())) if len(out) > 0 else 0.0
    return out


# ============================================================
# Experiment 1: DMD-U / LDA whitening relation
# ============================================================

def experiment_stage1_equivalence(dataset_name, Xtr, ytr, etr, seed=0, num_avg=8):
    r, _ = base.choose_r_q(dataset_name, ytr, etr)
    q_dummy = min(8, Xtr.shape[1] - 1)

    _, artifacts = fit_stage2_methods(
        dataset_name, Xtr, ytr, etr,
        q_max=q_dummy,
        num_avg=num_avg,
        seed=seed,
    )

    U_dmd = artifacts["U_dmd"]
    U_lda = artifacts["U_lda"]
    U_pls = artifacts["U_pls"]

    M = base.empirical_M_plugin(Xtr, ytr)
    U_between, _ = base.top_eigenspace(M, r)

    Ctot = base.sample_cov(Xtr)
    I = np.eye(Ctot.shape[0])
    Sigma_B = M
    Sigma_W = base.sym(Ctot - Sigma_B)

    whiten_fro_error = float(np.linalg.norm(Ctot - I, ord="fro") / np.sqrt(Ctot.shape[0]))
    comm = Sigma_B @ Sigma_W - Sigma_W @ Sigma_B
    comm_norm = float(
        np.linalg.norm(comm, ord="fro") /
        (np.linalg.norm(Sigma_B, ord="fro") * np.linalg.norm(Sigma_W, ord="fro") + 1e-12)
    )

    return [
        {
            "experiment": "stage1_equivalence",
            "method": "dmd_vs_lda",
            "r": int(r),
            "metric_name": "subspace_overlap",
            "value": projector_overlap(U_dmd, U_lda),
        },
        {
            "experiment": "stage1_equivalence",
            "method": "dmd_vs_between_cov",
            "r": int(r),
            "metric_name": "subspace_overlap",
            "value": projector_overlap(U_dmd, U_between),
        },
        {
            "experiment": "stage1_equivalence",
            "method": "lda_vs_between_cov",
            "r": int(r),
            "metric_name": "subspace_overlap",
            "value": projector_overlap(U_lda, U_between),
        },
        {
            "experiment": "stage1_equivalence",
            "method": "pls_vs_dmd",
            "r": int(r),
            "metric_name": "subspace_overlap",
            "value": projector_overlap(U_pls, U_dmd),
        },
        {
            "experiment": "stage1_equivalence",
            "method": "whitening_check",
            "r": int(r),
            "metric_name": "total_cov_fro_error_from_identity",
            "value": whiten_fro_error,
        },
        {
            "experiment": "stage1_equivalence",
            "method": "lda_whitened_geometry",
            "r": int(r),
            "metric_name": "between_within_commutator_norm",
            "value": comm_norm,
        },
    ]


# ============================================================
# Experiment 2: safe edit and raw overlap
# ============================================================

def experiment_safe_edit_and_raw_overlap(
    dataset_name,
    Xtr, ytr, etr,
    Xte, yte, ete,
    q_grid,
    num_avg=8,
    seed=0,
):
    q_grid = sorted(set(int(q) for q in q_grid if int(q) > 0))
    q_max = min(max(q_grid), Xtr.shape[1] - 1)

    methods, artifacts = fit_stage2_methods(
        dataset_name, Xtr, ytr, etr,
        q_max=q_max,
        num_avg=num_avg,
        seed=seed,
    )
    U_ref = artifacts["U_dmd"]

    full_label = base.fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 10)
    full_env = base.fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 11)

    records = []
    method_list = ["dmd", "raw_cov", "pca_first", "lda_first", "pls_first", "rand_first"]

    for method in method_list:
        Vfull = methods[method]["V"]
        for q in q_grid:
            q_eff = min(q, Vfull.shape[1])
            Vq = Vfull[:, :q_eff]

            Xtr_edit = base.residualize_subspace(Xtr, Vq)
            Xte_edit = base.residualize_subspace(Xte, Vq)

            lab = base.fit_logreg_representation(
                Xtr_edit, ytr, Xte_edit, yte, seed=seed + 100 + q_eff
            )
            env = base.fit_env_probe_quadratic(
                Xtr_edit, etr, Xte_edit, ete, seed=seed + 200 + q_eff
            )

            records.append({
                "experiment": "safe_edit_raw_overlap",
                "method": method,
                "q": int(q_eff),
                "quotient_preservation_error_train": quotient_preservation_error(Xtr, U_ref, Vq),
                "quotient_preservation_error_test": quotient_preservation_error(Xte, U_ref, Vq),
                "removed_subspace_overlap_with_U": projector_overlap(Vq, U_ref),
                "label_acc_after": float(lab["acc"]),
                "label_nll_after": float(lab["nll"]),
                "label_change_vs_full": float(lab["acc"] - full_label["acc"]),
                "label_acc_drop_vs_full": float(full_label["acc"] - lab["acc"]),
                "label_nll_change_vs_full": float(lab["nll"] - full_label["nll"]),
                "env_acc_after_quadratic": float(env["acc"]),
                "env_leak_reduction_vs_full_quadratic": float(full_env["acc"] - env["acc"]),
            })

    raw_df = pd.DataFrame([r for r in records if r["method"] == "raw_cov"])
    if len(raw_df) > 1:
        records.append({
            "experiment": "safe_edit_raw_overlap_summary",
            "method": "raw_cov",
            "q": -1,
            "metric_name": "corr_overlap_with_U_vs_label_acc_drop",
            "value": pearson_corr(
                raw_df["removed_subspace_overlap_with_U"].values,
                raw_df["label_acc_drop_vs_full"].values,
            ),
        })
        records.append({
            "experiment": "safe_edit_raw_overlap_summary",
            "method": "raw_cov",
            "q": -1,
            "metric_name": "corr_overlap_with_U_vs_nll_deterioration",
            "value": pearson_corr(
                raw_df["removed_subspace_overlap_with_U"].values,
                raw_df["label_nll_change_vs_full"].values,
            ),
        })

    return records


# ============================================================
# Experiment 3: quotient-visible vs residual leakage
# ============================================================

def experiment_quotient_leakage_decomposition(
    dataset_name,
    Xtr, ytr, etr,
    Xte, yte, ete,
    q_eval=8,
    probes=("linear", "quadratic", "mlp"),
    num_avg=8,
    seed=0,
):
    q_eval = min(int(q_eval), Xtr.shape[1] - 1)
    methods, artifacts = fit_stage2_methods(
        dataset_name, Xtr, ytr, etr,
        q_max=q_eval,
        num_avg=num_avg,
        seed=seed,
    )

    U = artifacts["U_dmd"]
    V_dmd = methods["dmd"]["V"][:, :q_eval]
    V_raw = methods["raw_cov"]["V"][:, :q_eval]

    reps = {
        "full_X": (Xtr, Xte),
        "quotient_UtX": (Xtr @ U, Xte @ U),
        "residual_PUperpX": (
            base.residualize_subspace(Xtr, U),
            base.residualize_subspace(Xte, U),
        ),
        "dmd_edited": (
            base.residualize_subspace(Xtr, V_dmd),
            base.residualize_subspace(Xte, V_dmd),
        ),
        "raw_edited": (
            base.residualize_subspace(Xtr, V_raw),
            base.residualize_subspace(Xte, V_raw),
        ),
    }

    records = []
    probe_acc_by_rep = {}

    for rep_name, (Xtr_rep, Xte_rep) in reps.items():
        lab = base.fit_logreg_representation(
            Xtr_rep, ytr, Xte_rep, yte, seed=seed + 100
        )
        envs = eval_env_probe_family(
            Xtr_rep, etr, Xte_rep, ete,
            probes=probes,
            seed=seed + 200,
        )
        probe_acc_by_rep[rep_name] = envs

        row = {
            "experiment": "quotient_leakage_decomposition",
            "representation": rep_name,
            "method": rep_name,
            "q": int(q_eval),
            "label_acc": float(lab["acc"]),
            "label_nll": float(lab["nll"]),
        }
        for p, v in envs.items():
            row[f"env_acc_{p}"] = float(v)
        records.append(row)

    base_acc = majority_baseline_accuracy(etr, ete)

    for p in list(probes) + ["max"]:
        full = probe_acc_by_rep["full_X"][p]
        denom = max(full - base_acc, 1e-12)
        qacc = probe_acc_by_rep["quotient_UtX"][p]
        racc = probe_acc_by_rep["residual_PUperpX"][p]

        # records.append({
        #     "experiment": "quotient_leakage_fraction",
        #     "representation": "summary",
        #     "method": "quotient_fraction",
        #     "probe": p,
        #     "q": int(q_eval),
        #     "majority_baseline_env_acc": float(base_acc),
        #     "full_env_acc": float(full),
        #     "quotient_env_acc": float(qacc),
        #     "residual_env_acc": float(racc),
        #     "quotient_visible_leakage_fraction": float(max(0.0, (qacc - base_acc) / denom)),
        #     "residual_leakage_fraction": float(max(0.0, (racc - base_acc) / denom)),
        # })

        q_ratio = float((qacc - base_acc) / denom)
        r_ratio = float((racc - base_acc) / denom)

        records.append({
            "experiment": "quotient_leakage_fraction",
            "representation": "summary",
            "method": "quotient_fraction",
            "probe": p,
            "q": int(q_eval),
            "majority_baseline_env_acc": float(base_acc),
            "full_env_acc": float(full),
            "quotient_env_acc": float(qacc),
            "residual_env_acc": float(racc),

            # Ratios can exceed 1 when a reduced representation is easier
            # for the selected auditor than the full representation, or when
            # full leakage is close to the baseline.
            "quotient_visible_leakage_ratio": q_ratio,
            "residual_leakage_ratio": r_ratio,

            # Capped versions are easier to report as "fractions".
            "quotient_visible_leakage_fraction_capped": float(np.clip(q_ratio, 0.0, 1.0)),
            "residual_leakage_fraction_capped": float(np.clip(r_ratio, 0.0, 1.0)),

            # Backward-compatible names; interpret as ratios, not bounded fractions.
            "quotient_visible_leakage_fraction": float(max(0.0, q_ratio)),
            "residual_leakage_fraction": float(max(0.0, r_ratio)),
        })

    return records


# ============================================================
# Experiment 4: q-frontier with label-loss budgets
# ============================================================

def experiment_frontier_budgeted(
    dataset_name,
    Xtr, ytr, etr,
    Xte, yte, ete,
    q_grid,
    acc_tols=(0.005, 0.01),
    nll_tols=(0.01, 0.02),
    num_avg=8,
    seed=0,
):
    q_grid = sorted(set(int(q) for q in q_grid if int(q) > 0))
    q_max = min(max(q_grid), Xtr.shape[1] - 1)

    methods, _ = fit_stage2_methods(
        dataset_name, Xtr, ytr, etr,
        q_max=q_max,
        num_avg=num_avg,
        seed=seed,
    )

    full_label = base.fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 10)
    full_env = base.fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 11)

    records = []

    for method, mf in methods.items():
        Vfull = mf["V"]
        for q in q_grid:
            q_eff = min(q, Vfull.shape[1])
            Vq = Vfull[:, :q_eff]

            Xtr_edit = base.residualize_subspace(Xtr, Vq)
            Xte_edit = base.residualize_subspace(Xte, Vq)

            lab = base.fit_logreg_representation(
                Xtr_edit, ytr, Xte_edit, yte, seed=seed + 100 + q_eff
            )
            env = base.fit_env_probe_quadratic(
                Xtr_edit, etr, Xte_edit, ete, seed=seed + 200 + q_eff
            )

            records.append({
                "experiment": "frontier_budget_curve",
                "method": method,
                "q": int(q_eff),
                "env_leak_reduction_vs_full": float(full_env["acc"] - env["acc"]),
                "label_change_vs_full": float(lab["acc"] - full_label["acc"]),
                "label_acc_drop_vs_full": float(full_label["acc"] - lab["acc"]),
                "label_nll_change_vs_full": float(lab["nll"] - full_label["nll"]),
                "label_acc_after": float(lab["acc"]),
                "label_nll_after": float(lab["nll"]),
            })

    df = pd.DataFrame(records)

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]

        for tol in acc_tols:
            eligible = sub[sub["label_acc_drop_vs_full"] <= tol]
            if len(eligible) == 0:
                best = sub.iloc[sub["env_leak_reduction_vs_full"].argmax()]
                feasible = 0
            else:
                best = eligible.iloc[eligible["env_leak_reduction_vs_full"].argmax()]
                feasible = 1

            records.append({
                "experiment": "frontier_budget_summary",
                "method": method,
                "budget_type": "acc_drop",
                "budget_tol": float(tol),
                "feasible": int(feasible),
                "best_q": int(best["q"]),
                "best_env_leak_reduction": float(best["env_leak_reduction_vs_full"]),
                "best_label_change": float(best["label_change_vs_full"]),
                "best_label_acc_drop": float(best["label_acc_drop_vs_full"]),
                "best_nll_change": float(best["label_nll_change_vs_full"]),
            })

        for tol in nll_tols:
            eligible = sub[sub["label_nll_change_vs_full"] <= tol]
            if len(eligible) == 0:
                best = sub.iloc[sub["env_leak_reduction_vs_full"].argmax()]
                feasible = 0
            else:
                best = eligible.iloc[eligible["env_leak_reduction_vs_full"].argmax()]
                feasible = 1

            records.append({
                "experiment": "frontier_budget_summary",
                "method": method,
                "budget_type": "nll_increase",
                "budget_tol": float(tol),
                "feasible": int(feasible),
                "best_q": int(best["q"]),
                "best_env_leak_reduction": float(best["env_leak_reduction_vs_full"]),
                "best_label_change": float(best["label_change_vs_full"]),
                "best_label_acc_drop": float(best["label_acc_drop_vs_full"]),
                "best_nll_change": float(best["label_nll_change_vs_full"]),
            })

    return records


# ============================================================
# Experiment 5: alternative auditor probes
# ============================================================

def experiment_alternative_probes(
    dataset_name,
    Xtr, ytr, etr,
    Xte, yte, ete,
    q_eval=8,
    probes=("linear", "quadratic", "mlp"),
    num_avg=8,
    seed=0,
):
    q_eval = min(int(q_eval), Xtr.shape[1] - 1)
    methods, _ = fit_stage2_methods(
        dataset_name, Xtr, ytr, etr,
        q_max=q_eval,
        num_avg=num_avg,
        seed=seed,
    )

    full_label = base.fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 10)
    full_envs = eval_env_probe_family(Xtr, etr, Xte, ete, probes=probes, seed=seed + 20)

    records = []

    full_row = {
        "experiment": "alternative_probes",
        "method": "full",
        "q": 0,
        "label_acc_after": float(full_label["acc"]),
        "label_nll_after": float(full_label["nll"]),
        "label_change_vs_full": 0.0,
        "label_nll_change_vs_full": 0.0,
    }
    for p, v in full_envs.items():
        full_row[f"env_acc_{p}"] = float(v)
        full_row[f"env_leak_reduction_{p}"] = 0.0
    records.append(full_row)

    for method in ["dmd", "raw_cov", "pca_first", "lda_first", "pls_first", "rand_first"]:
        Vq = methods[method]["V"][:, :q_eval]
        Xtr_edit = base.residualize_subspace(Xtr, Vq)
        Xte_edit = base.residualize_subspace(Xte, Vq)

        lab = base.fit_logreg_representation(
            Xtr_edit, ytr, Xte_edit, yte, seed=seed + 100
        )
        envs = eval_env_probe_family(
            Xtr_edit, etr, Xte_edit, ete,
            probes=probes,
            seed=seed + 200,
        )

        row = {
            "experiment": "alternative_probes",
            "method": method,
            "q": int(q_eval),
            "label_acc_after": float(lab["acc"]),
            "label_nll_after": float(lab["nll"]),
            "label_change_vs_full": float(lab["acc"] - full_label["acc"]),
            "label_nll_change_vs_full": float(lab["nll"] - full_label["nll"]),
        }
        for p, v in envs.items():
            row[f"env_acc_{p}"] = float(v)
            row[f"env_leak_reduction_{p}"] = float(full_envs[p] - v)
        records.append(row)

    return records


# ============================================================
# Experiment 6: r / alpha sensitivity
# ============================================================

def default_r_grid(dataset_name, ytr):
    dataset_name = dataset_name.lower()
    if dataset_name in ["celeba", "waterbirds"]:
        return [1, 2, 4]
    if dataset_name in ["pacs", "domainnet"]:
        K = base.unique_count(ytr)
        return sorted(set([2, 4, min(6, K - 1)]))
    K = base.unique_count(ytr)
    return sorted(set([1, min(2, K - 1), min(4, K - 1)]))


def experiment_r_alpha_sensitivity(
    dataset_name,
    Xtr, ytr, etr,
    Xte, yte, ete,
    r_grid=None,
    alpha_grid=(0.0, 0.05, 0.10, 0.20, 0.50),
    q_eval=8,
    num_avg=4,
    seed=0,
):
    if r_grid is None:
        r_grid = default_r_grid(dataset_name, ytr)

    q_eval = min(int(q_eval), Xtr.shape[1] - 1)

    full_label = base.fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 10)
    full_env = base.fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 11)

    records = []

    for r in r_grid:
        for alpha in alpha_grid:
            methods, _ = fit_stage2_methods(
                dataset_name,
                Xtr, ytr, etr,
                q_max=q_eval,
                num_avg=num_avg,
                seed=seed + int(1000 * alpha) + 17 * int(r),
                shrink_alpha=float(alpha),
                r_override=int(r),
            )

            for method in ["dmd", "raw_cov"]:
                Vq = methods[method]["V"][:, :q_eval]
                Xtr_edit = base.residualize_subspace(Xtr, Vq)
                Xte_edit = base.residualize_subspace(Xte, Vq)

                lab = base.fit_logreg_representation(
                    Xtr_edit, ytr, Xte_edit, yte,
                    seed=seed + 100 + int(r)
                )
                env = base.fit_env_probe_quadratic(
                    Xtr_edit, etr, Xte_edit, ete,
                    seed=seed + 200 + int(r)
                )

                records.append({
                    "experiment": "r_alpha_sensitivity",
                    "method": method,
                    "r": int(r),
                    "alpha": float(alpha),
                    "q": int(q_eval),
                    "env_leak_reduction_vs_full": float(full_env["acc"] - env["acc"]),
                    "label_change_vs_full": float(lab["acc"] - full_label["acc"]),
                    "label_acc_drop_vs_full": float(full_label["acc"] - lab["acc"]),
                    "label_nll_change_vs_full": float(lab["nll"] - full_label["nll"]),
                })

    return records


# ============================================================
# Combo runner
# ============================================================

def run_combo(args, dataset_name, backbone_name, seed, out_dir):
    print(f"[combo] start dataset={dataset_name} backbone={backbone_name} seed={seed}", flush=True)

    Xtr, ytr, etr, Xte, yte, ete, meta = load_preprocessed(
        data_root=args.data_root,
        dataset_name=dataset_name,
        backbone_name=backbone_name,
        pretrained=args.pretrained,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=seed,
    )

    q_grid = parse_csv_list(args.q_grid, int)
    probes = tuple(parse_csv_list(args.probes, str))
    alpha_grid = parse_csv_list(args.alpha_grid, float)

    records = []

    if args.do_stage1_equivalence:
        records.extend(experiment_stage1_equivalence(
            dataset_name, Xtr, ytr, etr,
            seed=seed + 10,
            num_avg=args.num_avg,
        ))

    if args.do_safe_overlap:
        records.extend(experiment_safe_edit_and_raw_overlap(
            dataset_name,
            Xtr, ytr, etr,
            Xte, yte, ete,
            q_grid=q_grid,
            num_avg=args.num_avg,
            seed=seed + 20,
        ))

    if args.do_quotient_leakage:
        records.extend(experiment_quotient_leakage_decomposition(
            dataset_name,
            Xtr, ytr, etr,
            Xte, yte, ete,
            q_eval=args.q_eval,
            probes=probes,
            num_avg=args.num_avg,
            seed=seed + 30,
        ))

    if args.do_frontier_budget:
        records.extend(experiment_frontier_budgeted(
            dataset_name,
            Xtr, ytr, etr,
            Xte, yte, ete,
            q_grid=q_grid,
            acc_tols=tuple(parse_csv_list(args.acc_tols, float)),
            nll_tols=tuple(parse_csv_list(args.nll_tols, float)),
            num_avg=args.num_avg,
            seed=seed + 40,
        ))

    if args.do_alt_probes:
        records.extend(experiment_alternative_probes(
            dataset_name,
            Xtr, ytr, etr,
            Xte, yte, ete,
            q_eval=args.q_eval,
            probes=probes,
            num_avg=args.num_avg,
            seed=seed + 50,
        ))

    if args.do_sensitivity:
        r_grid = None if args.r_grid.lower() == "auto" else parse_csv_list(args.r_grid, int)
        records.extend(experiment_r_alpha_sensitivity(
            dataset_name,
            Xtr, ytr, etr,
            Xte, yte, ete,
            r_grid=r_grid,
            alpha_grid=alpha_grid,
            q_eval=args.q_eval,
            num_avg=args.sensitivity_num_avg,
            seed=seed + 60,
        ))

    for r in records:
        r["dataset"] = dataset_name
        r["backbone"] = backbone_name
        r["seed"] = int(seed)
        r["pretrained"] = int(args.pretrained)
        for k, v in meta.items():
            if k in ["num_train", "num_test", "num_classes", "num_envs", "feature_dim", "pca_dim"]:
                r[f"meta_{k}"] = v

    combo_dir = os.path.join(out_dir, f"{dataset_name}__{backbone_name}__seed{seed}")
    ensure_dir(combo_dir)
    save_records_csv(os.path.join(combo_dir, "additional_records.csv"), records)
    save_json(os.path.join(combo_dir, "meta.json"), meta)

    print(f"[combo] done dataset={dataset_name} backbone={backbone_name} seed={seed} rows={len(records)}", flush=True)
    return records


def aggregate_records(records, out_dir):
    ensure_dir(out_dir)

    df0 = pd.DataFrame(records)
    if "metric" in df0.columns:
        df0 = df0.rename(columns={"metric": "record_metric"})
    records = df0.to_dict(orient="records")

    all_csv = os.path.join(out_dir, "all_additional_records.csv")
    df = save_records_csv(all_csv, records)

    if len(df) == 0:
        return {"all_csv": all_csv}

    id_cols_candidate = [
        "experiment", "dataset", "backbone", "method", "representation",
        "probe", "metric_name", "record_metric", "budget_type", "budget_tol",
        "q", "r", "alpha", "pretrained",
    ]
    id_cols = [c for c in id_cols_candidate if c in df.columns]

    numeric_cols = []
    for c in df.columns:
        if c in id_cols or c == "seed":
            continue
        if c.startswith("meta_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    long_rows = []
    for _, row in df.iterrows():
        base_id = {}
        for c in id_cols:
            if c in row and pd.notna(row[c]):
                base_id[c] = row[c]
            else:
                base_id[c] = ""
        for metric in numeric_cols:
            val = row.get(metric, np.nan)
            if pd.isna(val):
                continue
            long_rows.append({
                **base_id,
                "seed": int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else -1,
                "metric": metric,
                "value": float(val),
            })

    long_df = pd.DataFrame(long_rows)
    long_csv = os.path.join(out_dir, "all_additional_records_long.csv")
    long_df.to_csv(long_csv, index=False)

    group_cols = [c for c in id_cols if c in long_df.columns] + ["metric"]
    agg = long_df.groupby(group_cols, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg["mean_pm_std"] = agg.apply(
        lambda r: f"{round(float(r['mean']), 5)} ± {round(float(r['std']), 5)}",
        axis=1,
    )

    agg_csv = os.path.join(out_dir, "additional_mean_std.csv")
    agg.to_csv(agg_csv, index=False)

    safe = df[df["experiment"] == "safe_edit_raw_overlap"].copy()
    global_corrs = []
    if len(safe) > 0:
        raw = safe[safe["method"] == "raw_cov"]
        if len(raw) > 1:
            global_corrs.append({
                "experiment": "global_raw_overlap_correlation",
                "metric_name": "corr_overlap_with_U_vs_label_acc_drop",
                "value": pearson_corr(
                    raw["removed_subspace_overlap_with_U"].values,
                    raw["label_acc_drop_vs_full"].values,
                ),
            })
            global_corrs.append({
                "experiment": "global_raw_overlap_correlation",
                "metric_name": "corr_overlap_with_U_vs_nll_deterioration",
                "value": pearson_corr(
                    raw["removed_subspace_overlap_with_U"].values,
                    raw["label_nll_change_vs_full"].values,
                ),
            })

    global_corr_csv = os.path.join(out_dir, "global_raw_overlap_correlations.csv")
    save_records_csv(global_corr_csv, global_corrs)

    return {
        "all_csv": all_csv,
        "long_csv": long_csv,
        "agg_csv": agg_csv,
        "global_corr_csv": global_corr_csv,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--datasets", type=str, default="celeba,waterbirds,pacs")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=1)

    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--fast", type=int, default=0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_list", type=str, default="42,123,3407,2024,2025,7,17,29,314,2718")

    parser.add_argument("--num_avg", type=int, default=8)
    parser.add_argument("--sensitivity_num_avg", type=int, default=4)

    parser.add_argument("--q_grid", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--q_eval", type=int, default=8)

    parser.add_argument("--probes", type=str, default="linear,quadratic,mlp")
    parser.add_argument("--acc_tols", type=str, default="0.005,0.01")
    parser.add_argument("--nll_tols", type=str, default="0.01,0.02")

    parser.add_argument("--r_grid", type=str, default="auto")
    parser.add_argument("--alpha_grid", type=str, default="0,0.05,0.10,0.20,0.50")

    parser.add_argument("--do_stage1_equivalence", type=int, default=1)
    parser.add_argument("--do_safe_overlap", type=int, default=1)
    parser.add_argument("--do_quotient_leakage", type=int, default=1)
    parser.add_argument("--do_frontier_budget", type=int, default=1)
    parser.add_argument("--do_alt_probes", type=int, default=1)
    parser.add_argument("--do_sensitivity", type=int, default=1)

    parser.add_argument("--outdir", type=str, default="runs_labelonly_additional")

    args = parser.parse_args()

    if bool(args.fast):
        if args.max_train <= 0:
            args.max_train = 4000
        if args.max_test <= 0:
            args.max_test = 2000
        args.num_avg = min(args.num_avg, 4)
        args.sensitivity_num_avg = min(args.sensitivity_num_avg, 3)
        args.q_grid = "1,2,4,8"
        args.probes = "linear,quadratic"

    seeds = parse_csv_list(args.seed_list, int)
    if len(seeds) == 0:
        seeds = [int(args.seed)]

    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    backbones = [x.strip() for x in args.backbone.split(",") if x.strip()]

    root = os.path.join(args.outdir, f"additional_{now_tag()}")
    ensure_dir(root)

    print("=====================================================", flush=True)
    print("Running label-only additional experiments", flush=True)
    print(f"root={root}", flush=True)
    print(f"datasets={datasets}", flush=True)
    print(f"backbones={backbones}", flush=True)
    print(f"seeds={seeds}", flush=True)
    print(f"probes={args.probes}", flush=True)
    print(f"q_grid={args.q_grid}", flush=True)
    print("=====================================================", flush=True)

    all_records = []
    manifests = []

    for seed in seeds:
        set_seed(seed)
        for backbone in backbones:
            for dataset in datasets:
                try:
                    recs = run_combo(args, dataset, backbone, seed, root)
                    all_records.extend(recs)
                    manifests.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "seed": seed,
                        "status": "ok",
                        "num_records": len(recs),
                    })
                except Exception as ex:
                    print(f"[error] dataset={dataset} backbone={backbone} seed={seed}: {ex}", flush=True)
                    manifests.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "seed": seed,
                        "status": "error",
                        "error": str(ex),
                    })

    paths = aggregate_records(all_records, root)

    manifest = {
        "root": root,
        "args": vars(args),
        "seeds": seeds,
        "datasets": datasets,
        "backbones": backbones,
        "manifests": manifests,
        "aggregate_paths": paths,
    }
    save_json(os.path.join(root, "additional_manifest.json"), manifest)
    save_records_csv(os.path.join(root, "additional_manifest.csv"), manifests)

    print("=====================================================", flush=True)
    print("Finished additional experiments.", flush=True)
    print(f"All records: {paths.get('all_csv', '')}", flush=True)
    print(f"Mean/std:    {paths.get('agg_csv', '')}", flush=True)
    print(f"Manifest:    {os.path.join(root, 'additional_manifest.json')}", flush=True)
    print("=====================================================", flush=True)


if __name__ == "__main__":
    main()