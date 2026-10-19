  
import os    
import json    
import glob    
import time    
import argparse    
import random  
from datetime import datetime    
from collections import defaultdict    

  
import numpy as np  
import pandas as pd  
  
from sklearn.preprocessing import StandardScaler, normalize  
from sklearn.decomposition import PCA  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn.linear_model import LogisticRegression  
from sklearn.cross_decomposition import PLSRegression  
from sklearn.cluster import KMeans  
  
try:  
    import matplotlib.pyplot as plt  
    HAS_MPL = True  
except Exception:  
    HAS_MPL = False  
  
  
# ============================================================  
# Logger  
# ============================================================  
  
class SimpleLogger:  
    def __init__(self, root_dir="runs_labelonly_workflow_full", tag="labelonly_workflow_full"):  
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  
        self.run_dir = os.path.join(root_dir, f"{tag}_{ts}")  
        os.makedirs(self.run_dir, exist_ok=True)  
  
        self.jsonl_path = os.path.join(self.run_dir, "events.jsonl")  
        self.log_path = os.path.join(self.run_dir, "run.log")  
        self.final_report_path = os.path.join(self.run_dir, "final_report.json")  
        self.final_report_csv_path = os.path.join(self.run_dir, "final_report.csv")  
  
        self.f_jsonl = open(self.jsonl_path, "a", buffering=1, encoding="utf-8")  
        self.f_log = open(self.log_path, "a", buffering=1, encoding="utf-8")  
  
    def _jsonable(self, obj):  
        if isinstance(obj, dict):  
            return {str(k): self._jsonable(v) for k, v in obj.items()}  
        elif isinstance(obj, (list, tuple)):  
            return [self._jsonable(v) for v in obj]  
        elif isinstance(obj, np.ndarray):  
            return obj.tolist()  
        elif isinstance(obj, (np.floating,)):  
            return float(obj)  
        elif isinstance(obj, (np.integer,)):  
            return int(obj)  
        elif isinstance(obj, (np.bool_,)):  
            return bool(obj)  
        elif obj is None:  
            return None  
        else:  
            return obj  
  
    def log(self, rec):  
        rec = dict(rec)  
        rec["time"] = int(time.time())  
        rec = self._jsonable(rec)  
        self.f_jsonl.write(json.dumps(rec, sort_keys=True) + "\n")  
        self.f_jsonl.flush()  
  
    def print(self, msg):  
        line = str(msg)  
        print(line, flush=True)  
        self.f_log.write(line + "\n")  
        self.f_log.flush()  
        self.log({"event": "print", "msg": line})  
  
    def save_report(self, report):  
        report = round_recursive(self._jsonable(report), ndigits=4)  
        with open(self.final_report_path, "w", encoding="utf-8") as f:  
            json.dump(report, f, indent=2, sort_keys=True)  
  
    def close(self):  
        try:  
            self.f_jsonl.close()  
        except Exception:  
            pass  
        try:  
            self.f_log.close()  
        except Exception:  
            pass  
  
  
# ============================================================  
# IO / round helpers  
# ============================================================  
  
def ensure_dir(path):  
    os.makedirs(path, exist_ok=True)  
  
  
def parse_csv_list(s, cast=str):  
    if s is None:  
        return []  
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()]  
  
  
def round_recursive(obj, ndigits=4):  
    if isinstance(obj, dict):  
        return {k: round_recursive(v, ndigits) for k, v in obj.items()}  
    if isinstance(obj, list):  
        return [round_recursive(v, ndigits) for v in obj]  
    if isinstance(obj, tuple):  
        return [round_recursive(v, ndigits) for v in obj]  
    if isinstance(obj, np.ndarray):  
        return round_recursive(obj.tolist(), ndigits)  
    if isinstance(obj, (np.floating, float)):  
        return round(float(obj), ndigits)  
    if isinstance(obj, (np.integer, int)):  
        return int(obj)  
    if isinstance(obj, (np.bool_, bool)):  
        return bool(obj)  
    return obj  

  
  
def save_json(path, obj, ndigits=4):  
    with open(path, "w", encoding="utf-8") as f:  
        json.dump(round_recursive(obj, ndigits=ndigits), f, indent=2, sort_keys=True)  
  
  
def save_csv_from_records(path, records, ndigits=4):  
    if records is None:  
        return  
    df = pd.DataFrame(records)  
    if len(df) == 0:  
        df.to_csv(path, index=False)  
        return  
    for c in df.columns:  
        if pd.api.types.is_numeric_dtype(df[c]):  
            df[c] = df[c].round(ndigits)  
    df.to_csv(path, index=False)  
  
  
def save_npz_safe(path, **kwargs):  
    clean = {}  
    for k, v in kwargs.items():  
        if isinstance(v, np.ndarray):  
            clean[k] = v  
        elif np.isscalar(v):  
            clean[k] = np.asarray(v)  
    np.savez_compressed(path, **clean)  
  
  
def plot_if_available():  
    return HAS_MPL  
  
  
# ============================================================  
# Basic utils  
# ============================================================  
  
def set_seed(seed):    
    seed = int(seed)  
    np.random.seed(seed)    
    random.seed(seed)  
  
  
def sym(A):  
    return 0.5 * (A + A.T)  
  
  
def mean_std(arr):  
    arr = np.asarray(arr, dtype=float)  
    if arr.size == 0:  
        return 0.0, 0.0  
    return float(np.mean(arr)), float(np.std(arr))  
  
  
def pearson_corr(x, y):  
    x = np.asarray(x, dtype=float)  
    y = np.asarray(y, dtype=float)  
    if len(x) < 2 or len(y) < 2:  
        return 0.0  
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:  
        return 0.0  
    return float(np.corrcoef(x, y)[0, 1])  
  
  
def top_bottom_gap(score, target):  
    score = np.asarray(score, dtype=float)  
    target = np.asarray(target, dtype=float)  
    if len(score) < 2:  
        return 0.0  
    order = np.argsort(score)  
    m = max(1, len(score) // 2)  
    low = order[:m]  
    high = order[-m:]  
    return float(np.mean(target[high]) - np.mean(target[low]))  
  
  
def tercile_means(score, target):  
    score = np.asarray(score, dtype=float)  
    target = np.asarray(target, dtype=float)  
    if len(score) < 3:  
        return 0.0, 0.0, 0.0  
    q1 = np.quantile(score, 1.0 / 3.0)  
    q2 = np.quantile(score, 2.0 / 3.0)  
    low = target[score <= q1]  
    mid = target[(score > q1) & (score < q2)]  
    high = target[score >= q2]  
    return (  
        float(np.mean(low)) if low.size > 0 else 0.0,  
        float(np.mean(mid)) if mid.size > 0 else 0.0,  
        float(np.mean(high)) if high.size > 0 else 0.0,  
    )  
  
  
def unique_count(arr):  
    return len(np.unique(arr))  
  

def remap_labels_to_contiguous(y):  
    y = np.asarray(y)  
    classes = np.unique(y)  
    cls_to_idx = {c: i for i, c in enumerate(classes)}  
    y_new = np.array([cls_to_idx[v] for v in y], dtype=np.int64)  
    return y_new, classes  


def sample_cov(X):  
    if X.shape[0] <= 1:  
        return np.zeros((X.shape[1], X.shape[1]), dtype=float)  
    Xc = X - X.mean(axis=0, keepdims=True)  
    return (Xc.T @ Xc) / float(X.shape[0])  
  
  
def shrink_cov(C, alpha=0.10):  
    p = C.shape[0]  
    if p == 0:  
        return C  
    tau = np.trace(C) / max(1, p)  
    return (1.0 - alpha) * C + alpha * tau * np.eye(p)  
  
  
def top_eigenspace(A, k):  
    A = sym(A)  
    p = A.shape[0]  
    k = int(k)  
    if k <= 0:  
        return np.zeros((p, 0), dtype=float), np.zeros((0,), dtype=float)  
    vals, vecs = np.linalg.eigh(A)  
    idx = np.argsort(vals)[::-1][:k]  
    return vecs[:, idx], vals[idx]  
  
  
def top_eigenvalues(A, max_rank=None):  
    vals = np.linalg.eigvalsh(sym(A))  
    vals = np.sort(vals)[::-1]  
    if max_rank is not None:  
        vals = vals[:max_rank]  
    return vals.astype(float)  
  
  
def smallest_signal_eig(A, k):  
    k = int(k)  
    if k <= 0:  
        return 0.0  
    vals = np.linalg.eigvalsh(sym(A))  
    vals = np.sort(vals)  
    k = min(k, len(vals))  
    return float(vals[-k])  
  
  
def subspace_distance(Uhat, U):  
    if Uhat.shape[1] == 0 and U.shape[1] == 0:  
        return 0.0  
    if Uhat.shape[1] != U.shape[1]:  
        raise ValueError("Subspaces must have same dimension")  
    s = np.linalg.svd(Uhat.T @ U, compute_uv=False)  
    s = np.clip(s, 0.0, 1.0)  
    smin = float(np.min(s))  
    return float(np.sqrt(max(0.0, 1.0 - smin * smin)))  
  
  
def subspace_overlap(Uhat, U):  
    if Uhat.shape[1] == 0 and U.shape[1] == 0:  
        return 1.0  
    if Uhat.shape[1] == 0 or U.shape[1] == 0:  
        return 0.0  
    q = min(Uhat.shape[1], U.shape[1])  
    A = Uhat[:, :q].T @ U[:, :q]  
    return float(np.linalg.norm(A, ord="fro") ** 2 / max(1, q))  
  
  
def mean_pairwise_subspace_distance(subspaces):  
    vals = []  
    for i in range(len(subspaces)):  
        for j in range(i + 1, len(subspaces)):  
            vals.append(subspace_distance(subspaces[i], subspaces[j]))  
    return mean_std(vals)  
  
  
def mean_pairwise_subspace_overlap(subspaces):  
    vals = []  
    for i in range(len(subspaces)):  
        for j in range(i + 1, len(subspaces)):  
            vals.append(subspace_overlap(subspaces[i], subspaces[j]))  
    return mean_std(vals)  
  
  
def aggregate_subspaces(subspace_list, k):  
    p = subspace_list[0].shape[0]  
    if k <= 0:  
        return np.zeros((p, 0), dtype=float)  
    P = np.zeros((p, p), dtype=float)  
    for U in subspace_list:  
        P += U @ U.T  
    P /= float(len(subspace_list))  
    Uagg, _ = top_eigenspace(P, k)  
    return Uagg  
  
  
def random_projection_matrix(p, k, seed=0):  
    k = int(k)  
    if k <= 0:  
        return np.zeros((p, 0), dtype=float)  
    rng = np.random.default_rng(seed)  
    M = rng.normal(size=(p, k))  
    Q, _ = np.linalg.qr(M)  
    return Q[:, :k]  
  
  
def project_subspace(X, U):  
    return X @ U  
  
  
def residualize_subspace(X, U):  
    if U.shape[1] == 0:  
        return X.copy()  
    return X - (X @ U) @ U.T  
  
  
def project_cov(C, U):  
    if U.shape[1] == 0:  
        return C.copy()  
    P = np.eye(C.shape[0]) - U @ U.T  
    return P @ C @ P  
  
  
# ============================================================  
# Feature cache loading  
# ============================================================  
  
def resolve_feature_cache(data_root, dataset_name, split, backbone_name, pretrained_flag):  
    cache_dir = os.path.join(data_root, "feature_cache")  
    if not os.path.isdir(cache_dir):  
        raise FileNotFoundError(f"Missing feature cache dir: {cache_dir}")  
  
    pt_tag = "pt1" if bool(pretrained_flag) else "pt0"  
  
    exact_candidates = [  
        f"{dataset_name}_{split}_{backbone_name}_{pt_tag}.npz",  
        f"{dataset_name}{split}{backbone_name}{pt_tag}.npz",  
        f"{dataset_name}_{split}_{backbone_name}{pt_tag}.npz",  
        f"{dataset_name}{split}_{backbone_name}_{pt_tag}.npz",  
    ]  
    for fn in exact_candidates:  
        p = os.path.join(cache_dir, fn)  
        if os.path.exists(p):  
            return p  
  
    # strict pattern: backbone name must appear  
    patterns = [  
        f"*{dataset_name}*{split}*{backbone_name}*{pt_tag}*.npz",  
    ]  
    matches = []  
    for pat in patterns:  
        matches.extend(glob.glob(os.path.join(cache_dir, pat)))  
    matches = sorted(set(matches))  
  
    if len(matches) == 1:  
        return matches[0]  
    elif len(matches) > 1:  
        raise RuntimeError(  
            f"Multiple caches matched strictly for dataset={dataset_name}, split={split}, "  
            f"backbone={backbone_name}, pretrained={int(pretrained_flag)}:\n" +  
            "\n".join(matches)  
        )  
    else:  
        raise FileNotFoundError(  
            f"Could not find STRICT cache for dataset={dataset_name}, split={split}, "  
            f"backbone={backbone_name}, pretrained={int(pretrained_flag)} "  
            f"under {cache_dir}."  
        )  

  
  
def load_cached_features(data_root, dataset_name, split, backbone_name, pretrained_flag):  
    path = resolve_feature_cache(  
        data_root=data_root,  
        dataset_name=dataset_name,  
        split=split,  
        backbone_name=backbone_name,  
        pretrained_flag=pretrained_flag,  
    )  
    z = np.load(path)  
    X = z["X"].astype(np.float32)  
    y = z["y"].astype(np.int64)  
    e = z["e"].astype(np.int64)  
    return X, y, e, path  
  
  
# ============================================================  
# Preprocess  
# ============================================================  
  
def choose_pca_dim(dataset_name):  
    dataset_name = dataset_name.lower()  
    if dataset_name == "celeba":  
        return 128  
    if dataset_name == "waterbirds":  
        return 64  
    if dataset_name == "pacs":  
        return 128  
    if dataset_name == "domainnet":  
        return 128  
    return 128  
  
  
def preprocess_real_features(X_train, X_test, pca_dim=128, whiten=True, l2norm=True):  
    Xtr = X_train.copy()  
    Xte = X_test.copy()  
  
    if l2norm:  
        Xtr = normalize(Xtr, norm="l2")  
        Xte = normalize(Xte, norm="l2")  
  
    scaler = StandardScaler(with_mean=True, with_std=True)  
    Xtr = scaler.fit_transform(Xtr)  
    Xte = scaler.transform(Xte)  
  
    pca_dim = min(int(pca_dim), Xtr.shape[1], max(2, Xtr.shape[0] - 1))  
    if pca_dim >= 2:  
        pca = PCA(n_components=pca_dim, whiten=whiten, random_state=0)  
        Xtr = pca.fit_transform(Xtr)  
        Xte = pca.transform(Xte)  
  
    return Xtr.astype(np.float32), Xte.astype(np.float32)  
  
  
def subsample_if_needed(X, y, e, max_n=None, seed=0):  
    if max_n is None or max_n <= 0 or len(y) <= max_n:  
        return X, y, e  
    rng = np.random.default_rng(seed)  
    idx = np.arange(len(y))  
    rng.shuffle(idx)  
    idx = np.sort(idx[:max_n])  
    return X[idx], y[idx], e[idx]  
  
  
# ============================================================  
# Dataset-specific defaults  
# ============================================================  
  
def choose_r_q(dataset_name, y_train, e_train):  
    dataset_name = dataset_name.lower()  
    if dataset_name in ["celeba", "waterbirds"]:  
        return 1, 8  
    if dataset_name in ["pacs", "domainnet"]:  
        return min(unique_count(y_train) - 1, 4), 8  
    K = unique_count(y_train)  
    return min(K - 1, 4), 8  
  
  
# ============================================================  
# Splits / class stats  
# ============================================================  
  
def two_way_split_by_class(y, seed=0):  
    rng = np.random.default_rng(seed)  
    labels = sorted(np.unique(y).tolist())  
    blocks = {"A": {}, "B": {}}  
    for c in labels:  
        idx = np.where(y == c)[0].copy()  
        rng.shuffle(idx)  
        n = len(idx)  
        n2 = n // 2  
        blocks["A"][int(c)] = idx[:n2]  
        blocks["B"][int(c)] = idx[n2:2 * n2]  
    return blocks  
  
  
def train_val_split_by_class(y, fit_frac=0.7, seed=0):  
    rng = np.random.default_rng(seed)  
    fit_idx, val_idx = [], []  
    for c in sorted(np.unique(y).tolist()):  
        idx = np.where(y == c)[0].copy()  
        rng.shuffle(idx)  
        n = len(idx)  
        if n < 4:  
            n_fit = max(1, n - 1)  
        else:  
            n_fit = int(round(fit_frac * n))  
            n_fit = min(max(2, n_fit), n - 2)  
        fit_idx.extend(idx[:n_fit].tolist())  
        val_idx.extend(idx[n_fit:].tolist())  
    return np.array(sorted(fit_idx), dtype=int), np.array(sorted(val_idx), dtype=int)  
  
  
def stratified_subsample_by_class(y, frac=0.85, seed=0):  
    rng = np.random.default_rng(seed)  
    keep = []  
    for c in sorted(np.unique(y).tolist()):  
        idx = np.where(y == c)[0].copy()  
        rng.shuffle(idx)  
        m = max(2, int(round(frac * len(idx))))  
        keep.extend(idx[:m].tolist())  
    return np.array(sorted(keep), dtype=int)  
  
  
def binary_env_fixed_size_subset(y, e, total_per_class, minor_frac=0.1, seed=0):  
    rng = np.random.default_rng(seed)  
    classes = np.unique(y)  
    keep = []  
  
    for c in classes:  
        idx = np.where(y == c)[0]  
        idx0 = idx[e[idx] == 0]  
        idx1 = idx[e[idx] == 1]  
  
        if len(idx0) <= len(idx1):  
            minority = idx0.copy()  
            majority = idx1.copy()  
        else:  
            minority = idx1.copy()  
            majority = idx0.copy()  
  
        rng.shuffle(minority)  
        rng.shuffle(majority)  
  
        n_minor = int(round(total_per_class * minor_frac))  
        n_minor = min(n_minor, len(minority))  
        n_major = total_per_class - n_minor  
        n_major = min(n_major, len(majority))  
  
        chosen = list(minority[:n_minor]) + list(majority[:n_major])  
  
        if len(chosen) < total_per_class:  
            rem = total_per_class - len(chosen)  
            leftover = list(minority[n_minor:]) + list(majority[n_major:])  
            rng.shuffle(leftover)  
            chosen += leftover[:rem]  
  
        keep.extend(chosen)  
  
    return np.array(sorted(keep), dtype=int)  
  
  
# ============================================================  
# Operators  
# ============================================================  
  
def class_means_from_indices(X, class_indices):  
    means = {}  
    for c, idx in class_indices.items():  
        means[int(c)] = X[idx].mean(axis=0)  
    return means  
  
  
def class_covs_from_indices_shrunk(X, class_indices, shrink_alpha=0.10):  
    covs = {}  
    for c, idx in class_indices.items():  
        C = sample_cov(X[idx])  
        covs[int(c)] = shrink_cov(C, alpha=shrink_alpha)  
    return covs  
  
  
def empirical_M_plugin(X, y):  
    classes = sorted(np.unique(y).tolist())  
    pi = {c: float(np.mean(y == c)) for c in classes}  
    all_idx = {c: np.where(y == c)[0] for c in classes}  
    means = class_means_from_indices(X, all_idx)  
    global_mean = np.sum([pi[c] * means[c] for c in classes], axis=0)  
  
    p = X.shape[1]  
    M = np.zeros((p, p), dtype=float)  
    for c in classes:  
        mu = means[c] - global_mean  
        M += pi[c] * np.outer(mu, mu)  
    return sym(M)  
  
  
def empirical_T_perp_plugin(X, y, Uproj, shrink_alpha=0.10):  
    classes = sorted(np.unique(y).tolist())  
    pi = {c: float(np.mean(y == c)) for c in classes}  
    all_idx = {c: np.where(y == c)[0] for c in classes}  
    covs = class_covs_from_indices_shrunk(X, all_idx, shrink_alpha=shrink_alpha)  
    proj_covs = {c: project_cov(covs[c], Uproj) for c in classes}  
    bar = np.sum([pi[c] * proj_covs[c] for c in classes], axis=0)  
  
    p = X.shape[1]  
    T = np.zeros((p, p), dtype=float)  
    for c in classes:  
        D = proj_covs[c] - bar  
        T += pi[c] * (D @ D)  
    return sym(T)  
  
  
def empirical_T_raw_plugin(X, y, shrink_alpha=0.10):  
    classes = sorted(np.unique(y).tolist())  
    pi = {c: float(np.mean(y == c)) for c in classes}  
    all_idx = {c: np.where(y == c)[0] for c in classes}  
    covs = class_covs_from_indices_shrunk(X, all_idx, shrink_alpha=shrink_alpha)  
    bar = np.sum([pi[c] * covs[c] for c in classes], axis=0)  
  
    p = X.shape[1]  
    T = np.zeros((p, p), dtype=float)  
    for c in classes:  
        D = covs[c] - bar  
        T += pi[c] * (D @ D)  
    return sym(T)  
  
  
def empirical_T_env_oracle(X, y, e, Uproj, shrink_alpha=0.10):  
    classes = sorted(np.unique(y).tolist())  
    envs = sorted(np.unique(e).tolist())  
    pi_y = {c: float(np.mean(y == c)) for c in classes}  
  
    p = X.shape[1]  
    T = np.zeros((p, p), dtype=float)  
  
    for c in classes:  
        idx_c = np.where(y == c)[0]  
        if len(idx_c) <= 1:  
            continue  
  
        probs = {}  
        covs = {}  
        total_mass = 0.0  
        for env in envs:  
            idx_ce = np.where((y == c) & (e == env))[0]  
            if len(idx_ce) <= 1:  
                continue  
            prob = float(len(idx_ce)) / float(len(idx_c))  
            C = shrink_cov(sample_cov(X[idx_ce]), alpha=shrink_alpha)  
            C = project_cov(C, Uproj)  
            probs[env] = prob  
            covs[env] = C  
            total_mass += prob  
  
        if total_mass <= 0:  
            continue  
  
        for env in list(probs.keys()):  
            probs[env] /= total_mass  
  
        bar_c = np.sum([probs[env] * covs[env] for env in probs], axis=0)  
        for env in probs:  
            D = covs[env] - bar_c  
            T += pi_y[c] * probs[env] * (D @ D)  
  
    return sym(T)  
  
  
# ============================================================  
# DMD fitting  
# ============================================================  
  
def estimate_M_crossfit_two_way(X, y, blocks):  
    classes = sorted(blocks["A"].keys())  
    pi = {c: float(np.mean(y == c)) for c in classes}  
  
    means_A = class_means_from_indices(X, blocks["A"])  
    means_B = class_means_from_indices(X, blocks["B"])  
  
    global_A = np.sum([pi[c] * means_A[c] for c in classes], axis=0)  
    global_B = np.sum([pi[c] * means_B[c] for c in classes], axis=0)  
  
    p = X.shape[1]  
    Mhat = np.zeros((p, p), dtype=float)  
    for c in classes:  
        muA = means_A[c] - global_A  
        muB = means_B[c] - global_B  
        Mhat += pi[c] * sym(np.outer(muA, muB))  
    return sym(Mhat)  
  
  
def estimate_M_crossfit_ensemble(X, y, base_seed=0, num_avg=8):  
    Macc = np.zeros((X.shape[1], X.shape[1]), dtype=float)  
    for s in range(num_avg):  
        blocks = two_way_split_by_class(y, seed=base_seed + 1009 * (s + 1))  
        Macc += estimate_M_crossfit_two_way(X, y, blocks)  
    return Macc / float(num_avg)  
  
  
def fit_real_dmd_bagged_refit(X_train, y_train, r, q, split_seed=0, num_avg=8, shrink_alpha=0.10):  
    U_list = []  
    for s in range(num_avg):  
        blocks = two_way_split_by_class(y_train, seed=split_seed + 3001 * (s + 1))  
        Mhat = estimate_M_crossfit_two_way(X_train, y_train, blocks)  
        Uhat_s, _ = top_eigenspace(Mhat, r)  
        U_list.append(Uhat_s)  
    Uhat = aggregate_subspaces(U_list, r)  
  
    That = empirical_T_perp_plugin(X_train, y_train, Uhat, shrink_alpha=shrink_alpha)  
    Vhat, _ = top_eigenspace(That, q)  
  
    That_raw = empirical_T_raw_plugin(X_train, y_train, shrink_alpha=shrink_alpha)  
    Vhat_raw, _ = top_eigenspace(That_raw, q)  
  
    Mhat_ens = estimate_M_crossfit_ensemble(X_train, y_train, base_seed=split_seed, num_avg=num_avg)  
  
    return {  
        "Uhat": Uhat,  
        "Vhat": Vhat,  
        "Vhat_raw": Vhat_raw,  
        "Mhat": Mhat_ens,  
        "That": That,  
        "That_raw": That_raw,  
        "lambda_r_Mhat": smallest_signal_eig(Mhat_ens, r),  
        "lambda_q_That": smallest_signal_eig(That, q),  
        "lambda_q_That_raw": smallest_signal_eig(That_raw, q),  
    }  
  
  
def fit_real_dmd_single_split_refit(X_train, y_train, r, q, split_seed=0, shrink_alpha=0.10):  
    blocks = two_way_split_by_class(y_train, seed=split_seed)  
    Mhat = estimate_M_crossfit_two_way(X_train, y_train, blocks)  
    Uhat, _ = top_eigenspace(Mhat, r)  
    That = empirical_T_perp_plugin(X_train, y_train, Uhat, shrink_alpha=shrink_alpha)  
    Vhat, _ = top_eigenspace(That, q)  
    return {  
        "Uhat": Uhat,  
        "Vhat": Vhat,  
        "Mhat": Mhat,  
        "That": That,  
        "lambda_r_Mhat": smallest_signal_eig(Mhat, r),  
        "lambda_q_That": smallest_signal_eig(That, q),  
    }  
  
  
def fit_second_stage_given_U(X_train, y_train, Uproj, q, shrink_alpha=0.10):  
    That = empirical_T_perp_plugin(X_train, y_train, Uproj, shrink_alpha=shrink_alpha)  
    Vhat, _ = top_eigenspace(That, q)  
    return {  
        "Uhat": Uproj,  
        "Vhat": Vhat,  
        "That": That,  
        "lambda_q_That": smallest_signal_eig(That, q),  
    }  
  
  
# ============================================================  
# Stage-1 baseline subspaces  
# ============================================================  
  
def top_principal_subspace(X, k):  
    C = sample_cov(X)  
    U, _ = top_eigenspace(C, k)  
    return U  
  
  
def lda_subspace(X, y, k):  
    K = unique_count(y)  
    if K < 2 or k <= 0:  
        return np.zeros((X.shape[1], 0), dtype=float)  
    k_eff = min(k, K - 1, X.shape[1])  
    clf = LinearDiscriminantAnalysis(n_components=k_eff)  
    clf.fit(X, y)  
    scalings = clf.scalings_[:, :k_eff]  
    Q, _ = np.linalg.qr(scalings)  
    return Q[:, :k_eff]  
  
  
def random_subspace(X, k, seed=0):  
    return random_projection_matrix(X.shape[1], k, seed=seed)  
  
  
def pls_subspace(X, y, k):  
    y_enc, classes = remap_labels_to_contiguous(y)  
    K = len(classes)  
    if K < 2 or k <= 0:  
        return np.zeros((X.shape[1], 0), dtype=float)  
  
    k_eff = min(k, X.shape[1], max(1, X.shape[0] - 1))  
    Y_onehot = np.zeros((len(y_enc), K), dtype=float)  
    Y_onehot[np.arange(len(y_enc)), y_enc] = 1.0  
  
    try:  
        pls = PLSRegression(n_components=k_eff)  
        pls.fit(X, Y_onehot)  
        W = pls.x_weights_[:, :k_eff]  
        Q, _ = np.linalg.qr(W)  
        return Q[:, :k_eff]  
    except Exception:  
        return np.zeros((X.shape[1], 0), dtype=float)  

  
# ============================================================  
# Metrics / probes  
# ============================================================  
  
def multiclass_nll(y_true, prob, eps=1e-12):  
    prob = np.clip(prob, eps, 1.0)  
    return float(-np.mean(np.log(prob[np.arange(len(y_true)), y_true])))  
  
  
def multiclass_ece(y_true, prob, n_bins=10):  
    pred = np.argmax(prob, axis=1)  
    conf = np.max(prob, axis=1)  
    acc = (pred == y_true).astype(float)  
    bins = np.linspace(0.0, 1.0, n_bins + 1)  
    ece = 0.0  
    for i in range(n_bins):  
        lo, hi = bins[i], bins[i + 1]  
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)  
        if np.sum(mask) == 0:  
            continue  
        w = np.mean(mask)  
        ece += w * abs(np.mean(acc[mask]) - np.mean(conf[mask]))  
    return float(ece)  
  
  
def fit_logreg_representation(Xtr, ytr, Xte, yte, seed=0):  
    ytr = np.asarray(ytr)  
    yte = np.asarray(yte)  
  
    # map train labels to contiguous ids  
    train_classes = np.unique(ytr)  
    cls_to_idx = {c: i for i, c in enumerate(train_classes)}  
  
    # test labels must belong to train label set  
    if not np.all(np.isin(yte, train_classes)):  
        raise ValueError("yte contains labels not seen in ytr.")  
  
    ytr_enc = np.array([cls_to_idx[c] for c in ytr], dtype=np.int64)  
    yte_enc = np.array([cls_to_idx[c] for c in yte], dtype=np.int64)  
  
    if Xtr.shape[1] == 0 or len(train_classes) < 2:  
        K = len(train_classes)  
        counts = np.bincount(ytr_enc, minlength=K).astype(float)  
        probs = counts / max(1.0, np.sum(counts))  
        prob = np.tile(probs[None, :], (len(yte_enc), 1))  
        pred_enc = np.argmax(prob, axis=1)  
        pred = train_classes[pred_enc]  
        return {  
            "acc": float(np.mean(pred == yte)),  
            "nll": multiclass_nll(yte_enc, prob),  
            "ece": multiclass_ece(yte_enc, prob),  
            "pred": pred,  
            "pred_enc": pred_enc,  
            "prob": prob,  
            "classes": train_classes,  
        }  
  
    scaler = StandardScaler(with_mean=True, with_std=True)  
    Xtr_s = scaler.fit_transform(Xtr)  
    Xte_s = scaler.transform(Xte)  
  
    clf = LogisticRegression(  
        max_iter=2000,  
        random_state=seed,  
        class_weight="balanced",  
        solver="lbfgs",  
    )  
    clf.fit(Xtr_s, ytr_enc)  
    prob = clf.predict_proba(Xte_s)  
    pred_enc = np.argmax(prob, axis=1)  
    pred = train_classes[pred_enc]  
  
    return {  
        "acc": float(np.mean(pred == yte)),  
        "nll": multiclass_nll(yte_enc, prob),  
        "ece": multiclass_ece(yte_enc, prob),  
        "pred": pred,  
        "pred_enc": pred_enc,  
        "prob": prob,  
        "classes": train_classes,  
    }  

  
  
def full_quadratic_feature_map(X):  
    n, d = X.shape  
    if d == 0:  
        return np.zeros((n, 0), dtype=float)  
    idx_i, idx_j = np.triu_indices(d)  
    quad = X[:, idx_i] * X[:, idx_j]  
    return np.concatenate([X, quad], axis=1)  
  
  
def maybe_reduce_for_quadratic(Xtr, Xte, max_input_dim=16, seed=0):  
    if Xtr.shape[1] <= max_input_dim:  
        return Xtr, Xte  
    pca = PCA(n_components=max_input_dim, whiten=False, random_state=seed)  
    return pca.fit_transform(Xtr), pca.transform(Xte)  
  
  
def fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=0, max_input_dim=16):  
    etr = np.asarray(etr)  
    ete = np.asarray(ete)  
  
    env_classes = np.unique(etr)  
    env_to_idx = {c: i for i, c in enumerate(env_classes)}  
  
    if not np.all(np.isin(ete, env_classes)):  
        raise ValueError("ete contains labels not seen in etr.")  
  
    etr_enc = np.array([env_to_idx[c] for c in etr], dtype=np.int64)  
    ete_enc = np.array([env_to_idx[c] for c in ete], dtype=np.int64)  
  
    if Xtr.shape[1] == 0 or len(env_classes) < 2:  
        K = len(env_classes)  
        counts = np.bincount(etr_enc, minlength=K).astype(float)  
        probs = counts / max(1.0, np.sum(counts))  
        prob = np.tile(probs[None, :], (len(ete_enc), 1))  
        pred_enc = np.argmax(prob, axis=1)  
        pred = env_classes[pred_enc]  
        return {"acc": float(np.mean(pred == ete)), "pred": pred, "pred_enc": pred_enc}  
  
    Xtr_red, Xte_red = maybe_reduce_for_quadratic(Xtr, Xte, max_input_dim=max_input_dim, seed=seed)  
    Xtr_q = full_quadratic_feature_map(Xtr_red)  
    Xte_q = full_quadratic_feature_map(Xte_red)  
  
    scaler = StandardScaler(with_mean=True, with_std=True)  
    Xtr_s = scaler.fit_transform(Xtr_q)  
    Xte_s = scaler.transform(Xte_q)  
  
    clf = LogisticRegression(  
        max_iter=2000,  
        random_state=seed,  
        class_weight="balanced",  
        solver="lbfgs",  
    )  
    clf.fit(Xtr_s, etr_enc)  
    pred_enc = clf.predict(Xte_s)  
    pred = env_classes[pred_enc]  
    return {"acc": float(np.mean(pred == ete)), "pred": pred, "pred_enc": pred_enc}  

  
def worst_group_accuracy(y_true, y_pred, e_true):  
    groups = sorted(list(set((int(a), int(b)) for a, b in zip(y_true, e_true))))  
    vals = []  
    for yy, ee in groups:  
        mask = (y_true == yy) & (e_true == ee)  
        if np.sum(mask) == 0:  
            continue  
        vals.append(float(np.mean(y_pred[mask] == y_true[mask])))  
    return float(np.min(vals)) if vals else 0.0  
  
  
# ============================================================  
# Shared eval helper  
# ============================================================  
  
def mean_capture(U, M, eps=1e-12):  
    if U.shape[1] == 0:  
        return 0.0  
    return float(np.trace(U.T @ M @ U) / (np.trace(M) + eps))  
  
  
def hetero_capture(U, T, eps=1e-12):  
    if U.shape[1] == 0:  
        return 0.0  
    return float(np.trace(U.T @ T @ U) / (np.trace(T) + eps))  
  
  
def background_purity(hetero_cap, mean_cap, eps=1e-12):  
    return float(hetero_cap / (hetero_cap + mean_cap + eps))  
  
  
def oracle_purity(oracle_cap, mean_cap, eps=1e-12):  
    return float(oracle_cap / (oracle_cap + mean_cap + eps))  
  
  
def evaluate_v_subspace(  
    Vhat,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    M_test,  
    T_test_common,  
    T_env_common,  
    seed=0,  
):  
    mean_cap = mean_capture(Vhat, M_test)  
    hetero_cap = hetero_capture(Vhat, T_test_common)  
    oracle_cap = hetero_capture(Vhat, T_env_common)  
  
    purity = background_purity(hetero_cap, mean_cap)  
    oracle_p = oracle_purity(oracle_cap, mean_cap)  
  
    Xtr_v = residualize_subspace(Xtr, Vhat)  
    Xte_v = residualize_subspace(Xte, Vhat)  
  
    label_ev = fit_logreg_representation(Xtr_v, ytr, Xte_v, yte, seed=seed + 1)  
    env_ev = fit_env_probe_quadratic(Xtr_v, etr, Xte_v, ete, seed=seed + 2)  
    wg = worst_group_accuracy(yte, label_ev["pred"], ete)  
  
    return {  
        "mean_capture": float(mean_cap),  
        "hetero_capture": float(hetero_cap),  
        "oracle_capture": float(oracle_cap),  
        "purity": float(purity),  
        "oracle_purity": float(oracle_p),  
        "label_acc_after": float(label_ev["acc"]),  
        "label_nll_after": float(label_ev["nll"]),  
        "env_acc_after": float(env_ev["acc"]),  
        "worst_group_after": float(wg),  
    }  
  
  
# ============================================================  
# Existing experiments  
# ============================================================  
  
def experiment_semantic_auditing(dataset_name, Xtr, ytr, Xte, yte, seed=0):  
    r, _ = choose_r_q(dataset_name, ytr, np.zeros_like(ytr))  
    fit = fit_real_dmd_bagged_refit(Xtr, ytr, r=r, q=min(8, Xtr.shape[1] - 1), split_seed=seed, num_avg=6)  
    U_dmd = fit["Uhat"]  
  
    U_pca = top_principal_subspace(Xtr, r)  
    U_lda = lda_subspace(Xtr, ytr, r)  
    U_rand = random_subspace(Xtr, r, seed=seed + 13)  
    U_pls = pls_subspace(Xtr, ytr, r)  
  
    methods = {  
        "full": None,  
        "dmd_u": U_dmd,  
        "pca_u": U_pca,  
        "lda_u": U_lda,  
        "rand_u": U_rand,  
        "pls_u": U_pls,  
    }  
  
    full_eval = fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 1)  
    out = {}  
    rows = []  
  
    for name, U in methods.items():  
        if U is None:  
            Xtr_rep, Xte_rep = Xtr, Xte  
        else:  
            Xtr_rep, Xte_rep = project_subspace(Xtr, U), project_subspace(Xte, U)  
        ev = fit_logreg_representation(Xtr_rep, ytr, Xte_rep, yte, seed=seed + 11)  
        out[name] = {  
            "acc": float(ev["acc"]),  
            "nll": float(ev["nll"]),  
            "ece": float(ev["ece"]),  
            "dim": int(Xtr_rep.shape[1]),  
            "sufficiency_gap_nll": float(ev["nll"] - full_eval["nll"]),  
            "sufficiency_gap_acc": float(full_eval["acc"] - ev["acc"]),  
            "sufficiency_gap_ece": float(ev["ece"] - full_eval["ece"]),  
        }  
        rows.append({"method": name, **out[name]})  
  
    claims = {  
        "A1_dmd_u_better_than_rand_on_nll": out["dmd_u"]["nll"] < out["rand_u"]["nll"],  
        "A2_dmd_u_not_worse_than_lda_on_acc": out["dmd_u"]["acc"] >= out["lda_u"]["acc"] - 0.03,  
        "A3_dmd_u_closer_to_full_than_pca_on_nll": out["dmd_u"]["sufficiency_gap_nll"] <= out["pca_u"]["sufficiency_gap_nll"] + 0.02,  
        "A4_dmd_u_not_worse_than_pls_on_nll": out["dmd_u"]["nll"] <= out["pls_u"]["nll"] + 0.03 if out["pls_u"]["dim"] > 0 else True,  
    }  
  
    return {  
        "exp_name": "semantic_auditing",  
        "r": int(r),  
        "methods": out,  
        "methods_table": rows,  
        "claims": claims,  
    }  
  
  
def experiment_residual_editing(dataset_name, Xtr, ytr, etr, Xte, yte, ete, seed=0, num_avg=8):  
    r, q = choose_r_q(dataset_name, ytr, etr)  
    q = min(q, Xtr.shape[1] - 1)  
  
    fit_dmd = fit_real_dmd_bagged_refit(Xtr, ytr, r=r, q=q, split_seed=seed + 10, num_avg=num_avg)  
    U_dmd = fit_dmd["Uhat"]  
    V_dmd = fit_dmd["Vhat"]  
    V_raw = fit_dmd["Vhat_raw"]  
  
    U_pca = top_principal_subspace(Xtr, r)  
    U_lda = lda_subspace(Xtr, ytr, r)  
    U_rand = random_subspace(Xtr, r, seed=seed + 20)  
    U_pls = pls_subspace(Xtr, ytr, r)  
  
    fit_pca = fit_second_stage_given_U(Xtr, ytr, U_pca, q, shrink_alpha=0.10)  
    fit_lda = fit_second_stage_given_U(Xtr, ytr, U_lda, q, shrink_alpha=0.10)  
    fit_rand = fit_second_stage_given_U(Xtr, ytr, U_rand, q, shrink_alpha=0.10)  
    fit_pls = fit_second_stage_given_U(Xtr, ytr, U_pls, q, shrink_alpha=0.10) if U_pls.shape[1] > 0 else {"Vhat": np.zeros((Xtr.shape[1], 0))}  
    fit_oracle = fit_second_stage_given_U(Xtr, ytr, U_dmd, q, shrink_alpha=0.10)  
  
    M_test = empirical_M_plugin(Xte, yte)  
    T_test_common = empirical_T_perp_plugin(Xte, yte, U_dmd, shrink_alpha=0.10)  
    T_env_common = empirical_T_env_oracle(Xte, yte, ete, U_dmd, shrink_alpha=0.10)  
  
    full_label = fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 30)  
    full_env = fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 31)  
    full_wg = worst_group_accuracy(yte, full_label["pred"], ete)  
  
    methods = {  
        "dmd": V_dmd,  
        "raw_cov": V_raw,  
        "pca_first": fit_pca["Vhat"],  
        "lda_first": fit_lda["Vhat"],  
        "rand_first": fit_rand["Vhat"],  
        "pls_first": fit_pls["Vhat"],  
        "oracle_u": fit_oracle["Vhat"],  
    }  
  
    out = {}  
    rows = []  
    for i, (name, Vhat) in enumerate(methods.items()):  
        ev = evaluate_v_subspace(  
            Vhat, Xtr, ytr, etr, Xte, yte, ete,  
            M_test, T_test_common, T_env_common,  
            seed=seed + 100 + i  
        )  
        ev["env_leak_reduction_vs_full"] = float(full_env["acc"] - ev["env_acc_after"])  
        ev["label_change_vs_full"] = float(ev["label_acc_after"] - full_label["acc"])  
        ev["label_nll_change_vs_full"] = float(ev["label_nll_after"] - full_label["nll"])  
        ev["worst_group_change_vs_full"] = float(ev["worst_group_after"] - full_wg)  
        out[name] = ev  
        rows.append({"method": name, **ev})  
  
    claims = {  
        "B1_dmd_purer_than_raw": out["dmd"]["oracle_purity"] > out["raw_cov"]["oracle_purity"] + 0.01,  
        "B2_dmd_less_semantic_leak_than_raw": out["dmd"]["mean_capture"] < out["raw_cov"]["mean_capture"],  
        "B3_dmd_better_leak_reduction_than_rand": out["dmd"]["env_leak_reduction_vs_full"] > out["rand_first"]["env_leak_reduction_vs_full"] - 0.01,  
        "B4_dmd_label_retention_vs_raw": out["dmd"]["label_acc_after"] >= out["raw_cov"]["label_acc_after"] - 0.01,  
        "B5_dmd_competitive_with_pls_first": out["dmd"]["oracle_purity"] >= out["pls_first"]["oracle_purity"] - 0.03 if out["pls_first"]["oracle_capture"] > 0 else True,  
    }  
  
    return {  
        "exp_name": "residual_editing",  
        "r": int(r),  
        "q": int(q),  
        "methods": out,  
        "methods_table": rows,  
        "claims": claims,  
        "fit_artifacts": {  
            "U_dmd": U_dmd,  
            "V_dmd": V_dmd,  
            "V_raw": V_raw,  
            "That_dmd": fit_dmd["That"],  
            "That_raw": fit_dmd["That_raw"],  
        }  
    }  
  
  
def compute_bridge_trial_metrics(  
    dataset_name,  
    Xfit, yfit, efit,  
    Xval, yval,  
    Xte, yte, ete,  
    seed=0,  
):  
    r, q = choose_r_q(dataset_name, yfit, efit)  
    q = min(q, Xfit.shape[1] - 1)  
  
    fit = fit_real_dmd_bagged_refit(Xfit, yfit, r=r, q=q, split_seed=seed, num_avg=6)  
    Uhat = fit["Uhat"]  
    Vhat = fit["Vhat"]  
    Vraw = fit["Vhat_raw"]  
  
    M_val = empirical_M_plugin(Xval, yval)  
    T_val = empirical_T_perp_plugin(Xval, yval, Uhat, shrink_alpha=0.10)  
  
    mean_cap_dmd_val = mean_capture(Vhat, M_val)  
    mean_cap_raw_val = mean_capture(Vraw, M_val)  
    hetero_cap_dmd_val = hetero_capture(Vhat, T_val)  
    hetero_cap_raw_val = hetero_capture(Vraw, T_val)  
  
    proxy_purity_dmd = background_purity(hetero_cap_dmd_val, mean_cap_dmd_val)  
    proxy_purity_raw = background_purity(hetero_cap_raw_val, mean_cap_raw_val)  
    proxy_purity_gain = float(proxy_purity_dmd - proxy_purity_raw)  
  
    Xfit_v = residualize_subspace(Xfit, Vhat)  
    Xval_v = residualize_subspace(Xval, Vhat)  
    Xfit_rawv = residualize_subspace(Xfit, Vraw)  
    Xval_rawv = residualize_subspace(Xval, Vraw)  
  
    val_label_dmd = fit_logreg_representation(Xfit_v, yfit, Xval_v, yval, seed=seed + 1)  
    val_label_raw = fit_logreg_representation(Xfit_rawv, yfit, Xval_rawv, yval, seed=seed + 2)  
    proxy_label_gain = float(val_label_dmd["acc"] - val_label_raw["acc"])  
  
    M_test = empirical_M_plugin(Xte, yte)  
    T_test = empirical_T_perp_plugin(Xte, yte, Uhat, shrink_alpha=0.10)  
    T_env = empirical_T_env_oracle(Xte, yte, ete, Uhat, shrink_alpha=0.10)  
  
    dmd_ev = evaluate_v_subspace(Vhat, Xfit, yfit, efit, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 10)  
    raw_ev = evaluate_v_subspace(Vraw, Xfit, yfit, efit, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 20)  
  
    return {  
        "log_lambda_q_That": float(np.log10(max(fit["lambda_q_That"], 1e-12))),  
        "proxy_purity_gain_vs_raw": proxy_purity_gain,  
        "proxy_label_gain_vs_raw": proxy_label_gain,  
        "oracle_purity_gain_vs_raw": float(dmd_ev["oracle_purity"] - raw_ev["oracle_purity"]),  
        "env_leak_reduction_vs_raw": float(raw_ev["env_acc_after"] - dmd_ev["env_acc_after"]),  
        "label_gain_vs_raw": float(dmd_ev["label_acc_after"] - raw_ev["label_acc_after"]),  
    }  
  
  
def experiment_bridge_diagnostics(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    repeats=8,  
    fit_frac=0.7,  
    seed=0,  
):  
    records = []  
    for rep in range(repeats):  
        idx_fit, idx_val = train_val_split_by_class(ytr, fit_frac=fit_frac, seed=seed + 101 * rep)  
        Xfit, yfit, efit = Xtr[idx_fit], ytr[idx_fit], etr[idx_fit]  
        Xval, yval = Xtr[idx_val], ytr[idx_val]  
  
        rec = compute_bridge_trial_metrics(  
            dataset_name,  
            Xfit, yfit, efit,  
            Xval, yval,  
            Xte, yte, ete,  
            seed=seed + 1000 + rep,  
        )  
        rec["repeat_id"] = int(rep)  
        records.append(rec)  
  
    proxy = np.array([r["proxy_purity_gain_vs_raw"] for r in records], dtype=float)  
    proxy_lab = np.array([r["proxy_label_gain_vs_raw"] for r in records], dtype=float)  
    loglam = np.array([r["log_lambda_q_That"] for r in records], dtype=float)  
    oracle_gain = np.array([r["oracle_purity_gain_vs_raw"] for r in records], dtype=float)  
    leak_red = np.array([r["env_leak_reduction_vs_raw"] for r in records], dtype=float)  
    label_gain = np.array([r["label_gain_vs_raw"] for r in records], dtype=float)  
  
    metrics = {  
        "corr_proxy_vs_oracle_purity_gain": pearson_corr(proxy, oracle_gain),  
        "corr_proxy_vs_env_leak_reduction": pearson_corr(proxy, leak_red),  
        "corr_proxy_label_vs_test_label_gain": pearson_corr(proxy_lab, label_gain),  
        "corr_loglambda_vs_oracle_purity_gain": pearson_corr(loglam, oracle_gain),  
        "corr_loglambda_vs_env_leak_reduction": pearson_corr(loglam, leak_red),  
        "top_half_proxy_minus_bottom_half_oracle_gain": top_bottom_gap(proxy, oracle_gain),  
        "top_half_proxy_minus_bottom_half_env_leak_red": top_bottom_gap(proxy, leak_red),  
        "repeats": int(repeats),  
    }  
  
    claims = {  
        "D1_proxy_tracks_oracle_gain": bool(  
            (metrics["corr_proxy_vs_oracle_purity_gain"] > 0.25) and  
            (metrics["top_half_proxy_minus_bottom_half_oracle_gain"] > 0.01)  
        ),  
        "D2_proxy_tracks_env_leakage": bool(  
            (metrics["corr_proxy_vs_env_leak_reduction"] > 0.15) and  
            (metrics["top_half_proxy_minus_bottom_half_env_leak_red"] > 0.005)  
        ),  
    }  
  
    return {  
        "exp_name": "bridge_diagnostics",  
        "metrics": metrics,  
        "claims": claims,  
        "trial_records": records,  
    }  
  
  
def experiment_failure_cases(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    subset_fracs=(0.20, 0.35, 0.50, 0.70, 0.90, 1.00),  
    repeats=4,  
    fit_frac=0.7,  
    seed=0,  
):  
    records = []  
    trial_id = 0  
    for frac in subset_fracs:  
        for rep in range(repeats):  
            idx_sub = stratified_subsample_by_class(ytr, frac=float(frac), seed=seed + 101 * trial_id)  
            Xs, ys, es = Xtr[idx_sub], ytr[idx_sub], etr[idx_sub]  
  
            idx_fit, idx_val = train_val_split_by_class(ys, fit_frac=fit_frac, seed=seed + 1000 + trial_id)  
            Xfit, yfit, efit = Xs[idx_fit], ys[idx_fit], es[idx_fit]  
            Xval, yval = Xs[idx_val], ys[idx_val]  
  
            rec = compute_bridge_trial_metrics(  
                dataset_name,  
                Xfit, yfit, efit,  
                Xval, yval,  
                Xte, yte, ete,  
                seed=seed + 2000 + trial_id,  
            )  
            rec["subset_frac"] = float(frac)  
            rec["subset_n"] = int(len(idx_sub))  
            rec["failure_oracle"] = bool(rec["oracle_purity_gain_vs_raw"] < 0.0)  
            rec["failure_env"] = bool(rec["env_leak_reduction_vs_raw"] < 0.0)  
            records.append(rec)  
            trial_id += 1  
  
    loglam = np.array([r["log_lambda_q_That"] for r in records], dtype=float)  
    proxy = np.array([r["proxy_purity_gain_vs_raw"] for r in records], dtype=float)  
    oracle_gain = np.array([r["oracle_purity_gain_vs_raw"] for r in records], dtype=float)  
    leak_red = np.array([r["env_leak_reduction_vs_raw"] for r in records], dtype=float)  
    fail_env = np.array([float(r["failure_env"]) for r in records], dtype=float)  
    fail_orc = np.array([float(r["failure_oracle"]) for r in records], dtype=float)  
  
    low_env, mid_env, high_env = tercile_means(proxy, fail_env)  
    low_orc, mid_orc, high_orc = tercile_means(proxy, fail_orc)  
  
    metrics = {  
        "corr_proxy_vs_oracle_gain": pearson_corr(proxy, oracle_gain),  
        "corr_proxy_vs_env_leak_reduction": pearson_corr(proxy, leak_red),  
        "corr_loglambda_vs_oracle_gain": pearson_corr(loglam, oracle_gain),  
        "corr_loglambda_vs_env_leak_reduction": pearson_corr(loglam, leak_red),  
        "env_failure_rate_low_proxy": low_env,  
        "env_failure_rate_mid_proxy": mid_env,  
        "env_failure_rate_high_proxy": high_env,  
        "oracle_failure_rate_low_proxy": low_orc,  
        "oracle_failure_rate_mid_proxy": mid_orc,  
        "oracle_failure_rate_high_proxy": high_orc,  
        "top_half_proxy_minus_bottom_half_env_gain": top_bottom_gap(proxy, leak_red),  
        "top_half_proxy_minus_bottom_half_oracle_gain": top_bottom_gap(proxy, oracle_gain),  
        "num_trials": int(len(records)),  
    }  
  
    claims = {  
        "F1_proxy_separates_failure_regime": bool(  
            ((metrics["env_failure_rate_low_proxy"] > metrics["env_failure_rate_high_proxy"] + 0.10) or  
             (metrics["oracle_failure_rate_low_proxy"] > metrics["oracle_failure_rate_high_proxy"] + 0.10)) and  
            (metrics["top_half_proxy_minus_bottom_half_oracle_gain"] > 0.005)  
        )  
    }  
  
    return {  
        "exp_name": "failure_case_diagnostics",  
        "metrics": metrics,  
        "claims": claims,  
        "trial_records": records,  
    }  
  
  
def experiment_stability(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    repeats=8,  
    num_avg=8,  
    seed=0,  
):  
    r, q = choose_r_q(dataset_name, ytr, etr)  
    q = min(q, Xtr.shape[1] - 1)  
  
    U_single_list, V_single_list = [], []  
    U_bag_list, V_bag_list = [], []  
  
    purity_single_list, purity_bag_list = [], []  
    label_single_list, label_bag_list = [], []  
  
    for rep in range(repeats):  
        fit_single = fit_real_dmd_single_split_refit(  
            X_train=Xtr, y_train=ytr, r=r, q=q,  
            split_seed=seed + 100 * rep,  
            shrink_alpha=0.10,  
        )  
        fit_bag = fit_real_dmd_bagged_refit(  
            X_train=Xtr, y_train=ytr, r=r, q=q,  
            split_seed=seed + 1000 + rep,  
            num_avg=num_avg,  
            shrink_alpha=0.10,  
        )  
  
        U_single, V_single = fit_single["Uhat"], fit_single["Vhat"]  
        U_bag, V_bag = fit_bag["Uhat"], fit_bag["Vhat"]  
  
        M_test_single = empirical_M_plugin(Xte, yte)  
        T_test_single = empirical_T_perp_plugin(Xte, yte, U_single, shrink_alpha=0.10)  
        T_env_single = empirical_T_env_oracle(Xte, yte, ete, U_single, shrink_alpha=0.10)  
        ev_single = evaluate_v_subspace(V_single, Xtr, ytr, etr, Xte, yte, ete, M_test_single, T_test_single, T_env_single, seed=seed + 2000 + rep)  
  
        M_test_bag = empirical_M_plugin(Xte, yte)  
        T_test_bag = empirical_T_perp_plugin(Xte, yte, U_bag, shrink_alpha=0.10)  
        T_env_bag = empirical_T_env_oracle(Xte, yte, ete, U_bag, shrink_alpha=0.10)  
        ev_bag = evaluate_v_subspace(V_bag, Xtr, ytr, etr, Xte, yte, ete, M_test_bag, T_test_bag, T_env_bag, seed=seed + 3000 + rep)  
  
        U_single_list.append(U_single)  
        V_single_list.append(V_single)  
        U_bag_list.append(U_bag)  
        V_bag_list.append(V_bag)  
  
        purity_single_list.append(ev_single["purity"])  
        purity_bag_list.append(ev_bag["purity"])  
        label_single_list.append(ev_single["label_acc_after"])  
        label_bag_list.append(ev_bag["label_acc_after"])  
  
    U_single_mu, U_single_sd = mean_pairwise_subspace_distance(U_single_list)  
    U_bag_mu, U_bag_sd = mean_pairwise_subspace_distance(U_bag_list)  
    V_single_mu, V_single_sd = mean_pairwise_subspace_distance(V_single_list)  
    V_bag_mu, V_bag_sd = mean_pairwise_subspace_distance(V_bag_list)  
  
    purity_single_mu, purity_single_sd = mean_std(purity_single_list)  
    purity_bag_mu, purity_bag_sd = mean_std(purity_bag_list)  
    label_single_mu, label_single_sd = mean_std(label_single_list)  
    label_bag_mu, label_bag_sd = mean_std(label_bag_list)  
  
    metrics = {  
        "pairwise_U_single_mean": U_single_mu,  
        "pairwise_U_bagged_mean": U_bag_mu,  
        "pairwise_V_single_mean": V_single_mu,  
        "pairwise_V_bagged_mean": V_bag_mu,  
        "U_mean_reduction_ratio": float((U_single_mu - U_bag_mu) / max(U_single_mu, 1e-12)),  
        "V_mean_reduction_ratio": float((V_single_mu - V_bag_mu) / max(V_single_mu, 1e-12)),  
        "purity_single_std": purity_single_sd,  
        "purity_bagged_std": purity_bag_sd,  
        "label_single_std": label_single_sd,  
        "label_bagged_std": label_bag_sd,  
        "purity_std_reduction_ratio": float((purity_single_sd - purity_bag_sd) / max(purity_single_sd, 1e-12)),  
        "label_std_reduction_ratio": float((label_single_sd - label_bag_sd) / max(label_single_sd, 1e-12)),  
        "repeats": int(repeats),  
    }  
  
    claims = {  
        "S1_bagging_stabilizes_V": bool(metrics["V_mean_reduction_ratio"] > 0.20),  
        "S2_bagging_stabilizes_practical_metrics": bool(  
            (metrics["purity_std_reduction_ratio"] > 0.20) and  
            (metrics["label_std_reduction_ratio"] >= -0.10)  
        ),  
    }  
  
    return {  
        "exp_name": "stability",  
        "metrics": metrics,  
        "claims": claims,  
    }  
  
  
def experiment_waterbirds_coverage_sweep(  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    num_avg=8,  
    seed=0,  
):  
    if unique_count(etr) != 2:  
        return {  
            "exp_name": "waterbirds_coverage_sweep",  
            "skipped": True,  
            "reason": "env_not_binary",  
        }  
  
    r, q = choose_r_q("waterbirds", ytr, etr)  
    q = min(q, Xtr.shape[1] - 1)  
  
    class_counts = [np.sum(ytr == c) for c in np.unique(ytr)]  
    total_per_class = int(0.5 * min(class_counts))  
    total_per_class = max(total_per_class, 100)  
  
    coverage_grid = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]  
    num_repeats = 3  
  
    records = []  
    trial_id = 0  
    for frac_id, frac in enumerate(coverage_grid):  
        for rep in range(num_repeats):  
            idx = binary_env_fixed_size_subset(  
                ytr, etr,  
                total_per_class=total_per_class,  
                minor_frac=frac,  
                seed=seed + 1000 + 100 * frac_id + rep,  
            )  
            Xs, ys, es = Xtr[idx], ytr[idx], etr[idx]  
  
            fit = fit_real_dmd_bagged_refit(  
                X_train=Xs,  
                y_train=ys,  
                r=r,  
                q=q,  
                split_seed=seed + 2000 + trial_id,  
                num_avg=num_avg,  
                shrink_alpha=0.10,  
            )  
  
            Uhat = fit["Uhat"]  
            Vhat = fit["Vhat"]  
            Vraw = fit["Vhat_raw"]  
  
            M_test = empirical_M_plugin(Xte, yte)  
            T_test = empirical_T_perp_plugin(Xte, yte, Uhat, shrink_alpha=0.10)  
            T_env = empirical_T_env_oracle(Xte, yte, ete, Uhat, shrink_alpha=0.10)  
  
            dmd_ev = evaluate_v_subspace(Vhat, Xs, ys, es, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 3000 + trial_id)  
            raw_ev = evaluate_v_subspace(Vraw, Xs, ys, es, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 4000 + trial_id)  
  
            support_min = float(min(np.sum((ys == c) & (es == ee)) for c in np.unique(ys) for ee in np.unique(es)))  
  
            records.append({  
                "coverage_frac": float(frac),  
                "support_min_cell_count": support_min,  
                "subset_n": int(len(idx)),  
                "log_lambda_q_That": float(np.log10(max(fit["lambda_q_That"], 1e-12))),  
                "purity_gain_vs_raw": float(dmd_ev["purity"] - raw_ev["purity"]),  
                "oracle_purity_gain_vs_raw": float(dmd_ev["oracle_purity"] - raw_ev["oracle_purity"]),  
                "env_leak_reduction_vs_raw": float(raw_ev["env_acc_after"] - dmd_ev["env_acc_after"]),  
                "label_gain_vs_raw": float(dmd_ev["label_acc_after"] - raw_ev["label_acc_after"]),  
            })  
            trial_id += 1  
  
    frac_arr = np.array([r["coverage_frac"] for r in records], dtype=float)  
    support_arr = np.array([r["support_min_cell_count"] for r in records], dtype=float)  
    oracle_purity_gain = np.array([r["oracle_purity_gain_vs_raw"] for r in records], dtype=float)  
    leak_red = np.array([r["env_leak_reduction_vs_raw"] for r in records], dtype=float)  
    loglam = np.array([r["log_lambda_q_That"] for r in records], dtype=float)  
  
    metrics = {  
        "corr_frac_vs_oracle_purity_gain": pearson_corr(frac_arr, oracle_purity_gain),  
        "corr_frac_vs_env_leak_reduction": pearson_corr(frac_arr, leak_red),  
        "corr_support_vs_oracle_purity_gain": pearson_corr(support_arr, oracle_purity_gain),  
        "corr_support_vs_env_leak_reduction": pearson_corr(support_arr, leak_red),  
        "corr_loglambda_vs_oracle_purity_gain": pearson_corr(loglam, oracle_purity_gain),  
        "corr_loglambda_vs_env_leak_reduction": pearson_corr(loglam, leak_red),  
        "top_half_support_minus_bottom_half_oracle_gain": top_bottom_gap(support_arr, oracle_purity_gain),  
        "top_half_support_minus_bottom_half_leak_reduction": top_bottom_gap(support_arr, leak_red),  
        "num_trials": int(len(records)),  
        "total_per_class": int(total_per_class),  
    }  
  
    claims = {  
        "W1_support_quality_tracks_oracle_purity": bool(  
            (metrics["corr_support_vs_oracle_purity_gain"] > 0.15) and  
            (metrics["top_half_support_minus_bottom_half_oracle_gain"] > 0.01)  
        ),  
        "W2_support_quality_tracks_env_leak_reduction": bool(  
            (metrics["corr_support_vs_env_leak_reduction"] > 0.15) and  
            (metrics["top_half_support_minus_bottom_half_leak_reduction"] > 0.005)  
        ),  
    }  
  
    return {  
        "exp_name": "waterbirds_coverage_sweep",  
        "skipped": False,  
        "metrics": metrics,  
        "claims": claims,  
        "trial_records": records,  
    }  
  
  
def select_rank_by_permutation_M(X, y, num_perm=32, alpha=0.95, max_rank=20, seed=0):  
    max_rank = min(int(max_rank), X.shape[1], max(1, unique_count(y) - 1))  
    M_obs = empirical_M_plugin(X, y)  
    obs = top_eigenvalues(M_obs, max_rank=max_rank)  
  
    null_eigs = []  
    rng = np.random.default_rng(seed)  
    for _ in range(num_perm):  
        yp = y.copy()  
        rng.shuffle(yp)  
        M_perm = empirical_M_plugin(X, yp)  
        null_eigs.append(top_eigenvalues(M_perm, max_rank=max_rank))  
    null_eigs = np.asarray(null_eigs, dtype=float)  
    null_q = np.quantile(null_eigs, alpha, axis=0)  
  
    mask = obs > null_q  
    selected_rank = int(np.max(np.where(mask)[0]) + 1) if np.any(mask) else 0  
    return {  
        "selected_rank": int(selected_rank),  
        "observed_eigs": obs.tolist(),  
        "null_q_alpha": null_q.tolist(),  
        "num_perm": int(num_perm),  
        "alpha": float(alpha),  
    }  
  
  
def select_rank_by_permutation_T(X, y, Uproj, num_perm=32, alpha=0.95, max_rank=20, seed=0):  
    max_rank = min(int(max_rank), X.shape[1])  
    T_obs = empirical_T_perp_plugin(X, y, Uproj, shrink_alpha=0.10)  
    obs = top_eigenvalues(T_obs, max_rank=max_rank)  
  
    null_eigs = []  
    rng = np.random.default_rng(seed)  
    for _ in range(num_perm):  
        yp = y.copy()  
        rng.shuffle(yp)  
        T_perm = empirical_T_perp_plugin(X, yp, Uproj, shrink_alpha=0.10)  
        null_eigs.append(top_eigenvalues(T_perm, max_rank=max_rank))  
    null_eigs = np.asarray(null_eigs, dtype=float)  
    null_q = np.quantile(null_eigs, alpha, axis=0)  
  
    mask = obs > null_q  
    selected_rank = int(np.max(np.where(mask)[0]) + 1) if np.any(mask) else 0  
    return {  
        "selected_rank": int(selected_rank),  
        "observed_eigs": obs.tolist(),  
        "null_q_alpha": null_q.tolist(),  
        "num_perm": int(num_perm),  
        "alpha": float(alpha),  
    }  
  
  
def evaluate_fit_self_consistent(  
    fit,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    seed=0,  
):  
    Uhat = fit["Uhat"]  
    Vhat = fit["Vhat"]  
    Vraw = fit.get("Vhat_raw", np.zeros((Xtr.shape[1], 0), dtype=float))  
  
    M_test = empirical_M_plugin(Xte, yte)  
    T_test = empirical_T_perp_plugin(Xte, yte, Uhat, shrink_alpha=0.10)  
    T_env = empirical_T_env_oracle(Xte, yte, ete, Uhat, shrink_alpha=0.10)  
  
    dmd_ev = evaluate_v_subspace(Vhat, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 1)  
    raw_ev = evaluate_v_subspace(Vraw, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 2)  
  
    return {  
        "oracle_purity_gain_vs_raw": float(dmd_ev["oracle_purity"] - raw_ev["oracle_purity"]),  
        "env_leak_reduction_vs_raw": float(raw_ev["env_acc_after"] - dmd_ev["env_acc_after"]),  
        "label_acc_after": float(dmd_ev["label_acc_after"]),  
    }  
  
  
def experiment_rank_selection(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    num_avg=8,  
    num_perm=32,  
    alpha=0.95,  
    max_rank=20,  
    seed=0,  
):  
    heur_r, heur_q = choose_r_q(dataset_name, ytr, etr)  
    heur_q = min(heur_q, Xtr.shape[1] - 1)  
  
    rank_M = select_rank_by_permutation_M(Xtr, ytr, num_perm=num_perm, alpha=alpha, max_rank=max_rank, seed=seed + 11)  
    r_sel = int(rank_M["selected_rank"])  
  
    U_sel_obs, _ = top_eigenspace(empirical_M_plugin(Xtr, ytr), r_sel)  
  
    rank_T = select_rank_by_permutation_T(Xtr, ytr, U_sel_obs, num_perm=num_perm, alpha=alpha, max_rank=max_rank, seed=seed + 29)  
    q_sel = int(rank_T["selected_rank"])  
  
    fit_heur = fit_real_dmd_bagged_refit(Xtr, ytr, heur_r, heur_q, split_seed=seed + 101, num_avg=num_avg)  
    fit_sel = fit_real_dmd_bagged_refit(Xtr, ytr, r_sel, q_sel, split_seed=seed + 202, num_avg=num_avg)  
  
    heur_metrics = evaluate_fit_self_consistent(fit_heur, Xtr, ytr, etr, Xte, yte, ete, seed=seed + 300)  
    sel_metrics = evaluate_fit_self_consistent(fit_sel, Xtr, ytr, etr, Xte, yte, ete, seed=seed + 400)  
  
    claims = {  
        "Rk1_permnull_not_worse_than_heuristic": bool(  
            (sel_metrics["oracle_purity_gain_vs_raw"] >= heur_metrics["oracle_purity_gain_vs_raw"] - 0.02) and  
            (sel_metrics["label_acc_after"] >= heur_metrics["label_acc_after"] - 0.01)  
        )  
    }  
  
    return {  
        "exp_name": "rank_selection",  
        "heuristic_r": int(heur_r),  
        "heuristic_q": int(heur_q),  
        "selected_r": int(r_sel),  
        "selected_q": int(q_sel),  
        "M_rank_report": rank_M,  
        "T_rank_report": rank_T,  
        "heuristic_metrics": heur_metrics,  
        "selected_metrics": sel_metrics,  
        "claims": claims,  
    }  
  
  
def experiment_baseline_positioning(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    num_avg=8,  
    rand_repeats=6,  
    seed=0,  
):  
    r, q = choose_r_q(dataset_name, ytr, etr)  
    q = min(q, Xtr.shape[1] - 1)  
  
    fit_dmd = fit_real_dmd_bagged_refit(Xtr, ytr, r, q, split_seed=seed + 100, num_avg=num_avg)  
    U_dmd = fit_dmd["Uhat"]  
    V_dmd = fit_dmd["Vhat"]  
    V_raw = fit_dmd["Vhat_raw"]  
  
    M_test = empirical_M_plugin(Xte, yte)  
    T_test_common = empirical_T_perp_plugin(Xte, yte, U_dmd, shrink_alpha=0.10)  
    T_env_common = empirical_T_env_oracle(Xte, yte, ete, U_dmd, shrink_alpha=0.10)  
  
    full_label = fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 1)  
    full_env = fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 2)  
    full_wg = worst_group_accuracy(yte, full_label["pred"], ete)  
  
    def finalize(ev):  
        ev["env_leak_reduction_vs_full"] = float(full_env["acc"] - ev["env_acc_after"])  
        ev["wg_gain_vs_full"] = float(ev["worst_group_after"] - full_wg)  
        ev["label_change_vs_full"] = float(ev["label_acc_after"] - full_label["acc"])  
        return ev  
  
    methods = {}  
  
    methods["dmd"] = finalize(evaluate_v_subspace(V_dmd, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test_common, T_env_common, seed=seed + 10))  
    methods["raw_cov"] = finalize(evaluate_v_subspace(V_raw, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test_common, T_env_common, seed=seed + 20))  
  
    U_pca = top_principal_subspace(Xtr, r)  
    V_pca = fit_second_stage_given_U(Xtr, ytr, U_pca, q)["Vhat"]  
    methods["pca_first"] = finalize(evaluate_v_subspace(V_pca, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test_common, T_env_common, seed=seed + 30))  
  
    U_lda = lda_subspace(Xtr, ytr, r)  
    V_lda = fit_second_stage_given_U(Xtr, ytr, U_lda, q)["Vhat"]  
    methods["lda_first"] = finalize(evaluate_v_subspace(V_lda, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test_common, T_env_common, seed=seed + 40))  
  
    U_pls = pls_subspace(Xtr, ytr, r)  
    V_pls = fit_second_stage_given_U(Xtr, ytr, U_pls, q)["Vhat"] if U_pls.shape[1] > 0 else np.zeros((Xtr.shape[1], 0))  
    methods["pls_first"] = finalize(evaluate_v_subspace(V_pls, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test_common, T_env_common, seed=seed + 50))  
  
    rand_trials = []  
    for rep in range(rand_repeats):  
        U_rand = random_subspace(Xtr, r, seed=seed + 1000 + rep)  
        V_rand = fit_second_stage_given_U(Xtr, ytr, U_rand, q)["Vhat"]  
        rand_trials.append(finalize(evaluate_v_subspace(V_rand, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test_common, T_env_common, seed=seed + 60 + rep)))  
  
    rand_mean = {}  
    if len(rand_trials) > 0:  
        keys = rand_trials[0].keys()  
        for k in keys:  
            rand_mean[k] = float(np.mean([r[k] for r in rand_trials]))  
            rand_mean[k + "_std"] = float(np.std([r[k] for r in rand_trials]))  
    methods["rand_first"] = rand_mean  
  
    claims = {  
        "P1_dmd_stronger_than_raw_and_rand_on_oracle_purity": bool(  
            (methods["dmd"]["oracle_purity"] > methods["raw_cov"]["oracle_purity"] + 0.01) and  
            (methods["dmd"]["oracle_purity"] > methods["rand_first"]["oracle_purity"] + 0.01)  
        ),  
        "P2_dmd_better_than_pca_on_leakage": bool(  
            methods["dmd"]["env_leak_reduction_vs_full"] >= methods["pca_first"]["env_leak_reduction_vs_full"] - 0.01  
        ),  
    }  
  
    return {  
        "exp_name": "baseline_positioning",  
        "methods": methods,  
        "rand_first_trials": rand_trials,  
        "claims": claims,  
    }  
  
  
def experiment_representation_quality_control(  
    data_root,  
    dataset_name,  
    backbone_name,  
    max_train=0,  
    max_test=0,  
    seed=0,  
):  
    out = {  
        "exp_name": "representation_quality_control",  
        "dataset": dataset_name,  
        "backbone": backbone_name,  
        "status": "ok",  
    }  
  
    def load_proc(pretrained_flag):    
        Xtr_raw, ytr, etr, _ = load_cached_features(    
            data_root=data_root,    
            dataset_name=dataset_name,    
            split="train",    
            backbone_name=backbone_name,    
            pretrained_flag=pretrained_flag,    
        )    
        Xte_raw, yte, ete, _ = load_cached_features(    
            data_root=data_root,    
            dataset_name=dataset_name,    
            split="test",    
            backbone_name=backbone_name,    
            pretrained_flag=pretrained_flag,    
        )    
        if max_train > 0:    
            Xtr_raw, ytr, etr = subsample_if_needed(  
                Xtr_raw, ytr, etr, max_n=max_train, seed=seed + 11  
            )    
        if max_test > 0:    
            Xte_raw, yte, ete = subsample_if_needed(  
                Xte_raw, yte, ete, max_n=max_test, seed=seed + 17  
            )    
        pca_dim = choose_pca_dim(dataset_name)    
        Xtr, Xte = preprocess_real_features(  
            Xtr_raw, Xte_raw, pca_dim=pca_dim, whiten=True, l2norm=True  
        )    
        return Xtr, ytr, etr, Xte, yte, ete    
  
  
    try:  
        Xtr_pt, ytr_pt, etr_pt, Xte_pt, yte_pt, ete_pt = load_proc(True)  
        Xtr_rd, ytr_rd, etr_rd, Xte_rd, yte_rd, ete_rd = load_proc(False)  
  
        sem_pt = experiment_semantic_auditing(dataset_name, Xtr_pt, ytr_pt, Xte_pt, yte_pt, seed=seed)  
        sem_rd = experiment_semantic_auditing(dataset_name, Xtr_rd, ytr_rd, Xte_rd, yte_rd, seed=seed + 100)  
  
        edit_pt = experiment_residual_editing(dataset_name, Xtr_pt, ytr_pt, etr_pt, Xte_pt, yte_pt, ete_pt, seed=seed + 200)  
        edit_rd = experiment_residual_editing(dataset_name, Xtr_rd, ytr_rd, etr_rd, Xte_rd, yte_rd, ete_rd, seed=seed + 300)  
  
        out["pretrained"] = {"semantic": sem_pt, "editing": edit_pt}  
        out["random"] = {"semantic": sem_rd, "editing": edit_rd}  
  
        claims = {  
            "Q1_pretrained_better_stage1_than_random": bool(  
                sem_pt["methods"]["dmd_u"]["nll"] < sem_rd["methods"]["dmd_u"]["nll"]  
            ),  
            "Q2_pretrained_better_stage2_than_random": bool(  
                edit_pt["methods"]["dmd"]["oracle_purity"] > edit_rd["methods"]["dmd"]["oracle_purity"]  
            ),  
        }  
        out["claims"] = claims  
        return out  
  
    except Exception as ex:  
        out["status"] = "error"  
        out["error"] = str(ex)  
        return out  
  
  
# ============================================================  
# New experiment helpers  
# ============================================================  
  
def prepare_stage2_method_fits(dataset_name, Xtr, ytr, etr, q_max, num_avg=8, seed=0):  
    r, _ = choose_r_q(dataset_name, ytr, etr)  
    q_max = min(int(q_max), max(1, Xtr.shape[1] - 1))  
  
    fit_dmd = fit_real_dmd_bagged_refit(Xtr, ytr, r=r, q=q_max, split_seed=seed + 10, num_avg=num_avg)  
    U_dmd = fit_dmd["Uhat"]  
  
    U_pca = top_principal_subspace(Xtr, r)  
    U_lda = lda_subspace(Xtr, ytr, r)  
    U_rand = random_subspace(Xtr, r, seed=seed + 20)  
  
    fit_pca = fit_second_stage_given_U(Xtr, ytr, U_pca, q_max, shrink_alpha=0.10)  
    fit_lda = fit_second_stage_given_U(Xtr, ytr, U_lda, q_max, shrink_alpha=0.10)  
    fit_rand = fit_second_stage_given_U(Xtr, ytr, U_rand, q_max, shrink_alpha=0.10)  
  
    methods = {  
        "dmd": {  
            "U": U_dmd,  
            "Vfull": fit_dmd["Vhat"],  
            "T": fit_dmd["That"],  
        },  
        "raw_cov": {  
            "U": np.zeros((Xtr.shape[1], 0), dtype=float),  
            "Vfull": fit_dmd["Vhat_raw"],  
            "T": fit_dmd["That_raw"],  
        },  
        "pca_first": {  
            "U": U_pca,  
            "Vfull": fit_pca["Vhat"],  
            "T": fit_pca["That"],  
        },  
        "lda_first": {  
            "U": U_lda,  
            "Vfull": fit_lda["Vhat"],  
            "T": fit_lda["That"],  
        },  
        "rand_first": {  
            "U": U_rand,  
            "Vfull": fit_rand["Vhat"],  
            "T": fit_rand["That"],  
        },  
    }  
    return r, q_max, methods  
  
  
def eval_editing_after_removal(Vremove, Xtr, ytr, etr, Xte, yte, ete, full_label, full_env, full_wg, U_common_for_purity, seed=0):  
    M_test = empirical_M_plugin(Xte, yte)  
    T_test = empirical_T_perp_plugin(Xte, yte, U_common_for_purity, shrink_alpha=0.10)  
    T_env = empirical_T_env_oracle(Xte, yte, ete, U_common_for_purity, shrink_alpha=0.10)  
    ev = evaluate_v_subspace(Vremove, Xtr, ytr, etr, Xte, yte, ete, M_test, T_test, T_env, seed=seed)  
    ev["env_leak_reduction_vs_full"] = float(full_env["acc"] - ev["env_acc_after"])  
    ev["label_change_vs_full"] = float(ev["label_acc_after"] - full_label["acc"])  
    ev["label_acc_drop_vs_full"] = float(full_label["acc"] - ev["label_acc_after"])  
    ev["label_nll_deterioration_vs_full"] = float(ev["label_nll_after"] - full_label["nll"])  
    ev["label_retention_vs_full"] = float(ev["label_acc_after"] / max(full_label["acc"], 1e-12))  
    ev["worst_group_change_vs_full"] = float(ev["worst_group_after"] - full_wg)  
    return ev  
  
  
def pareto_area(x, y):  
    x = np.asarray(x, dtype=float)  
    y = np.asarray(y, dtype=float)  
    if len(x) < 2:  
        return 0.0  
    order = np.argsort(x)  
    x = x[order]  
    y = y[order]  
    x = x - np.min(x)  
    return float(np.trapz(y, x))  
  
  
def plot_pareto_curve(records, out_png, dataset_name, backbone_name):  
    if not plot_if_available():  
        return  
    df = pd.DataFrame(records)  
    if len(df) == 0:  
        return  
    plt.figure(figsize=(7.5, 5.5))  
    for method in sorted(df["method"].unique()):  
        tmp = df[df["method"] == method].sort_values("q_remove")  
        plt.plot(  
            tmp["env_leak_reduction_vs_full"],  
            tmp["label_retention_vs_full"],  
            marker="o",  
            linewidth=2,  
            label=method,  
        )  
    plt.xlabel("Environment leakage reduction")  
    plt.ylabel("Label retention")  
    plt.title(f"{dataset_name} | {backbone_name} | editing frontier")  
    plt.grid(alpha=0.3)  
    plt.legend()  
    plt.tight_layout()  
    plt.savefig(out_png, dpi=180)  
    plt.close()  
  
    plt.figure(figsize=(7.5, 5.5))  
    for method in sorted(df["method"].unique()):  
        tmp = df[df["method"] == method].sort_values("q_remove")  
        plt.plot(  
            tmp["env_leak_reduction_vs_full"],  
            tmp["label_nll_deterioration_vs_full"],  
            marker="o",  
            linewidth=2,  
            label=method,  
        )  
    plt.xlabel("Environment leakage reduction")  
    plt.ylabel("Label NLL deterioration")  
    plt.title(f"{dataset_name} | {backbone_name} | editing frontier (NLL)")  
    plt.grid(alpha=0.3)  
    plt.legend()  
    plt.tight_layout()  
    plt.savefig(out_png.replace(".png", "_nll.png"), dpi=180)  
    plt.close()  
  
  
def experiment_editing_frontier(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    q_grid=(1, 2, 3, 5, 8, 12),  
    num_avg=8,  
    seed=0,  
):  
    q_grid = sorted(list(set([int(q) for q in q_grid if int(q) > 0])))  
    if len(q_grid) == 0:  
        q_grid = [1, 2, 3, 5, 8]  
  
    q_max = min(max(q_grid), Xtr.shape[1] - 1)  
    r, q_max, methods = prepare_stage2_method_fits(dataset_name, Xtr, ytr, etr, q_max=q_max, num_avg=num_avg, seed=seed)  
  
    full_label = fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 100)  
    full_env = fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 101)  
    full_wg = worst_group_accuracy(yte, full_label["pred"], ete)  
  
    records = []  
    summaries = []  
    method_artifacts = {}  
  
    for m_id, (mname, mfit) in enumerate(methods.items()):  
        U_method = mfit["U"]  
        Vfull = mfit["Vfull"]  
        T_method = mfit["T"]  
        method_artifacts[mname] = {  
            "U": U_method,  
            "Vfull": Vfull,  
            "T": T_method,  
        }  
  
        for q in q_grid:  
            q_eff = min(q, Vfull.shape[1])  
            Vq = Vfull[:, :q_eff]  
            ev = eval_editing_after_removal(  
                Vremove=Vq,  
                Xtr=Xtr, ytr=ytr, etr=etr,  
                Xte=Xte, yte=yte, ete=ete,  
                full_label=full_label,  
                full_env=full_env,  
                full_wg=full_wg,  
                U_common_for_purity=U_method,  
                seed=seed + 500 + 13 * m_id + q_eff,  
            )  
            rec = {"method": mname, "q_remove": int(q_eff), **ev}  
            records.append(rec)  
  
        mdf = pd.DataFrame([r for r in records if r["method"] == mname]).sort_values("q_remove")  
        area_ret = pareto_area(mdf["env_leak_reduction_vs_full"].values, mdf["label_retention_vs_full"].values)  
        area_nll = pareto_area(mdf["env_leak_reduction_vs_full"].values, -mdf["label_nll_deterioration_vs_full"].values)  
        summaries.append({  
            "method": mname,  
            "pareto_area_label_retention": float(area_ret),  
            "pareto_area_neg_nll_deterioration": float(area_nll),  
            "best_env_leak_reduction": float(np.max(mdf["env_leak_reduction_vs_full"].values)),  
            "best_label_retention": float(np.max(mdf["label_retention_vs_full"].values)),  
            "best_oracle_purity": float(np.max(mdf["oracle_purity"].values)),  
        })  
  
    sum_by_method = {r["method"]: r for r in summaries}  
    claims = {  
        "PF1_dmd_frontier_area_vs_raw": bool(sum_by_method["dmd"]["pareto_area_label_retention"] >= sum_by_method["raw_cov"]["pareto_area_label_retention"] - 0.005),  
        "PF2_dmd_frontier_area_vs_pca": bool(sum_by_method["dmd"]["pareto_area_label_retention"] >= sum_by_method["pca_first"]["pareto_area_label_retention"] - 0.005),  
        "PF3_dmd_frontier_area_vs_rand": bool(sum_by_method["dmd"]["pareto_area_label_retention"] >= sum_by_method["rand_first"]["pareto_area_label_retention"] - 0.005),  
    }  
  
    return {  
        "exp_name": "editing_frontier",  
        "r": int(r),  
        "q_grid": q_grid,  
        "curve_records": records,  
        "method_summary": summaries,  
        "claims": claims,  
        "artifacts": method_artifacts,  
    }  
  
  
def residual_top3_eig_sum(T):  
    vals = np.maximum(top_eigenvalues(T, max_rank=10), 0.0)  
    return float(np.sum(vals[:min(3, len(vals))]))  
  
  
def residual_eigengap_ratio(T, q=3):  
    vals = np.maximum(top_eigenvalues(T, max_rank=max(q + 1, 6)), 0.0)  
    if len(vals) <= q:  
        if len(vals) >= 2:  
            return float(vals[0] / max(vals[1], 1e-12))  
        return 0.0  
    return float(vals[q - 1] / max(vals[q], 1e-12))  
  
  
def residual_mass_above_null(X, y, Uproj, shrink_alpha=0.10, q=3, num_perm=24, seed=0):  
    T_obs = empirical_T_perp_plugin(X, y, Uproj, shrink_alpha=shrink_alpha)  
    obs_top3 = residual_top3_eig_sum(T_obs)  
    obs_trace = float(max(np.trace(T_obs), 1e-12))  
  
    rng = np.random.default_rng(seed)  
    null_top3 = []  
    for _ in range(num_perm):  
        yp = y.copy()  
        rng.shuffle(yp)  
        T_perm = empirical_T_perp_plugin(X, yp, Uproj, shrink_alpha=shrink_alpha)  
        null_top3.append(residual_top3_eig_sum(T_perm))  
    null_top3 = np.asarray(null_top3, dtype=float)  
  
    above_null = max(0.0, obs_top3 - float(np.mean(null_top3)))  
    explained_above_null = float(above_null / obs_trace)  
    return {  
        "top3_obs": float(obs_top3),  
        "null_top3_mean": float(np.mean(null_top3)) if len(null_top3) > 0 else 0.0,  
        "null_top3_std": float(np.std(null_top3)) if len(null_top3) > 0 else 0.0,  
        "explained_spectral_mass_above_null": explained_above_null,  
        "null_top3_samples": null_top3,  
        "T_obs": T_obs,  
    }  
  
  
def support_quality_labelonly(X, y, max_clusters=3, reduce_dim=8, seed=0):  
    vals = []  
    for c in sorted(np.unique(y).tolist()):  
        idx = np.where(y == c)[0]  
        Xc = X[idx]  
        if Xc.shape[0] < 12:  
            continue  
        d_eff = min(reduce_dim, Xc.shape[1], max(2, Xc.shape[0] - 1))  
        if d_eff < Xc.shape[1]:  
            try:  
                pca = PCA(n_components=d_eff, whiten=False, random_state=seed)  
                Xcr = pca.fit_transform(Xc)  
            except Exception:  
                Xcr = Xc  
        else:  
            Xcr = Xc  
  
        k = min(max_clusters, max(2, Xcr.shape[0] // 40))  
        k = min(k, Xcr.shape[0] - 1)  
        if k <= 1:  
            vals.append(1.0)  
            continue  
        try:  
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)  
            lab = km.fit_predict(Xcr)  
            cnt = np.bincount(lab, minlength=k).astype(float)  
            p = cnt / np.sum(cnt)  
            H = -np.sum(p * np.log(p + 1e-12))  
            eff = np.exp(H) / float(k)  
            vals.append(float(eff))  
        except Exception:  
            vals.append(0.0)  
    if len(vals) == 0:  
        return 0.0  
    return float(np.mean(vals))  
  
  
def split_stability_dmd(dataset_name, Xtr, ytr, etr, q_eval=3, repeats=5, subset_frac=0.85, num_avg=6, seed=0):  
    Vs = []  
    r, q = choose_r_q(dataset_name, ytr, etr)  
    q = min(max(q_eval, 1), q, Xtr.shape[1] - 1)  
    for rep in range(repeats):  
        idx = stratified_subsample_by_class(ytr, frac=subset_frac, seed=seed + 99 * rep)  
        Xs, ys, es = Xtr[idx], ytr[idx], etr[idx]  
        fit = fit_real_dmd_bagged_refit(Xs, ys, r=r, q=q, split_seed=seed + 1000 + rep, num_avg=num_avg)  
        Vs.append(fit["Vhat"][:, :min(q_eval, fit["Vhat"].shape[1])])  
  
    if len(Vs) <= 1:  
        return {"split_stability": 0.0, "split_instability": 1.0}  
  
    dist_mu, dist_sd = mean_pairwise_subspace_distance(Vs)  
    ov_mu, ov_sd = mean_pairwise_subspace_overlap(Vs)  
    stability = float(ov_mu)  
    instability = float(dist_mu)  
    return {  
        "split_stability": stability,  
        "split_instability": instability,  
        "pairwise_overlap_mean": float(ov_mu),  
        "pairwise_overlap_std": float(ov_sd),  
        "pairwise_distance_mean": float(dist_mu),  
        "pairwise_distance_std": float(dist_sd),  
    }  
  
  
def experiment_residual_diagnostics_E(  
    dataset_name,  
    Xtr, ytr, etr,  
    num_avg=8,  
    q_eval=3,  
    num_perm=24,  
    split_repeats=5,  
    seed=0,  
):  
    q_max = max(q_eval, 8)  
    r, q_max, methods = prepare_stage2_method_fits(dataset_name, Xtr, ytr, etr, q_max=q_max, num_avg=num_avg, seed=seed)  
    rows = []  
    artifacts = {}  
  
    dmd_stab = split_stability_dmd(  
        dataset_name, Xtr, ytr, etr,  
        q_eval=q_eval,  
        repeats=split_repeats,  
        subset_frac=0.85,  
        num_avg=min(num_avg, 6),  
        seed=seed + 800,  
    )  
  
    for mname, mfit in methods.items():  
        U = mfit["U"]  
        T = mfit["T"]  
        Xres = residualize_subspace(Xtr, U)  
  
        top3 = residual_top3_eig_sum(T)  
        gap = residual_eigengap_ratio(T, q=q_eval)  
  
        if mname == "raw_cov":  
            # raw_cov also has label-conditioned T, but to keep consistency we can use U=0 in T_perp null  
            mass_rep = residual_mass_above_null(Xtr, ytr, Uproj=np.zeros((Xtr.shape[1], 0)), q=q_eval, num_perm=num_perm, seed=seed + 11)  
        else:  
            mass_rep = residual_mass_above_null(Xtr, ytr, Uproj=U, q=q_eval, num_perm=num_perm, seed=seed + 11)  
  
        support = support_quality_labelonly(Xres, ytr, max_clusters=3, reduce_dim=8, seed=seed + 12)  
        if mname == "dmd":  
            stability = dmd_stab["split_stability"]  
            instability = dmd_stab["split_instability"]  
        else:  
            stability = np.nan  
            instability = np.nan  
  
        row = {  
            "method": mname,  
            "top3_residual_eig_sum": float(top3),  
            "explained_spectral_mass_above_null": float(mass_rep["explained_spectral_mass_above_null"]),  
            "eigengap_ratio": float(gap),  
            "support_quality": float(support),  
            "split_stability": float(stability) if not np.isnan(stability) else np.nan,  
            "split_instability": float(instability) if not np.isnan(instability) else np.nan,  
            "null_top3_mean": float(mass_rep["null_top3_mean"]),  
            "null_top3_std": float(mass_rep["null_top3_std"]),  
        }  
        rows.append(row)  
        artifacts[mname] = {  
            "U": U,  
            "T": T,  
            "null_top3_samples": mass_rep["null_top3_samples"],  
            "eigvals": top_eigenvalues(T, max_rank=20),  
        }  
  
    claims = {  
        "E1_dmd_above_null_positive": bool(next(r for r in rows if r["method"] == "dmd")["explained_spectral_mass_above_null"] > 0.0),  
    }  
  
    return {  
        "exp_name": "residual_diagnostics_E",  
        "r": int(r),  
        "q_eval": int(q_eval),  
        "methods_table": rows,  
        "claims": claims,  
        "artifacts": artifacts,  
        "dmd_stability_report": dmd_stab,  
    }  
  
  
def plot_abstention_scatter(records, out_png, dataset_name, backbone_name):  
    if not plot_if_available():  
        return  
    df = pd.DataFrame(records)  
    if len(df) == 0:  
        return  
  
    plt.figure(figsize=(7.5, 5.5))  
    plt.scatter(  
        df["support_quality"],  
        df["always_env_leak_reduction_vs_full"],  
        c=df["split_stability"],  
        cmap="viridis",  
        alpha=0.8  
    )  
    plt.colorbar(label="split_stability")  
    plt.xlabel("support_quality")  
    plt.ylabel("always-edit env leakage reduction")  
    plt.title(f"{dataset_name} | {backbone_name} | abstention diagnostics")  
    plt.grid(alpha=0.3)  
    plt.tight_layout()  
    plt.savefig(out_png, dpi=180)  
    plt.close()  
  
  
def experiment_abstention_protocol(  
    dataset_name,  
    Xtr, ytr, etr,  
    Xte, yte, ete,  
    subset_fracs=(1.0, 0.7, 0.4, 0.2),  
    repeats=5,  
    num_avg=8,  
    q_eval=3,  
    num_perm=24,  
    gate_support=0.45,  
    gate_stability=0.55,  
    gate_mass=0.01,  
    label_damage_tol=0.02,  
    seed=0,  
):  
    records = []  
    trial_id = 0  
    r, q = choose_r_q(dataset_name, ytr, etr)  
    q = min(q, Xtr.shape[1] - 1)  
  
    full_label = fit_logreg_representation(Xtr, ytr, Xte, yte, seed=seed + 1)  
    full_env = fit_env_probe_quadratic(Xtr, etr, Xte, ete, seed=seed + 2)  
  
    for frac in subset_fracs:  
        for rep in range(repeats):  
            idx_sub = stratified_subsample_by_class(ytr, frac=float(frac), seed=seed + 100 * trial_id)  
            Xs, ys, es = Xtr[idx_sub], ytr[idx_sub], etr[idx_sub]  
  
            fit = fit_real_dmd_bagged_refit(Xs, ys, r=r, q=q, split_seed=seed + 2000 + trial_id, num_avg=num_avg)  
            Uhat = fit["Uhat"]  
            Vhat = fit["Vhat"][:, :min(q_eval, fit["Vhat"].shape[1])]  
            Vraw = fit["Vhat_raw"][:, :min(q_eval, fit["Vhat_raw"].shape[1])]  
  
            mass_rep = residual_mass_above_null(Xs, ys, Uhat, q=q_eval, num_perm=num_perm, seed=seed + 3000 + trial_id)  
            support = support_quality_labelonly(residualize_subspace(Xs, Uhat), ys, seed=seed + 3001 + trial_id)  
            stab = split_stability_dmd(  
                dataset_name, Xs, ys, es,  
                q_eval=q_eval,  
                repeats=min(4, repeats + 1),  
                subset_frac=0.85,  
                num_avg=min(num_avg, 6),  
                seed=seed + 3100 + trial_id,  
            )  
  
            M_test = empirical_M_plugin(Xte, yte)  
            T_test = empirical_T_perp_plugin(Xte, yte, Uhat, shrink_alpha=0.10)  
            T_env = empirical_T_env_oracle(Xte, yte, ete, Uhat, shrink_alpha=0.10)  
  
            dmd_ev = evaluate_v_subspace(Vhat, Xs, ys, es, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 3200 + trial_id)  
            raw_ev = evaluate_v_subspace(Vraw, Xs, ys, es, Xte, yte, ete, M_test, T_test, T_env, seed=seed + 3300 + trial_id)  
  
            always_env_red = float(full_env["acc"] - dmd_ev["env_acc_after"])  
            always_label_drop = float(full_label["acc"] - dmd_ev["label_acc_after"])  
            always_oracle_gain_vs_raw = float(dmd_ev["oracle_purity"] - raw_ev["oracle_purity"])  
  
            gate_pass = bool(  
                (support >= gate_support) and  
                (stab["split_stability"] >= gate_stability) and  
                (mass_rep["explained_spectral_mass_above_null"] >= gate_mass)  
            )  
  
            if gate_pass:  
                gated_env_red = always_env_red  
                gated_label_drop = always_label_drop  
                gated_oracle_gain_vs_raw = always_oracle_gain_vs_raw  
                action = "edit"  
            else:  
                gated_env_red = 0.0  
                gated_label_drop = 0.0  
                gated_oracle_gain_vs_raw = 0.0  
                action = "abstain"  
  
            always_harm = bool((always_env_red <= 0.0) or (always_label_drop > label_damage_tol))  
            gated_harm = bool((action == "edit") and ((gated_env_red <= 0.0) or (gated_label_drop > label_damage_tol)))  
  
            records.append({  
                "trial_id": int(trial_id),  
                "subset_frac": float(frac),  
                "subset_n": int(len(idx_sub)),  
                "gate_pass": int(gate_pass),  
                "action": action,  
                "support_quality": float(support),  
                "split_stability": float(stab["split_stability"]),  
                "split_instability": float(stab["split_instability"]),  
                "top3_residual_eig_sum": float(mass_rep["top3_obs"]),  
                "explained_spectral_mass_above_null": float(mass_rep["explained_spectral_mass_above_null"]),  
                "eigengap_ratio": float(residual_eigengap_ratio(fit["That"], q=q_eval)),  
                "always_env_leak_reduction_vs_full": float(always_env_red),  
                "always_label_acc_drop_vs_full": float(always_label_drop),  
                "always_oracle_purity_gain_vs_raw": float(always_oracle_gain_vs_raw),  
                "always_harm": int(always_harm),  
                "gated_env_leak_reduction_vs_full": float(gated_env_red),  
                "gated_label_acc_drop_vs_full": float(gated_label_drop),  
                "gated_oracle_purity_gain_vs_raw": float(gated_oracle_gain_vs_raw),  
                "gated_harm": int(gated_harm),  
            })  
            trial_id += 1  
  
    df = pd.DataFrame(records)  
    metrics = {  
        "always_mean_env_leak_reduction": float(df["always_env_leak_reduction_vs_full"].mean()) if len(df) > 0 else 0.0,  
        "gated_mean_env_leak_reduction": float(df["gated_env_leak_reduction_vs_full"].mean()) if len(df) > 0 else 0.0,  
        "always_mean_label_drop": float(df["always_label_acc_drop_vs_full"].mean()) if len(df) > 0 else 0.0,  
        "gated_mean_label_drop": float(df["gated_label_acc_drop_vs_full"].mean()) if len(df) > 0 else 0.0,  
        "always_harm_rate": float(df["always_harm"].mean()) if len(df) > 0 else 0.0,  
        "gated_harm_rate": float(df["gated_harm"].mean()) if len(df) > 0 else 0.0,  
        "gate_pass_rate": float(df["gate_pass"].mean()) if len(df) > 0 else 0.0,  
        "always_mean_oracle_gain_vs_raw": float(df["always_oracle_purity_gain_vs_raw"].mean()) if len(df) > 0 else 0.0,  
        "gated_mean_oracle_gain_vs_raw": float(df["gated_oracle_purity_gain_vs_raw"].mean()) if len(df) > 0 else 0.0,  
    }  
  
    claims = {  
        "AB1_gate_reduces_harm_rate": bool(metrics["gated_harm_rate"] <= metrics["always_harm_rate"] + 1e-12),  
        "AB2_gate_not_worse_label_damage": bool(metrics["gated_mean_label_drop"] <= metrics["always_mean_label_drop"] + 1e-12),  
    }  
  
    return {  
        "exp_name": "abstention_protocol",  
        "gate": {  
            "support_threshold": float(gate_support),  
            "stability_threshold": float(gate_stability),  
            "mass_threshold": float(gate_mass),  
            "label_damage_tol": float(label_damage_tol),  
        },  
        "metrics": metrics,  
        "claims": claims,  
        "trial_records": records,  
    }  
  
  
# ============================================================  
# Save helpers per dataset-backbone  
# ============================================================  
  
def save_dataset_backbone_results(base_dir, dataset_name, backbone_name, ds_report):  
    ds_dir = os.path.join(base_dir, f"{dataset_name}__{backbone_name}")  
    tables_dir = os.path.join(ds_dir, "tables")  
    arrays_dir = os.path.join(ds_dir, "arrays")  
    plots_dir = os.path.join(ds_dir, "plots")  
    ensure_dir(ds_dir)  
    ensure_dir(tables_dir)  
    ensure_dir(arrays_dir)  
    ensure_dir(plots_dir)  
  
    save_json(os.path.join(ds_dir, "report.json"), ds_report)  
  
    if "semantic_auditing" in ds_report:  
        sem = ds_report["semantic_auditing"]  
        save_csv_from_records(os.path.join(tables_dir, "semantic_auditing_methods.csv"), sem.get("methods_table", []))  
  
    if "residual_editing" in ds_report:  
        edit = ds_report["residual_editing"]  
        save_csv_from_records(os.path.join(tables_dir, "residual_editing_methods.csv"), edit.get("methods_table", []))  
        art = edit.get("fit_artifacts", {})  
        if len(art) > 0:  
            save_npz_safe(  
                os.path.join(arrays_dir, "residual_editing_fit_artifacts.npz"),  
                U_dmd=art.get("U_dmd", np.zeros((0, 0))),  
                V_dmd=art.get("V_dmd", np.zeros((0, 0))),  
                V_raw=art.get("V_raw", np.zeros((0, 0))),  
                That_dmd=art.get("That_dmd", np.zeros((0, 0))),  
                That_raw=art.get("That_raw", np.zeros((0, 0))),  
            )  
  
    if "bridge_diagnostics" in ds_report:  
        bridge = ds_report["bridge_diagnostics"]  
        save_csv_from_records(os.path.join(tables_dir, "bridge_diagnostics_trials.csv"), bridge.get("trial_records", []))  
        save_json(os.path.join(ds_dir, "bridge_diagnostics_summary.json"), bridge)  
  
    if "failure_case_diagnostics" in ds_report:  
        failure = ds_report["failure_case_diagnostics"]  
        save_csv_from_records(os.path.join(tables_dir, "failure_case_trials.csv"), failure.get("trial_records", []))  
        save_json(os.path.join(ds_dir, "failure_case_summary.json"), failure)  
  
    if "stability" in ds_report:  
        save_json(os.path.join(ds_dir, "stability_summary.json"), ds_report["stability"])  
  
    if "coverage_sweep" in ds_report:  
        cov = ds_report["coverage_sweep"]  
        if not cov.get("skipped", False):  
            save_csv_from_records(os.path.join(tables_dir, "coverage_sweep_trials.csv"), cov.get("trial_records", []))  
        save_json(os.path.join(ds_dir, "coverage_sweep_summary.json"), cov)  
  
    if "rank_selection" in ds_report:  
        save_json(os.path.join(ds_dir, "rank_selection_summary.json"), ds_report["rank_selection"])  
  
    if "baseline_positioning" in ds_report:  
        base = ds_report["baseline_positioning"]  
        base_rows = []  
        for m, rec in base.get("methods", {}).items():  
            if isinstance(rec, dict):  
                rr = {"method": m}  
                rr.update(rec)  
                base_rows.append(rr)  
        save_csv_from_records(os.path.join(tables_dir, "baseline_positioning_methods.csv"), base_rows)  
        save_json(os.path.join(ds_dir, "baseline_positioning_summary.json"), base)  
  
    if "editing_frontier" in ds_report:  
        pf = ds_report["editing_frontier"]  
        save_csv_from_records(os.path.join(tables_dir, "editing_frontier_curve.csv"), pf.get("curve_records", []))  
        save_csv_from_records(os.path.join(tables_dir, "editing_frontier_summary.csv"), pf.get("method_summary", []))  
        save_json(os.path.join(ds_dir, "editing_frontier_summary.json"), pf)  
        arts = pf.get("artifacts", {})  
        for mname, art in arts.items():  
            save_npz_safe(  
                os.path.join(arrays_dir, f"editing_frontier_{mname}.npz"),  
                U=art.get("U", np.zeros((0, 0))),  
                Vfull=art.get("Vfull", np.zeros((0, 0))),  
                T=art.get("T", np.zeros((0, 0))),  
            )  
        plot_pareto_curve(  
            pf.get("curve_records", []),  
            os.path.join(plots_dir, "editing_frontier.png"),  
            dataset_name,  
            backbone_name,  
        )  
  
    if "residual_diagnostics_E" in ds_report:  
        diag = ds_report["residual_diagnostics_E"]  
        save_csv_from_records(os.path.join(tables_dir, "residual_diagnostics_E.csv"), diag.get("methods_table", []))  
        save_json(os.path.join(ds_dir, "residual_diagnostics_E_summary.json"), diag)  
        arts = diag.get("artifacts", {})  
        for mname, art in arts.items():  
            save_npz_safe(  
                os.path.join(arrays_dir, f"diagnostics_E_{mname}.npz"),  
                U=art.get("U", np.zeros((0, 0))),  
                T=art.get("T", np.zeros((0, 0))),  
                eigvals=np.asarray(art.get("eigvals", np.zeros((0,)))),  
                null_top3_samples=np.asarray(art.get("null_top3_samples", np.zeros((0,)))),  
            )  
  
    if "abstention_protocol" in ds_report:  
        ab = ds_report["abstention_protocol"]  
        save_csv_from_records(os.path.join(tables_dir, "abstention_trials.csv"), ab.get("trial_records", []))  
        save_json(os.path.join(ds_dir, "abstention_summary.json"), ab)  
        plot_abstention_scatter(  
            ab.get("trial_records", []),  
            os.path.join(plots_dir, "abstention_scatter.png"),  
            dataset_name,  
            backbone_name,  
        )  
  
    if "representation_quality_control" in ds_report:  
        save_json(os.path.join(ds_dir, "representation_quality_control.json"), ds_report["representation_quality_control"])  
  
  
def append_summary_rows(summary_rows, dataset_name, backbone_name, exp_name, payload):  
    if exp_name == "semantic_auditing":  
        for m, rec in payload.get("methods", {}).items():  
            for key in ["acc", "nll", "ece", "sufficiency_gap_nll", "sufficiency_gap_acc"]:  
                if key in rec:  
                    summary_rows.append({  
                        "dataset": dataset_name,  
                        "backbone": backbone_name,  
                        "experiment": exp_name,  
                        "method": m,  
                        "metric": key,  
                        "value": rec[key],  
                    })  
  
    elif exp_name == "residual_editing":  
        for m, rec in payload.get("methods", {}).items():  
            for key in ["oracle_purity", "mean_capture", "env_leak_reduction_vs_full", "label_change_vs_full", "label_nll_change_vs_full"]:  
                if key in rec:  
                    summary_rows.append({  
                        "dataset": dataset_name,  
                        "backbone": backbone_name,  
                        "experiment": exp_name,  
                        "method": m,  
                        "metric": key,  
                        "value": rec[key],  
                    })  
  
    elif exp_name == "editing_frontier":  
        for rec in payload.get("method_summary", []):  
            m = rec["method"]  
            for key in ["pareto_area_label_retention", "pareto_area_neg_nll_deterioration", "best_env_leak_reduction", "best_label_retention", "best_oracle_purity"]:  
                summary_rows.append({  
                    "dataset": dataset_name,  
                    "backbone": backbone_name,  
                    "experiment": exp_name,  
                    "method": m,  
                    "metric": key,  
                    "value": rec[key],  
                })  
  
    elif exp_name == "residual_diagnostics_E":  
        for rec in payload.get("methods_table", []):  
            m = rec["method"]  
            for key in ["top3_residual_eig_sum", "explained_spectral_mass_above_null", "eigengap_ratio", "support_quality", "split_stability"]:  
                if key in rec:  
                    summary_rows.append({  
                        "dataset": dataset_name,  
                        "backbone": backbone_name,  
                        "experiment": exp_name,  
                        "method": m,  
                        "metric": key,  
                        "value": rec[key],  
                    })  
  
    elif exp_name == "abstention_protocol":  
        for key, val in payload.get("metrics", {}).items():  
            summary_rows.append({  
                "dataset": dataset_name,  
                "backbone": backbone_name,  
                "experiment": exp_name,  
                "method": "dmd_gate",  
                "metric": key,  
                "value": val,  
            })  
  
    else:  
        if "metrics" in payload and isinstance(payload["metrics"], dict):  
            for key, val in payload["metrics"].items():  
                if isinstance(val, (int, float, np.integer, np.floating)):  
                    summary_rows.append({  
                        "dataset": dataset_name,  
                        "backbone": backbone_name,  
                        "experiment": exp_name,  
                        "method": "summary",  
                        "metric": key,  
                        "value": float(val),  
                    })  
  
  
# ============================================================  
# Main  
# ============================================================  
  
def maybe_fast_override(args):  
    if not bool(args.fast):  
        return args  
    if args.max_train <= 0:  
        args.max_train = 4000  
    if args.max_test <= 0:  
        args.max_test = 2000  
    args.num_avg = min(args.num_avg, 6)  
    args.bridge_repeats = min(args.bridge_repeats, 5)  
    args.failure_repeats = min(args.failure_repeats, 3)  
    args.stability_repeats = min(args.stability_repeats, 6)  
    args.coverage_repeats = min(args.coverage_repeats, 2)  
    args.rank_perms = min(args.rank_perms, 24)  
    args.baseline_rand_repeats = min(args.baseline_rand_repeats, 4)  
    args.diagE_perms = min(args.diagE_perms, 16)  
    args.abstention_repeats = min(args.abstention_repeats, 3)  
    return args  
  
def resolve_run_seeds(args):  
    if getattr(args, "seed_list", None):  
        seeds = parse_csv_list(args.seed_list, cast=int)  
        if len(seeds) == 0:  
            raise ValueError("seed_list was provided but no valid seeds were parsed.")  
        return seeds  
  
    num_runs = max(1, int(getattr(args, "num_runs", 1)))  
    base_seed = int(getattr(args, "seed", 0))  
    seed_stride = int(getattr(args, "seed_stride", 1))  
    return [base_seed + i * seed_stride for i in range(num_runs)]  

def aggregate_multiseed_final_csvs(run_manifests, out_dir, ndigits=4):  
    ensure_dir(out_dir)  
  
    all_rows = []  
    for rec in run_manifests:  
        if rec.get("status", "") != "ok":  
            continue  
  
        csv_path = rec.get("final_report_csv", "")  
        seed = int(rec.get("seed", -1))  
  
        if (not csv_path) or (not os.path.exists(csv_path)):  
            continue  
  
        try:  
            df = pd.read_csv(csv_path)  
        except Exception:  
            continue  
  
        if len(df) == 0:  
            continue  
  
        required_cols = {"dataset", "backbone", "experiment", "method", "metric", "value"}  
        if not required_cols.issubset(set(df.columns)):  
            continue  
  
        df = df.copy()  
        df["seed"] = seed  
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  
        df = df.dropna(subset=["value"])  
  
        if len(df) == 0:  
            continue  
  
        all_rows.append(df)  
  
    out_csv = os.path.join(out_dir, "final_report_mean_std.csv")  
    out_json = os.path.join(out_dir, "final_report_mean_std.json")  
  
    if len(all_rows) == 0:  
        empty_df = pd.DataFrame(columns=[  
            "dataset", "backbone", "experiment", "method", "metric",  
            "mean", "std", "count", "mean_pm_std"  
        ])  
        empty_df.to_csv(out_csv, index=False)  
        save_json(out_json, {"rows": []}, ndigits=ndigits)  
        return {  
            "csv": out_csv,  
            "json": out_json,  
            "num_successful_runs": 0,  
        }  
  
    big_df = pd.concat(all_rows, axis=0, ignore_index=True)  
  
    group_cols = ["dataset", "backbone", "experiment", "method", "metric"]  
    agg_df = big_df.groupby(group_cols)["value"].agg(["mean", "std", "count"]).reset_index()  
    agg_df["std"] = agg_df["std"].fillna(0.0)  
  
    agg_df["mean_pm_std"] = agg_df.apply(  
        lambda r: f"{round(float(r['mean']), ndigits)} ± {round(float(r['std']), ndigits)}",  
        axis=1  
    )  
  
    for c in ["mean", "std"]:  
        agg_df[c] = agg_df[c].astype(float).round(ndigits)  
    agg_df["count"] = agg_df["count"].astype(int)  
  
    agg_df.to_csv(out_csv, index=False)  
  
    save_json(  
        out_json,  
        {  
            "num_successful_runs": int(len(set(big_df["seed"].tolist()))),  
            "rows": agg_df.to_dict(orient="records"),  
        },  
        ndigits=ndigits,  
    )  
  
    return {  
        "csv": out_csv,  
        "json": out_json,  
        "num_successful_runs": int(len(set(big_df["seed"].tolist()))),  
    }  
 
  
def run_one_seed(args, run_seed, run_root=None):  
    run_seed = int(run_seed)  
    set_seed(run_seed)  
  
    if run_root is None:  
        run_root = args.outdir  
  
    logger = SimpleLogger(  
        root_dir=run_root,  
        tag=f"labelonly_workflow_full_seed{run_seed}"  
    )  
    logger.print(f"[info] run_dir={logger.run_dir}")  
    logger.print(f"[info] run_seed={run_seed}")  
    logger.print(f"[info] datasets={args.datasets}")  
    logger.print(f"[info] backbones={args.backbone}, pretrained={args.pretrained}, fast={args.fast}")  
    logger.log({"event": "seed_start", "seed": run_seed})  
  
    dataset_names = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]  
    backbone_names = [x.strip() for x in args.backbone.split(",") if x.strip()]  
    failure_fracs = [float(x) for x in args.failure_fracs.split(",") if x.strip()]  
    frontier_qs = [int(x) for x in args.frontier_qs.split(",") if x.strip()]  
    abstention_fracs = [float(x) for x in args.abstention_fracs.split(",") if x.strip()]  
  
    final_report = {  
        "run_dir": logger.run_dir,  
        "seed": run_seed,  
        "datasets_requested": dataset_names,  
        "backbones_requested": backbone_names,  
        "pretrained": int(args.pretrained),  
        "config": {  
            "seed": run_seed,  
            "max_train": int(args.max_train),  
            "max_test": int(args.max_test),  
            "num_avg": int(args.num_avg),  
            "bridge_repeats": int(args.bridge_repeats),  
            "failure_repeats": int(args.failure_repeats),  
            "stability_repeats": int(args.stability_repeats),  
            "coverage_repeats": int(args.coverage_repeats),  
            "rank_perms": int(args.rank_perms),  
            "diagE_perms": int(args.diagE_perms),  
            "abstention_repeats": int(args.abstention_repeats),  
            "fast": int(args.fast),  
        },  
        "dataset_reports": {},  
        "aggregate_claims": {},  
    }  
  
    agg = defaultdict(list)  
    summary_rows = []  
  
    try:  
        for backbone_name in backbone_names:  
            for dataset_name in dataset_names:  
                combo_key = f"{dataset_name}__{backbone_name}"  
                logger.print(f"[dataset-backbone] start {combo_key} | seed={run_seed}")  
                try:  
                    Xtr_raw, ytr, etr, train_path = load_cached_features(  
                        data_root=args.data_root,  
                        dataset_name=dataset_name,  
                        split="train",  
                        backbone_name=backbone_name,  
                        pretrained_flag=bool(args.pretrained),  
                    )  
                    Xte_raw, yte, ete, test_path = load_cached_features(  
                        data_root=args.data_root,  
                        dataset_name=dataset_name,  
                        split="test",  
                        backbone_name=backbone_name,  
                        pretrained_flag=bool(args.pretrained),  
                    )  
                    logger.print(f"[cache] {combo_key} train={train_path}")  
                    logger.print(f"[cache] {combo_key} test={test_path}")  
  
                    if args.max_train > 0:  
                        Xtr_raw, ytr, etr = subsample_if_needed(  
                            Xtr_raw, ytr, etr, max_n=args.max_train, seed=run_seed + 11  
                        )  
                    if args.max_test > 0:  
                        Xte_raw, yte, ete = subsample_if_needed(  
                            Xte_raw, yte, ete, max_n=args.max_test, seed=run_seed + 17  
                        )  
  
                    pca_dim = choose_pca_dim(dataset_name)  
                    Xtr, Xte = preprocess_real_features(  
                        Xtr_raw, Xte_raw, pca_dim=pca_dim, whiten=True, l2norm=True  
                    )  
  
                    ds_report = {  
                        "status": "ok",  
                        "dataset": dataset_name,  
                        "backbone": backbone_name,  
                        "seed": run_seed,  
                        "cache_paths": {"train": train_path, "test": test_path},  
                        "meta": {  
                            "num_train": int(len(ytr)),  
                            "num_test": int(len(yte)),  
                            "num_classes": int(unique_count(ytr)),  
                            "num_envs": int(unique_count(etr)),  
                            "feature_dim_after_preprocess": int(Xtr.shape[1]),  
                            "pca_dim_requested": int(pca_dim),  
                        },  
                    }  
  
                    # E1 Semantic auditing  
                    sem = experiment_semantic_auditing(  
                        dataset_name, Xtr, ytr, Xte, yte, seed=run_seed + 0  
                    )  
                    ds_report["semantic_auditing"] = sem  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "semantic_auditing", sem)  
                    for k, v in sem["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(f"[done] {combo_key} semantic auditing | seed={run_seed}")  
  
                    # E2 Residual editing  
                    edit = experiment_residual_editing(  
                        dataset_name, Xtr, ytr, etr, Xte, yte, ete,  
                        seed=run_seed + 100, num_avg=args.num_avg  
                    )  
                    ds_report["residual_editing"] = edit  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "residual_editing", edit)  
                    for k, v in edit["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(  
                        f"[done] {combo_key} residual editing | seed={run_seed} "  
                        f"B1={edit['claims']['B1_dmd_purer_than_raw']} "  
                        f"B2={edit['claims']['B2_dmd_less_semantic_leak_than_raw']} "  
                        f"B3={edit['claims']['B3_dmd_better_leak_reduction_than_rand']} "  
                        f"B4={edit['claims']['B4_dmd_label_retention_vs_raw']}"  
                    )  
  
                    # NEW A: editing frontier / Pareto  
                    if bool(args.do_frontier):  
                        frontier = experiment_editing_frontier(  
                            dataset_name,  
                            Xtr, ytr, etr,  
                            Xte, yte, ete,  
                            q_grid=frontier_qs,  
                            num_avg=args.num_avg,  
                            seed=run_seed + 120,  
                        )  
                        ds_report["editing_frontier"] = frontier  
                        append_summary_rows(summary_rows, dataset_name, backbone_name, "editing_frontier", frontier)  
                        for k, v in frontier["claims"].items():  
                            agg[k].append(bool(v))  
                        logger.print(  
                            f"[done] {combo_key} frontier | seed={run_seed} "  
                            f"PF1={frontier['claims']['PF1_dmd_frontier_area_vs_raw']} "  
                            f"PF2={frontier['claims']['PF2_dmd_frontier_area_vs_pca']} "  
                            f"PF3={frontier['claims']['PF3_dmd_frontier_area_vs_rand']}"  
                        )  
  
                    # NEW E: residual diagnostics  
                    if bool(args.do_diagE):  
                        diagE = experiment_residual_diagnostics_E(  
                            dataset_name,  
                            Xtr, ytr, etr,  
                            num_avg=args.num_avg,  
                            q_eval=args.diagE_q,  
                            num_perm=args.diagE_perms,  
                            split_repeats=args.diagE_split_repeats,  
                            seed=run_seed + 140,  
                        )  
                        ds_report["residual_diagnostics_E"] = diagE  
                        append_summary_rows(summary_rows, dataset_name, backbone_name, "residual_diagnostics_E", diagE)  
                        for k, v in diagE["claims"].items():  
                            agg[k].append(bool(v))  
                        logger.print(  
                            f"[done] {combo_key} diagE | seed={run_seed} "  
                            f"E1={diagE['claims']['E1_dmd_above_null_positive']}"  
                        )  
  
                    # E3 Bridge diagnostics  
                    bridge = experiment_bridge_diagnostics(  
                        dataset_name,  
                        Xtr, ytr, etr,  
                        Xte, yte, ete,  
                        repeats=args.bridge_repeats,  
                        fit_frac=args.bridge_fitfrac,  
                        seed=run_seed + 200,  
                    )  
                    ds_report["bridge_diagnostics"] = bridge  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "bridge_diagnostics", bridge)  
                    for k, v in bridge["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(  
                        f"[done] {combo_key} bridge | seed={run_seed} "  
                        f"D1={bridge['claims']['D1_proxy_tracks_oracle_gain']} "  
                        f"D2={bridge['claims']['D2_proxy_tracks_env_leakage']}"  
                    )  
  
                    # E4 Failure cases  
                    failure = experiment_failure_cases(  
                        dataset_name,  
                        Xtr, ytr, etr,  
                        Xte, yte, ete,  
                        subset_fracs=tuple(failure_fracs),  
                        repeats=args.failure_repeats,  
                        fit_frac=args.failure_fitfrac,  
                        seed=run_seed + 300,  
                    )  
                    ds_report["failure_case_diagnostics"] = failure  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "failure_case_diagnostics", failure)  
                    for k, v in failure["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(  
                        f"[done] {combo_key} failure | seed={run_seed} "  
                        f"F1={failure['claims']['F1_proxy_separates_failure_regime']}"  
                    )  
  
                    # E5 Stability  
                    stability = experiment_stability(  
                        dataset_name,  
                        Xtr, ytr, etr,  
                        Xte, yte, ete,  
                        repeats=args.stability_repeats,  
                        num_avg=args.num_avg,  
                        seed=run_seed + 400,  
                    )  
                    ds_report["stability"] = stability  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "stability", stability)  
                    for k, v in stability["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(  
                        f"[done] {combo_key} stability | seed={run_seed} "  
                        f"S1={stability['claims']['S1_bagging_stabilizes_V']} "  
                        f"S2={stability['claims']['S2_bagging_stabilizes_practical_metrics']}"  
                    )  
  
                    # E6 Waterbirds coverage sweep  
                    if dataset_name == "waterbirds":  
                        coverage = experiment_waterbirds_coverage_sweep(  
                            Xtr, ytr, etr,  
                            Xte, yte, ete,  
                            num_avg=args.num_avg,  
                            seed=run_seed + 500,  
                        )  
                        ds_report["coverage_sweep"] = coverage  
                        append_summary_rows(summary_rows, dataset_name, backbone_name, "coverage_sweep", coverage)  
                        if not coverage.get("skipped", False):  
                            for k, v in coverage["claims"].items():  
                                agg[k].append(bool(v))  
                            logger.print(  
                                f"[done] {combo_key} coverage | seed={run_seed} "  
                                f"W1={coverage['claims']['W1_support_quality_tracks_oracle_purity']} "  
                                f"W2={coverage['claims']['W2_support_quality_tracks_env_leak_reduction']}"  
                            )  
  
                    # E7 Rank selection  
                    ranksel = experiment_rank_selection(  
                        dataset_name,  
                        Xtr, ytr, etr,  
                        Xte, yte, ete,  
                        num_avg=args.num_avg,  
                        num_perm=args.rank_perms,  
                        alpha=args.rank_alpha,  
                        max_rank=args.rank_maxeig,  
                        seed=run_seed + 600,  
                    )  
                    ds_report["rank_selection"] = ranksel  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "rank_selection", ranksel)  
                    for k, v in ranksel["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(  
                        f"[done] {combo_key} rank | seed={run_seed} "  
                        f"heur=({ranksel['heuristic_r']},{ranksel['heuristic_q']}) "  
                        f"sel=({ranksel['selected_r']},{ranksel['selected_q']}) "  
                        f"Rk1={ranksel['claims']['Rk1_permnull_not_worse_than_heuristic']}"  
                    )  
  
                    # E8 Baseline positioning  
                    basepos = experiment_baseline_positioning(  
                        dataset_name,  
                        Xtr, ytr, etr,  
                        Xte, yte, ete,  
                        num_avg=args.num_avg,  
                        rand_repeats=args.baseline_rand_repeats,  
                        seed=run_seed + 700,  
                    )  
                    ds_report["baseline_positioning"] = basepos  
                    append_summary_rows(summary_rows, dataset_name, backbone_name, "baseline_positioning", basepos)  
                    for k, v in basepos["claims"].items():  
                        agg[k].append(bool(v))  
                    logger.print(  
                        f"[done] {combo_key} baseline | seed={run_seed} "  
                        f"P1={basepos['claims']['P1_dmd_stronger_than_raw_and_rand_on_oracle_purity']} "  
                        f"P2={basepos['claims']['P2_dmd_better_than_pca_on_leakage']}"  
                    )  
  
                    # NEW B: Abstention protocol  
                    if bool(args.do_abstention):  
                        abst = experiment_abstention_protocol(  
                            dataset_name,  
                            Xtr, ytr, etr,  
                            Xte, yte, ete,  
                            subset_fracs=tuple(abstention_fracs),  
                            repeats=args.abstention_repeats,  
                            num_avg=args.num_avg,  
                            q_eval=args.diagE_q,  
                            num_perm=args.diagE_perms,  
                            gate_support=args.abstention_support_thresh,  
                            gate_stability=args.abstention_stability_thresh,  
                            gate_mass=args.abstention_mass_thresh,  
                            label_damage_tol=args.abstention_label_damage_tol,  
                            seed=run_seed + 820,  
                        )  
                        ds_report["abstention_protocol"] = abst  
                        append_summary_rows(summary_rows, dataset_name, backbone_name, "abstention_protocol", abst)  
                        for k, v in abst["claims"].items():  
                            agg[k].append(bool(v))  
                        logger.print(  
                            f"[done] {combo_key} abstention | seed={run_seed} "  
                            f"AB1={abst['claims']['AB1_gate_reduces_harm_rate']} "  
                            f"AB2={abst['claims']['AB2_gate_not_worse_label_damage']}"  
                        )  
  
                    final_report["dataset_reports"][combo_key] = ds_report  
                    save_dataset_backbone_results(logger.run_dir, dataset_name, backbone_name, ds_report)  
  
                except Exception as ex:  
                    final_report["dataset_reports"][combo_key] = {  
                        "status": "error",  
                        "dataset": dataset_name,  
                        "backbone": backbone_name,  
                        "seed": run_seed,  
                        "error": str(ex),  
                    }  
                    logger.print(f"[dataset-backbone] error {combo_key} | seed={run_seed}: {str(ex)}")  
  
        # Random control  
        if bool(args.do_random_control):  
            final_report["random_control"] = {}  
            for backbone_name in backbone_names:  
                for dataset_name in [d for d in dataset_names if d in ["waterbirds", "pacs", "celeba"]]:  
                    rc_key = f"{dataset_name}__{backbone_name}"  
                    logger.print(f"[random-control] {rc_key} | seed={run_seed}")  
                    rc = experiment_representation_quality_control(  
                        data_root=args.data_root,  
                        dataset_name=dataset_name,  
                        backbone_name=backbone_name,  
                        max_train=args.max_train,  
                        max_test=args.max_test,  
                        seed=run_seed + 900,  
                    )  
                    final_report["random_control"][rc_key] = rc  
                    if rc.get("status", "ok") == "ok":  
                        for k, v in rc.get("claims", {}).items():  
                            agg[k].append(bool(v))  
                        logger.print(  
                            f"[done] random-control {rc_key} | seed={run_seed} "  
                            f"Q1={rc['claims']['Q1_pretrained_better_stage1_than_random']} "  
                            f"Q2={rc['claims']['Q2_pretrained_better_stage2_than_random']}"  
                        )  
                    else:  
                        logger.print(f"[skip] random-control {rc_key} | seed={run_seed}: {rc.get('error', 'unknown')}")  
  
        for k, vals in agg.items():  
            final_report["aggregate_claims"][k] = bool(sum(vals) > (len(vals) / 2.0)) if len(vals) > 0 else False  
  
        logger.save_report(final_report)  
  
        if len(summary_rows) > 0:  
            save_csv_from_records(logger.final_report_csv_path, summary_rows, ndigits=4)  
  
        logger.print("[final] aggregate claims")  
        for k in sorted(final_report["aggregate_claims"].keys()):  
            logger.print(f"[final] {k}: {'PASS' if final_report['aggregate_claims'][k] else 'FAIL'}")  
        logger.print(f"[final] seed={run_seed}")  
        logger.print(f"[final] json report saved to {logger.final_report_path}")  
        logger.print(f"[final] csv report saved to {logger.final_report_csv_path}")  
        logger.print(f"[final] text log saved to {logger.log_path}")  
  
        return {  
            "seed": run_seed,  
            "run_dir": logger.run_dir,  
            "final_report_json": logger.final_report_path,  
            "final_report_csv": logger.final_report_csv_path,  
            "log_path": logger.log_path,  
            "status": "ok",  
        }  
  
    finally:  
        logger.close()  
        

def main():  
    parser = argparse.ArgumentParser()  
  
    parser.add_argument("--data_root", type=str, default="./data")  
    parser.add_argument("--datasets", type=str, default="celeba,waterbirds,pacs")  
    parser.add_argument("--backbone", type=str, default="resnet18")  
    parser.add_argument("--pretrained", type=int, default=1)  
  
    parser.add_argument("--max_train", type=int, default=0)  
    parser.add_argument("--max_test", type=int, default=0)  
  
    parser.add_argument("--num_avg", type=int, default=8)  
    parser.add_argument("--bridge_repeats", type=int, default=8)  
    parser.add_argument("--bridge_fitfrac", type=float, default=0.7)  
    parser.add_argument("--failure_repeats", type=int, default=4)  
    parser.add_argument("--failure_fracs", type=str, default="0.20,0.35,0.50,0.70,0.90,1.00")  
    parser.add_argument("--failure_fitfrac", type=float, default=0.7)  
    parser.add_argument("--stability_repeats", type=int, default=8)  
    parser.add_argument("--coverage_repeats", type=int, default=3)  
    parser.add_argument("--rank_perms", type=int, default=32)  
    parser.add_argument("--rank_alpha", type=float, default=0.95)  
    parser.add_argument("--rank_maxeig", type=int, default=20)  
    parser.add_argument("--baseline_rand_repeats", type=int, default=6)  
  
    parser.add_argument("--do_random_control", type=int, default=0)  
    parser.add_argument("--fast", type=int, default=0)  
  
    # multi-seed controls  
    parser.add_argument("--seed", type=int, default=0)  
    parser.add_argument("--num_runs", type=int, default=1)  
    parser.add_argument("--seed_stride", type=int, default=1)  
    parser.add_argument("--seed_list", type=str, default="")  
  
    # new toggles  
    parser.add_argument("--do_frontier", type=int, default=1)  
    parser.add_argument("--do_abstention", type=int, default=1)  
    parser.add_argument("--do_diagE", type=int, default=1)  
  
    parser.add_argument("--frontier_qs", type=str, default="1,2,3,5,8,12")  
  
    parser.add_argument("--diagE_q", type=int, default=3)  
    parser.add_argument("--diagE_perms", type=int, default=24)  
    parser.add_argument("--diagE_split_repeats", type=int, default=5)  
  
    parser.add_argument("--abstention_fracs", type=str, default="1.0,0.7,0.4,0.2")  
    parser.add_argument("--abstention_repeats", type=int, default=5)  
    parser.add_argument("--abstention_support_thresh", type=float, default=0.45)  
    parser.add_argument("--abstention_stability_thresh", type=float, default=0.55)  
    parser.add_argument("--abstention_mass_thresh", type=float, default=0.01)  
    parser.add_argument("--abstention_label_damage_tol", type=float, default=0.02)  
  
    parser.add_argument("--outdir", type=str, default="runs_labelonly_workflow_full")  
  
    args = parser.parse_args()  
    args = maybe_fast_override(args)  
  
    seeds = resolve_run_seeds(args)  
  
    if len(seeds) > 1:  
        multi_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  
        multi_root = os.path.join(args.outdir, f"multiseed_{multi_ts}")  
    else:  
        multi_root = args.outdir  
  
    ensure_dir(multi_root)  
  
    print(f"[multi-seed] seeds={seeds}", flush=True)  
    print(f"[multi-seed] output_root={multi_root}", flush=True)  
  
    run_manifests = []  
    for s in seeds:  
        print(f"[multi-seed] start seed={s}", flush=True)  
        try:  
            manifest = run_one_seed(args, run_seed=s, run_root=multi_root)  
        except Exception as ex:  
            manifest = {  
                "seed": int(s),  
                "run_dir": "",  
                "final_report_json": "",  
                "final_report_csv": "",  
                "log_path": "",  
                "status": "error",  
                "error": str(ex),  
            }  
            print(f"[multi-seed] error seed={s}: {str(ex)}", flush=True)  
        run_manifests.append(manifest)  
        print(f"[multi-seed] done seed={s} status={manifest.get('status', 'unknown')}", flush=True)  
  
        agg_paths = aggregate_multiseed_final_csvs(run_manifests, multi_root, ndigits=4)  
  
    multiseed_report = {  
        "multi_seed_root": multi_root,  
        "seeds": [int(s) for s in seeds],  
        "num_runs": int(len(seeds)),  
        "runs": run_manifests,  
        "aggregate_files": agg_paths,  
    }  
  
    save_json(os.path.join(multi_root, "multiseed_manifest.json"), multiseed_report)  
    save_csv_from_records(os.path.join(multi_root, "multiseed_manifest.csv"), run_manifests)  
  
    print(f"[multi-seed] manifest json saved to {os.path.join(multi_root, 'multiseed_manifest.json')}", flush=True)  
    print(f"[multi-seed] manifest csv saved to {os.path.join(multi_root, 'multiseed_manifest.csv')}", flush=True)  
    print(f"[multi-seed] aggregate mean±std csv saved to {agg_paths['csv']}", flush=True)  
    print(f"[multi-seed] aggregate mean±std json saved to {agg_paths['json']}", flush=True)  
  
if __name__ == "__main__":  
    main()  

