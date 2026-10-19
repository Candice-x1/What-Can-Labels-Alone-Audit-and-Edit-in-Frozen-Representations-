import os
import re
import json
import hashlib
import argparse
import random
import platform
import importlib.metadata

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# A fixed, transparent lexicon for an evaluation-only
# identity-mention proxy. These are text mentions, not the
# demographic identity of the commenter.
IDENTITY_MENTION_TERMS = [
    # Gender
    "man",
    "men",
    "male",
    "boy",
    "boys",
    "father",
    "husband",
    "woman",
    "women",
    "female",
    "girl",
    "girls",
    "mother",
    "wife",

    # Sexual orientation / gender identity
    "gay",
    "gays",
    "lesbian",
    "lesbians",
    "bisexual",
    "bisexuals",
    "homosexual",
    "homosexuals",
    "transgender",
    "transsexual",
    "nonbinary",
    "non-binary",
    "queer",
    "lgbt",
    "lgbtq",

    # Religion
    "christian",
    "christians",
    "christianity",
    "muslim",
    "muslims",
    "islam",
    "islamic",
    "jew",
    "jews",
    "jewish",
    "judaism",
    "hindu",
    "hindus",
    "hinduism",
    "buddhist",
    "buddhists",
    "buddhism",
    "sikh",
    "sikhs",
    "sikhism",
    "atheist",
    "atheists",
    "atheism",

    # Race / ethnicity / nationality-group mentions
    "black",
    "blacks",
    "white",
    "whites",
    "asian",
    "asians",
    "latino",
    "latinos",
    "latina",
    "latinas",
    "hispanic",
    "hispanics",
    "african american",
    "african americans",
    "native american",
    "native americans",
    "indigenous",
    "arab",
    "arabs",
    "middle eastern",

    # Disability / mental-health identity mentions
    "disabled",
    "disability",
    "autistic",
    "autism",
    "blind",
    "deaf",
    "mentally ill",
    "mental illness",
]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def numeric_array(values):
    out = np.full(len(values), np.nan, dtype=np.float32)

    for i, value in enumerate(values):
        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        if np.isfinite(value):
            out[i] = value

    return out


def compile_identity_regex(terms):
    """
    Compile a case-insensitive identity-term matcher.

    Multi-word phrases permit arbitrary whitespace between words.
    Word boundaries are implemented using non-word lookarounds.
    """
    unique_terms = sorted(
        set(str(term).strip().lower() for term in terms),
        key=lambda x: (-len(x), x),
    )

    patterns = []

    for term in unique_terms:
        escaped = re.escape(term)
        escaped = escaped.replace(r"\ ", r"\s+")
        patterns.append(escaped)

    pattern = (
        r"(?<!\w)(?:"
        + "|".join(patterns)
        + r")(?!\w)"
    )

    return re.compile(pattern, flags=re.IGNORECASE)


def identity_mention_statistics(texts, regex):
    """
    Return:
      - binary identity-mention proxy score;
      - number of lexicon matches.

    E=1 iff at least one lexicon term is matched.
    """
    mention_score = np.zeros(
        len(texts),
        dtype=np.float32,
    )
    match_count = np.zeros(
        len(texts),
        dtype=np.int32,
    )

    for i, text in enumerate(texts):
        matches = regex.findall(text)
        count = len(matches)

        match_count[i] = count
        mention_score[i] = 1.0 if count > 0 else 0.0

    return mention_score, match_count


def select_uniform_indices(n_total, n_samples, seed):
    """
    Uniform random sample from the official split.

    Neither Y nor E is used for sampling.
    """
    if n_total <= 0:
        raise ValueError("The requested split is empty.")

    if (
        n_samples is None
        or int(n_samples) <= 0
        or int(n_samples) >= n_total
    ):
        return np.arange(n_total, dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    indices = rng.choice(
        n_total,
        size=int(n_samples),
        replace=False,
    )

    return np.sort(indices.astype(np.int64))


def prepare_split(
    split,
    split_name,
    n_samples,
    sample_seed,
    toxicity_threshold,
    identity_threshold,
    identity_regex,
):
    """
    Uniformly sample from an official split, then construct:
      Y = thresholded official toxicity score;
      E = lexicon-defined identity-mention proxy.

    No Y/E balancing or conditioning is used.
    """
    n_total = len(split)

    selected_indices = select_uniform_indices(
        n_total=n_total,
        n_samples=n_samples,
        seed=sample_seed,
    )

    subset = split.select(selected_indices.tolist())

    texts = [
        "" if value is None else str(value)
        for value in subset["text"]
    ]

    if any(len(text.strip()) == 0 for text in texts):
        raise ValueError(
            f"{split_name}: an empty text was encountered "
            "in the uniformly sampled subset."
        )

    toxicity = numeric_array(subset["toxicity"])

    if not np.all(np.isfinite(toxicity)):
        bad = int(np.sum(~np.isfinite(toxicity)))
        raise ValueError(
            f"{split_name}: {bad} selected examples have "
            "missing/non-finite toxicity scores."
        )

    y = (
        toxicity >= float(toxicity_threshold)
    ).astype(np.int64)

    mention_score, match_count = (
        identity_mention_statistics(
            texts,
            identity_regex,
        )
    )

    e = (
        mention_score >= float(identity_threshold)
    ).astype(np.int64)

    return {
        "texts": texts,
        "y": y,
        "e": e,
        "toxicity": toxicity.astype(np.float32),
        "identity_mention_score":
            mention_score.astype(np.float32),
        "identity_match_count":
            match_count.astype(np.int32),
        "selected_indices": selected_indices,
        "num_total_rows": int(n_total),
    }


def count_values(values):
    unique, counts = np.unique(
        values,
        return_counts=True,
    )

    return {
        str(int(k)): int(v)
        for k, v in zip(unique, counts)
    }


def compute_cell_counts(y, e):
    counts = {}

    for yy in [0, 1]:
        for ee in [0, 1]:
            counts[f"y={yy},e={ee}"] = int(
                np.sum((y == yy) & (e == ee))
            )

    return counts


def validate_split(prepared, split_name, min_cell_count):
    y = prepared["y"]
    e = prepared["e"]

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{split_name}: only one toxicity class "
            "was observed."
        )

    if len(np.unique(e)) < 2:
        raise ValueError(
            f"{split_name}: only one identity-mention "
            "proxy class was observed."
        )

    cell_counts = compute_cell_counts(y, e)
    smallest_cell = min(cell_counts.values())

    print(
        f"[{split_name}] official split rows="
        f"{prepared['num_total_rows']}",
        flush=True,
    )
    print(
        f"[{split_name}] selected n={len(y)}",
        flush=True,
    )
    print(
        f"[{split_name}] y counts={count_values(y)}",
        flush=True,
    )
    print(
        f"[{split_name}] e counts={count_values(e)}",
        flush=True,
    )
    print(
        f"[{split_name}] y/e cells={cell_counts}",
        flush=True,
    )
    print(
        f"[{split_name}] identity-match count: "
        f"mean={prepared['identity_match_count'].mean():.4f}, "
        f"max={prepared['identity_match_count'].max()}",
        flush=True,
    )

    if smallest_cell < int(min_cell_count):
        raise ValueError(
            f"{split_name}: minimum y/e cell count is "
            f"{smallest_cell}, below --min_cell_count="
            f"{min_cell_count}. Increase n_train/n_test. "
            "Do not rebalance using E."
        )

    return cell_counts


def resolve_device(requested_device):
    requested_device = str(requested_device).strip().lower()

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but "
                "torch.cuda.is_available() is False."
            )

    return requested_device


def load_frozen_encoder(model_name, device, max_length):
    print(
        f"[encoder] loading {model_name} on {device}",
        flush=True,
    )

    model = SentenceTransformer(
        model_name,
        device=device,
    )
    model.max_seq_length = int(max_length)
    model.eval()

    embedding_dim = (
        model.get_sentence_embedding_dimension()
    )

    print(
        f"[encoder] embedding dimension={embedding_dim}",
        flush=True,
    )
    print(
        f"[encoder] max sequence length="
        f"{model.max_seq_length}",
        flush=True,
    )

    return model


def encode_texts(model, texts, batch_size, split_name):
    print(
        f"[encoder] encoding {split_name}: n={len(texts)}",
        flush=True,
    )

    embeddings = model.encode(
        texts,
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"{split_name}: expected 2-D embeddings, "
            f"got {embeddings.shape}."
        )

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"{split_name}: number of embeddings does "
            "not equal number of texts."
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            f"{split_name}: non-finite embeddings found."
        )

    return embeddings


def save_feature_cache(
    path,
    X,
    prepared,
    compress=False,
):
    ensure_dir(os.path.dirname(path))

    payload = {
        # Fields required by the existing label-only pipeline
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(
            prepared["y"],
            dtype=np.int64,
        ),
        "e": np.asarray(
            prepared["e"],
            dtype=np.int64,
        ),

        # Reproducibility and audit fields
        "source_indices": np.asarray(
            prepared["selected_indices"],
            dtype=np.int64,
        ),
        "toxicity_score": np.asarray(
            prepared["toxicity"],
            dtype=np.float32,
        ),
        "identity_mention_score": np.asarray(
            prepared["identity_mention_score"],
            dtype=np.float32,
        ),
        "identity_match_count": np.asarray(
            prepared["identity_match_count"],
            dtype=np.int32,
        ),
    }

    if bool(compress):
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)

    size_mb = os.path.getsize(path) / (1024.0 ** 2)

    print(
        f"[save] {path}",
        flush=True,
    )
    print(
        f"[save] X shape={X.shape}, "
        f"size={size_mb:.1f} MB",
        flush=True,
    )


def verify_feature_cache(path, expected_n, expected_dim):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved cache was not found: {path}"
        )

    with np.load(path) as cache:
        required = {
            "X",
            "y",
            "e",
            "source_indices",
            "toxicity_score",
            "identity_mention_score",
            "identity_match_count",
        }

        missing = required - set(cache.files)

        if missing:
            raise ValueError(
                f"{path}: missing fields {sorted(missing)}"
            )

        X = cache["X"]
        y = cache["y"]
        e = cache["e"]

        expected_shape = (
            int(expected_n),
            int(expected_dim),
        )

        if X.shape != expected_shape:
            raise ValueError(
                f"{path}: X shape={X.shape}; expected "
                f"{expected_shape}."
            )

        if y.shape != (int(expected_n),):
            raise ValueError(
                f"{path}: unexpected y shape={y.shape}."
            )

        if e.shape != (int(expected_n),):
            raise ValueError(
                f"{path}: unexpected e shape={e.shape}."
            )

        if X.dtype != np.float32:
            raise ValueError(
                f"{path}: expected X float32; "
                f"got {X.dtype}."
            )

        if y.dtype != np.int64:
            raise ValueError(
                f"{path}: expected y int64; "
                f"got {y.dtype}."
            )

        if e.dtype != np.int64:
            raise ValueError(
                f"{path}: expected e int64; "
                f"got {e.dtype}."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"{path}: non-finite embeddings found."
            )

        if not set(np.unique(y).tolist()).issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{path}: y is not binary."
            )

        if not set(np.unique(e).tolist()).issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{path}: e is not binary."
            )

    print(
        f"[verify] cache verified: {path}",
        flush=True,
    )


def sha256_int_array(values):
    values = np.ascontiguousarray(
        np.asarray(values, dtype=np.int64)
    )

    return hashlib.sha256(
        values.tobytes()
    ).hexdigest()


def load_dataset_robust(
    dataset_id,
    dataset_config,
    hf_cache_dir,
):
    kwargs = {}

    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir

    if dataset_config:
        return load_dataset(
            dataset_id,
            dataset_config,
            **kwargs,
        )

    return load_dataset(
        dataset_id,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="google/civil_comments",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=(
            "sentence-transformers/"
            "all-mpnet-base-v2"
        ),
    )
    parser.add_argument(
        "--backbone_tag",
        type=str,
        default="all_mpnet_base_v2",
    )

    parser.add_argument(
        "--n_train",
        type=int,
        default=40000,
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--toxicity_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--identity_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_cell_count",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    print(
        f"[dataset] loading {args.dataset_id}",
        flush=True,
    )

    dataset = load_dataset_robust(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        hf_cache_dir=args.hf_cache_dir,
    )

    available_splits = list(dataset.keys())

    print(
        f"[dataset] available splits={available_splits}",
        flush=True,
    )

    if args.train_split not in dataset:
        raise KeyError(
            f"Missing train split '{args.train_split}'. "
            f"Available splits: {available_splits}"
        )

    if args.test_split not in dataset:
        if "validation" in dataset:
            print(
                f"[warning] split '{args.test_split}' "
                "not found; using validation.",
                flush=True,
            )
            args.test_split = "validation"
        else:
            raise KeyError(
                f"Missing test split '{args.test_split}'. "
                f"Available splits: {available_splits}"
            )

    train_split = dataset[args.train_split]
    test_split = dataset[args.test_split]

    print(
        f"[dataset] train columns="
        f"{train_split.column_names}",
        flush=True,
    )

    required_columns = {"text", "toxicity"}

    for split_name, split in [
        (args.train_split, train_split),
        (args.test_split, test_split),
    ]:
        missing = (
            required_columns
            - set(split.column_names)
        )

        if missing:
            raise KeyError(
                f"{split_name}: missing required "
                f"columns {sorted(missing)}."
            )

    identity_regex = compile_identity_regex(
        IDENTITY_MENTION_TERMS
    )

    print(
        f"[environment] using fixed lexicon-defined "
        f"identity-mention proxy with "
        f"{len(IDENTITY_MENTION_TERMS)} terms",
        flush=True,
    )

    train = prepare_split(
        split=train_split,
        split_name=args.train_split,
        n_samples=args.n_train,
        sample_seed=args.seed + 11,
        toxicity_threshold=args.toxicity_threshold,
        identity_threshold=args.identity_threshold,
        identity_regex=identity_regex,
    )

    test = prepare_split(
        split=test_split,
        split_name=args.test_split,
        n_samples=args.n_test,
        sample_seed=args.seed + 17,
        toxicity_threshold=args.toxicity_threshold,
        identity_threshold=args.identity_threshold,
        identity_regex=identity_regex,
    )

    train_cells = validate_split(
        train,
        "train",
        min_cell_count=args.min_cell_count,
    )

    test_cells = validate_split(
        test,
        "test",
        min_cell_count=args.min_cell_count,
    )

    model = load_frozen_encoder(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
    )

    X_train = encode_texts(
        model=model,
        texts=train["texts"],
        batch_size=args.batch_size,
        split_name="train",
    )

    X_test = encode_texts(
        model=model,
        texts=test["texts"],
        batch_size=args.batch_size,
        split_name="test",
    )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Train and test embedding dimensions differ."
        )

    cache_dir = os.path.join(
        args.data_root,
        "feature_cache",
    )
    ensure_dir(cache_dir)

    train_cache = os.path.join(
        cache_dir,
        (
            f"civilcomments_train_"
            f"{args.backbone_tag}_pt1.npz"
        ),
    )

    test_cache = os.path.join(
        cache_dir,
        (
            f"civilcomments_test_"
            f"{args.backbone_tag}_pt1.npz"
        ),
    )

    save_feature_cache(
        path=train_cache,
        X=X_train,
        prepared=train,
        compress=bool(args.compress),
    )

    save_feature_cache(
        path=test_cache,
        X=X_test,
        prepared=test,
        compress=bool(args.compress),
    )

    verify_feature_cache(
        path=train_cache,
        expected_n=len(train["y"]),
        expected_dim=X_train.shape[1],
    )

    verify_feature_cache(
        path=test_cache,
        expected_n=len(test["y"]),
        expected_dim=X_test.shape[1],
    )

    metadata = {
        "dataset": {
            "dataset_id": args.dataset_id,
            "dataset_config": (
                args.dataset_config
                if args.dataset_config
                else None
            ),
            "official_train_split":
                args.train_split,
            "official_test_split":
                args.test_split,
            "train_split_fingerprint":
                getattr(
                    train_split,
                    "_fingerprint",
                    "unavailable",
                ),
            "test_split_fingerprint":
                getattr(
                    test_split,
                    "_fingerprint",
                    "unavailable",
                ),
            "train_total_rows":
                int(train["num_total_rows"]),
            "test_total_rows":
                int(test["num_total_rows"]),
        },

        "encoder": {
            "model_name": args.model_name,
            "backbone_tag": args.backbone_tag,
            "frozen": True,
            "device_used": device,
            "max_sequence_length":
                int(args.max_length),
            "embedding_dimension":
                int(X_train.shape[1]),
            "normalize_embeddings": False,
        },

        "sampling": {
            "description": (
                "Uniform random sampling within each "
                "official split. Neither task-label values "
                "Y nor evaluation-attribute values E are "
                "used for sampling or balancing."
            ),
            "sample_seed": int(args.seed),
            "train_sample_seed":
                int(args.seed + 11),
            "test_sample_seed":
                int(args.seed + 17),
            "requested_train_size":
                int(args.n_train),
            "requested_test_size":
                int(args.n_test),
            "actual_train_size":
                int(len(train["y"])),
            "actual_test_size":
                int(len(test["y"])),
            "train_source_indices_sha256":
                sha256_int_array(
                    train["selected_indices"]
                ),
            "test_source_indices_sha256":
                sha256_int_array(
                    test["selected_indices"]
                ),
        },

        "task_label": {
            "name": "binary_toxicity",
            "source_column": "toxicity",
            "definition": (
                "y=1 iff official toxicity score >= "
                f"{args.toxicity_threshold}"
            ),
            "threshold":
                float(args.toxicity_threshold),
            "usage": (
                "Used with frozen embeddings to fit "
                "the label-only workflow."
            ),
        },

        "evaluation_attribute": {
            "name": (
                "lexicon_defined_identity_mention"
            ),
            "definition": (
                "e=1 iff the comment contains at least "
                "one exact word/phrase match from the "
                "fixed identity-term lexicon."
            ),
            "score_definition": (
                "identity_mention_score is binary: "
                "1 iff identity_match_count > 0."
            ),
            "threshold":
                float(args.identity_threshold),
            "lexicon_terms":
                IDENTITY_MENTION_TERMS,
            "lexicon_size":
                int(len(IDENTITY_MENTION_TERMS)),
            "usage": (
                "Evaluation only. E is not used to "
                "estimate the stage-one surrogate, "
                "construct residual covariance operators, "
                "select DMD directions, or fit edits."
            ),
            "interpretation_warning": (
                "E is a transparent lexical proxy for "
                "identity-term mention. It is not an "
                "official demographic annotation, not the "
                "identity of the commenter, and not a "
                "uniquely identified nuisance factor."
            ),
        },

        "counts": {
            "train_y_counts":
                count_values(train["y"]),
            "test_y_counts":
                count_values(test["y"]),
            "train_e_counts":
                count_values(train["e"]),
            "test_e_counts":
                count_values(test["e"]),
            "train_y_e_cell_counts":
                train_cells,
            "test_y_e_cell_counts":
                test_cells,
        },

        "cache_files": {
            "train_cache":
                os.path.abspath(train_cache),
            "test_cache":
                os.path.abspath(test_cache),
            "fields": [
                "X",
                "y",
                "e",
                "source_indices",
                "toxicity_score",
                "identity_mention_score",
                "identity_match_count",
            ],
        },

        "software": {
            "python":
                platform.python_version(),
            "platform":
                platform.platform(),
            "numpy":
                package_version("numpy"),
            "torch":
                package_version("torch"),
            "datasets":
                package_version("datasets"),
            "transformers":
                package_version("transformers"),
            "sentence_transformers":
                package_version(
                    "sentence-transformers"
                ),
            "scikit_learn":
                package_version("scikit-learn"),
        },
    }

    metadata_path = os.path.join(
        cache_dir,
        (
            f"civilcomments_"
            f"{args.backbone_tag}_metadata.json"
        ),
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
            ensure_ascii=False,
        )

    print(
        f"[save] metadata: {metadata_path}",
        flush=True,
    )
    print(
        "[done] CivilComments frozen-language "
        "caches are ready.",
        flush=True,
    )


if __name__ == "__main__":
    main()