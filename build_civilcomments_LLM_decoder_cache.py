import os
import json
import argparse
import platform
from contextlib import nullcontext

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

import build_civilcomments_frozen_cache as common


def resolve_dtype(device, dtype_name):
    dtype_name = str(dtype_name).lower()

    if not str(device).startswith("cuda"):
        return torch.float32

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected GPU does not support bfloat16. "
                "Use --dtype float16."
            )
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32

    raise ValueError(
        f"Unsupported dtype: {dtype_name}"
    )


def load_decoder_encoder(
    model_name,
    revision,
    device,
    dtype,
    attn_implementation,
):
    print(
        f"[encoder] loading decoder-only model: {model_name}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        use_fast=True,
        trust_remote_code=False,
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor "
                "eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    actual_attn_implementation = str(
        attn_implementation
    )

    # Older PyTorch installations may not provide SDPA.
    if (
        actual_attn_implementation == "sdpa"
        and not hasattr(
            torch.nn.functional,
            "scaled_dot_product_attention",
        )
    ):
        print(
            "[warning] PyTorch does not expose "
            "scaled_dot_product_attention; falling back "
            "from SDPA to eager attention.",
            flush=True,
        )
        actual_attn_implementation = "eager"

    print(
        f"[encoder] revision={revision}, device={device}, "
        f"dtype={dtype}, "
        f"attention={actual_attn_implementation}",
        flush=True,
    )

    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation=actual_attn_implementation,
    )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    hidden_size = int(model.config.hidden_size)

    commit_hash = getattr(
        model.config,
        "_commit_hash",
        None,
    )

    configured_attn = getattr(
        model.config,
        "_attn_implementation",
        actual_attn_implementation,
    )

    print(
        f"[encoder] loaded class="
        f"{model.__class__.__name__}",
        flush=True,
    )
    print(
        f"[encoder] hidden size={hidden_size}",
        flush=True,
    )
    print(
        f"[encoder] resolved commit={commit_hash}",
        flush=True,
    )
    print(
        f"[encoder] configured attention={configured_attn}",
        flush=True,
    )

    return (
        tokenizer,
        model,
        hidden_size,
        commit_hash,
        str(configured_attn),
    )

def pool_hidden_states(
    hidden,
    attention_mask,
    pooling,
):
    """
    Pool decoder hidden states.

    last:
        Last non-padding token. This position has causal access
        to all preceding tokens.

    mean:
        Attention-mask-weighted mean over non-padding tokens.
    """
    if pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp_min(0)

        batch_indices = torch.arange(
            hidden.shape[0],
            device=hidden.device,
        )

        return hidden[batch_indices, lengths]

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden.dtype
        )
        denominator = mask.sum(dim=1).clamp_min(1.0)

        return (
            (hidden * mask).sum(dim=1)
            / denominator
        )

    raise ValueError(
        f"Unknown pooling method: {pooling}"
    )


def encode_texts(
    tokenizer,
    model,
    texts,
    device,
    hidden_size,
    batch_size,
    max_length,
    pooling,
    split_name,
):
    """
    Extract frozen decoder hidden representations.

    If CUDA runs out of memory, the batch size is halved and the
    current batch is retried.
    """
    n = len(texts)

    embeddings = np.empty(
        (n, int(hidden_size)),
        dtype=np.float32,
    )

    current_batch_size = int(batch_size)
    start = 0

    progress = tqdm(
        total=n,
        desc=f"Encoding {split_name}",
    )

    while start < n:
        end = min(
            start + current_batch_size,
            n,
        )

        batch_texts = texts[start:end]

        tokens = None
        outputs = None
        pooled = None

        try:
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
                add_special_tokens=True,
            )

            tokens = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value in tokens.items()
            }

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=model.dtype,
                )
                if str(device).startswith("cuda")
                and model.dtype in {
                    torch.float16,
                    torch.bfloat16,
                }
                else nullcontext()
            )

            with torch.inference_mode():
                with autocast_context:
                    outputs = model(
                        **tokens,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                    pooled = pool_hidden_states(
                        outputs.last_hidden_state,
                        tokens["attention_mask"],
                        pooling=pooling,
                    )

            pooled = (
                pooled.detach()
                .float()
                .cpu()
                .numpy()
            )

            if pooled.shape != (
                end - start,
                int(hidden_size),
            ):
                raise ValueError(
                    f"Unexpected pooled shape: {pooled.shape}"
                )

            embeddings[start:end] = pooled

            progress.update(end - start)
            start = end

            del outputs
            del pooled
            del tokens

        except RuntimeError as exc:
            message = str(exc).lower()

            if (
                "out of memory" not in message
                and "cuda error: out of memory" not in message
            ):
                raise

            if not torch.cuda.is_available():
                raise

            # Release references owned by the failed batch before
            # retrying with a smaller batch size.
            outputs = None
            pooled = None
            tokens = None

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if current_batch_size <= 1:
                raise RuntimeError(
                    "CUDA OOM even at batch size 1. "
                    "Reduce --max_length or use "
                    "Qwen/Qwen2.5-0.5B."
                ) from exc

            current_batch_size = max(
                1,
                current_batch_size // 2,
            )

            print(
                f"\n[warning] CUDA OOM; retrying the same "
                f"batch with batch_size="
                f"{current_batch_size}",
                flush=True,
            )

    progress.close()

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            f"{split_name}: non-finite embeddings detected."
        )

    return embeddings


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
        default="Qwen/Qwen2.5-1.5B",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
    )
    parser.add_argument(
        "--backbone_tag",
        type=str,
        default="qwen2_5_1_5b_last",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["last", "mean"],
        default="last",
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
        default=32,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
        default="float16",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["eager", "sdpa"],
        default="sdpa",
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

    common.set_seed(args.seed)

    device = common.resolve_device(
        args.device
    )
    dtype = resolve_dtype(
        device,
        args.dtype,
    )

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

    required_columns = {
        "text",
        "toxicity",
    }

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
                f"{split_name}: missing required columns "
                f"{sorted(missing)}"
            )

    identity_regex = common.compile_identity_regex(
        common.IDENTITY_MENTION_TERMS
    )

    # These offsets exactly match the existing MPNet builder.
    # Therefore the same seed gives the same sampled cohort.
    train = common.prepare_split(
        split=train_split,
        split_name=args.train_split,
        n_samples=args.n_train,
        sample_seed=args.seed + 11,
        toxicity_threshold=args.toxicity_threshold,
        identity_threshold=args.identity_threshold,
        identity_regex=identity_regex,
    )

    test = common.prepare_split(
        split=test_split,
        split_name=args.test_split,
        n_samples=args.n_test,
        sample_seed=args.seed + 17,
        toxicity_threshold=args.toxicity_threshold,
        identity_threshold=args.identity_threshold,
        identity_regex=identity_regex,
    )

    train_cells = common.validate_split(
        train,
        "train",
        min_cell_count=args.min_cell_count,
    )
    test_cells = common.validate_split(
        test,
        "test",
        min_cell_count=args.min_cell_count,
    )

    (
        tokenizer,
        model,
        hidden_size,
        commit_hash,
        actual_attn_implementation,
    ) = load_decoder_encoder(
        model_name=args.model_name,
        revision=args.revision,
        device=device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    X_train = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=train["texts"],
        device=device,
        hidden_size=hidden_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        split_name="train",
    )

    X_test = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=test["texts"],
        device=device,
        hidden_size=hidden_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        split_name="test",
    )

    cache_dir = os.path.join(
        args.data_root,
        "feature_cache",
    )
    common.ensure_dir(cache_dir)

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

    common.save_feature_cache(
        path=train_cache,
        X=X_train,
        prepared=train,
        compress=bool(args.compress),
    )
    common.save_feature_cache(
        path=test_cache,
        X=X_test,
        prepared=test,
        compress=bool(args.compress),
    )

    common.verify_feature_cache(
        path=train_cache,
        expected_n=len(train["y"]),
        expected_dim=hidden_size,
    )
    common.verify_feature_cache(
        path=test_cache,
        expected_n=len(test["y"]),
        expected_dim=hidden_size,
    )

    metadata = {
        "dataset": {
            "dataset_id": args.dataset_id,
            "official_train_split": args.train_split,
            "official_test_split": args.test_split,
            "train_total_rows": int(
                train["num_total_rows"]
            ),
            "test_total_rows": int(
                test["num_total_rows"]
            ),
        },
        "encoder": {
            "model_name": args.model_name,
            "revision_requested": args.revision,
            "resolved_commit_hash": commit_hash,
            "architecture": (
                model.config.architectures
                if hasattr(
                    model.config,
                    "architectures",
                )
                else None
            ),
            "model_type": getattr(
                model.config,
                "model_type",
                None,
            ),
            "backbone_tag": args.backbone_tag,
            "frozen": True,
            "representation": (
                "last-layer hidden state pooled over "
                f"tokens using '{args.pooling}' pooling"
            ),
            "pooling": args.pooling,
            "hidden_size": int(hidden_size),
            "max_length": int(args.max_length),
            "dtype": str(dtype),
            "device": device,
            "attention_implementation_requested":
                args.attn_implementation,
            "attention_implementation_used":
                actual_attn_implementation,
        },
        "sampling": {
            "description": (
                "Uniform sampling from the official splits; "
                "Y and E are not used for sampling."
            ),
            "seed": int(args.seed),
            "train_sample_seed": int(
                args.seed + 11
            ),
            "test_sample_seed": int(
                args.seed + 17
            ),
            "train_size": int(
                len(train["y"])
            ),
            "test_size": int(
                len(test["y"])
            ),
            "train_source_indices_sha256":
                common.sha256_int_array(
                    train["selected_indices"]
                ),
            "test_source_indices_sha256":
                common.sha256_int_array(
                    test["selected_indices"]
                ),
        },
        "task_label": {
            "name": "binary_toxicity",
            "definition": (
                "y=1 iff official toxicity score >= "
                f"{args.toxicity_threshold}"
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
            "usage": "Evaluation only.",
            "interpretation_warning": (
                "E is a lexical identity-mention proxy, "
                "not demographic identity and not a "
                "uniquely identified nuisance."
            ),
        },
        "counts": {
            "train_y_counts":
                common.count_values(train["y"]),
            "test_y_counts":
                common.count_values(test["y"]),
            "train_e_counts":
                common.count_values(train["e"]),
            "test_e_counts":
                common.count_values(test["e"]),
            "train_y_e_cells": train_cells,
            "test_y_e_cells": test_cells,
        },
        "software": {
            "python": platform.python_version(),
            "torch":
                common.package_version("torch"),
            "transformers":
                common.package_version("transformers"),
            "datasets":
                common.package_version("datasets"),
            "numpy":
                common.package_version("numpy"),
        },
        "cache_files": {
            "train": os.path.abspath(
                train_cache
            ),
            "test": os.path.abspath(
                test_cache
            ),
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
        "[done] Frozen decoder-only LLM caches are ready.",
        flush=True,
    )


if __name__ == "__main__":
    main()