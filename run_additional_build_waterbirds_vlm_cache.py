import os
import json
import glob
import argparse
import platform
import hashlib
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
# from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModel


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def package_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def resolve_dtype(device, dtype_name):
    dtype_name = str(dtype_name).lower()

    if not str(device).startswith("cuda"):
        return torch.float32

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "GPU does not support bfloat16. Use --dtype float16."
            )
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_device(requested_device):
    requested_device = str(requested_device).strip().lower()

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    return requested_device


def find_metadata_path(waterbirds_root):
    if waterbirds_root:
        root = os.path.abspath(waterbirds_root)
        direct = os.path.join(root, "metadata.csv")
        if os.path.exists(direct):
            return direct

        matches = glob.glob(
            os.path.join(root, "**", "metadata.csv"),
            recursive=True,
        )
        if matches:
            return sorted(matches)[0]

        raise FileNotFoundError(
            f"No metadata.csv found under {root}"
        )

    raise ValueError("--waterbirds_root must be provided.")


def normalize_split_value(value):
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"train", "tr", "0"}:
            return "0"
        if s in {"val", "valid", "validation", "1"}:
            return "1"
        if s in {"test", "te", "2"}:
            return "2"
        return s

    if pd.isna(value):
        return ""

    try:
        return str(int(value))
    except Exception:
        return str(value)


def select_split(df, split_value):
    split_value = normalize_split_value(split_value)

    split_col = None
    for candidate in ["split", "Split", "SPLIT"]:
        if candidate in df.columns:
            split_col = candidate
            break

    if split_col is None:
        raise KeyError(
            "metadata.csv must contain a split column."
        )

    normalized = df[split_col].apply(normalize_split_value)
    out = df[normalized == split_value].copy()

    if len(out) == 0:
        raise ValueError(
            f"No rows found for split={split_value}. "
            f"Available normalized splits: "
            f"{sorted(set(normalized.tolist()))}"
        )

    return out


def infer_columns(df):
    filename_col = None
    for candidate in [
        "img_filename",
        "filename",
        "image_filename",
        "filepath",
        "path",
        "image_path",
    ]:
        if candidate in df.columns:
            filename_col = candidate
            break

    if filename_col is None:
        raise KeyError(
            "Could not find image filename/path column in metadata.csv."
        )

    y_col = None
    for candidate in ["y", "label", "target"]:
        if candidate in df.columns:
            y_col = candidate
            break

    if y_col is None:
        raise KeyError(
            "Could not find task label column y/label/target."
        )

    e_col = None
    for candidate in ["place", "e", "env", "environment", "background"]:
        if candidate in df.columns:
            e_col = candidate
            break

    if e_col is None:
        raise KeyError(
            "Could not find environment column place/e/env/environment."
        )

    return filename_col, y_col, e_col


class ImagePathResolver:
    def __init__(self, root, metadata_dir):
        self.root = os.path.abspath(root)
        self.metadata_dir = os.path.abspath(metadata_dir)
        self.basename_map = None

    def build_basename_map(self):
        print("[resolver] building basename map by recursive scan", flush=True)
        patterns = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            patterns.extend(
                glob.glob(
                    os.path.join(self.root, "**", ext),
                    recursive=True,
                )
            )
        mapping = {}
        for path in patterns:
            base = os.path.basename(path)
            if base not in mapping:
                mapping[base] = path
        self.basename_map = mapping
        print(
            f"[resolver] indexed {len(mapping)} image basenames",
            flush=True,
        )

    def resolve(self, filename):
        filename = str(filename)

        candidates = [
            os.path.join(self.root, filename),
            os.path.join(self.root, "images", filename),
            os.path.join(self.metadata_dir, filename),
            os.path.join(self.metadata_dir, "images", filename),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        if self.basename_map is None:
            self.build_basename_map()

        base = os.path.basename(filename)
        if base in self.basename_map:
            return self.basename_map[base]

        raise FileNotFoundError(
            f"Could not resolve image path for {filename}"
        )


def resolve_model_name_or_path(model_name):
    """
    If model_name points to a local directory, return its absolute
    path and force local loading. This prevents HuggingFace Hub from
    trying to parse Windows paths such as '.\\hf_models\\...' as repo ids.
    """
    raw = str(model_name)

    expanded = os.path.expanduser(raw)
    absolute = os.path.abspath(expanded)

    if os.path.isdir(absolute):
        return absolute, True

    if os.path.isdir(expanded):
        return os.path.abspath(expanded), True

    return raw, False


def load_vlm(model_name, revision, device, dtype):
    model_ref, is_local = resolve_model_name_or_path(model_name)

    print(
        f"[model] loading image_processor/model: {model_ref}",
        flush=True,
    )
    print(
        f"[model] local_files_only={is_local}",
        flush=True,
    )

    processor_kwargs = {
        "trust_remote_code": False,
    }

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }

    if is_local:
        processor_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True
    else:
        processor_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    # We only need image features. AutoProcessor may try to load
    # the text tokenizer as well, which can fail on local SigLIP
    # folders if sentencepiece / processor metadata are inconsistent.
    # AutoImageProcessor avoids the tokenizer path entirely.
    processor = AutoImageProcessor.from_pretrained(
        model_ref,
        **processor_kwargs,
    )

    model = AutoModel.from_pretrained(
        model_ref,
        **model_kwargs,
    )

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(
        f"[model] class={model.__class__.__name__}, "
        f"dtype={dtype}, device={device}",
        flush=True,
    )
    print(
        f"[processor] class={processor.__class__.__name__}",
        flush=True,
    )

    return processor, model


def extract_image_features(model, inputs):
    if hasattr(model, "get_image_features"):
        return model.get_image_features(**inputs)

    outputs = model(
        **inputs,
        output_hidden_states=False,
        return_dict=True,
    )

    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0]

    raise RuntimeError(
        "Could not extract image features from model outputs."
    )


def encode_images(
    processor,
    model,
    image_paths,
    device,
    batch_size,
    split_name,
):
    features = []
    current_batch_size = int(batch_size)
    start = 0

    progress = tqdm(
        total=len(image_paths),
        desc=f"Encoding {split_name}",
    )

    while start < len(image_paths):
        end = min(start + current_batch_size, len(image_paths))
        batch_paths = image_paths[start:end]

        images = None
        inputs = None
        feats = None

        try:
            images = [
                Image.open(path).convert("RGB")
                for path in batch_paths
            ]

            inputs = processor(
                images=images,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in inputs.items()
            }

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=model.dtype,
                )
                if str(device).startswith("cuda")
                and model.dtype in {torch.float16, torch.bfloat16}
                else nullcontext()
            )

            with torch.inference_mode():
                with autocast_context:
                    feats = extract_image_features(model, inputs)

            feats = feats.detach().float().cpu().numpy()

            if feats.ndim != 2:
                raise ValueError(f"Expected 2-D features, got {feats.shape}")

            features.append(feats)

            progress.update(end - start)
            start = end

            del images
            del inputs
            del feats

        except RuntimeError as exc:
            message = str(exc).lower()

            if (
                "out of memory" not in message
                and "cuda error: out of memory" not in message
            ):
                raise

            if not torch.cuda.is_available():
                raise

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if current_batch_size <= 1:
                raise RuntimeError(
                    "CUDA OOM even at batch size 1."
                ) from exc

            current_batch_size = max(1, current_batch_size // 2)
            print(
                f"\n[warning] CUDA OOM; retrying with batch_size={current_batch_size}",
                flush=True,
            )

    progress.close()

    X = np.concatenate(features, axis=0).astype(np.float32)

    if X.shape[0] != len(image_paths):
        raise ValueError(
            f"Feature row mismatch: got {X.shape[0]}, expected {len(image_paths)}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite features detected.")

    return X


def save_cache(path, X, y, e, source_indices):
    ensure_dir(os.path.dirname(path))

    np.savez(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        e=np.asarray(e, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int64),
    )

    print(
        f"[save] {path}, X={X.shape}, size={os.path.getsize(path)/(1024**2):.1f} MB",
        flush=True,
    )


def sha256_int_array(values):
    values = np.ascontiguousarray(
        np.asarray(values, dtype=np.int64)
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def count_values(values):
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--waterbirds_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--backbone_tag", type=str, default="openclip_vit_l14")
    parser.add_argument("--train_split_value", type=str, default="0")
    parser.add_argument("--test_split_value", type=str, default="2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)

    metadata_path = find_metadata_path(args.waterbirds_root)
    metadata_dir = os.path.dirname(metadata_path)
    root = os.path.abspath(args.waterbirds_root)

    print(f"[data] metadata={metadata_path}", flush=True)

    df = pd.read_csv(metadata_path)
    filename_col, y_col, e_col = infer_columns(df)

    train_df = select_split(df, args.train_split_value)
    test_df = select_split(df, args.test_split_value)

    if args.max_train > 0 and len(train_df) > args.max_train:
        idx = rng.choice(len(train_df), size=args.max_train, replace=False)
        train_df = train_df.iloc[np.sort(idx)].copy()

    if args.max_test > 0 and len(test_df) > args.max_test:
        idx = rng.choice(len(test_df), size=args.max_test, replace=False)
        test_df = test_df.iloc[np.sort(idx)].copy()

    resolver = ImagePathResolver(root=root, metadata_dir=metadata_dir)

    train_paths = [
        resolver.resolve(x)
        for x in train_df[filename_col].tolist()
    ]
    test_paths = [
        resolver.resolve(x)
        for x in test_df[filename_col].tolist()
    ]

    y_train = train_df[y_col].to_numpy(dtype=np.int64)
    y_test = test_df[y_col].to_numpy(dtype=np.int64)
    e_train = train_df[e_col].to_numpy(dtype=np.int64)
    e_test = test_df[e_col].to_numpy(dtype=np.int64)

    print(f"[data] train n={len(y_train)}, test n={len(y_test)}", flush=True)
    print(f"[data] train y={count_values(y_train)}, e={count_values(e_train)}", flush=True)
    print(f"[data] test y={count_values(y_test)}, e={count_values(e_test)}", flush=True)

    processor, model = load_vlm(
        model_name=args.model_name,
        revision=args.revision,
        device=device,
        dtype=dtype,
    )

    X_train = encode_images(
        processor=processor,
        model=model,
        image_paths=train_paths,
        device=device,
        batch_size=args.batch_size,
        split_name="train",
    )

    X_test = encode_images(
        processor=processor,
        model=model,
        image_paths=test_paths,
        device=device,
        batch_size=args.batch_size,
        split_name="test",
    )

    cache_dir = os.path.join(args.data_root, "feature_cache")
    ensure_dir(cache_dir)

    train_cache = os.path.join(
        cache_dir,
        f"waterbirds_train_{args.backbone_tag}_pt1.npz",
    )
    test_cache = os.path.join(
        cache_dir,
        f"waterbirds_test_{args.backbone_tag}_pt1.npz",
    )

    save_cache(
        train_cache,
        X_train,
        y_train,
        e_train,
        train_df.index.to_numpy(dtype=np.int64),
    )
    save_cache(
        test_cache,
        X_test,
        y_test,
        e_test,
        test_df.index.to_numpy(dtype=np.int64),
    )

    metadata = {
        "dataset": "waterbirds",
        "metadata_path": os.path.abspath(metadata_path),
        "waterbirds_root": root,
        "filename_col": filename_col,
        "task_label_col": y_col,
        "environment_col": e_col,
        "train_split_value": args.train_split_value,
        "test_split_value": args.test_split_value,
        "model_name": args.model_name,
        "revision": args.revision,
        "backbone_tag": args.backbone_tag,
        "frozen": True,
        "representation": "image features from frozen VLM/vision-language encoder",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "train_y_counts": count_values(y_train),
        "test_y_counts": count_values(y_test),
        "train_e_counts": count_values(e_train),
        "test_e_counts": count_values(e_test),
        "train_source_indices_sha256": sha256_int_array(train_df.index.to_numpy(dtype=np.int64)),
        "test_source_indices_sha256": sha256_int_array(test_df.index.to_numpy(dtype=np.int64)),
        "software": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "Pillow": package_version("Pillow"),
        },
        "cache_files": {
            "train": os.path.abspath(train_cache),
            "test": os.path.abspath(test_cache),
        },
    }

    metadata_out = os.path.join(
        cache_dir,
        f"waterbirds_{args.backbone_tag}_metadata.json",
    )

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[save] metadata: {metadata_out}", flush=True)
    print("[done] Waterbirds VLM cache is ready.", flush=True)


if __name__ == "__main__":
    main()

    