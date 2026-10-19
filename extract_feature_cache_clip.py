# extract_feature_cache_clip.py  
# Generate CLIP feature caches compatible with run_labelonly_workflow_full.py  
# Output npz keys: X, y, e  
# Output names: {dataset}_{split}_clip_vit_b32_pt1.npz or pt0  
  
import os  
import glob  
import json  
import time  
import argparse  
from datetime import datetime  
  
import numpy as np  
import pandas as pd  
from PIL import Image  
  
import torch  
from torch.utils.data import Dataset, DataLoader  
  
import open_clip  
  
  
# ============================================================  
# Logger  
# ============================================================  
  
class SimpleLogger:  
    def __init__(self, root_dir="runs_feature_cache_clip", tag="extract_feature_cache_clip"):  
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  
        self.run_dir = os.path.join(root_dir, f"{tag}_{ts}")  
        os.makedirs(self.run_dir, exist_ok=True)  
        self.log_path = os.path.join(self.run_dir, "run.log")  
        self.jsonl_path = os.path.join(self.run_dir, "events.jsonl")  
        self.f_log = open(self.log_path, "a", buffering=1, encoding="utf-8")  
        self.f_jsonl = open(self.jsonl_path, "a", buffering=1, encoding="utf-8")  
  
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
  
    def close(self):  
        try:  
            self.f_log.close()  
        except Exception:  
            pass  
        try:  
            self.f_jsonl.close()  
        except Exception:  
            pass  
  
  
# ============================================================  
# Utils  
# ============================================================  
  
def ensure_dir(path):  
    os.makedirs(path, exist_ok=True)  
  
  
def parse_csv_list(s, cast=str):  
    if s is None:  
        return []  
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()]  
  
  
def set_seed(seed):  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
  
  
def find_existing(paths):  
    for p in paths:  
        if os.path.exists(p):  
            return p  
    return None  
  
  
# ============================================================  
# Dataset objects  
# ============================================================  
  
class PathDataset(Dataset):  
    def __init__(self, records, preprocess):  
        self.records = records  
        self.preprocess = preprocess  
  
    def __len__(self):  
        return len(self.records)  
  
    def __getitem__(self, idx):  
        rec = self.records[idx]  
        img = Image.open(rec["path"]).convert("RGB")  
        x = self.preprocess(img)  
        return x, int(rec["y"]), int(rec["e"]), rec["path"]  
  
  
# ============================================================  
# Dataset parsing  
# ============================================================  
  
def load_celeba_records(data_root, logger):  
    from torchvision.datasets import CelebA  
  
    root_candidates = [  
        os.path.join(data_root, "celeba"),  
        os.path.join(data_root, "CelebA"),  
    ]  
    root = find_existing(root_candidates)  
    if root is None:  
        raise FileNotFoundError("Could not find CelebA root under data_root/celeba or data_root/CelebA")  
  
    logger.print(f"[celeba] root={root}")  
  
    ds_train = CelebA(root=data_root, split="train", target_type="attr", download=False)  
    ds_valid = CelebA(root=data_root, split="valid", target_type="attr", download=False)  
    ds_test = CelebA(root=data_root, split="test", target_type="attr", download=False)  

    attr_names = ds_train.attr_names  
    if "Blond_Hair" not in attr_names or "Male" not in attr_names:  
        raise RuntimeError("CelebA attrs Blond_Hair / Male not found.")  
  
    y_idx = attr_names.index("Blond_Hair")  
    e_idx = attr_names.index("Male")  
  
    img_dir = find_existing([  
        os.path.join(root, "img_align_celeba"),  
        os.path.join(root, "celeba", "img_align_celeba"),  
    ])  
    if img_dir is None:  
        raise FileNotFoundError("Could not find img_align_celeba directory.")  
  
    def build(ds):  
        recs = []  
        for i in range(len(ds)):  
            fn = ds.filename[i]  
            attrs = ds.attr[i].numpy()  
            y = int(attrs[y_idx])  
            e = int(attrs[e_idx])  
            path = os.path.join(img_dir, fn)  
            recs.append({"path": path, "y": y, "e": e})  
        return recs  
  
    train_records = build(ds_train) + build(ds_valid)  
    test_records = build(ds_test)  
  
    return {"train": train_records, "test": test_records}  
  
  
def load_waterbirds_records(data_root, logger):  
    root_candidates = [  
        os.path.join(data_root, "waterbirds_v1.0"),  
        os.path.join(data_root, "waterbirds"),  
        os.path.join(data_root, "Waterbirds"),  
    ]  
    root = find_existing(root_candidates)  
    if root is None:  
        raise FileNotFoundError("Could not find Waterbirds root.")  
  
    metadata_path = find_existing([  
        os.path.join(root, "metadata.csv"),  
        os.path.join(root, "waterbirds_v1.0", "metadata.csv"),  
    ])  
    if metadata_path is None:  
        raise FileNotFoundError("Could not find Waterbirds metadata.csv")  
  
    logger.print(f"[waterbirds] metadata={metadata_path}")  
  
    meta = pd.read_csv(metadata_path)  
  
    img_col = None  
    for c in ["img_filename", "filename", "path"]:  
        if c in meta.columns:  
            img_col = c  
            break  
    if img_col is None:  
        raise RuntimeError("Waterbirds metadata must contain img_filename / filename / path")  
  
    y_col = None  
    for c in ["y", "label", "target"]:  
        if c in meta.columns:  
            y_col = c  
            break  
    if y_col is None:  
        raise RuntimeError("Waterbirds metadata must contain y / label / target")  
  
    e_col = None  
    for c in ["place", "background", "env", "environment"]:  
        if c in meta.columns:  
            e_col = c  
            break  
    if e_col is None:  
        raise RuntimeError("Waterbirds metadata must contain place / background / env / environment")  
  
    split_col = None  
    for c in ["split", "split_id"]:  
        if c in meta.columns:  
            split_col = c  
            break  
    if split_col is None:  
        raise RuntimeError("Waterbirds metadata must contain split / split_id")  
  
    def split_name(v):  
        # common convention: 0 train, 1 val, 2 test  
        return "train" if int(v) in [0, 1] else "test"  
  
    base_dirs = [  
        root,  
        os.path.join(root, "images"),  
        os.path.join(root, "waterbirds_v1.0"),  
    ]  
  
    out = {"train": [], "test": []}  
    for _, row in meta.iterrows():  
        rel = str(row[img_col])  
        abspath = None  
        for bd in base_dirs:  
            p = os.path.join(bd, rel)  
            if os.path.exists(p):  
                abspath = p  
                break  
        if abspath is None:  
            continue  
  
        sname = split_name(row[split_col])  
        out[sname].append({  
            "path": abspath,  
            "y": int(row[y_col]),  
            "e": int(row[e_col]),  
        })  
  
    if len(out["train"]) == 0 or len(out["test"]) == 0:  
        raise RuntimeError("Parsed Waterbirds train/test is empty. Check metadata path columns.")  
  
    return out  
  
  
def load_pacs_records(data_root, logger, seed=0):  
    root_candidates = [  
        os.path.join(data_root, "PACS"),  
        os.path.join(data_root, "pacs"),  
    ]  
    root = find_existing(root_candidates)  
    if root is None:  
        raise FileNotFoundError("Could not find PACS root.")  
  
    logger.print(f"[pacs] root={root}")  
  
    domains = []  
    for d in sorted(os.listdir(root)):  
        dpath = os.path.join(root, d)  
        if os.path.isdir(dpath):  
            domains.append(d)  
  
    if len(domains) == 0:  
        raise RuntimeError("No PACS domains found.")  
  
    class_names = set()  
    for dom in domains:  
        dom_path = os.path.join(root, dom)  
        for cls in sorted(os.listdir(dom_path)):  
            cls_path = os.path.join(dom_path, cls)  
            if os.path.isdir(cls_path):  
                class_names.add(cls)  
  
    class_names = sorted(list(class_names))  
    class_to_idx = {c: i for i, c in enumerate(class_names)}  
    env_to_idx = {d: i for i, d in enumerate(domains)}  
  
    rng = np.random.default_rng(seed)  
    out = {"train": [], "test": []}  
  
    for dom in domains:  
        dom_path = os.path.join(root, dom)  
        e = env_to_idx[dom]  
        for cls in sorted(os.listdir(dom_path)):  
            cls_path = os.path.join(dom_path, cls)  
            if not os.path.isdir(cls_path):  
                continue  
            y = class_to_idx[cls]  
  
            files = []  
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:  
                files.extend(glob.glob(os.path.join(cls_path, ext)))  
            files = sorted(files)  
            if len(files) == 0:  
                continue  
  
            idx = np.arange(len(files))  
            rng.shuffle(idx)  
            cut = max(1, int(0.8 * len(files)))  
            tr_idx = idx[:cut]  
            te_idx = idx[cut:] if cut < len(files) else idx[-1:]  
  
            for j in tr_idx:  
                out["train"].append({"path": files[j], "y": y, "e": e})  
            for j in te_idx:  
                out["test"].append({"path": files[j], "y": y, "e": e})  
  
    if len(out["train"]) == 0 or len(out["test"]) == 0:  
        raise RuntimeError("Parsed PACS train/test is empty.")  
  
    return out  
  
  
def load_records_for_dataset(dataset_name, data_root, logger, seed=0):  
    dn = dataset_name.lower()  
    if dn == "celeba":  
        return load_celeba_records(data_root, logger)  
    elif dn == "waterbirds":  
        return load_waterbirds_records(data_root, logger)  
    elif dn == "pacs":  
        return load_pacs_records(data_root, logger, seed=seed)  
    else:  
        raise ValueError(f"Unsupported dataset: {dataset_name}")  
  
  
# ============================================================  
# CLIP model  
# ============================================================  
  
def build_clip_model(pretrained_flag, device, local_ckpt=""):  
    model_name = "ViT-B-32"  
  
    if local_ckpt is not None and str(local_ckpt).strip() != "":  
        local_ckpt = str(local_ckpt).strip()  
        if not os.path.isfile(local_ckpt):  
            raise FileNotFoundError(f"Local CLIP checkpoint not found: {local_ckpt}")  
        pretrained_arg = local_ckpt  
    else:  
        pretrained_arg = "openai" if bool(pretrained_flag) else None  
  
    model, _, preprocess = open_clip.create_model_and_transforms(  
        model_name,  
        pretrained=pretrained_arg  
    )  
    model = model.to(device)  
    model.eval()  
    return model, preprocess  

  
  
@torch.no_grad()  
def extract_features(records, model, preprocess, batch_size, num_workers, device, logger, split_name):  
    ds = PathDataset(records, preprocess)  
    loader = DataLoader(  
        ds,  
        batch_size=batch_size,  
        shuffle=False,  
        num_workers=num_workers,  
        pin_memory=True,  
    )  
  
    feats = []  
    ys = []  
    es = []  
    paths = []  
  
    for bi, (imgs, y, e, pths) in enumerate(loader):  
        imgs = imgs.to(device, non_blocking=True)  
        z = model.encode_image(imgs)  
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)  
        z = z.detach().cpu().numpy().astype(np.float32)  
  
        feats.append(z)  
        ys.append(y.numpy().astype(np.int64))  
        es.append(e.numpy().astype(np.int64))  
        paths.extend(list(pths))  
  
        if (bi + 1) % 20 == 0 or (bi + 1) == len(loader):  
            logger.print(f"[extract] split={split_name} batch={bi+1}/{len(loader)}")  
  
    X = np.concatenate(feats, axis=0) if len(feats) > 0 else np.zeros((0, 512), dtype=np.float32)  
    y = np.concatenate(ys, axis=0) if len(ys) > 0 else np.zeros((0,), dtype=np.int64)  
    e = np.concatenate(es, axis=0) if len(es) > 0 else np.zeros((0,), dtype=np.int64)  
    return X, y, e, np.array(paths, dtype=object)  
  
  
# ============================================================  
# Main  
# ============================================================  
  
def main():  
    parser = argparse.ArgumentParser()  
  
    parser.add_argument("--data_root", type=str, default="./data")  
    parser.add_argument("--datasets", type=str, default="celeba,waterbirds,pacs")  
    parser.add_argument("--backbone_name", type=str, default="clip_vit_b32")  
    parser.add_argument("--pretrained", type=int, default=1)  
    parser.add_argument("--clip_ckpt", type=str, default="")  

  
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--num_workers", type=int, default=4)  
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")  
    parser.add_argument("--seed", type=int, default=42)  
  
    parser.add_argument("--max_train", type=int, default=0)  
    parser.add_argument("--max_test", type=int, default=0)  
  
    parser.add_argument("--outdir", type=str, default="runs_feature_cache_clip")  
  
    args = parser.parse_args()  
    set_seed(args.seed)  
  
    logger = SimpleLogger(root_dir=args.outdir, tag="extract_feature_cache_clip")  
    logger.print(f"[info] run_dir={logger.run_dir}")  
    logger.print(f"[info] data_root={args.data_root}")  
    logger.print(f"[info] datasets={args.datasets}")  
    logger.print(f"[info] backbone_name={args.backbone_name}")  
    logger.print(f"[info] pretrained={args.pretrained}")  
    logger.print(f"[info] device={args.device}")  
  
    datasets = parse_csv_list(args.datasets, str)  
  
    if args.backbone_name.lower() != "clip_vit_b32":  
        raise ValueError("This script is designed for backbone_name=clip_vit_b32")  
  
    model, preprocess = build_clip_model(args.pretrained, args.device, args.clip_ckpt)  


    cache_dir = os.path.join(args.data_root, "feature_cache")  
    ensure_dir(cache_dir)  
  
    summary = {  
        "run_dir": logger.run_dir,  
        "data_root": args.data_root,  
        "backbone_name": args.backbone_name,  
        "pretrained": int(args.pretrained),  
        "device": args.device,  
        "datasets": {},  
    }  
  
    try:  
        for dataset_name in datasets:  
            logger.print("=" * 80)  
            logger.print(f"[dataset] start {dataset_name}")  
            logger.print("=" * 80)  
  
            ds_out = {  
                "status": "ok",  
                "train": {},  
                "test": {},  
            }  
  
            try:  
                records = load_records_for_dataset(dataset_name, args.data_root, logger, seed=args.seed)  
  
                for split in ["train", "test"]:  
                    split_records = records[split]  
  
                    if split == "train" and args.max_train > 0:  
                        split_records = split_records[:args.max_train]  
                    if split == "test" and args.max_test > 0:  
                        split_records = split_records[:args.max_test]  
  
                    logger.print(f"[dataset={dataset_name}] split={split} num_records={len(split_records)}")  
  
                    X, y, e, paths = extract_features(  
                        split_records,  
                        model=model,  
                        preprocess=preprocess,  
                        batch_size=args.batch_size,  
                        num_workers=args.num_workers,  
                        device=args.device,  
                        logger=logger,  
                        split_name=split,  
                    )  
  
                    pt_tag = "pt1" if bool(args.pretrained) else "pt0"  
                    out_name = f"{dataset_name}_{split}_{args.backbone_name}_{pt_tag}.npz"  
                    out_path = os.path.join(cache_dir, out_name)  
  
                    np.savez_compressed(  
                        out_path,  
                        X=X,  
                        y=y,  
                        e=e,  
                        paths=paths,  
                    )  
  
                    logger.print(f"[saved] {out_path}")  
                    logger.print(  
                        f"[shape] dataset={dataset_name} split={split} "  
                        f"X={tuple(X.shape)} y={tuple(y.shape)} e={tuple(e.shape)} "  
                        f"num_classes={len(np.unique(y))} num_envs={len(np.unique(e))}"  
                    )  
  
                    ds_out[split] = {  
                        "path": out_path,  
                        "num_samples": int(len(y)),  
                        "feature_dim": int(X.shape[1]) if X.ndim == 2 and X.shape[0] > 0 else 0,  
                        "num_classes": int(len(np.unique(y))),  
                        "num_envs": int(len(np.unique(e))),  
                    }  
  
                summary["datasets"][dataset_name] = ds_out  
  
            except Exception as ex:  
                summary["datasets"][dataset_name] = {  
                    "status": "error",  
                    "error": str(ex),  
                }  
                logger.print(f"[dataset] error {dataset_name}: {str(ex)}")  
  
        with open(os.path.join(logger.run_dir, "summary.json"), "w", encoding="utf-8") as f:  
            json.dump(summary, f, indent=2, sort_keys=True)  
  
        logger.print("[final] cache generation finished")  
        logger.print(f"[final] summary saved to {os.path.join(logger.run_dir, 'summary.json')}")  
        logger.print(f"[final] log saved to {logger.log_path}")  
  
    finally:  
        logger.close()  
  
  
if __name__ == "__main__":  
    main()  
