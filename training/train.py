"""Train MobileFaceNet + ArcFace on the rendered chunks (approach A synthetic people only).

    python -m training.train --root <synthface_output> --out <synthface_output>/models/<run> \
        [--chunks p1_chunk01,p1_chunk02] [--epochs 40] [--batch 128] [--stop-at <unix time>]

- People / images / exclusions / train-holdout split: training/data.py (written to <out>/index.json).
- All aligned crops are loaded once into memory and cached as <root>/training_cache/images_<hash>.npy.
- Augmentation at training time only (training/augment.py).
- Resumable: <out>/last.pt is saved after every epoch; running again continues. --stop-at ends cleanly
  before a Colab session runs out (the next run resumes).
- Evaluation: 1:1 verification on held-out people (never trained on) every --eval-every epochs:
  <out>/metrics.json (history), <out>/best.pt (best TAR@FAR=1e-3), at the end <out>/mobilefacenet.onnx
  and mobilefacenet.ts (embedding model only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training import augment, data, metrics
from training.model import EMBEDDING_DIM, ArcFace, MobileFaceNet, normalise


class FaceDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, seed: int):
        self.images, self.labels, self.indices, self.seed = images, labels, indices, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        rng = np.random.default_rng([self.seed, self.epoch, idx])   # deterministic per (epoch, image)
        return torch.from_numpy(augment.augment(self.images[idx], rng)), int(self.labels[idx])


def load_or_cache(root: Path, people: list[data.Person], cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / f"images_{data.fingerprint(people)}"
    files = {k: stem.with_name(f"{stem.name}_{k}.npy") for k in ("images", "labels", "split", "keys")}
    if all(f.exists() for f in files.values()):
        print(f"CACHE using {stem.name}", flush=True)
        return {k: np.load(f, mmap_mode="r" if k == "images" else None) for k, f in files.items()}
    arrays = data.load_arrays(root, people)
    for k, f in files.items():
        tmp = f.with_name(f.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, arrays[k])
        os.replace(tmp, f)
    print(f"CACHE wrote {stem.name} ({arrays['images'].nbytes / 1e6:.0f} MB)", flush=True)
    return arrays


@torch.inference_mode()
def embed(model: nn.Module, images: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    """Embeddings with flip test-time augmentation (sum of the image and its mirror)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = normalise(torch.from_numpy(np.ascontiguousarray(images[i:i + batch])).to(device))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            e = model(x).float() + model(torch.flip(x, dims=[3])).float()
        out.append(e.cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, EMBEDDING_DIM), np.float32)


def evaluate(model: nn.Module, arrays: dict, device: str, train_people_sample: int, seed: int) -> dict:
    labels, split = np.asarray(arrays["labels"]), np.asarray(arrays["split"])
    result = {}
    hold = np.flatnonzero(split == 1)
    if len(hold):
        result["holdout"] = metrics.verification(embed(model, arrays["images"][hold], device), labels[hold], seed)
    train_labels = np.unique(labels[split == 0])
    pick = np.random.default_rng(seed).choice(train_labels, min(train_people_sample, len(train_labels)), replace=False)
    tr = np.flatnonzero(np.isin(labels, pick) & (split == 0))
    result["train_sample"] = metrics.verification(embed(model, arrays["images"][tr], device), labels[tr], seed)
    return result


def param_groups(model: nn.Module, head: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            (decay if isinstance(m, (nn.Conv2d, nn.Linear)) and name == "weight" else no_decay).append(p)
    return [{"params": decay + list(head.parameters()), "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def export(model: nn.Module, out: Path) -> None:
    model = model.cpu().eval()
    example = torch.zeros(1, 3, 112, 112)
    try:
        torch.jit.trace(model, example).save(str(out / "mobilefacenet.ts"))
        print("EXPORT mobilefacenet.ts", flush=True)
    except Exception as exc:  # export is a convenience; the state dict in best.pt is the result
        print(f"EXPORT TorchScript failed: {exc}", flush=True)
    try:
        torch.onnx.export(model, example, str(out / "mobilefacenet.onnx"), input_names=["image"],
                          output_names=["embedding"], dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=17, dynamo=False)
        print("EXPORT mobilefacenet.onnx", flush=True)
    except Exception as exc:
        print(f"EXPORT ONNX failed: {exc}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="folder with the chunk folders (synthface_output)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunks", help="comma-separated chunk names (default: every ready chunk)")
    p.add_argument("--cache-dir", type=Path, help="default <root>/training_cache")
    p.add_argument("--holdout", type=float, default=data.DEFAULT_HOLDOUT_FRACTION)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=32.0)
    p.add_argument("--margin-warmup-epochs", type=int, default=3, help="ArcFace margin grows 0 -> margin")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--stop-at", type=float, help="unix time: start no epoch that would end after it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    people, report = data.collect(args.root, args.chunks.split(",") if args.chunks else None, args.holdout)
    data.save_index(args.out / "index.json", people, report)
    print(f"DATA chunks {[c['chunk'] for c in report['chunks']]}  people {report['people']}  images {report['images']}  "
          f"excluded {len(report['excluded'])}", flush=True)
    for e in report["excluded"]:
        print(f"  excluded {e['chunk']}/{e['identity']}: {e['reason']}", flush=True)
    arrays = load_or_cache(args.root, people, args.cache_dir or args.root / "training_cache")
    labels, split = np.asarray(arrays["labels"]), np.asarray(arrays["split"])
    train_idx = np.flatnonzero(split == 0)
    classes = int(labels[train_idx].max()) + 1

    model = MobileFaceNet().to(args.device).to(memory_format=torch.channels_last)
    head = ArcFace(EMBEDDING_DIM, classes, args.scale, args.margin).to(args.device)
    opt = torch.optim.SGD(param_groups(model, head, args.weight_decay), lr=args.lr, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
    dataset = FaceDataset(arrays["images"], labels, train_idx, args.seed)
    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                        pin_memory=args.device == "cuda", generator=loader_gen, persistent_workers=False)
    steps_per_epoch = len(loader)
    total, warmup = args.epochs * steps_per_epoch, steps_per_epoch
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {
        "classes": classes, "train_images": int(len(train_idx)), "steps_per_epoch": steps_per_epoch}

    start_epoch, history, best = 0, [], None
    last = args.out / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=args.device, weights_only=False)
        if ckpt["config"]["classes"] != classes or ckpt["fingerprint"] != data.fingerprint(people):
            raise SystemExit(f"{last} was trained on other data: use a new --out")
        model.load_state_dict(ckpt["model"])
        head.load_state_dict(ckpt["head"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, history, best = ckpt["epoch"], ckpt["history"], ckpt["best"]
        print(f"RESUME from epoch {start_epoch}", flush=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    epoch_seconds = 0.0
    for epoch in range(start_epoch, args.epochs):
        if args.stop_at and time.time() + epoch_seconds * 1.2 > args.stop_at:
            print(f"STOPPED before epoch {epoch + 1} (session time); run again to resume", flush=True)
            return
        t0 = time.time()
        dataset.epoch = epoch
        loader_gen.manual_seed(args.seed + epoch)
        margin = args.margin * min(1.0, (epoch + 1) / max(1, args.margin_warmup_epochs))
        loss_sum, correct, seen = 0.0, 0, 0
        for i, (x, y) in enumerate(loader):
            step = epoch * steps_per_epoch + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, args.lr)
            x = normalise(x.to(args.device, non_blocking=True)).contiguous(memory_format=torch.channels_last)
            y = y.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device == "cuda"):
                emb = model(x)
            logits = head(emb, y, margin)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss) * len(y)
            with torch.no_grad():
                correct += int((head(emb.detach(), y, 0.0).argmax(1) == y).sum())
            seen += len(y)
        entry = {"epoch": epoch + 1, "loss": loss_sum / max(seen, 1), "train_top1": correct / max(seen, 1),
                 "margin": margin, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            entry["eval"] = evaluate(model, arrays, args.device, 100, args.seed)
            h = entry["eval"].get("holdout", {})
            score = h.get("tar@far=0.001", -1.0)
            if best is None or score > best["tar@far=0.001"]:
                best = {"epoch": epoch + 1, "tar@far=0.001": score, "eer": h.get("eer")}
                save_atomic({"model": model.state_dict(), "config": config, "epoch": epoch + 1, "eval": entry["eval"]},
                            args.out / "best.pt")
        history.append(entry)
        save_atomic({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history, "best": best,
                     "config": config, "fingerprint": data.fingerprint(people)}, last)
        (args.out / "metrics.json").write_text(json.dumps({"history": history, "best": best}, indent=1) + "\n")
        epoch_seconds = time.time() - t0
        line = (f"EPOCH {epoch + 1}/{args.epochs} loss {entry['loss']:.3f} train_top1 {entry['train_top1']:.3f} "
                f"margin {margin:.2f} {epoch_seconds:.0f}s")
        if "eval" in entry and "holdout" in entry["eval"]:
            h = entry["eval"]["holdout"]
            line += (f" | HOLDOUT auc {h['auc']:.4f} eer {h['eer']:.4f} tar@1e-3 {h['tar@far=0.001']:.4f} "
                     f"acc {h['best_accuracy']:.4f}")
        print(line, flush=True)

    best_ckpt = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    export(model, args.out)
    print(f"DONE best epoch {best['epoch']} tar@far=1e-3 {best['tar@far=0.001']:.4f} eer {best['eer']}", flush=True)


if __name__ == "__main__":
    main()
