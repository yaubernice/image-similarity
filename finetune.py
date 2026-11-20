"""Fine-tune DINOv2 on the labelled Caltech-101 subset.

Training uses only the database images. The held-out queries stay out of
the loss so evaluate.py still measures them fairly.

The last two transformer blocks and a 256-d projection are trained with
triplet loss on every triplet in the batch. The checkpoint is models/dinov2-ft.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "caltech101"
CHECKPOINT = ROOT / "models" / "dinov2-ft.pt"
LOG = ROOT / "results" / "finetune_dinov2.json"


def training_pairs(ground_truth: Path) -> tuple[list[Path], list[int]]:
    """Database images only. Queries are the evaluation hold-out."""
    ground = json.loads(ground_truth.read_text())
    paths: list[Path] = []
    labels: list[int] = []
    names: dict[str, int] = {}
    seen = set()
    for query in ground["queries"]:
        class_name = query["class"]
        if class_name not in names:
            names[class_name] = len(names)
        label = names[class_name]
        for relative in query["relevant"]:
            if relative in seen:
                continue
            seen.add(relative)
            paths.append(DATA / relative)
            labels.append(label)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training image: {missing[0]}")
    return paths, labels


def unfreeze_tail(model, blocks: int) -> list:
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.encoder.layer
    chosen = list(layers[-blocks:])
    if hasattr(model, "layernorm"):
        chosen.append(model.layernorm)
    trainable = []
    for module in chosen:
        for parameter in module.parameters():
            parameter.requires_grad = True
            trainable.append(parameter)
    return trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Triplet fine-tune of DINOv2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--images-per-class", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--ground-truth", type=Path, default=DATA / "ground_truth.json")
    args = parser.parse_args()

    import torch
    from pytorch_metric_learning import losses, miners
    from pytorch_metric_learning.samplers import MPerClassSampler
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoImageProcessor, AutoModel

    from extractors import torch_device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths, labels = training_pairs(args.ground_truth)
    if args.batch_size % args.images_per_class != 0:
        raise SystemExit("--batch-size must be divisible by --images-per-class")
    print(f"Training images: {len(paths)}  classes: {len(set(labels))}  queries excluded")

    device = torch_device()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    backbone_parameters = unfreeze_tail(model, args.blocks)
    head = torch.nn.Linear(1536, args.dim).to(device)
    print(f"Device: {device}  trainable tensors: {len(backbone_parameters) + len(list(head.parameters()))}")

    class ImageSet(Dataset):
        def __getitem__(self, index):
            image = Image.open(paths[index]).convert("RGB")
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            pixels = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            return pixels, labels[index]

        def __len__(self):
            return len(paths)

    sampler = MPerClassSampler(
        labels,
        m=args.images_per_class,
        batch_size=args.batch_size,
    )
    loader = DataLoader(ImageSet(), batch_sampler=sampler)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    miner = miners.TripletMarginMiner(margin=args.margin, type_of_triplets="all")
    loss_fn = losses.TripletMarginLoss(margin=args.margin)

    history = []
    model.train()
    head.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        batches = 0
        triplets = 0
        for pixels, batch_labels in loader:
            pixels = pixels.to(device)
            batch_labels = batch_labels.to(device)
            tokens = model(pixel_values=pixels).last_hidden_state
            descriptor = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
            embedding = torch.nn.functional.normalize(head(descriptor), dim=1)
            mined = miner(embedding, batch_labels)
            if mined[0].numel() == 0:
                continue
            loss = loss_fn(embedding, batch_labels, mined)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_parameters + list(head.parameters()), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            triplets += int(mined[0].shape[0]) if len(mined) else 0
        mean_loss = total / max(batches, 1)
        history.append({"epoch": epoch, "loss": round(mean_loss, 4), "triplets": triplets})
        print(f"Epoch {epoch}/{args.epochs}  loss {mean_loss:.4f}  triplets {triplets}")

    model.eval()
    head.eval()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    backbone = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {"backbone": backbone, "head": head.state_dict(), "dim": args.dim},
        CHECKPOINT,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "checkpoint": str(CHECKPOINT),
        "images": len(paths),
        "classes": len(set(labels)),
        "epochs": args.epochs,
        "margin": args.margin,
        "dim": args.dim,
        "blocks": args.blocks,
        "history": history,
    }
    LOG.write_text(json.dumps(record, indent=2))
    print(f"Wrote {CHECKPOINT}")
    print(f"Wrote {LOG}")


if __name__ == "__main__":
    main()
