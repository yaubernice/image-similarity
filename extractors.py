"""Pluggable image embedders.

vgg19  GeM over the block5 conv map, short-side resize then centre crop.
dinov2 CLS token concatenated with the mean patch token.
clip   image (and text) tower of CLIP ViT-B/32.

embed() returns one L2-normalised float32 vector per image.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from PIL import Image

from retrieval import l2_normalize

Progress = Optional[Callable[[int, int], None]]
MODELS = ("vgg19", "dinov2", "dinov2-ft", "clip")
ROOT = Path(__file__).resolve().parent
FINETUNE_CHECKPOINT = ROOT / "models" / "dinov2-ft.pt"


def torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def center_crop(image: Image.Image, size: int = 224, short_side: int = 256) -> Image.Image:
    """Resize the short side, then take the centre crop. No squashing."""
    image = image.convert("RGB")
    width, height = image.size
    scale = short_side / min(width, height)
    resized = image.resize((max(size, round(width * scale)), max(size, round(height * scale))), Image.BICUBIC)
    width, height = resized.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def gem_pool(feature_map: np.ndarray, p: float = 3.0) -> np.ndarray:
    """GeM over a channels-last conv map (batch, height, width, channels)."""
    clipped = np.maximum(feature_map.astype(np.float32), 1e-6)
    pooled = np.mean(np.power(clipped, p), axis=(1, 2))
    return np.power(pooled, 1.0 / p).astype(np.float32)


class VGG19Extractor:
    name = "vgg19"
    label = "VGG19 block5 GeM"

    def __init__(self) -> None:
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

        self._preprocess = preprocess_input
        self._model = VGG19(weights="imagenet", include_top=False, pooling=None)
        self._model.trainable = False

    def embed(self, paths: Sequence[Path], batch_size: int = 16, tta: bool = False, progress: Progress = None) -> np.ndarray:
        from tensorflow.keras.preprocessing import image as keras_image

        images = []
        for path in paths:
            crop = center_crop(Image.open(path))
            array = keras_image.img_to_array(crop)
            images.append(self._preprocess(array))
        batch = np.stack(images).astype(np.float32)
        features = self._embed_array(batch, batch_size, progress, len(paths))
        if tta:
            flipped = self._embed_array(batch[:, :, ::-1, :].copy(), batch_size, None, len(paths))
            features = l2_normalize(features + flipped)
        return features

    def _embed_array(self, batch: np.ndarray, batch_size: int, progress: Progress, total: int) -> np.ndarray:
        blocks = []
        done = 0
        for start in range(0, len(batch), batch_size):
            maps = self._model.predict(batch[start:start + batch_size], verbose=0)
            blocks.append(gem_pool(maps))
            done = min(start + batch_size, len(batch))
            if progress:
                progress(done, total)
        return l2_normalize(np.vstack(blocks))


class DINOv2Extractor:
    name = "dinov2"
    label = "DINOv2 ViT-B (CLS + mean patch)"

    def __init__(self, checkpoint: Optional[Path] = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self._device = torch_device()
        self._processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self._model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(self._device)
        self._head = None
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self._model.load_state_dict(payload["backbone"], strict=False)
            self._head = torch.nn.Linear(1536, int(payload["dim"]))
            self._head.load_state_dict(payload["head"])
            self._head.eval().to(self._device)

    def embed(self, paths: Sequence[Path], batch_size: int = 8, tta: bool = False, progress: Progress = None) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in paths]
        features = self._embed_images(images, batch_size, progress, len(paths))
        if tta:
            flipped = self._embed_images(
                [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images],
                batch_size,
                None,
                len(paths),
            )
            features = l2_normalize(features + flipped)
        return features

    def _embed_images(self, images: List[Image.Image], batch_size: int, progress: Progress, total: int) -> np.ndarray:
        torch = self._torch
        blocks = []
        done = 0
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                inputs = self._processor(images=chunk, return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                tokens = self._model(**inputs).last_hidden_state
                descriptor = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
                if self._head is not None:
                    descriptor = self._head(descriptor)
                descriptor = torch.nn.functional.normalize(descriptor, dim=1)
                blocks.append(descriptor.cpu().numpy().astype(np.float32))
                done = min(start + batch_size, len(images))
                if progress:
                    progress(done, total)
        return l2_normalize(np.vstack(blocks))


class FineTunedDINOv2Extractor(DINOv2Extractor):
    name = "dinov2-ft"
    label = "DINOv2 fine-tuned (triplet)"

    def __init__(self) -> None:
        if not FINETUNE_CHECKPOINT.exists():
            raise FileNotFoundError(
                "No fine-tuned checkpoint. Run finetune.py before using dinov2-ft."
            )
        super().__init__(checkpoint=FINETUNE_CHECKPOINT)


class CLIPExtractor:
    name = "clip"
    label = "CLIP ViT-B/32"

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._device = torch_device()
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(self._device)

    def embed(self, paths: Sequence[Path], batch_size: int = 16, tta: bool = False, progress: Progress = None) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in paths]
        features = self._embed_images(images, batch_size, progress, len(paths))
        if tta:
            flipped = self._embed_images(
                [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images],
                batch_size,
                None,
                len(paths),
            )
            features = l2_normalize(features + flipped)
        return features

    def embed_text(self, text: str) -> np.ndarray:
        torch = self._torch
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self._tensor(self._model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, dim=1)
        return l2_normalize(features.cpu().numpy().astype(np.float32))

    def _embed_images(self, images: List[Image.Image], batch_size: int, progress: Progress, total: int) -> np.ndarray:
        torch = self._torch
        blocks = []
        done = 0
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                inputs = self._processor(images=chunk, return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items() if key != "input_ids"}
                features = self._tensor(self._model.get_image_features(**inputs))
                features = torch.nn.functional.normalize(features, dim=1)
                blocks.append(features.cpu().numpy().astype(np.float32))
                done = min(start + batch_size, len(images))
                if progress:
                    progress(done, total)
        return l2_normalize(np.vstack(blocks))

    @staticmethod
    def _tensor(output):
        import torch

        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple):
            for value in reversed(output):
                if torch.is_tensor(value) and value.ndim == 2:
                    return value
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                return value
        raise TypeError(f"CLIP returned {type(output)}, not a feature tensor")


_CACHE = {}


def is_loaded(name: str) -> bool:
    return name in _CACHE


def release_others(keep: str) -> None:
    """Drop every cached extractor except one, so two DINOv2 copies are not both resident."""
    import gc

    dropped = [name for name in list(_CACHE) if name != keep]
    for name in dropped:
        del _CACHE[name]
    if not dropped:
        return
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        return


def get_extractor(name: str):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {', '.join(MODELS)}.")
    if name not in _CACHE:
        _CACHE[name] = {
            "vgg19": VGG19Extractor,
            "dinov2": DINOv2Extractor,
            "dinov2-ft": FineTunedDINOv2Extractor,
            "clip": CLIPExtractor,
        }[name]()
    return _CACHE[name]


def _worker(model_name: str) -> None:
    """Embed in a process that has not imported FAISS.

    Loading the fine-tuned checkpoint after FAISS has initialised segfaults
    on this machine. The studio already uses FAISS, so it sends images here.
    """
    extractor = get_extractor(model_name)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        features = extractor.embed(
            [Path(path) for path in request["paths"]],
            batch_size=int(request.get("batch", 8)),
            tta=bool(request.get("tta", False)),
        )
        np.save(request["out"], features)
        sys.stdout.write(json.dumps({"ok": True, "shape": list(features.shape)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed images outside the FAISS process")
    parser.add_argument("--worker", choices=MODELS, required=True)
    args = parser.parse_args()
    _worker(args.worker)
