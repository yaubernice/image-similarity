# Image Similarity

Visual search studio for photographs, comparing VGG19, DINOv2, and CLIP side by side.

## Overview

The project finds visually similar images with deep descriptors and exact nearest-neighbour search. A FastAPI studio lets you switch between VGG19, DINOv2, CLIP, and a fine-tuned DINOv2, rebuild per-model indexes, and score retrieval on a labelled Caltech-101 subset. The original VGG19 global-average-pool path is kept as a measured baseline.

## Features

- Studio UI for query upload, library management, model compare, and evaluation
- Pluggable extractors: VGG19 (GeM), DINOv2, CLIP (image + text), DINOv2 fine-tuned
- Flip test-time augmentation, PCA whitening, α-query expansion, ORB + RANSAC re-ranking
- Separate FAISS `IndexFlatIP` per model
- Precision@k, Recall@k, mAP, and nDCG@10 on held-out Caltech queries
- Triplet fine-tuning of DINOv2 on database images only

## Stack

- Python 3.12, FastAPI, Uvicorn
- TensorFlow (VGG19), PyTorch + Transformers (DINOv2, CLIP)
- FAISS, OpenCV, NumPy, Pillow
- pytorch-metric-learning for fine-tuning

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Start the studio:

```bash
source venv/bin/activate
python app.py
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787).

- **Studio** — drop, paste, or pick a query. Choose a model. Optional query expansion and geometric re-ranking. CLIP also accepts a text query.
- **Compare** — one photograph searched against every model that already has an index.
- **Evaluation** — Caltech-101 metrics, including the original VGG19 baseline.
- **Library** — add or remove photographs and rebuild the selected model’s index.
- **Method** — the six pipeline stages.

The first launch may download model weights. Search stays locked until the active model is ready. The status pill shows whether the model and index are ready.

### CLI

Place database images in `images/database/` and queries in `images/query/`.

```bash
python image_similarity.py build --database images/database --model vgg19
python image_similarity.py query --image images/query/your_image.jpg --top-k 5
python image_similarity.py query-folder --folder images/query --top-k 5
```

`--model` is `vgg19`, `dinov2`, `dinov2-ft`, or `clip`. Each model writes `index/<model>/`. Flip TTA and PCA whitening are on unless you pass `--no-tta` or `--no-pca`.

### Evaluate on Caltech-101

```bash
python scripts/prepare_caltech101.py
python evaluate.py --model vgg19 --variant baseline
python evaluate.py --model vgg19 --variant all
```

`baseline` is the original global-average-pool VGG19 index. `all` scores plain, flip + whitening, query expansion, and geometric re-ranking.

### Fine-tune DINOv2

```bash
python finetune.py
python evaluate.py --model dinov2-ft --variant plain
```

Trains the last two DINOv2 blocks and a 256-d projection with triplet loss. Only database images are used. Queries stay out of training. The checkpoint is `models/dinov2-ft.pt`.

### Programmatic use

```python
from image_similarity import build_database, query_image

build_database("images/database")
results = query_image("images/query/cat.jpg", top_k=5)
for path, distance, similarity in results:
    print(path.name, similarity)
```

## Pipeline

| Step | Description |
|------|-------------|
| **1. Image loading** | Collect images from the library folders |
| **2. Feature extraction** | VGG19 GeM, DINOv2, CLIP, or fine-tuned DINOv2 → descriptor |
| **3. Normalization** | L2-normalize; optional flip average and PCA whitening |
| **4. Indexing** | FAISS `IndexFlatIP` per model |
| **5. Similarity matching** | Inner product on unit vectors; optional αQE and geometric re-rank |
| **6. Results** | Ranked neighbours with distance and similarity `1 / (1 + L2)` |

## Project layout

```
├── app.py                 # FastAPI studio
├── image_similarity.py    # CLI build / query
├── extractors.py          # VGG19, DINOv2, CLIP, dinov2-ft
├── retrieval.py           # Whitening, αQE, geometric re-rank
├── metrics.py             # Precision, recall, mAP, nDCG
├── evaluate.py            # Caltech-101 evaluation
├── finetune.py            # Triplet fine-tune for DINOv2
├── static/                # Studio UI
├── scripts/               # Dataset preparation
├── images/database/       # Library photographs
├── images/query/          # Sample queries
├── index/                 # Per-model FAISS indexes (generated)
└── results/               # Evaluation JSON and plots
```
