# Image Similarity Recognition

**VGG19 Feature Extraction + FAISS Indexing**

Deep learning-based image similarity system that finds visually similar images using pretrained VGG19 features and FAISS nearest-neighbour search.

## Pipeline

| Step | Description |
|------|-------------|
| **1. Image loading** | Collect images from specified folders for processing |
| **2. Feature extraction** | Pretrained VGG19 analyses each image → feature vector (colours, shapes, textures) |
| **3. Normalization** | L2-normalize vectors to unit length (compare direction, not magnitude) |
| **4. Indexing** | FAISS indexes the normalized feature vectors |
| **5. Similarity matching** | Compare query vector against index; smaller distance = higher similarity |
| **6. Results display** | Show top matches with distance / similarity scores |

### VGG19

Breaks images into numerical feature vectors:

- **Early layers** – edges and colours  
- **Middle layers** – shapes and textures  
- **Deeper layers** – objects and scenes  

### FAISS

Library for efficient similarity search / clustering of dense vectors.

### Benefits

1. **High accuracy & efficiency** – pretrained on ImageNet; automates evaluation  
2. **Scalability** – easy to update as new images are added  
3. **Robustness** – handles variation in lighting, scale, and orientation  

## Studio

The interface is a local site:

- **Studio** — drop, paste, or pick a query. Choose VGG19, DINOv2, or CLIP. Optional query expansion and geometric re-ranking. CLIP also accepts a text query.
- **Compare** — one photograph searched against every model that already has an index.
- **Evaluation** — Precision@k, Recall@k, mAP, and nDCG@10 on the Caltech-101 subset, including the original VGG19 baseline.
- **Library** — add or remove photographs and rebuild the selected model’s index.
- **Method** — the six pipeline stages.

```bash
source venv/bin/activate
python app.py
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787).

The first launch downloads VGG19’s ImageNet weights and keeps the model in memory. Search stays locked until that finishes. The status pill in the header shows whether the model and the index are ready.

## Setup

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Folder layout

```
Image Similarity Recognition/
├── image_similarity.py      # Main system
├── requirements.txt
├── README.md
├── images/
│   ├── database/            # Put your database images here
│   └── query/               # Put query images here
├── index/                   # Generated FAISS index (created on build)
└── results/                 # Similarity result figures + JSON
```

## Usage

### 1. Add images

Place database images in `images/database/` and query images in `images/query/` (subfolders are fine).

### 2. Build the index

```bash
python image_similarity.py build --database images/database --model vgg19
```

`--model` is `vgg19`, `dinov2`, or `clip`. Each model writes `index/<model>/`. Flip test-time augmentation and PCA whitening are on unless you pass `--no-tta` or `--no-pca`.

### 3. Search

Single query:

```bash
python image_similarity.py query --image images/query/your_image.jpg --top-k 5
```

All images in a folder:

```bash
python image_similarity.py query-folder --folder images/query --top-k 5
```

Results are saved under `results/` as a plot (PNG) and a JSON summary.

### 4. Evaluate on Caltech-101

```bash
python scripts/prepare_caltech101.py
python evaluate.py --model vgg19 --variant baseline
python evaluate.py --model vgg19 --variant all
```

`baseline` is the original global-average-pool VGG19 index. `all` scores the current extractor plain, with flip augmentation and whitening, then with query expansion, then with geometric re-ranking.

### 5. Fine-tune DINOv2

```bash
python finetune.py
python evaluate.py --model dinov2-ft --variant plain
```

`finetune.py` trains the last two DINOv2 blocks and a 256-d projection with triplet loss. Only the database images are used. The held-out queries stay out of training. The checkpoint is `models/dinov2-ft.pt`, available in the studio as DINOv2 fine-tuned.

## Programmatic use

```python
from image_similarity import build_database, query_image

# Build index once
build_database("images/database")

# Query
results = query_image("images/query/cat.jpg", top_k=5)
for path, distance, similarity in results:
    print(path.name, similarity)
```
