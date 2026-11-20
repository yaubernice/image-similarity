"""
Image Similarity Recognition
VGG19 Feature Extraction + FAISS Indexing

Pipeline:
  1. Image loading      – collect images from specified folders
  2. Feature extraction – pretrained VGG19 → feature vectors
  3. Normalization      – L2-normalize vectors (unit length)
  4. Indexing           – FAISS index of normalized features
  5. Similarity matching – nearest-neighbour search by distance
  6. Results display    – show most similar images + similarity scores
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from tqdm import tqdm

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}

# Default paths
DEFAULT_DATABASE_DIR = Path("images/database")
DEFAULT_QUERY_DIR = Path("images/query")
DEFAULT_INDEX_DIR = Path("index")
DEFAULT_RESULTS_DIR = Path("results")


# ─────────────────────────────────────────────
# 1. Image loading
# ─────────────────────────────────────────────

def collect_images(folder: Path) -> List[Path]:
    """Recursively collect image paths from a folder."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {folder}")
    return images


# ─────────────────────────────────────────────
# 2. Feature extraction (VGG19)
# ─────────────────────────────────────────────

def build_feature_extractor() -> Model:
    """
    Build a VGG19-based feature extractor.

    Uses the pretrained ImageNet weights and removes the final
    classification layers, taking the output of the last pooling
    layer (fc2 precursor) as a 4096-d feature vector via include_top=True
    and cutting before the softmax, OR using avg-pooled conv features.

    Here we use VGG19 with include_top=False and GlobalAveragePooling,
    giving a compact 512-d descriptor that captures shapes, textures,
    colours and high-level concepts from the deeper layers.
    """
    base = VGG19(weights="imagenet", include_top=False, pooling="avg")
    # Freeze weights – we only extract features, no training
    base.trainable = False
    return base


def load_and_preprocess(img_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load an image and preprocess it for VGG19."""
    img = keras_image.load_img(img_path, target_size=target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def extract_features(
    model: Model,
    img_paths: List[Path],
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Extract feature vectors for a list of images.

    Early VGG19 layers capture edges/colours; middle layers shapes/textures;
    deeper layers high-level objects and scenes. The pooled output encodes
    all of these into a single dense vector per image.
    """
    if not img_paths:
        raise ValueError("No images to extract features from")

    features = []
    total = len(img_paths)
    processed = 0
    for i in tqdm(range(0, total, batch_size), desc="Extracting features"):
        batch_paths = img_paths[i : i + batch_size]
        batch = np.vstack([load_and_preprocess(p) for p in batch_paths])
        feats = model.predict(batch, verbose=0)
        features.append(feats)
        processed += len(batch_paths)
        if progress_callback:
            progress_callback(processed, total)

    return np.vstack(features).astype(np.float32)


# ─────────────────────────────────────────────
# 3. Normalization
# ─────────────────────────────────────────────

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    L2-normalize feature vectors so each has length 1.

    This focuses comparison on direction (angular similarity)
    rather than magnitude, ensuring fair distance comparisons.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    return features / norms


# ─────────────────────────────────────────────
# 4. FAISS indexing
# ─────────────────────────────────────────────

def build_faiss_index(features: np.ndarray, metric: str = "l2") -> faiss.Index:
    """
    Build an exact FAISS index over the feature vectors.

    "l2" is Euclidean distance (smaller is closer). "ip" is inner product,
    which is cosine similarity when the vectors are L2-normalised
    (larger is closer). On unit vectors the two rankings match.
    """
    dim = features.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unknown FAISS metric: {metric}")
    index.add(np.ascontiguousarray(features, dtype=np.float32))
    return index


def save_index(
    index: faiss.Index,
    image_paths: List[Path],
    features: np.ndarray,
    index_dir: Path,
    model: str = "VGG19 (ImageNet, avg-pooled)",
    metric: str = "l2",
    extra: Optional[dict] = None,
) -> None:
    """Persist FAISS index, image paths, and features to disk."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "faiss.index"))

    with open(index_dir / "image_paths.pkl", "wb") as f:
        pickle.dump([str(p) for p in image_paths], f)

    np.save(index_dir / "features.npy", features)

    meta = {
        "num_images": len(image_paths),
        "feature_dim": int(features.shape[1]),
        "model": model,
        "metric": metric,
    }
    if extra:
        meta.update(extra)
    with open(index_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Index saved → {index_dir}  ({len(image_paths)} images, dim={features.shape[1]})")


def load_index(index_dir: Path) -> Tuple[faiss.Index, List[Path], np.ndarray]:
    """Load a previously saved FAISS index and metadata."""
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))

    with open(index_dir / "image_paths.pkl", "rb") as f:
        paths = [Path(p) for p in pickle.load(f)]

    features = np.load(index_dir / "features.npy")
    return index, paths, features


# ─────────────────────────────────────────────
# 5. Similarity matching
# ─────────────────────────────────────────────

def search_similar(
    model: Model,
    index: faiss.Index,
    image_paths: List[Path],
    query_path: Path,
    top_k: int = 5,
) -> List[Tuple[Path, float, float]]:
    """
    Compare a query image against the indexed database.

    Returns a list of (image_path, distance, similarity) sorted by
    ascending distance. Smaller distance ⇒ higher similarity.
    Similarity is mapped to [0, 1] via  1 / (1 + distance).
    """
    query_feat = extract_features(model, [query_path])
    query_feat = normalize_features(query_feat)

    distances, indices = index.search(query_feat, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        similarity = 1.0 / (1.0 + float(dist))
        results.append((image_paths[idx], float(dist), similarity))

    return results


# ─────────────────────────────────────────────
# 6. Results display
# ─────────────────────────────────────────────

def display_results(
    query_path: Path,
    results: List[Tuple[Path, float, float]],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Display the query image alongside its top similar matches."""
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.5 * (n + 1), 4))
    if n == 0:
        print("No results to display.")
        return

    # Query
    axes[0].imshow(Image.open(query_path).convert("RGB"))
    axes[0].set_title("Query", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Matches
    for ax, (path, dist, sim) in zip(axes[1:], results):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(f"{path.name}\nsim={sim:.3f}  dist={dist:.3f}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Image Similarity Results (VGG19 + FAISS)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Results figure saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_results(query_path: Path, results: List[Tuple[Path, float, float]]) -> None:
    """Print similarity results to the console."""
    print(f"\n{'─' * 60}")
    print(f"Query: {query_path}")
    print(f"{'─' * 60}")
    print(f"{'Rank':<6}{'Image':<40}{'Distance':>10}{'Similarity':>12}")
    print(f"{'─' * 60}")
    for rank, (path, dist, sim) in enumerate(results, 1):
        print(f"{rank:<6}{path.name:<40}{dist:>10.4f}{sim:>12.4f}")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────
# High-level pipelines
# ─────────────────────────────────────────────

def build_database(
    database_dir: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    batch_size: int = 16,
    model_name: str = "vgg19",
    tta: bool = True,
    pca: bool = True,
) -> None:
    """Encode a folder and store one inner-product index for that model."""
    from extractors import get_extractor
    from retrieval import apply_whitening, fit_whitening, save_whitening

    print("═══ Building image database index ═══")
    print(f"Database folder: {database_dir}")
    print(f"Model: {model_name}  tta={tta}  pca={pca}")

    image_paths = [path.resolve() for path in collect_images(database_dir)]
    print(f"Found {len(image_paths)} images")
    extractor = get_extractor(model_name)
    features = extractor.embed(image_paths, batch_size=batch_size, tta=tta)
    print(f"Feature matrix shape: {features.shape}")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    if pca:
        state = fit_whitening(features)
        features = apply_whitening(features, state)
        save_whitening(index_dir / "pca.npz", state)
        print(f"Whitened to {features.shape[1]} dimensions")

    index = build_faiss_index(features, metric="ip")
    print(f"FAISS index built – {index.ntotal} vectors")
    save_index(
        index,
        image_paths,
        features,
        index_dir,
        model=extractor.label,
        metric="ip",
        extra={
            "model_key": model_name,
            "tta": tta,
            "pca": pca,
            "pooling": "gem" if model_name == "vgg19" else "native",
        },
    )


def query_image(
    query_path: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    top_k: int = 5,
    show: bool = True,
) -> List[Tuple[Path, float, float]]:
    """Search for images similar to a query image."""
    print("═══ Similarity search ═══")
    query_path = Path(query_path)
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    index, image_paths, features = load_index(index_dir)
    meta_path = Path(index_dir) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("model_key"):
        from extractors import get_extractor
        from retrieval import apply_whitening, cosine_to_l2, load_whitening, similarity_from_l2

        extractor = get_extractor(meta["model_key"])
        vector = extractor.embed([query_path], tta=bool(meta.get("tta", True)))
        pca_path = Path(index_dir) / "pca.npz"
        if meta.get("pca") and pca_path.exists():
            vector = apply_whitening(vector, load_whitening(pca_path))
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        scores, ids = index.search(np.ascontiguousarray(vector), top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if meta.get("metric") == "ip":
                distance = float(cosine_to_l2(float(score)))
            else:
                distance = float(score)
            results.append((image_paths[int(idx)], distance, similarity_from_l2(distance)))
    else:
        model = build_feature_extractor()
        results = search_similar(model, index, image_paths, query_path, top_k=top_k)

    # 6. Display
    print_results(query_path, results)

    save_path = Path(results_dir) / f"result_{query_path.stem}.png"
    display_results(query_path, results, save_path=save_path, show=show)

    # Also save a JSON summary
    summary = {
        "query": str(query_path),
        "results": [
            {"rank": i, "image": str(p), "distance": d, "similarity": s}
            for i, (p, d, s) in enumerate(results, 1)
        ],
    }
    json_path = Path(results_dir) / f"result_{query_path.stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved → {json_path}")

    return results


def query_folder(
    query_dir: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    top_k: int = 5,
    show: bool = False,
) -> None:
    """Run similarity search for every image in a query folder."""
    queries = collect_images(query_dir)
    print(f"Processing {len(queries)} query image(s)…")
    for q in queries:
        query_image(q, index_dir=index_dir, results_dir=results_dir, top_k=top_k, show=show)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image Similarity Recognition – VGG19 + FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build the FAISS index from a database folder
  python image_similarity.py build --database images/database

  # Search with a single query image
  python image_similarity.py query --image images/query/cat.jpg --top-k 5

  # Search with every image in a query folder
  python image_similarity.py query-folder --folder images/query --top-k 5
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build FAISS index from database images")
    p_build.add_argument("--database", type=Path, default=DEFAULT_DATABASE_DIR,
                         help="Folder of database images")
    p_build.add_argument("--model", default="vgg19", choices=["vgg19", "dinov2", "dinov2-ft", "clip"])
    p_build.add_argument("--index-dir", type=Path, default=None,
                         help="Where to save the FAISS index (default: index/<model>)")
    p_build.add_argument("--batch-size", type=int, default=16)
    p_build.add_argument("--no-tta", action="store_true")
    p_build.add_argument("--no-pca", action="store_true")

    # query
    p_query = sub.add_parser("query", help="Find similar images for one query")
    p_query.add_argument("--image", type=Path, required=True, help="Query image path")
    p_query.add_argument("--model", default="vgg19", choices=["vgg19", "dinov2", "dinov2-ft", "clip"])
    p_query.add_argument("--index-dir", type=Path, default=None)
    p_query.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--no-show", action="store_true", help="Don't open a plot window")

    # query-folder
    p_qfold = sub.add_parser("query-folder", help="Query every image in a folder")
    p_qfold.add_argument("--folder", type=Path, default=DEFAULT_QUERY_DIR)
    p_qfold.add_argument("--model", default="vgg19", choices=["vgg19", "dinov2", "dinov2-ft", "clip"])
    p_qfold.add_argument("--index-dir", type=Path, default=None)
    p_qfold.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p_qfold.add_argument("--top-k", type=int, default=5)
    p_qfold.add_argument("--show", action="store_true", help="Open plot windows")

    args = parser.parse_args()

    if args.command == "build":
        index_dir = args.index_dir or (DEFAULT_INDEX_DIR / args.model)
        build_database(
            args.database,
            index_dir=index_dir,
            batch_size=args.batch_size,
            model_name=args.model,
            tta=not args.no_tta,
            pca=not args.no_pca,
        )
    elif args.command == "query":
        query_image(
            args.image,
            index_dir=args.index_dir or (DEFAULT_INDEX_DIR / args.model),
            results_dir=args.results_dir,
            top_k=args.top_k,
            show=not args.no_show,
        )
    elif args.command == "query-folder":
        query_folder(
            args.folder,
            index_dir=args.index_dir or (DEFAULT_INDEX_DIR / args.model),
            results_dir=args.results_dir,
            top_k=args.top_k,
            show=args.show,
        )


if __name__ == "__main__":
    main()
