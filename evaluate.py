"""Score a retrieval model on the Caltech-101 subset.

The baseline variant uses the original VGG19 global-average-pool
features and an L2 index. Later variants go through extractors.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from metrics import summarise

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "caltech101"
RESULTS = ROOT / "results"


def load_ground_truth(path: Path) -> dict:
    return json.loads(path.read_text())


def ranking_rows(retrieved_lists: Sequence[Sequence[str]], queries: Sequence[dict]) -> List[dict]:
    rows = []
    for retrieved, query in zip(retrieved_lists, queries):
        rows.append({"retrieved": list(retrieved), "relevant": query["relevant"]})
    return rows


def search_order(scores: np.ndarray, metric: str) -> np.ndarray:
    """Return indexes from best to worst."""
    if metric == "ip":
        return np.argsort(-scores)
    return np.argsort(scores)


def evaluate_baseline(ground: dict, batch_size: int = 16) -> dict:
    """Original VGG19: squash to 224, global average pool, L2 index."""
    import image_similarity as sim

    database = [DATA / item for item in _database_paths(ground)]
    model = sim.build_feature_extractor()
    features = sim.normalize_features(sim.extract_features(model, database, batch_size=batch_size))
    index = sim.build_faiss_index(features, metric="l2")
    rels = [path.relative_to(DATA).as_posix() for path in database]

    retrieved_lists = []
    query_paths = [DATA / query["path"] for query in ground["queries"]]
    query_features = sim.normalize_features(
        sim.extract_features(model, query_paths, batch_size=batch_size)
    )
    distances, indices = index.search(query_features, len(database))
    for row_ids in indices:
        retrieved_lists.append([rels[int(i)] for i in row_ids if i >= 0])

    summary = summarise(ranking_rows(retrieved_lists, ground["queries"]))
    summary.update({
        "model": "vgg19",
        "variant": "baseline",
        "pooling": "gap",
        "tta": False,
        "pca": False,
        "aqe": False,
        "geometric": False,
        "metric": "l2",
        "feature_dim": int(features.shape[1]),
        "database": len(database),
    })
    return summary


def _database_paths(ground: dict) -> List[str]:
    seen = []
    known = set()
    for query in ground["queries"]:
        for path in query["relevant"]:
            if path not in known:
                known.add(path)
                seen.append(path)
    return seen


VARIANTS = {
    "plain": {"tta": False, "pca": False, "aqe": False, "geom": False},
    "tta_pca": {"tta": True, "pca": True, "aqe": False, "geom": False},
    "aqe": {"tta": True, "pca": True, "aqe": True, "geom": False},
    "geom": {"tta": True, "pca": True, "aqe": False, "geom": True},
}


def _cached_embed(extractor, paths: List[Path], tta: bool, batch_size: int, split: str) -> np.ndarray:
    cache_dir = DATA / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{extractor.name}_{split}_{'tta' if tta else 'plain'}.npz"
    rels = np.array([str(path) for path in paths])
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=False)
        if list(saved["paths"]) == list(rels) and saved["features"].shape[0] == len(paths):
            print(f"Using cached {extractor.name} {split} features ({'tta' if tta else 'no tta'})")
            return saved["features"]
    print(f"Embedding {len(paths)} {split} images with {extractor.label}")
    features = extractor.embed(paths, batch_size=batch_size, tta=tta)
    np.savez(cache_path, paths=rels, features=features)
    return features


def evaluate_variant(ground: dict, model_name: str, variant: str, batch_size: int, data_root: Path) -> dict:
    from extractors import get_extractor
    from retrieval import (
        AQE_K,
        apply_whitening,
        expand_query,
        fit_whitening,
        rerank_geometric,
    )

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {', '.join(VARIANTS)}.")
    flags = VARIANTS[variant]
    database_rels = _database_paths(ground)
    database_paths = [data_root / rel for rel in database_rels]
    query_paths = [data_root / query["path"] for query in ground["queries"]]
    extractor = get_extractor(model_name)
    database_features = _cached_embed(extractor, database_paths, flags["tta"], batch_size, "database")
    query_features = _cached_embed(extractor, query_paths, flags["tta"], batch_size, "queries")
    if flags["pca"]:
        state = fit_whitening(database_features)
        database_features = apply_whitening(database_features, state)
        query_features = apply_whitening(query_features, state)

    # Exact inner product. FAISS in this same process segfaults once PyTorch
    # has initialised, and 360 vectors do not need an approximate index.
    similarities = query_features @ database_features.T
    indices = np.argsort(-similarities, axis=1)
    retrieved_lists = []
    for query_index, query in enumerate(ground["queries"]):
        order = [int(item) for item in indices[query_index]]
        cosines = similarities[query_index]
        vector = query_features[query_index]
        if flags["aqe"]:
            top = order[:AQE_K]
            vector = expand_query(vector, database_features[top], cosines[top])
            new_scores = vector @ database_features.T
            order = [int(item) for item in np.argsort(-new_scores)]
            cosines = new_scores
        if flags["geom"]:
            order = rerank_geometric(order, cosines, query_paths[query_index], database_paths)
        retrieved_lists.append([database_rels[item] for item in order])

    summary = summarise(ranking_rows(retrieved_lists, ground["queries"]))
    summary.update({
        "model": model_name,
        "variant": variant,
        "pooling": "gem" if model_name == "vgg19" else "triplet" if model_name == "dinov2-ft" else "native",
        "tta": flags["tta"],
        "pca": flags["pca"],
        "aqe": flags["aqe"],
        "geometric": flags["geom"],
        "metric": "ip",
        "feature_dim": int(database_features.shape[1]),
        "database": len(database_paths),
    })
    return summary


def write_summary(summary: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"eval_{summary['model']}_{summary['variant']}.json"
    path = RESULTS / name
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image retrieval on Caltech-101")
    parser.add_argument("--model", default="vgg19", choices=["vgg19", "dinov2", "dinov2-ft", "clip"])
    parser.add_argument(
        "--variant",
        default="baseline",
        help="baseline, plain, tta_pca, aqe, geom, or all",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ground-truth", type=Path, default=DATA / "ground_truth.json")
    args = parser.parse_args()

    ground = load_ground_truth(args.ground_truth)
    if args.variant == "all":
        variants = ["plain", "tta_pca", "aqe", "geom"]
    else:
        variants = [args.variant]
    for variant in variants:
        if variant == "baseline":
            if args.model != "vgg19":
                raise SystemExit("The baseline variant is the original VGG19 GAP model.")
            summary = evaluate_baseline(ground, batch_size=args.batch_size)
        else:
            summary = evaluate_variant(
                ground,
                model_name=args.model,
                variant=variant,
                batch_size=args.batch_size,
                data_root=DATA,
            )
        write_summary(summary)


if __name__ == "__main__":
    main()
